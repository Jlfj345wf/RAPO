import embeddings
from cupy_utils import *
import argparse
import collections
import numpy as np
import sys
import pickle
import time
import torch
import random
from dssm_trainer import DssmTrainer
import faiss
import json
import copy
import os

def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    if supports_cupy():
        xp = get_cupy()
        xp.random.seed(seed)

def get_NN(src, X, Z, num_NN, cuda = False, batch_size = 100, return_scores = True, method="cos"):

    def get_faiss_index(embeddings, nlist=250, nprobe=150):
        d = embeddings.shape[1]
        ngpus = faiss.get_num_gpus()
        print("number of GPUs:", ngpus)
        cpu_index = faiss.IndexFlatL2(d)
        cpu_index = faiss.IndexIVFFlat(cpu_index, d, nlist, faiss.METRIC_INNER_PRODUCT)
        gpu_index = faiss.index_cpu_to_all_gpus(  # build the index
            cpu_index
        )
        gpu_index.train(embeddings)
        gpu_index.nprobe = nprobe
        gpu_index.add(embeddings)
        return gpu_index

    embeddingsx = asnumpy(X)
    embeddingsy = asnumpy(Z)
    tgt_embs_index = get_faiss_index(embeddingsy)
    sim_x2y_index = {}
    nq = len(src)
    begin_id = 0
    while begin_id < nq:
        batch_src = src[begin_id:begin_id+batch_size]
        dis, ind = tgt_embs_index.search(embeddingsx[batch_src], num_NN)
        dis = dis.tolist()
        ind = ind.tolist()
        for i in range(len(batch_src)):
            if return_scores:
                sim_x2y_index[batch_src[i]] = [(ind[i][_], dis[i][_]) for _ in range(num_NN)]
            else:
                sim_x2y_index[batch_src[i]] = [ind[i][_] for _ in range(num_NN)]
        begin_id += batch_size  
    del embeddingsx, embeddingsy, tgt_embs_index
    return sim_x2y_index  

def generate_negative_samplewise_examples_v2(positive_examples, src_w2ind, tar_w2ind, x, z, top_k, num_neg_per_pos, hard_neg_random=True, method="cos"):
  l = len(positive_examples)
  pos_pair2neg_words = collections.defaultdict(list)

  pos_src_word_indexes = [src_w2ind[e[0]] for e in positive_examples]
  pos_tar_word_indexes = [tar_w2ind[e[1]] for e in positive_examples]
  
  # src_word_ind -> tgt_word_ind:set 的dict
  correct_mapping = collections.defaultdict(set)
  for s, t in zip(pos_src_word_indexes, pos_tar_word_indexes):
    correct_mapping[s].add(t)

  # nns是一个dict，src_word_ind -> (tgt_word_ind, sim_score) 的topk list
  # nns = get_NN(pos_src_word_indexes, x, z, top_k, cuda = True, batch_size = 100, return_scores = True)
  tgt_nns = get_NN(list(set(pos_tar_word_indexes)), z, z, top_k, cuda = True, batch_size = 100, return_scores = True, method=method)
  
  # 对于训练数据中pos_src -> pos_tgt1, pos_tgt2,... 
  # 将mean(pos_tgt)在tgt语言空间内最近邻作为pos_src的负例
  # 为每个(pos_src, pos_tgt) pair挑num_neg_per_pos个负例
  for src_ind, tgt_ind in zip(pos_src_word_indexes, pos_tar_word_indexes):
    tgt_nn_wi = tgt_nns[tgt_ind][:top_k]
    tgt_nn_wi_filter_lowscore = [(w, s) for w, s in tgt_nn_wi]
    # 去掉groundtruth与重复单词
    candidate_neg_wi = [(w, s) for w, s in tgt_nn_wi_filter_lowscore if w not in correct_mapping[src_ind]]
    # 采样num_neg_per_pos个单词
    if not hard_neg_random:
      hard_neg_inds_sample = random.sample(candidate_neg_wi, num_neg_per_pos)
    else:
      hard_neg_inds_sample = candidate_neg_wi
    pos_pair2neg_words[(src_ind, tgt_ind)] = hard_neg_inds_sample

  return pos_pair2neg_words, None

def get_adapter_feat(xw, k=100, threshold=0.9):
  xp = get_array_module(xw)
  x_sim_topn = get_NN(list(range(xw.shape[0])), xw, xw, k, cuda = True, batch_size = 100, return_scores = True)
  x_cluster_feat = []
  for i in range(len(x_sim_topn)):
    knn_list = [wi for wi, si in x_sim_topn[i] if si >= threshold]
    x_cluster_feat.append(xp.mean(xw[knn_list], 0))
  x_cluster_feat = xp.array(x_cluster_feat)
  return x_cluster_feat


def expand_dict(best_model, torch_src, torch_tgt, torch_src_afeat, torch_tgt_afeat, expand_dict_size=10000, expand_rank=20000):

    src_index_list = list(range(expand_rank))
    tgt_index_list = list(range(expand_rank))

    csls_s2t_index = best_model.predict(torch_src, torch_tgt, src_index_list, torch_src_afeat, torch_tgt_afeat)
    csls_t2s_index = best_model.predict_y2x(torch_src, torch_tgt, tgt_index_list, torch_src_afeat, torch_tgt_afeat)
    add_train_ind_pair = []
    for i in src_index_list:
      t_y_index = csls_s2t_index[i][0]
      if t_y_index >= expand_rank:
        continue
      if csls_t2s_index[t_y_index][0] == i:
        add_train_ind_pair.append((i, t_y_index))
    
    return add_train_ind_pair

def load_model(model_file_path):
    with(open(model_file_path, "rb")) as infile:
        model = pickle.load(infile)   
    return model

def run_dssm_trainning(args, is_optuna=False):

  print(args)
  set_seed(args.random_seed)
  SL_start_time = time.time()

  # 对于每个src，从tgt单词的cos相似度最高的neg_top_k个单词中随机采样neg_per_pos个
  neg_top_k = args.hard_neg_top_k
  debug = args.debug

  # load up the embeddings
  print("Loading embeddings from disk ...")
  dtype = "float32"
  srcfile = open(args.in_src, encoding="utf-8", errors='surrogateescape')
  trgfile = open(args.in_tar, encoding="utf-8", errors='surrogateescape')
  src_words, x = embeddings.read(srcfile, 200000, dtype=dtype)
  trg_words, z = embeddings.read(trgfile, 200000, dtype=dtype)

  # load the supervised dictionary
  src_word2ind = {word: i for i, word in enumerate(src_words)}
  trg_word2ind = {word: i for i, word in enumerate(trg_words)}
  src_ind2word = {i: word for i, word in enumerate(src_words)}
  trg_ind2word = {i: word for i, word in enumerate(trg_words)}

  # 读入训练集
  pos_examples = []
  f = open(args.train_dict, encoding="utf-8", errors='surrogateescape')
  for line in f:
      src, trg = [_.lower().strip() for _ in line.split()]
      if src in src_word2ind and trg in trg_word2ind:
          pos_examples.append((src,trg))
      else:
          print(src, src in src_word2ind)
          print(trg, trg in trg_word2ind)

  val_examples = []
  f = open(args.val_dict, encoding="utf-8", errors='surrogateescape')
  for line in f:
      src, trg = [_.lower().strip() for _ in line.split()]
      if src in src_word2ind and trg in trg_word2ind:
          val_examples.append((src,trg))

  init_pos_examples = copy.deepcopy(pos_examples)

  # 调用vecmap
  # call artetxe to get the initial alignment on the initial train dict
  xw, zw = x, z

  embeddings.normalize(xw, ['unit', 'center', 'unit'])
  embeddings.normalize(zw, ['unit', 'center', 'unit'])

  x_adapter_feat = None
  z_adapter_feat = None
  if args.adapter_type != "none":
    x_adapter_feat = get_adapter_feat(xw, k=args.adapter_src_cluster_k, threshold=args.adapter_src_cluster_threshold)
    z_adapter_feat = get_adapter_feat(zw, k=args.adapter_tgt_cluster_k, threshold=args.adapter_tgt_cluster_threshold)
    print(x_adapter_feat.shape)
    print(z_adapter_feat.shape)

  # 生成负例,hard neg examples
  # 返回的是负例word pair的list
  # generate negative examples for the current


  with torch.no_grad():
    torch_xw = torch.from_numpy(asnumpy(xw))
    torch_zw = torch.from_numpy(asnumpy(zw))
    if args.adapter_type != "none":
      x_adapter_feat = torch.from_numpy(asnumpy(x_adapter_feat))
      z_adapter_feat = torch.from_numpy(asnumpy(z_adapter_feat))

  if args.load_model != -1 and os.path.exists(f"{args.model_filename}_{args.load_model}"):
    best_model = load_model(f"{args.model_filename}_{args.load_model}")
    print(f"load model from {args.model_filename}_{args.load_model}")
    val_set = [[src_word2ind[_s], trg_word2ind[_t]] for _s, _t in val_examples]
    eval_src2tgts = collections.defaultdict(set)
    for s, t in val_set:
      eval_src2tgts[s].add(t)
    eval_src = list(set([_[0] for _ in val_set])) 
    acc = best_model.eval(torch_xw, torch_zw, eval_src, eval_src2tgts, src_afeat=x_adapter_feat, tgt_afeat=z_adapter_feat)
    print(f"load model top1 acc in val: {acc}")
    new_index_pairs = expand_dict(best_model, torch_xw, torch_zw, x_adapter_feat, z_adapter_feat)
    dump_pair = 0
    new_pair = 0
    pos_examples = copy.deepcopy(init_pos_examples)
    for s_i, t_i in new_index_pairs:
      w_pair = (src_ind2word[s_i], trg_ind2word[t_i])
      if w_pair in init_pos_examples:
        dump_pair += 1
      else:
        new_pair += 1
      pos_examples.append(w_pair)
    print(f"expand dict count {len(new_index_pairs)} : dump pair count {dump_pair} | new pair count {new_pair} ")
    print(f"pos examples size: {len(init_pos_examples)} -> {len(pos_examples)} ")    
  if args.load_model != -1:
    T = args.load_model
  else:
    T = 5
  while T > 0:
    T -= 1
    this_iter_model_file_name = f"{args.model_filename}_{T}"
    train_set = [[src_word2ind[_s], trg_word2ind[_t]] for _s, _t in pos_examples] 
    val_set = [[src_word2ind[_s], trg_word2ind[_t]] for _s, _t in val_examples]

    print("train data size: ", len(pos_examples))
    print("test data size: ", len(val_examples))

    # pos_examples是word piar的list
    src_indices = [src_word2ind[t[0]] for t in pos_examples]
    trg_indices = [trg_word2ind[t[1]] for t in pos_examples]

    print("unique source words in train data: ", len(set(src_indices)))

    val_src_indices = [src_word2ind[t[0]] for t in val_examples]
    val_trg_indices = [trg_word2ind[t[1]] for t in val_examples] 

    print("Generating negative examples ...")  
    src2negtgts, tgt2negsrcs = generate_negative_samplewise_examples_v2(pos_examples, 
                                            src_word2ind, 
                                            trg_word2ind, 
                                            copy.deepcopy(xw), 
                                            copy.deepcopy(zw), 
                                            top_k = neg_top_k, 
                                            num_neg_per_pos = args.hard_neg_per_pos,
                                            hard_neg_random = args.hard_neg_random,
                                            method = args.hard_sim_method)
    
    # 去掉score，score是用来debug的
    if not args.hard_neg_random_with_prob:
      for key in src2negtgts:
        src2negtgts[key] = [_[0] for _ in src2negtgts[key]]
      if tgt2negsrcs is not None:
        for key in tgt2negsrcs:
          tgt2negsrcs[key] = [_[0] for _ in tgt2negsrcs[key]]

    if is_optuna:
      args.model_filename = None

    model = DssmTrainer(torch_xw.shape[1], 
                          torch_zw.shape[1], 
                          args.h_dim, 
                          random_neg_per_pos=args.random_neg_per_pos, 
                          epochs=args.train_epochs,
                          eval_every_epoch=args.eval_every_epoch,
                          shuffle_in_train=args.shuffle_in_train,
                          lr=args.lr,
                          train_batch_size=args.train_batch_size,
                          model_save_file=this_iter_model_file_name,
                          is_single_tower=args.is_single_tower,
                          hard_neg_per_pos=args.hard_neg_per_pos,
                          hard_neg_random=args.hard_neg_random,
                          update_neg_every_epoch=args.update_neg_every_epoch,
                          random_warmup_epoches=args.random_warmup_epoches,
                          loss_metric=args.loss_metric,
                          args=args)

    model.fit(torch_xw, torch_zw, train_set, src2negtgts, tgt2negsrcs, val_set, src_afeat=x_adapter_feat, tgt_afeat=z_adapter_feat)
    score = model.best_val_acc
    print(f"best model file name : {this_iter_model_file_name}")
    print(f"best score at semi iter: {5 - T} is {score}")
    if is_optuna:
      return model.best_val_acc
    # write res to disk
    # 保存xw, zw
    if T > 0:
      best_model = load_model(this_iter_model_file_name)
      new_index_pairs = expand_dict(best_model, torch_xw, torch_zw, x_adapter_feat, z_adapter_feat)
      dump_pair = 0
      new_pair = 0
      pos_examples = copy.deepcopy(init_pos_examples)
      for s_i, t_i in new_index_pairs:
        w_pair = (src_ind2word[s_i], trg_ind2word[t_i])
        if w_pair in init_pos_examples:
          dump_pair += 1
        else:
          new_pair += 1
        pos_examples.append(w_pair)
      print(f"expand dict count {len(new_index_pairs)} : dump pair count {dump_pair} | new pair count {new_pair} ")
      print(f"pos examples size: {len(init_pos_examples)} -> {len(pos_examples)} ")
    

if __name__ == "__main__":  

  parser = argparse.ArgumentParser(description='Run classification based self learning for aligning embedding spaces in two languages.')

  parser.add_argument('--train_dict', type=str, help='Name of the input dictionary file.', required = True)
  parser.add_argument('--val_dict', type=str, help='Name of the input dictionary file.', required = True)
  parser.add_argument('--in_src', type=str, help='Name of the input source languge embeddings file.', required = True)
  parser.add_argument('--in_tar', type=str, help='Name of the input target language embeddings file.', required = True)
  parser.add_argument('--out_src', type=str, help='Name of the output source languge embeddings file.', required = True)
  parser.add_argument('--out_tar', type=str, help='Name of the output target language embeddings file.', required = True)
  # parameters

  # model related para
  parser.add_argument('--is_single_tower', action='store_true', help='use single tower')
  parser.add_argument('--h_dim', type=int, default=300, help='hidden states dim in GNN')
  parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')

  parser.add_argument('--hard_neg_per_pos', type=int, default=256, help='number of hard negative examples')
  parser.add_argument('--hard_neg_sampling_threshold', type=float, default=-10, help='filter low similarity neg word in sampling hard word')
  parser.add_argument('--hard_neg_random', action='store_true', help='random sampling hard in every epoch')
  parser.add_argument('--hard_neg_top_k', type=int, default=500, help='number of topk examples for select hard neg word')
  parser.add_argument('--hard_sim_method', type=str, default="cos", help='number of topk examples for select hard neg word')
  parser.add_argument('--hard_neg_random_with_prob', action='store_true', help='random sampling hard in every epoch')
  parser.add_argument('--random_neg_per_pos', type=int, default=256, help='number of random negative examples')
  parser.add_argument('--update_neg_every_epoch', type=int, default=1, help='recalculate hard neg examples. 0 means fixed hard exmaples.')
  parser.add_argument('--random_warmup_epoches', type=int, default=0, help='only use random neg sampling at begin epoches')
  
  
  parser.add_argument('--train_batch_size', type=int, default=256, help='train batch size')
  parser.add_argument('--train_epochs', type=int, default=70, help='train epochs')
  parser.add_argument('--eval_every_epoch', type=int, default=5, help='eval epochs')
  parser.add_argument('--shuffle_in_train', action='store_true', help='use shuffle in train')
  parser.add_argument('--loss_metric', type=str, default="cos", help='number of topk examples for select hard neg word')

  
  parser.add_argument('--model_filename', type=str, help='Name of file where the model will be stored..', required = True)
  parser.add_argument('--debug', action='store_true', help='store debug info')
  parser.add_argument('--random_seed', type=int, default=2021, help='random seed')

  parser.add_argument('--adapter_src_cluster_k', type=int, default=5, help='hidden states dim in GNN')
  parser.add_argument('--adapter_src_cluster_threshold', type=float, default=0.8, help='learning rate')
  parser.add_argument('--adapter_tgt_cluster_k', type=int, default=5, help='hidden states dim in GNN')
  parser.add_argument('--adapter_tgt_cluster_threshold', type=float, default=0.8, help='learning rate')
  parser.add_argument('--adapter_actfunc', type=str, default="sigmoid", help='learning rate')
  parser.add_argument('--adapter_norm', action='store_true', help='use single tower')
  parser.add_argument('--adapter_regular', type=float, default=0.1, help='learning rate')
  parser.add_argument('--adapter_regular_method', type=str, default="para", help='learning rate')
  parser.add_argument('--adapter_type', type=str, default="none", help='learning rate')
  args = parser.parse_args()

  run_dssm_trainning(args)


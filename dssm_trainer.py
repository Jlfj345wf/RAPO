import torch
from torch import embedding, nn
import numpy as np
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, Dataset, RandomSampler, SequentialSampler, Sampler
import json
from tqdm import tqdm
import random
import collections
import pickle
import faiss
import time
#from transformers import WarmupLinearSchedule


def get_rand_list_with_p(a, size, p):
    p = 1/(1+np.exp(-np.array(p)))
    p = p / np.sum(p)
    sample_list = np.random.choice(a, size, False, p)
    return sample_list.tolist()


def get_faiss_index(embeddings, nlist=200, nprobe=200, gpu_id=1):
    d = embeddings.shape[1]
    
    res = faiss.StandardGpuResources()
    res.setTempMemory(16 * 1024 * 1024 * 1024) 
    # build a flat (CPU) index
    cpu_index = faiss.IndexFlatL2(d)
    cpu_index = faiss.IndexIVFFlat(cpu_index, d, nlist, faiss.METRIC_INNER_PRODUCT)
    # make it into a gpu index
    gpu_index = faiss.index_cpu_to_gpu(res, gpu_id, cpu_index)
    
    gpu_index.train(embeddings)
    gpu_index.nprobe = nprobe
    gpu_index.add(embeddings)
    return gpu_index


def calc_rt_rs(embeddingsx, embeddingsy):
    embeddingsx = embeddingsx.detach().cpu().numpy()
    embeddingsy = embeddingsy.detach().cpu().numpy()

    tgt_embs_index = get_faiss_index(embeddingsy, nlist=2048, nprobe=512, gpu_id=1)
    nx = embeddingsx.shape[0]
    ny = embeddingsy.shape[0]
    begin_id = 0
    batch_size = 2048
    rtd = np.zeros(nx)
    while begin_id < nx:
      d, _ = tgt_embs_index.search(embeddingsx[begin_id:begin_id+batch_size], 10)
      rtd[begin_id:min(begin_id+batch_size, nx)] = np.mean(d, axis=1)
      begin_id += batch_size
    del tgt_embs_index

    src_embs_index = get_faiss_index(embeddingsx, nlist=2048, nprobe=512, gpu_id=1)
    rsd = np.zeros(ny)
    begin_id = 0
    while begin_id < ny:
      d, _ = src_embs_index.search(embeddingsy[begin_id:begin_id+batch_size], 10)
      rsd[begin_id:min(begin_id+batch_size, ny)] = np.mean(d, axis=1)
      begin_id += batch_size
    del src_embs_index
    return rsd, rtd


def calc_csls_sim(embeddingsx, embeddingsy, qx=None):
    embeddingsx = embeddingsx.detach().cpu().numpy()
    embeddingsy = embeddingsy.detach().cpu().numpy()

    src_embs_index = get_faiss_index(embeddingsx, nlist=2048, nprobe=512, gpu_id=1)
    nx = embeddingsx.shape[0]
    ny = embeddingsy.shape[0]
    batch_size = 2048

    rsd = np.zeros(ny)
    begin_id = 0
    while begin_id < ny:
      d, _ = src_embs_index.search(embeddingsy[begin_id:begin_id+batch_size], 10)
      rsd[begin_id:min(begin_id+batch_size, ny)] = np.mean(d, axis=1)
      begin_id += batch_size
    del src_embs_index

    tgt_embs_index = get_faiss_index(embeddingsy, nlist=2048, nprobe=512, gpu_id=1)
    rtd = np.zeros(nx)
    begin_id = 0
    while begin_id < nx:
      d, i = tgt_embs_index.search(embeddingsx[begin_id:begin_id+batch_size], 10)
      rtd[begin_id:min(begin_id+batch_size, nx)] = np.mean(d, axis=1)
      begin_id += batch_size

    sim_x2y_index = {}
    if qx is not None:
      nq = len(qx)
      begin_id = 0
      while begin_id < nq:
        batch_src = qx[begin_id:begin_id+batch_size]
        batch_embeddings = embeddingsx[batch_src]
        d, i = tgt_embs_index.search(batch_embeddings, 1024)
        csls_d = d * 2 - rsd[i]
        csls_i = np.argsort(-csls_d)
        for _ in range(len(batch_src)):
          sim_x2y_index[batch_src[_]] = i[_][csls_i[_]]
        begin_id += batch_size  

    del tgt_embs_index
    return rsd, rtd, sim_x2y_index


def calc_cos_sim(embeddingsx, embeddingsy, qx, qy=None):
    embeddingsx = embeddingsx.detach().cpu().numpy()
    embeddingsy = embeddingsy.detach().cpu().numpy()
    tgt_embs_index = get_faiss_index(embeddingsy, nlist=2048, nprobe=512, gpu_id=1)    
    
    batch_size = 2048
    sim_x2y_index = {}
    nq = len(qx)
    begin_id = 0
    while begin_id < nq:
        batch_src = qx[begin_id:begin_id+batch_size]
        batch_embeddings = embeddingsx[batch_src]
        d, i = tgt_embs_index.search(batch_embeddings, 1024)
        for _ in range(len(batch_src)):
            sim_x2y_index[batch_src[_]] = i[_]
        begin_id += batch_size
    del tgt_embs_index
    return None, None, sim_x2y_index  

def evaluate_use_csls(embeddingsx, embeddingsy, qx):
    embeddingsx = embeddingsx.detach().cpu().numpy()
    embeddingsy = embeddingsy.detach().cpu().numpy()
    src_embs_index = get_faiss_index(embeddingsx, nlist=200, nprobe=200, gpu_id=1)
    nx = embeddingsx.shape[0]
    ny = embeddingsy.shape[0]
    begin_id = 0
    batch_size = 2048

    rsd = np.zeros(ny)
    begin_id = 0
    while begin_id < ny:
      d, _ = src_embs_index.search(embeddingsy[begin_id:begin_id+batch_size], 10)
      rsd[begin_id:min(begin_id+batch_size, ny)] = np.mean(d, axis=1)
      begin_id += batch_size
    del src_embs_index

    tgt_embs_index = get_faiss_index(embeddingsy, nlist=200, nprobe=200, gpu_id=1)

    sim_x2y_index = {}
    nq = len(qx)
    begin_id = 0
    while begin_id < nq:
        batch_src = qx[begin_id:begin_id+batch_size]
        batch_embeddings = embeddingsx[batch_src]
        d, i = tgt_embs_index.search(batch_embeddings, 2048)
        csls_d = d * 2 - rsd[i]
        csls_i = np.argsort(-csls_d)
        for _ in range(len(batch_src)):
            sim_x2y_index[batch_src[_]] = i[_][csls_i[_]]
        begin_id += batch_size  
    del tgt_embs_index
    return sim_x2y_index   


def evaluate_use_csls_y2x(embeddingsx, embeddingsy, qy):
    embeddingsx = embeddingsx.detach().cpu().numpy()
    embeddingsy = embeddingsy.detach().cpu().numpy()
    tgt_embs_index = get_faiss_index(embeddingsy, nlist=200, nprobe=200, gpu_id=1)
    nx = embeddingsx.shape[0]
    ny = embeddingsy.shape[0]
    begin_id = 0
    batch_size = 2048

    rtd = np.zeros(nx)
    begin_id = 0
    while begin_id < nx:
      d, _ = tgt_embs_index.search(embeddingsx[begin_id:begin_id+batch_size], 10)
      rtd[begin_id:min(begin_id+batch_size, nx)] = np.mean(d, axis=1)
      begin_id += batch_size
    del tgt_embs_index

    src_embs_index = get_faiss_index(embeddingsx, nlist=200, nprobe=200, gpu_id=1)

    sim_y2x_index = {}
    nq = len(qy)
    begin_id = 0
    while begin_id < nq:
        batch_src = qy[begin_id:begin_id+batch_size]
        batch_embeddings = embeddingsy[batch_src]
        d, i = src_embs_index.search(batch_embeddings, 2048)
        csls_d = d * 2 - rtd[i]
        csls_i = np.argsort(-csls_d)
        for _ in range(len(batch_src)):
            sim_y2x_index[batch_src[_]] = i[_][csls_i[_]]
        begin_id += batch_size  
    del src_embs_index
    return sim_y2x_index   

class DssmDatasets(Dataset):
    def __init__(self, pos_examples, src2negtgts, tgt2negsrcs=None, vocab_size=30000, 
                random_neg_per_pos=1000, hard_neg_per_pos=256, hard_neg_random=True):
      self.lens = len(pos_examples)
      self.vocab_size = vocab_size
      self.datas, self.src2gold, self.tgt2gold, \
        self.sample2negtgts, self.sample2negsrcs = self._build_dataset(
                                                  pos_examples, src2negtgts, tgt2negsrcs)
      self.random_neg_per_pos = random_neg_per_pos
      self.hard_neg_per_pos = hard_neg_per_pos
      self.hard_neg_random = hard_neg_random
      random.seed(2021)

    def _build_dataset(self, pos_examples, src2negtgts, tgt2negsrcs):
      datas = []
      src2gold = collections.defaultdict(set)
      tgt2gold = collections.defaultdict(set)

      sample2negtgts = collections.defaultdict(set)
      sample2negsrcs = collections.defaultdict(set)
      if tgt2negsrcs is None:
        sample2negsrcs = None
      
      # 如果src在sample2negtgts说明是word-wise的
      # 将word-wise调整为sample-wise，为了之后的处理更加统一
      is_word_wise = pos_examples[0][0] in src2negtgts 
      if not is_word_wise:
        sample2negtgts = src2negtgts
        sample2negsrcs = tgt2negsrcs
      else:
        for pos_src, pos_tgt in pos_examples:
          sample2negtgts[(pos_src, pos_tgt)] = src2negtgts[pos_src]
          if tgt2negsrcs is not None:
            sample2negsrcs[(pos_src, pos_tgt)] = tgt2negsrcs[pos_tgt]

      for pos_src, pos_tgt in pos_examples:
        datas.append([pos_src, pos_tgt])
        src2gold[pos_src].add(pos_tgt)
        tgt2gold[pos_tgt].add(pos_src)

      return datas, src2gold, tgt2gold, sample2negtgts, sample2negsrcs

    def __getitem__(self, i):
      # 在getitem的时候随机采样，是为了保证每个epoch采样得到的负例都不相同
      orig = self.datas[i]
      src = orig[0]
      tgt = orig[1]
      negtgts = list(self.sample2negtgts[(src, tgt)])
      negtgts_prob = None
      if len(negtgts) > 0:
        # 有给定的概率，按照给定概率采样
        if not isinstance(negtgts[0], int):
          negtgts_prob = [_[1] for _ in negtgts]
          negtgts = [_[0] for _ in negtgts]

        if self.hard_neg_random:
          if negtgts_prob is not None:
            hard_neg_tgts_list = get_rand_list_with_p(negtgts, min(self.hard_neg_per_pos, len(negtgts)), negtgts_prob)
          else:
            #hard_neg_tgts_list = random.sample(negtgts, min(self.hard_neg_per_pos, len(negtgts)))
            hard_neg_tgts_list = negtgts[:self.hard_neg_per_pos]
          hard_neg_tgts_set = set(hard_neg_tgts_list)
        else:
          hard_neg_tgts_list = negtgts
          hard_neg_tgts_set = set(hard_neg_tgts_list)
      else:
        hard_neg_tgts_list = []
        hard_neg_tgts_set = set()

      rand_sampling_tgts = random.sample(list(range(self.vocab_size)), self.random_neg_per_pos)
      no_dup_random_tgts = list(set(rand_sampling_tgts) - hard_neg_tgts_set - self.src2gold[src])
      combi_tgts = hard_neg_tgts_list + no_dup_random_tgts

      # 双向
      combi_srcs = []
      if self.sample2negsrcs is not None:
        negsrcs = list(self.sample2negsrcs[(src, tgt)])
        if len(negsrcs) > 0:
          negsrcs_prob = None
          if not isinstance(negsrcs[0], int):
            negsrcs_prob = [_[1] for _ in negsrcs]
            negsrcs = [_[0] for _ in negsrcs]        
          if self.hard_neg_random: 
            #print(len(orig[2:]))
            if negsrcs_prob is not None:
              hard_neg_srcs_list = get_rand_list_with_p(negsrcs, min(self.hard_neg_per_pos, len(negsrcs)), negsrcs_prob)
            else:
              hard_neg_srcs_list = random.sample(negsrcs, min(self.hard_neg_per_pos, len(negsrcs)))
            hard_neg_srcs_set = set(hard_neg_srcs_list)
          else:
            hard_neg_srcs_list = negsrcs
            hard_neg_srcs_set = set(hard_neg_srcs_list)
        else:
          hard_neg_srcs_list = []
          hard_neg_srcs_set = set()

        rand_sampling_srcs = random.sample(list(range(self.vocab_size)), self.random_neg_per_pos)
        no_dup_random_srcs = list(set(rand_sampling_srcs) - hard_neg_srcs_set - self.tgt2gold[tgt])
        combi_srcs = hard_neg_srcs_list + no_dup_random_srcs
      # 每个item是一个tuple，由两个list组成，第一个是src的list，第二个是tgt的list
      # 正例永远位于list首位
      new_item = ([src] + combi_srcs, [tgt] + combi_tgts)
      return new_item

    def __len__(self):
      return len(self.datas)

    def collate(self, features):
      # ground truth 总是位于tgts_list的首位
      srcs_list = [_[0] for _ in features]
      tgts_list = [_[1] for _ in features]
      labels_list = [[0, 0] for _ in features]

      # batch内data cut到同一长度      
      min_neg_tgt_size = min([len(_) for _ in tgts_list])
      for i in range(len(tgts_list)):
        tgts_list[i] = tgts_list[i][:min_neg_tgt_size]

      # batch内data cut到同一长度      
      min_neg_src_size = min([len(_) for _ in srcs_list])
      for i in range(len(srcs_list)):
        srcs_list[i] = srcs_list[i][:min_neg_src_size]      

      # to_tensor
      srcs_list_index = torch.tensor(srcs_list, dtype=torch.long)
      tgts_list_index = torch.tensor(tgts_list, dtype=torch.long)
      labels_list = torch.tensor(labels_list, dtype=torch.long)
      return srcs_list_index, tgts_list_index, labels_list
      
    def update_hard_neg_faiss(self, similarity_x2y_index, similarity_y2x_index=None):
      neg_candi_size = max([len(_) + 3 for _ in self.sample2negtgts.values()])
      src2hard_neg_candi = {}
      index_x2y = similarity_x2y_index
      for src in self.src2gold:
        neg_candi = index_x2y[src][:neg_candi_size].tolist()
        neg_candi = [_ for _ in neg_candi if _ not in self.src2gold[src]]
        src2hard_neg_candi[src] = neg_candi
      for (pos_src, pos_tgt) in self.datas:
        self.sample2negtgts[(pos_src, pos_tgt)] = src2hard_neg_candi[pos_src]
      if self.sample2negsrcs is not None and similarity_y2x_index is not None:
        index_y2x = similarity_y2x_index
        tgt2hard_neg_candi = {}
        for tgt in self.tgt2gold:
          neg_candi = index_y2x[tgt][:neg_candi_size].tolist()
          neg_candi = [_ for _ in neg_candi if _ not in self.tgt2gold[tgt]]
          tgt2hard_neg_candi[tgt] = neg_candi
        for (pos_src, pos_tgt) in self.datas:
          self.sample2negsrcs[(pos_src, pos_tgt)] = tgt2hard_neg_candi[pos_tgt]


def l2_penalty(var):
    return torch.sqrt(torch.pow(var, 2).sum())

class Mahalanobisdistance(nn.Module):
    # XH
    def __init__(self, feat_dim):
        super(Mahalanobisdistance, self).__init__()
        print("use Mahalanobisdistance")
        self.matrix = nn.Parameter(torch.randn((feat_dim, feat_dim), requires_grad=True))

    def forward(self, feat1, feat2):
        B = torch.matmul(self.matrix, self.matrix.transpose(0, 1))
        B = B / torch.max(torch.abs(B))
        md_feat = torch.matmul(torch.matmul(feat1, B), feat2.transpose(-2, -1))
        return md_feat

class AdapterForInit(nn.Module):

    def __init__(self, in_dim, out_dim, bias=False, actfunc="sigmoid", norm=True):
      super(AdapterForInit, self).__init__()
      self.layer = nn.Linear(in_dim, out_dim, bias=bias)
      self.actfunc = actfunc
      self.norm = norm

    def get_l2_penalty(self):
      return l2_penalty(self.layer.weight)

    def forward(self, x, feature):
      mx = self.layer(feature)
      if self.actfunc == "sigmoid":
        smx = (F.sigmoid(mx) - 0.5) * 2
      elif self.actfunc == "tanh":
        smx = F.tanh(mx)
      else:
        smx = mx
      if self.norm:
        nmx = F.normalize(x + smx, dim=-1)
      else:
        nmx = x + smx
      return nmx


class AdapterForRotation(nn.Module):
    def __init__(self, in_dim, out_dim, bias=False, actfunc="sigmoid", norm=True):
      super(AdapterForRotation, self).__init__()
      self.layer = nn.Linear(in_dim, out_dim, bias=bias)
      self.actfunc = actfunc
      self.norm = norm

    def get_l2_penalty(self):
      return l2_penalty(self.layer.weight)

    def forward(self, x, feature):
      mx = self.layer(feature)
      if self.actfunc == "sigmoid":
        smx = (F.sigmoid(mx) - 0.5) * 2
      elif self.actfunc == "tanh":
        smx = F.tanh(mx)
      else:
        smx = mx
      v = F.normalize(smx, dim=-1)
      x = x - 2 * (x * v).sum(dim=-1, keepdim=True) * v 
      return x


class AdapterForAdjust(nn.Module):
    def __init__(self, in_dim, out_dim, bias=False, actfunc="sigmoid", norm=True):
      super(AdapterForAdjust, self).__init__()
      self.layer = nn.Linear(in_dim, out_dim, bias=bias)
      self.actfunc = actfunc
      self.norm = norm

    def get_l2_penalty(self):
      return l2_penalty(self.layer.weight)

    def forward(self, feature):
      mx = self.layer(feature)
      if self.actfunc == "sigmoid":
        smx = F.sigmoid(mx)
      else:
        smx = mx
      return smx



class HouseholderTower(nn.Module):
    # XH
    def __init__(self, in_feat_dim, hhr_number):
        super(HouseholderTower, self).__init__()
        print("use HouseholderTower")
        self.mapping_vectors = nn.ParameterList([nn.Parameter(torch.randn((1, in_feat_dim), requires_grad=True)) 
                                                      for _ in range(hhr_number)])

    def _householderReflection(self, v, x, w=None):
        v = F.normalize(v, dim=-1)
        if w is not None:
          x = x - (2 - w) * torch.matmul(x, v.T) * v
          x = F.normalize(x, dim=-1)
        else:
          x = x - 2 * torch.matmul(x, v.T) * v
        return x

    def forward(self, node_feat, adjust_w=None):
        h = node_feat
        if adjust_w is not None:
          weights = torch.chunk(adjust_w, adjust_w.shape[-1], dim=-1)
        for i, v in enumerate(self.mapping_vectors):
            if adjust_w is None:
                h = self._householderReflection(v, h)
            else:
                h = self._householderReflection(v, h, weights[i])
        return h  

class GDSSM(nn.Module):
    
    def __init__(self, src_in_feat_dim, tgt_in_feat_dim, h_feat_dim, is_single_tower=False, args=None):
        super(GDSSM, self).__init__()
        self.src_tower = HouseholderTower(src_in_feat_dim, h_feat_dim)
        self.args = args
        if args is not None and args.adapter_type == "shift":
          self.src_adapter = AdapterForInit(src_in_feat_dim, src_in_feat_dim, actfunc=args.adapter_actfunc, norm=args.adapter_norm)
          self.tgt_adapter = AdapterForInit(tgt_in_feat_dim, tgt_in_feat_dim, actfunc=args.adapter_actfunc, norm=args.adapter_norm)
        elif args is not None and args.adapter_type == "rotation":
          self.src_adapter = AdapterForRotation(src_in_feat_dim, src_in_feat_dim, actfunc=args.adapter_actfunc, norm=args.adapter_norm)
          self.tgt_adapter = AdapterForRotation(tgt_in_feat_dim, tgt_in_feat_dim, actfunc=args.adapter_actfunc, norm=args.adapter_norm)
        elif args is not None and args.adapter_type == "adjust":
          self.src_adapter = AdapterForAdjust(src_in_feat_dim, src_in_feat_dim, actfunc=args.adapter_actfunc, norm=args.adapter_norm)
          self.tgt_adapter = AdapterForAdjust(tgt_in_feat_dim, tgt_in_feat_dim, actfunc=args.adapter_actfunc, norm=args.adapter_norm)


        if is_single_tower == True:
          self.tgt_tower = self._straight_forwoard
        else:
          self.tgt_tower = HouseholderTower(tgt_in_feat_dim, h_feat_dim)

    def _straight_forwoard(self, x):
        return x

    def _get_hidden(self, node_feat, t="src", adapter_feat=None):
        if self.args is not None and self.args.adapter_type in ["shift", "rotation"]:
            if adapter_feat is None:
                adapter_feat = node_feat        
            if t == "src":
                return self.src_tower(self.src_adapter(node_feat, adapter_feat))
            else:
                return self.tgt_tower(self.tgt_adapter(node_feat, adapter_feat))
        elif self.args is not None and self.args.adapter_type in ["shift", "rotation"]:
            if adapter_feat is None:
                adapter_feat = node_feat        
            if t == "src":
                return self.src_tower(node_feat, self.src_adapter(adapter_feat))
            else:
                return self.tgt_tower(node_feat, self.tgt_adapter(adapter_feat))
        else:
            if t == "src":
                return self.src_tower(node_feat)
            else:
                return self.tgt_tower(node_feat)

    def forward(self, node_feat_src, node_feat_tgt, srcs_index, tgts_index, rs=None, rt=None, src_afeat=None, tgt_afeat=None):
        
        if self.args is not None and self.args.adapter_type in ["shift", "rotation"]:
            if src_afeat is None:
                src_afeat = node_feat_src
            if tgt_afeat is None:
                tgt_afeat = node_feat_tgt
            src_hidden_norm = F.normalize(self.src_tower(self.src_adapter(node_feat_src, src_afeat)), dim=-1)
            tgt_hidden_norm = F.normalize(self.tgt_tower(self.tgt_adapter(node_feat_tgt, tgt_afeat)), dim=-1)
        elif self.args is not None and self.args.adapter_type in ["adjust"]:
            src_hidden_norm = F.normalize(self.src_tower(node_feat_src, self.src_adapter(src_afeat)), dim=-1)
            tgt_hidden_norm = F.normalize(self.tgt_tower(node_feat_tgt, self.tgt_adapter(tgt_afeat)), dim=-1)
        else:
            src_hidden_norm = F.normalize(self.src_tower(node_feat_src), dim=-1)
            tgt_hidden_norm = F.normalize(self.tgt_tower(node_feat_tgt), dim=-1)

        pos_src_norm = src_hidden_norm[:, 0]
        pos_tgt_norm = tgt_hidden_norm[:, 0]

        tgt_list_norm = tgt_hidden_norm
        sim_src2tgt = torch.matmul(pos_src_norm.unsqueeze(1), tgt_list_norm.transpose(1,2))
        if self.args.loss_metric == "csls":
            srcs_rt = rt[srcs_index]
            tgts_rs = rs[tgts_index]        
            logits_src2tgt = sim_src2tgt.squeeze() * 2 - srcs_rt[:, 0:1] - tgts_rs 
        else:
            logits_src2tgt = sim_src2tgt.squeeze()

        return logits_src2tgt, pos_src_norm, pos_tgt_norm


class DssmTrainer:
    def __init__(self, src_in_feat_dim, tgt_in_feat_dim, h_feat_dim, 
                  device='gpu', epochs=100, eval_every_epoch=5, lr=0.0001, train_batch_size=256,
                  model_save_file='tmp_model.pickle', is_single_tower=False, shuffle_in_train=True,
                  random_neg_per_pos=256, hard_neg_per_pos=256, hard_neg_random=True, 
                  update_neg_every_epoch=1, random_warmup_epoches=0, loss_metric="cos", args=None):
        # train config
        self.args = args
        self.epochs = epochs
        self.eval_every_epoch = eval_every_epoch
        self.train_batch_size = train_batch_size
        self.random_neg_per_pos = random_neg_per_pos
        self.model_save_file = model_save_file
        self.device = torch.device("cuda" if torch.cuda.is_available() and device == 'gpu' else "cpu")
        self.loss_metric = loss_metric

        self.hard_neg_per_pos = hard_neg_per_pos
        self.hard_neg_random = hard_neg_random
        self.shuffle_in_train = shuffle_in_train
        self.update_neg_every_epoch = update_neg_every_epoch
        self.random_warmup_epoches = random_warmup_epoches
        # model config
        self.model = GDSSM(src_in_feat_dim, tgt_in_feat_dim, h_feat_dim, is_single_tower, args)
        #self.loss_func = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        #self.scheduler = WarmupLinearSchedule(
        #self.optimizer, warmup_steps=0, t_total=epochs
        #)

    def _bpr_loss_func(self, logits):
        new_logits = logits
        pos_si = new_logits[:, 0]
        neg_si = new_logits[:, 1:]
        diff = pos_si[:, None] - neg_si
        bpr_loss = - diff.sigmoid().log().mean(1)
        bpr_loss_batch_mean = bpr_loss.mean()
        return bpr_loss_batch_mean

    def _calc_similarity_info(self, src_x, tgt_x, unique_src_list=None, src_afeat=None, tgt_afeat=None, type="csls"):
        with torch.no_grad():
            batch_size = 4096
            begin_id = 0
            src_hidden = torch.zeros_like(src_x)
            tgt_hidden = torch.zeros_like(tgt_x)
            while begin_id < src_x.shape[0]:
                end_id = min(begin_id+batch_size, src_x.shape[0])
                batch_x = src_x[begin_id:end_id].to(self.device)
                if self.args.adapter_type != "none" and src_afeat is not None:
                    batch_x_afeat = src_afeat[begin_id:end_id].to(self.device)
                    batch_src_h = self.model._get_hidden(batch_x, "src", batch_x_afeat)
                else:
                    batch_src_h = self.model._get_hidden(batch_x, "src")
                src_hidden[begin_id:end_id] = F.normalize(batch_src_h).cpu()
                begin_id += batch_size
          
            begin_id = 0
            while begin_id < tgt_x.shape[0]:
                end_id = min(begin_id+batch_size, tgt_x.shape[0])
                batch_x = tgt_x[begin_id:end_id].to(self.device)
                if self.args.adapter_type != "none" and tgt_afeat is not None:
                    batch_x_afeat = tgt_afeat[begin_id:end_id].to(self.device)
                    batch_tgt_h = self.model._get_hidden(batch_x, "tgt", batch_x_afeat)
                else:
                    batch_tgt_h = self.model._get_hidden(batch_x, "tgt")
                tgt_hidden[begin_id:end_id] = F.normalize(batch_tgt_h).cpu()
                begin_id += batch_size
        if type == "csls":
            rs, rt, sim_x2y_index = calc_csls_sim(src_hidden, tgt_hidden, unique_src_list)
        else:
            rs, rt, sim_x2y_index = calc_cos_sim(src_hidden, tgt_hidden, unique_src_list)

        return rs, rt, sim_x2y_index

    def fit(self, src_x, tgt_x, train_set, src2negtgts, tgt2negsrcs, val_set, src_afeat=None, tgt_afeat=None):
        
        # for evaluate and debug
        # eval_data_set = train_set
        eval_data_set = val_set   
        eval_src2tgts = collections.defaultdict(set)
        for s, t in eval_data_set:
          eval_src2tgts[s].add(t)
        eval_src = list(set([_[0] for _ in eval_data_set])) 

        model = self.model
        model.to(self.device)        

        train_dataset = DssmDatasets(train_set, src2negtgts, tgt2negsrcs,
                                      vocab_size=tgt_x.shape[0], 
                                      random_neg_per_pos=self.random_neg_per_pos,
                                      hard_neg_per_pos=self.hard_neg_per_pos,
                                      hard_neg_random=self.hard_neg_random)

        train_dataloader = DataLoader(train_dataset, 
                                batch_size=self.train_batch_size,
                                shuffle=self.shuffle_in_train, 
                                collate_fn=train_dataset.collate)
        
        optimizer = self.optimizer
        loss_func = self._bpr_loss_func
        if self.args.use_mseloss:
          mse_loss_func = nn.MSELoss()

        unique_src_list = sorted(list(train_dataset.src2gold.keys()))
        best_val_acc = [0, 0, 0, 0, 0]
        save_best_acc = [0, 0, 0, 0, 0]
        best_epoch = 0
        global_step = 0
        total_step = ((len(train_set) + self.train_batch_size - 1) // self.train_batch_size) * self.epochs
        
        for e in range(self.epochs):
          # Forward
          rs, rt, sim_x2y_index = self._calc_similarity_info(src_x, tgt_x, unique_src_list, type=self.loss_metric, src_afeat=src_afeat, tgt_afeat=tgt_afeat)
          if self.loss_metric == "csls":
            rs = torch.from_numpy(rs).to(self.device)
            rt = torch.from_numpy(rt).to(self.device)

          if self.update_neg_every_epoch > 0 and e % self.update_neg_every_epoch == 0:
            train_dataset.update_hard_neg_faiss(sim_x2y_index)      
          
          if e < self.random_warmup_epoches:
            train_dataset.hard_neg_per_pos = 0
            train_dataset.random_neg_per_pos = self.hard_neg_per_pos + self.random_neg_per_pos
          else:
            train_dataset.hard_neg_per_pos = self.hard_neg_per_pos
            train_dataset.random_neg_per_pos = self.random_neg_per_pos      

          model.train()
          for step, batch in enumerate(train_dataloader):
            srcs_index, tgts_index, labels_index = batch
            src_feat = src_x[srcs_index]
            src_feat = src_feat.to(self.device)
            tgt_feat = tgt_x[tgts_index]
            tgt_feat = tgt_feat.to(self.device)

            if self.args.adapter_type != "none":
              batch_src_afeat = src_afeat[srcs_index]
              batch_tgt_afeat = tgt_afeat[tgts_index]
              batch_src_afeat = batch_src_afeat.to(self.device)
              batch_tgt_afeat = batch_tgt_afeat.to(self.device)
            else:
              batch_src_afeat = None
              batch_tgt_afeat = None

            srcs_index = srcs_index.to(self.device)
            tgts_index = tgts_index.to(self.device)
            logits_src2tgt, pos_src_norm, pos_tgt_norm = model(src_feat, tgt_feat, srcs_index, tgts_index, rs=rs, rt=rt, src_afeat=batch_src_afeat, tgt_afeat=batch_tgt_afeat)

            loss1 = loss_func(logits_src2tgt)
            loss2 = 0
            loss3 = 0
            loss4 = 0
            if self.args.adapter_type != "none" and self.args.adapter_regular_method == "para":
              adapter_regular_loss = model.src_adapter.get_l2_penalty() + model.tgt_adapter.get_l2_penalty()
              loss2 = self.args.adapter_regular * adapter_regular_loss
            if self.args.use_mseloss:
              loss3 = self.args.mse_loss_lambda * mse_loss_func(pos_src_norm, pos_tgt_norm)
            if self.args.use_geomm:
              loss4 = self.args.geomm_regular * 0
            loss = loss1 + loss2 + loss3 + loss4
            
            loss.backward()
            optimizer.step()
            #scheduler.step()
            optimizer.zero_grad()      
            global_step += 1
            #self._liner_adjust_lr(optimizer, total_step, 0.01, 0.0001, global_step)
            print('In epoch {}, step: {}, loss: {:.5f}, loss1: {:.5f}, loss2: {:.5f}, loss3: {:.5f}, loss4: {:.5f}'.format(
              e, step, loss, loss1, loss2, loss3, loss4))
            # if step % 50 == 0 and step != 0:
            #   model.eval()
            #   print(f'In epoch {e} step {step} evaluate:')
            #   acc = self.eval(src_x, tgt_x, eval_src, eval_src2tgts, src_afeat=src_afeat, tgt_afeat=tgt_afeat)

            #   if best_val_acc < acc:
            #     if acc[0] - save_best_acc[0] > 0.0001:
            #       self.save()
            #       save_best_acc = acc
            #     best_val_acc = acc
            #     best_epoch = e
            #     self.best_val_acc = best_val_acc
            #   print(f"best result at epoch {best_epoch}: {best_val_acc}")
            #   model.train()              
            # evaluate test set
          if e % self.eval_every_epoch == 0 or e == self.epochs - 1:
            model.eval()
            print(f'In epoch {e} evaluate:')
            acc = self.eval(src_x, tgt_x, eval_src, eval_src2tgts, src_afeat=src_afeat, tgt_afeat=tgt_afeat)

            if best_val_acc < acc:
              if acc[0] - save_best_acc[0] > 0.0005:
                self.save()
                save_best_acc = acc
              best_val_acc = acc
              best_epoch = e
            print(f"best result at epoch {best_epoch}: {best_val_acc}")
          
          self.best_val_acc = best_val_acc
          if e - best_epoch > 20:
            print("early-stop for no gain latest 10 epoches")
            break   


    def eval(self, src_x, tgt_x, val_src, val_src2tgts, src_afeat=None, tgt_afeat=None):
        with torch.no_grad():
            batch_size = 4096
            begin_id = 0
            src_hidden = torch.zeros_like(src_x)
            tgt_hidden = torch.zeros_like(tgt_x)
            while begin_id < src_x.shape[0]:
                end_id = min(begin_id+batch_size, src_x.shape[0])
                batch_x = src_x[begin_id:end_id].to(self.device)
                if self.args.adapter_type != "none" and src_afeat is not None:
                    batch_x_afeat = src_afeat[begin_id:end_id].to(self.device)
                    batch_src_h = self.model._get_hidden(batch_x, "src", batch_x_afeat)
                else:
                    batch_src_h = self.model._get_hidden(batch_x, "src")
                src_hidden[begin_id:end_id] = F.normalize(batch_src_h).cpu()
                begin_id += batch_size
            
            begin_id = 0
            while begin_id < tgt_x.shape[0]:
                end_id = min(begin_id+batch_size, tgt_x.shape[0])
                batch_x = tgt_x[begin_id:end_id].to(self.device)
                if self.args.adapter_type != "none" and tgt_afeat is not None:
                    batch_x_afeat = tgt_afeat[begin_id:end_id].to(self.device)
                    batch_tgt_h = self.model._get_hidden(batch_x, "tgt", batch_x_afeat)
                else:
                    batch_tgt_h = self.model._get_hidden(batch_x, "tgt")
                tgt_hidden[begin_id:end_id] = F.normalize(batch_tgt_h).cpu()
                begin_id += batch_size

        csls_x2y_index = evaluate_use_csls(src_hidden, tgt_hidden, val_src)
        acc = []
        positions = []
        for s in val_src:
            flag = 0
            for i, wi in enumerate(csls_x2y_index[s].tolist()):
                if wi in val_src2tgts[s]:
                    positions.append(i+1)
                    flag = 1
                    break
            if flag == 0:
                positions.append(2048)
        for k in [1, 5, 10, 50, 100]:
            patk = len([p for p in positions if p <= k]) / len(positions)
            print(f'top_{k} acc: {patk}')
            acc.append(patk)
        mrr = sum([1.0/p for p in positions]) / len(positions)
        print(f'mrr: {mrr}')
        acc.append(mrr)
        return acc

    def predict(self, src_x, tgt_x, test_src, src_afeat=None, tgt_afeat=None):
        model = self.model
        model.to(self.device)

        with torch.no_grad():
            batch_size = 4096
            begin_id = 0
            src_hidden = torch.zeros_like(src_x)
            tgt_hidden = torch.zeros_like(tgt_x)
            while begin_id < src_x.shape[0]:
                end_id = min(begin_id+batch_size, src_x.shape[0])
                batch_x = src_x[begin_id:end_id].to(self.device)
                if self.args.adapter_type != "none" and src_afeat is not None:
                    batch_x_afeat = src_afeat[begin_id:end_id].to(self.device)
                    batch_src_h = self.model._get_hidden(batch_x, "src", batch_x_afeat)
                else:
                    batch_src_h = self.model._get_hidden(batch_x, "src")
                src_hidden[begin_id:end_id] = F.normalize(batch_src_h).cpu()
                begin_id += batch_size
            
            begin_id = 0
            while begin_id < tgt_x.shape[0]:
                end_id = min(begin_id+batch_size, tgt_x.shape[0])
                batch_x = tgt_x[begin_id:end_id].to(self.device)
                if self.args.adapter_type != "none" and tgt_afeat is not None:
                    batch_x_afeat = tgt_afeat[begin_id:end_id].to(self.device)
                    batch_tgt_h = self.model._get_hidden(batch_x, "tgt", batch_x_afeat)
                else:
                    batch_tgt_h = self.model._get_hidden(batch_x, "tgt")
                tgt_hidden[begin_id:end_id] = F.normalize(batch_tgt_h).cpu()
                begin_id += batch_size

        csls_x2y_index = evaluate_use_csls(src_hidden, tgt_hidden, test_src)
        return csls_x2y_index

    def predict_y2x(self, src_x, tgt_x, test_tgt, src_afeat=None, tgt_afeat=None):
        model = self.model
        model.to(self.device)

        with torch.no_grad():
            batch_size = 4096
            begin_id = 0
            src_hidden = torch.zeros_like(src_x)
            tgt_hidden = torch.zeros_like(tgt_x)
            while begin_id < src_x.shape[0]:
                end_id = min(begin_id+batch_size, src_x.shape[0])
                batch_x = src_x[begin_id:end_id].to(self.device)
                if self.args.adapter_type != "none" and src_afeat is not None:
                    batch_x_afeat = src_afeat[begin_id:end_id].to(self.device)
                    batch_src_h = self.model._get_hidden(batch_x, "src", batch_x_afeat)
                else:
                    batch_src_h = self.model._get_hidden(batch_x, "src")
                src_hidden[begin_id:end_id] = F.normalize(batch_src_h).cpu()
                begin_id += batch_size
            
            begin_id = 0
            while begin_id < tgt_x.shape[0]:
                end_id = min(begin_id+batch_size, tgt_x.shape[0])
                batch_x = tgt_x[begin_id:end_id].to(self.device)
                if self.args.adapter_type != "none" and tgt_afeat is not None:
                    batch_x_afeat = tgt_afeat[begin_id:end_id].to(self.device)
                    batch_tgt_h = self.model._get_hidden(batch_x, "tgt", batch_x_afeat)
                else:
                    batch_tgt_h = self.model._get_hidden(batch_x, "tgt")
                tgt_hidden[begin_id:end_id] = F.normalize(batch_tgt_h).cpu()
                begin_id += batch_size

        csls_y2x_index = evaluate_use_csls_y2x(src_hidden, tgt_hidden, test_tgt)
        return csls_y2x_index

    def save(self):
      print("Saving the best model to disk ...")
      if self.model_save_file is None:
        print("Save failed for model_save_file para is None !!!!")
      else:
        with open("./" + self.model_save_file, "wb") as outfile:
          pickle.dump(self, outfile)
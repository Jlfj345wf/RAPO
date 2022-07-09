import optuna
import argparse
from train import run_dssm_trainning
import sys
from easydict import EasyDict

class Objective(object):
    def __init__(self, predefine_para):
        # Hold this implementation specific arguments as the fields of the class.
        self.predefine_para = predefine_para

    def __call__(self, trial):
        # Calculate an objective value by using the extra arguments.
        args = EasyDict({
            "is_single_tower": False,
            "h_dim": 300,
            "hard_neg_top_k": 1000,
            "hard_neg_random_with_prob" : False,
            "hard_sim_method" : "cos",
            "hard_neg_random": True,
            "debug": False,
            "eval_every_epoch": 1,
            "update_neg_every_epoch": 1,
            "train_epochs": 200,
            "use_geomm": False,
            "geomm_regular": 0,
            "use_mseloss": True,
            "loss_metric": "csls",
            "adapter_type": "shift",
            "adapter_actfunc": "tanh",
            "adapter_regular_method": "para",
            "adapter_src_cluster_k": 10,
            "adapter_tgt_cluster_k": 10,
            "adapter_norm": True

        })

        # predefine_para
        predefine_para = self.predefine_para
        lang_pair = f"{predefine_para.src_lang}-{predefine_para.tgt_lang}"
        args.train_dict = f"./data/{lang_pair}/{lang_pair}.0-5000.txt"
        args.val_dict = f"./data/{lang_pair}/{lang_pair}.5000-6500.txt"
        args.in_src = f"./data/embeddings/wiki.{predefine_para.src_lang}.vec"
        args.in_tar = f"./data/embeddings/wiki.{predefine_para.tgt_lang}.vec"
        
        args.shuffle_in_train = trial.suggest_categorical(name="shuffle_in_train", choices=[True, False])
        args.lr = trial.suggest_float("learning_rate", 0.001, 0.003, step=0.0005)
        args.train_batch_size = trial.suggest_int(name="train_batch_size", low=64, high=128, step=32)
        args.hard_neg_per_pos = trial.suggest_int(name="hard_neg_per_pos", low=64, high=384, step=64)
        args.random_neg_per_pos = trial.suggest_int(name="random_neg_per_pos", low=64, high=256, step=64)
        args.random_seed = trial.suggest_categorical(name="random seed", choices=[9997237, 5644829, 7777967, 65533, 3422427, 6319423])
        args.random_warmup_epoches = trial.suggest_int(name="random_warmup_epoches", low=0, high=20, step=4)
        args.mse_loss_lambda = trial.suggest_float("mse_loss_lambda", 0.5, 2.5, step=0.5) 

        # adapter related
        args.adapter_src_cluster_threshold = trial.suggest_float("adapter_src_cluster_threshold", 0.85, 0.99) 
        args.adapter_tgt_cluster_threshold = trial.suggest_float("adapter_tgt_cluster_threshold", 0.85, 0.99) 
        args.adapter_regular = trial.suggest_float("adapter_regular", 0.001, 0.1)          

        # geomm 10k没效果暂时先不用了
        """args.use_geomm = predefine_para.geomm
        if args.use_geomm:
            args.geomm_regular = trial.suggest_categorical(name="geomm_regular", choices=[0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5])
        else:
            args.geomm_regular = 0"""      
        
        score = run_dssm_trainning(args, is_optuna=True)
        return score[0]

def main(predefine_para):

  iden = str(predefine_para.src_lang) + "_" + str(predefine_para.tgt_lang) + "_" + str(predefine_para.adapter_type)
  print(iden)
  print('=======================================================')
  import os
  if os.path.exists(f'{iden}.db'):
    print("load from exist optuna study")
    study = optuna.load_study(study_name=iden, 
                              storage=f'sqlite:///{iden}.db')     
  else:
    print("create new optuna study")
    study = optuna.create_study(pruner=optuna.pruners.MedianPruner(n_warmup_steps=10), 
                                direction="maximize",
                                study_name=iden, 
                                storage=f'sqlite:///{iden}.db')

  study.optimize(Objective(predefine_para), n_trials=predefine_para.n_trail)

  print("Number of finished trials: {}".format(len(study.trials)))
  print("Best trial:")
  trial = study.best_trial
  print("  Value: {}".format(trial.value))
  print("  Params: ")
  for key, value in trial.params.items():
      print("    {}: {}".format(key, value))

if __name__ == "__main__":

  parser = argparse.ArgumentParser(description='Run classification based self learning for aligning embedding spaces in two languages.')

  parser.add_argument('--src_lang', type=str, help='Name of the input dictionary file.', required = True)
  parser.add_argument('--tgt_lang', type=str, help='Name of the input dictionary file.', required = True)
  parser.add_argument('--adapter_type', type=str, help='Name of the output source languge embeddings file.', required = True)
  parser.add_argument('--n_trail', type=int, help='Name of the output target language embeddings file.', required = True)
  # parameters
  predefine_para = parser.parse_args()
  main(predefine_para)





import json
from train_semi import run_dssm_trainning
import sys
import os
from easydict import EasyDict

default_config_file = "configs/default.json"
language_config_file = sys.argv[1]

with open(default_config_file, 'r') as f:
    default_config = json.load(f)
with open(language_config_file, 'r') as f:
    language_config = json.load(f)

config = default_config
config.update(language_config)
args = EasyDict(config)

lang_pair = f"{args.src_lang}-{args.tgt_lang}"
args.train_dict = f"./data/dictionaries/{lang_pair}.0-5000.txt"
args.val_dict = f"./data/dictionaries/{lang_pair}.5000-6500.txt"
args.in_src = f"./data/embeddings/wiki.{args.src_lang}.vec"
args.in_tar = f"./data/embeddings/wiki.{args.tgt_lang}.vec"
args.model_filename = f"./models/semi/{lang_pair}_model.pkl"
if not os.path.exists("./models/semi"):
    os.mkdir("./models/semi")
args.load_model = 1
score = run_dssm_trainning(args, is_optuna=False)
print(score)

# RAPO: An Adaptive Ranking Paradigm for Bilingual Lexicon Induction

This is the code of the paper *RAPO: An Adaptive Ranking Paradigm for Bilingual Lexicon Induction*.

A novel Ranking-based method with Adaptive Personalized Offsets for BLI.

## Requirements

- python == 3.7
- pytorch == 1.6
- cupy == 10.2
- faiss
- easydict
- optuna
- pandas
- tqdm

## Data

We conduct extensive experiments over multiple language pairs in the public [MUSE](https://github.com/facebookresearch/MUSE) dataset, including five popular rich-resource language pairs (French (**fr**), Spanish(**es**), Italian (**it**), Russian (**ru**), Chinese (**zh**) from and to English(**en**)) and five low-resource language pairs (Faroese(**fa**), Turkish(**tr**), Hebrew(**he**), Arabic(**ar**), Estonian(**er**) from and to English(**en**)), totally 20 BLI datasets considering bidirectional translation. 

We use the data splits in the original MUSE dataset.

Please put the downloaded [MUSE](https://github.com/facebookresearch/MUSE) dataset in the `data` folder and organize it as follows:

```
RAPO
|-- data
|   |-- embeddings
|       |-- wiki.en.vec
|       |-- wiki.zh.vec
|       |-- ...
|       |-- wiki.tr.vec
|   |-- dictionaries
|       |-- en-zh.0-5000.txt
|       |-- en-zh.5000-6500.txt
|       |-- en-zh.txt
|       |-- ....
|       |-- en-tr.txt 
```

## Usage

All training configs are listed in folder `configs`. For example, you can run the following commands to train RAPO on en-zh datasets in supervised or semi-supervised settings.

```
# train RAPO on en-zh dataset in supervised setting
bash train_sup.sh configs/en-zh-sup.json

# train RAPO on en-zh dataset in semi-supervised setting
bash train_semi.sh configs/semi-en-zh-sup.json
```

The best checkpoint will be save in folder `models`
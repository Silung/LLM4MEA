# Introduction

Fork from https://github.com/Yueeeeeeee/RecSys-Extraction-Attack


## Requirements

PyTorch, pandas, wget, libarchive-c, faiss-cpu, tqdm, tensorboard. For our running environment see requirements.txt


## Train Black-Box Recommender Models

```bash
python train.py
```
Excecute the above command (with arguments) to train a black-box model, select datasets from Movielens 1M/20M, Beauty, Games, Steam and Yoochoose. Availabel models are NARM, SASRec and BERT4Rec. Trained black-box recommenders could be found under ./experiments/model-code/dataset-code/models/best_acc_model.pth


## Extract trained Black-Box Recommender Models

```bash
python distill.py
```
Excecute the above command (with arguments) to extract a white-box model, white-box model can also be chosen from NARM, SASRec and BERT4Rec. Trained models could be found under ./experiments/distillation_rank/distillation-specification/dataset-code/models/best_acc_model.pth



#!/bin/bash

# 88.08
#python predict.py -models BERT_fold1 BERT_fold2

# 88.39
#python predict.py -models BERT_fold1 BERT_fold2 BERT_fold3

# 88.11
#python predict.py -models BERT_PLoss0.03_fold1 BERT_PLoss0.03_fold2 BERT_PLoss0.03_fold3 BERT_PLoss0.03_fold4 BERT_PLoss0.03_fold5 

# 88.56   /epoch 50
#python predict.py -models BERT_PLoss0.24_fold1 BERT_PLoss0.24_fold2 BERT_PLoss0.24_fold3 BERT_PLoss0.24_fold4 BERT_PLoss0.24_fold5 

# 88.56   /epoch 50
python predict.py -models BERT_PLoss0.24_fold1 BERT_PLoss0.24_fold2 BERT_PLoss0.24_fold3 BERT_PLoss0.24_fold4 BERT_PLoss0.24_fold5 

# 89.4
#python predict.py -models BERT_fold1 BERT_fold2 BERT_fold3 TextRCNN_fold1_loss-eda_alpha0.1-n_aug0.5-embedding500 TextRCNN_fold2_loss-eda_alpha0.1-n_aug0.5-embedding500 TextRCNN_fold3_loss-eda_alpha0.1-n_aug0.5-embedding500 TextRCNN_fold4_loss-eda_alpha0.1-n_aug0.5-embedding500 TextRCNN_fold5_loss-eda_alpha0.1-n_aug0.5-embedding500

# 90.3
#python predict.py -models BERT_fold1 BERT_fold2 BERT_fold3 BERT_fold4 BERT_fold5 \
#TextRCNN_fold1_loss-eda_alpha0.1-n_aug0.5-embedding500 TextRCNN_fold2_loss-eda_alpha0.1-n_aug0.5-embedding500 TextRCNN_fold3_loss-eda_alpha0.1-n_aug0.5-embedding500 TextRCNN_fold4_loss-eda_alpha0.1-n_aug0.5-embedding500 TextRCNN_fold5_loss-eda_alpha0.1-n_aug0.5-embedding500 \
#TextCNN_fold1_loss-eda_alpha0.1-n_aug0.5-embedding500 TextCNN_fold2_loss-eda_alpha0.1-n_aug0.5-embedding500 TextCNN_fold3_loss-eda_alpha0.1-n_aug0.5-embedding500 TextCNN_fold4_loss-eda_alpha0.1-n_aug0.5-embedding500 TextCNN_fold5_loss-eda_alpha0.1-n_aug0.5-embedding500 \
#DPCNN_fold1_loss-eda_alpha0.1-n_aug0.5-embedding500 DPCNN_fold2_loss-eda_alpha0.1-n_aug0.5-embedding500 DPCNN_fold3_loss-eda_alpha0.1-n_aug0.5-embedding500 DPCNN_fold4_loss-eda_alpha0.1-n_aug0.5-embedding500 DPCNN_fold5_loss-eda_alpha0.1-n_aug0.5-embedding500

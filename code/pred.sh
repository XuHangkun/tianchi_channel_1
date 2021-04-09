#!/bin/bash

# 88.08
# python predict.py -models BERT_fold1

# 88.3 # max length 70, EDA
# python predict.py -models TextRCNN_fold1 TextRCNN_fold2 TextRCNN_fold3 TextRCNN_fold4 TextRCNN_fold5

# 88.3 # max length 60,no EDA
# python predict.py -models TextRCNN_fold1 TextRCNN_fold2 TextRCNN_fold3 TextRCNN_fold4 TextRCNN_fold5

# 90.11 # max length 60,no EDA
# python predict.py -models TextRCNN_fold1 TextRCNN_fold2 TextRCNN_fold3 TextRCNN_fold4 TextRCNN_fold5

# 88.39
#python predict.py -models BERT_fold1 BERT_fold2 BERT_fold3

# 88.11
#python predict.py -models BERT_PLoss0.03_fold1 BERT_PLoss0.03_fold2 BERT_PLoss0.03_fold3 BERT_PLoss0.03_fold4 BERT_PLoss0.03_fold5 

# 88.56   /epoch 50
#python predict.py -models BERT_PLoss0.24_fold1 BERT_PLoss0.24_fold2 BERT_PLoss0.24_fold3 BERT_PLoss0.24_fold4 BERT_PLoss0.24_fold5 

# 87.99   /epoch 80
# python predict.py -models BERT_PLoss0.24_fold1 BERT_PLoss0.24_fold2 BERT_PLoss0.24_fold3 BERT_PLoss0.24_fold4 BERT_PLoss0.24_fold5 

# 88.27   /epoch 50
# python predict.py -models BERT_PLoss0.24_fold1_epoch_50_loss_0.033 \
# BERT_PLoss0.24_fold2_epoch_50_loss_0.031 BERT_PLoss0.24_fold4_epoch_50_loss_0.032 BERT_PLoss0.24_fold5_epoch_50_loss_0.031 \
# -model_path /hpcfs/juno/junogpu/xuhangkun/ML/MyselfProject/tianchi_channel_1/user_data/model_data/Bert

# 89.19   /epoch 35
# python predict.py -models BERT_PLoss0.24_fold1_epoch_35_loss_0.038 BERT_PLoss0.24_fold2_epoch_35_loss_0.036 \
# BERT_PLoss0.24_fold3_epoch_35_loss_0.039 BERT_PLoss0.24_fold4_epoch_35_loss_0.036 BERT_PLoss0.24_fold5_epoch_35_loss_0.035 \
# -model_path /hpcfs/juno/junogpu/xuhangkun/ML/MyselfProject/tianchi_channel_1/user_data/model_data/Bert

# 89.54   /epoch 25
# python predict.py -models BERT_PLoss0.24_fold1_epoch_25_loss_0.046 BERT_PLoss0.24_fold2_epoch_25_loss_0.045 \
# BERT_PLoss0.24_fold3_epoch_25_loss_0.048 BERT_PLoss0.24_fold4_epoch_25_loss_0.045 BERT_PLoss0.24_fold5_epoch_25_loss_0.044 \
# -model_path /hpcfs/juno/junogpu/xuhangkun/ML/MyselfProject/tianchi_channel_1/user_data/model_data/Bert

# 90.6   /
# python predict.py -models Bert/BERT_PLoss0.24_fold1_epoch_25_loss_0.046 Bert/BERT_PLoss0.24_fold2_epoch_25_loss_0.045 \
# Bert/BERT_PLoss0.24_fold3_epoch_25_loss_0.048 Bert/BERT_PLoss0.24_fold4_epoch_25_loss_0.045 Bert/BERT_PLoss0.24_fold5_epoch_25_loss_0.044 \
# TextRCNN/TextRCNN_fold1 TextRCNN/TextRCNN_fold2 TextRCNN/TextRCNN_fold3 TextRCNN/TextRCNN_fold4 TextRCNN/TextRCNN_fold5 \
# -model_path /hpcfs/juno/junogpu/xuhangkun/ML/MyselfProject/tianchi_channel_1/user_data/model_data

# 90.70   /
#python predict.py -models Bert_new/BERT_PLoss0.64_fold1_epoch_13_loss_0.037 Bert_new/BERT_PLoss0.64_fold2_epoch_13_loss_0.035 \
#Bert_new/BERT_PLoss0.64_fold3_epoch_13_loss_0.040 Bert_new/BERT_PLoss0.64_fold4_epoch_13_loss_0.036 Bert_new/BERT_PLoss0.64_fold5_epoch_13_loss_0.036 \
#TextRCNN/TextRCNN_fold1 TextRCNN/TextRCNN_fold2 TextRCNN/TextRCNN_fold3 TextRCNN/TextRCNN_fold4 TextRCNN/TextRCNN_fold5 \
#-model_path /hpcfs/juno/junogpu/xuhangkun/ML/MyselfProject/tianchi_channel_1/user_data/model_data

# 90.76   /potential 
# python predict.py -models Bert_new/BERT_PLoss0.64_fold1_epoch_13_loss_0.037 Bert_new/BERT_PLoss0.64_fold2_epoch_13_loss_0.035 \
# Bert_new/BERT_PLoss0.64_fold3_epoch_13_loss_0.040 Bert_new/BERT_PLoss0.64_fold4_epoch_13_loss_0.036 Bert_new/BERT_PLoss0.64_fold5_epoch_13_loss_0.036 \
# TextRCNN/TextRCNN_fold1 TextRCNN/TextRCNN_fold2 TextRCNN/TextRCNN_fold3 TextRCNN/TextRCNN_fold4 TextRCNN/TextRCNN_fold5 \
# DPCNN/DPCNN_fold1_epoch_8_loss_0.049_enhance_True_pretrain_False \
# DPCNN/DPCNN_fold2_epoch_8_loss_0.053_enhance_True_pretrain_False \
# DPCNN/DPCNN_fold3_epoch_7_loss_0.060_enhance_True_pretrain_False \
# DPCNN/DPCNN_fold4_epoch_8_loss_0.056_enhance_True_pretrain_False \
# DPCNN/DPCNN_fold5_epoch_8_loss_0.047_enhance_True_pretrain_False \
# -model_path /hpcfs/juno/junogpu/xuhangkun/ML/MyselfProject/tianchi_channel_1/user_data/model_data

# 90.7   /
#python predict.py -models Bert_new/BERT_PLoss0.64_fold1_epoch_13_loss_0.037 Bert_new/BERT_PLoss0.64_fold2_epoch_13_loss_0.035 \
#Bert_new/BERT_PLoss0.64_fold3_epoch_13_loss_0.040 Bert_new/BERT_PLoss0.64_fold4_epoch_13_loss_0.036 Bert_new/BERT_PLoss0.64_fold5_epoch_13_loss_0.036 \
#TextRCNN/TextRCNN_fold1_epoch_30_loss_0.030_enhance_True_pretrain_False \
#TextRCNN/TextRCNN_fold2_epoch_30_loss_0.028_enhance_True_pretrain_False \
#TextRCNN/TextRCNN_fold3_epoch_30_loss_0.035_enhance_True_pretrain_False \
#TextRCNN/TextRCNN_fold4_epoch_30_loss_0.032_enhance_True_pretrain_False \
#TextRCNN/TextRCNN_fold5_epoch_30_loss_0.026_enhance_True_pretrain_False \
#DPCNN/DPCNN_fold1_epoch_8_loss_0.049_enhance_True_pretrain_False \
#DPCNN/DPCNN_fold2_epoch_8_loss_0.053_enhance_True_pretrain_False \
#DPCNN/DPCNN_fold3_epoch_7_loss_0.060_enhance_True_pretrain_False \
#DPCNN/DPCNN_fold4_epoch_8_loss_0.056_enhance_True_pretrain_False \
#DPCNN/DPCNN_fold5_epoch_8_loss_0.047_enhance_True_pretrain_False \
#-model_path /hpcfs/juno/junogpu/xuhangkun/ML/MyselfProject/tianchi_channel_1/user_data/model_data


# 90.4   /
# python predict.py -models Bert/BERT_PLoss0.24_fold1_epoch_25_loss_0.046 Bert/BERT_PLoss0.24_fold2_epoch_25_loss_0.045 \
# Bert/BERT_PLoss0.24_fold3_epoch_25_loss_0.048 Bert/BERT_PLoss0.24_fold4_epoch_25_loss_0.045 Bert/BERT_PLoss0.24_fold5_epoch_25_loss_0.044 \
# TextRCNN/TextRCNN_fold1 TextRCNN/TextRCNN_fold2 TextRCNN/TextRCNN_fold3 TextRCNN/TextRCNN_fold4 TextRCNN/TextRCNN_fold5 \
# TextCNN/TextCNN_fold1_epoch_25_loss_0.051_enhance_False_pretrain_False \
# TextCNN/TextCNN_fold2_epoch_25_loss_0.051_enhance_False_pretrain_False \
# TextCNN/TextCNN_fold3_epoch_25_loss_0.055_enhance_False_pretrain_False \
# TextCNN/TextCNN_fold4_epoch_25_loss_0.050_enhance_False_pretrain_False \
# TextCNN/TextCNN_fold5_epoch_25_loss_0.048_enhance_False_pretrain_False \
# -model_path /hpcfs/juno/junogpu/xuhangkun/ML/MyselfProject/tianchi_channel_1/user_data/model_data

# 90.4   /
#python predict.py -models Bert/BERT_PLoss0.24_fold1_epoch_25_loss_0.046 Bert/BERT_PLoss0.24_fold2_epoch_25_loss_0.045 \
#Bert/BERT_PLoss0.24_fold3_epoch_25_loss_0.048 Bert/BERT_PLoss0.24_fold4_epoch_25_loss_0.045 Bert/BERT_PLoss0.24_fold5_epoch_25_loss_0.044 \
#TextRCNN/TextRCNN_fold1 TextRCNN/TextRCNN_fold2 TextRCNN/TextRCNN_fold3 TextRCNN/TextRCNN_fold4 TextRCNN/TextRCNN_fold5 \
#DPCNN/DPCNN_fold1_epoch_12_loss_0.060_enhance_False_pretrain_False \
#DPCNN/DPCNN_fold2_epoch_14_loss_0.066_enhance_False_pretrain_False \
#DPCNN/DPCNN_fold3_epoch_13_loss_0.060_enhance_False_pretrain_False \
#DPCNN/DPCNN_fold4_epoch_13_loss_0.066_enhance_False_pretrain_False \
#DPCNN/DPCNN_fold5_epoch_17_loss_0.057_enhance_False_pretrain_False \
#-model_path /hpcfs/juno/junogpu/xuhangkun/ML/MyselfProject/tianchi_channel_1/user_data/model_data


# 89.48   /epoch 20
# python predict.py -models BERT_PLoss0.24_fold1_epoch_20_loss_0.054 BERT_PLoss0.24_fold2_epoch_20_loss_0.054 \
# BERT_PLoss0.24_fold3_epoch_20_loss_0.056 BERT_PLoss0.24_fold4_epoch_20_loss_0.053 BERT_PLoss0.24_fold5_epoch_20_loss_0.052 \
# -model_path /hpcfs/juno/junogpu/xuhangkun/ML/MyselfProject/tianchi_channel_1/user_data/model_data/Bert

# 89.48   /epoch 20
# python predict.py -models BERT_PLoss0.24_fold1_epoch_14_loss_0.034 BERT_PLoss0.24_fold2_epoch_14_loss_0.034 \
# BERT_PLoss0.24_fold3_epoch_14_loss_0.037 BERT_PLoss0.24_fold4_epoch_14_loss_0.034 BERT_PLoss0.24_fold5_epoch_14_loss_0.034 \
# -model_path /hpcfs/juno/junogpu/xuhangkun/ML/MyselfProject/tianchi_channel_1/user_data/model_data/Bert_new

# 89.4
#python predict.py -models BERT_fold1 BERT_fold2 BERT_fold3 TextRCNN_fold1_loss-eda_alpha0.1-n_aug0.5-embedding500 TextRCNN_fold2_loss-eda_alpha0.1-n_aug0.5-embedding500 TextRCNN_fold3_loss-eda_alpha0.1-n_aug0.5-embedding500 TextRCNN_fold4_loss-eda_alpha0.1-n_aug0.5-embedding500 TextRCNN_fold5_loss-eda_alpha0.1-n_aug0.5-embedding500

# 90.3
#python predict.py -models BERT_fold1 BERT_fold2 BERT_fold3 BERT_fold4 BERT_fold5 \
#TextRCNN_fold1_loss-eda_alpha0.1-n_aug0.5-embedding500 TextRCNN_fold2_loss-eda_alpha0.1-n_aug0.5-embedding500 TextRCNN_fold3_loss-eda_alpha0.1-n_aug0.5-embedding500 TextRCNN_fold4_loss-eda_alpha0.1-n_aug0.5-embedding500 TextRCNN_fold5_loss-eda_alpha0.1-n_aug0.5-embedding500 \
#TextCNN_fold1_loss-eda_alpha0.1-n_aug0.5-embedding500 TextCNN_fold2_loss-eda_alpha0.1-n_aug0.5-embedding500 TextCNN_fold3_loss-eda_alpha0.1-n_aug0.5-embedding500 TextCNN_fold4_loss-eda_alpha0.1-n_aug0.5-embedding500 TextCNN_fold5_loss-eda_alpha0.1-n_aug0.5-embedding500 \
#DPCNN_fold1_loss-eda_alpha0.1-n_aug0.5-embedding500 DPCNN_fold2_loss-eda_alpha0.1-n_aug0.5-embedding500 DPCNN_fold3_loss-eda_alpha0.1-n_aug0.5-embedding500 DPCNN_fold4_loss-eda_alpha0.1-n_aug0.5-embedding500 DPCNN_fold5_loss-eda_alpha0.1-n_aug0.5-embedding500

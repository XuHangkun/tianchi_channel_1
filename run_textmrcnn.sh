#!/bin/bash

# we need to set a enviroment variable here
cd .
export PROJTOP=$(pwd)
cd -

nvidia-smi

############ TRAIN ################
# 5-fold train TextRCNN
python ./code/train.py -model TextMRCNN -epoch 30 -no_word2vec_pretrain -seed 7 \
-output_dir ./user_data/model_data/TextRCNN -lr 1.e-3 -input ./tcdata/train.csv

# 5-fold train DPCNN
#python ./code/train.py -model DPCNN -eda_alpha 0.1 -n_aug 0.5 -epoch 8 -no_word2vec_pretrain \
#-output_dir ./user_data/model_data/DPCNN -lr 1.e-3 -save_mode all -save_start_epoch 1 \
#-save_per_epoch 1

# 5-fold train BERT
#python ./code/train.py -model BERT -epoch 15 -no_word2vec_pretrain \
#-bert_path ./user_data/bert/checkpoint-15800 \
#-save_mode all -save_start_epoch 10 -save_per_epoch 3 \
#-output_dir ./user_data/model_data/Bert_new \
#-lr 5.e-5



########### TEST ###################
# python ./code/predict.py -input ./tcdata/test.csv \
# -models Bert_new/BERT_fold1_epoch_13 Bert_new/BERT_fold2_epoch_13 \
# Bert_new/BERT_fold3_epoch_13 Bert_new/BERT_fold4_epoch_13 Bert_new/BERT_fold5_epoch_13 \
# TextRCNN/TextRCNN_fold1 TextRCNN/TextRCNN_fold2 TextRCNN/TextRCNN_fold3 TextRCNN/TextRCNN_fold4 TextRCNN/TextRCNN_fold5 \
# DPCNN/DPCNN_fold1_epoch_8 \
# DPCNN/DPCNN_fold2_epoch_8 \
# DPCNN/DPCNN_fold3_epoch_7 \
# DPCNN/DPCNN_fold4_epoch_8 \
# DPCNN/DPCNN_fold5_epoch_8 \
# -model_path ./user_data/model_data \
# -tokenizer_path ./user_data/bert \
# -output ./prediction_result/result.csv

python ./code/predict.py -input ./tcdata/testA.csv \
-models TextRCNN/TextMRCNN_fold1 TextRCNN/TextMRCNN_fold2 TextRCNN/TextMRCNN_fold3 TextRCNN/TextMRCNN_fold4 TextRCNN/TextMRCNN_fold5 \
-model_path ./user_data/model_data \
-output ./result.csv

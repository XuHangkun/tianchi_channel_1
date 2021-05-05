#!/bin/bash

# we need to set a enviroment variable here
cd .
export PROJTOP=$(pwd)
cd -

nvidia-smi

############ TRAIN ################

# 5-fold train TextRCNN
# python ./code/train.py -model TextRCNN -epoch 30 -no_word2vec_pretrain \
# -output_dir ./user_data/model_data/TextRCNN -lr 1.e-3 -input ./tcdata/train.csv

# 5-fold train DPCNN
#python ./code/train.py -model DPCNN -eda_alpha 0.1 -n_aug 0.5 -epoch 8 -no_word2vec_pretrain \
#-output_dir ./user_data/model_data/DPCNN -lr 1.e-3 -save_mode all -save_start_epoch 1 \
#-save_per_epoch 1

# BERT pretrain
#python ./code/bert_pretrain.py -epoch 100

# 5-fold train BERT
#python ./code/train.py -model BERT -epoch 15 -no_word2vec_pretrain \
#-bert_path ./user_data/bert \
#-save_mode best \
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

# TEST by TextRCNN
#python ./code/predict.py -input ./tcdata/testA.csv \
#-models TextRCNN/TextRCNN_fold1 TextRCNN/TextRCNN_fold2 TextRCNN/TextRCNN_fold3 TextRCNN/TextRCNN_fold4 TextRCNN/TextRCNN_fold5 \
#-model_path ./user_data/model_data \
#-output ./result.csv

########## PREDICTION #############
python ./code/predict.py -input ./tcdata/testA.csv \
-models TextRCNN/TextRCNN_task1_fold1 TextRCNN/TextRCNN_task1_fold2 TextRCNN/TextRCNN_task1_fold3 TextRCNN/TextRCNN_task1_fold4 TextRCNN/TextRCNN_task1_fold5 \
-model_path ./user_data/model_data \
-output ./user_data/tmp_data/result_1.csv

########## PREDICTION #############
python ./code/predict.py -input ./tcdata/testA.csv \
-models TextRCNN/TextRCNN_task2_fold1 TextRCNN/TextRCNN_task2_fold2 TextRCNN/TextRCNN_task2_fold3 TextRCNN/TextRCNN_task2_fold4 TextRCNN/TextRCNN_task2_fold5 \
-model_path ./user_data/model_data \
-output ./user_data/tmp_data/result_2.csv
### concat all result
python code/concat_two_result.py -task1_file ./user_data/tmp_data/result_1.csv -task2_file ./user_data/tmp_data/result_2.csv -output ./user_data/tmp_data/result_rcnn.csv   


# TEST by Bert
python ./code/predict.py -input ./tcdata/testA.csv \
-models Bert_new/BERT_fold1 Bert_new/BERT_fold2 \
Bert_new/BERT_fold3 Bert_new/BERT_fold4 Bert_new/BERT_fold5 \
-model_path ./user_data/model_data \
-tokenizer_path ./user_data/bert \
-output ./user_data/tmp_data/result_bert.csv


### concat all result
python code/mix_results.py -in_files ./user_data/tmp_data/result_bert.csv ./user_data/tmp_data/result_rcnn.csv -output ./result.csv 

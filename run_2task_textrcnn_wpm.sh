#!/bin/bash

# we need to set a enviroment variable here
cd .
export PROJTOP=$(pwd)
cd -

nvidia-smi

## Par
word_size=100
pre_train_epoch=5
train_epoch=3

####### TRAIN WORD VEC ############
# python ./code/train_word_vector.py -epoch 12 -word_size ${word_size}

####### PRETRAIN  ################
python ./code/train.py -model TextRCNN_task1 -model_task 1 -epoch ${pre_train_epoch} -seed 13 \
-output_dir ./user_data/model_data/TextRCNN -lr 5.e-4 -input ./tcdata/train.csv \
-dropout 0.5 -nemb ${word_size} -max_len 100 -hidden_size 1024 -lstm_dropout 0.1 \
-eda_alpha 0.2 -n_aug 4.0 -save_mode best -is_pretrain -no_word2vec_pretrain

############ TRAIN ################
# 5-fold train TextRCNN
python ./code/train.py -model TextRCNN_task1 -model_task 1 -epoch ${train_epoch} -seed 7 -no_word2vec_pretrain \
-output_dir ./user_data/model_data/TextRCNN -lr 5.e-4 -input ./tcdata/train.csv \
-dropout 0.5 -nemb ${word_size} -max_len 100 -hidden_size 1024 -lstm_dropout 0.1 \
-eda_alpha 0.0 -n_aug 0 -model_path ./user_data/model_data/TextRCNN/TextRCNN_task1.chkpt

########## PREDICTION #############
python ./code/predict.py -input ./tcdata/testA.csv \
-models TextRCNN/TextRCNN_task1_fold1 TextRCNN/TextRCNN_task1_fold2 TextRCNN/TextRCNN_task1_fold3 TextRCNN/TextRCNN_task1_fold4 TextRCNN/TextRCNN_task1_fold5 \
-model_path ./user_data/model_data \
-output ./user_data/tmp_data/result_1.csv


####### PRETRAIN  ################
python ./code/train.py -model TextRCNN_task2 -model_task 2 -epoch ${pre_train_epoch} -seed 13 \
-output_dir ./user_data/model_data/TextRCNN -lr 5.e-4 -input ./tcdata/train.csv \
-dropout 0.5 -nemb ${word_size} -max_len 100 -hidden_size 1024 -lstm_dropout 0.1 \
-eda_alpha 0.2 -n_aug 4.0 -save_mode best -is_pretrain -no_word2vec_pretrain

############ TRAIN ################
# 5-fold train TextRCNN
python ./code/train.py -model TextRCNN_task2 -model_task 2 -epoch ${train_epoch} -seed 7 -no_word2vec_pretrain \
-output_dir ./user_data/model_data/TextRCNN -lr 5.e-4 -input ./tcdata/train.csv \
-dropout 0.5 -nemb ${word_size} -max_len 100 -hidden_size 1024 -lstm_dropout 0.1 \
-eda_alpha 0.0 -n_aug 0 -model_path ./user_data/model_data/TextRCNN/TextRCNN_task2.chkpt

########## PREDICTION #############
python ./code/predict.py -input ./tcdata/testA.csv \
-models TextRCNN/TextRCNN_task2_fold1 TextRCNN/TextRCNN_task2_fold2 TextRCNN/TextRCNN_task2_fold3 TextRCNN/TextRCNN_task2_fold4 TextRCNN/TextRCNN_task2_fold5 \
-model_path ./user_data/model_data \
-output ./user_data/tmp_data/result_2.csv

### concat all result
python code/concat_two_result.py -task1_file ./user_data/tmp_data/result_1.csv -task2_file ./user_data/tmp_data/result_2.csv -output ./result.csv   

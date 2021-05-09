#!/bin/bash

# we need to set a enviroment variable here
cd .
export PROJTOP=$(pwd)
cd -

nvidia-smi

########## Tokenizer ##############
python ./code/generate_tokenizer.py -input ./tcdata/train.csv -output ./user_data/tokenizer/tokenizer.pkl 

########## TextRCNN  ##############
############ TRAIN ################
# 5-fold train TextRCNN
python ./code/train.py -model TextRCNN_task1 -model_task 1 -epoch 5 -no_word2vec_pretrain -seed 7 \
-output_dir ./user_data/model_data/TextRCNN -lr 5.e-4 -input ./tcdata/train.csv \
-dropout 0.5 -nemb 100 -max_len 100 -hidden_size 1024 -eda_alpha 0.15 -n_aug 4 \
-lstm_dropout 0.1 -tokenizer_path ./user_data/tokenizer/tokenizer.pkl -batch_size 128
########## PREDICTION #############
python ./code/predict.py -input ./tcdata/testB.csv \
-models TextRCNN/TextRCNN_task1_fold1 TextRCNN/TextRCNN_task1_fold2 TextRCNN/TextRCNN_task1_fold3 TextRCNN/TextRCNN_task1_fold4 TextRCNN/TextRCNN_task1_fold5 \
-model_path ./user_data/model_data -tokenizer_file ./user_data/tokenizer/tokenizer.pkl \
-output ./user_data/result_textrcnn_task1.csv
# 5-fold train TextRCNN
python ./code/train.py -model TextRCNN_task2 -model_task 2 -epoch 5 -no_word2vec_pretrain -seed 7 \
-output_dir ./user_data/model_data/TextRCNN -lr 5.e-4 -input ./tcdata/train.csv \
-dropout 0.5 -nemb 100 -max_len 100 -hidden_size 1024 -eda_alpha 0.15 -n_aug 4 \
-lstm_dropout 0.1 -tokenizer_path ./user_data/tokenizer/tokenizer.pkl -batch_size 128
########## PREDICTION #############
python ./code/predict.py -input ./tcdata/testB.csv \
-models TextRCNN/TextRCNN_task2_fold1 TextRCNN/TextRCNN_task2_fold2 TextRCNN/TextRCNN_task2_fold3 TextRCNN/TextRCNN_task2_fold4 TextRCNN/TextRCNN_task2_fold5 \
-model_path ./user_data/model_data -tokenizer_file ./user_data/tokenizer/tokenizer.pkl \
-output ./user_data/result_textrcnn_task2.csv
### concat all result
python code/concat_two_result.py -task1_file ./user_data/result_textrcnn_task1.csv -task2_file ./user_data/result_textrcnn_task2.csv -output ./user_data/result_textrcnn.csv

########## TextRCNNCs  ##############
############ TRAIN ################
# 5-fold train TextRCNNCs
python ./code/train.py -model TextRCNNCs -epoch 5 -no_word2vec_pretrain -seed 7 \
-output_dir ./user_data/model_data/TextRCNN -lr 5.e-4 -input ./tcdata/train.csv \
-dropout 0.5 -nemb 100 -max_len 100 -hidden_size 1024 -eda_alpha 0.15 -n_aug 4 \
-lstm_dropout 0.1 -tokenizer_path ./user_data/tokenizer/tokenizer.pkl -batch_size 128
########## PREDICTION #############
python ./code/predict.py -input ./tcdata/testB.csv \
-models TextRCNN/TextRCNNCs_fold1 TextRCNN/TextRCNNCs_fold2 TextRCNN/TextRCNNCs_fold3 TextRCNN/TextRCNNCs_fold4 TextRCNN/TextRCNNCs_fold5 \
-model_path ./user_data/model_data -tokenizer_file ./user_data/tokenizer/tokenizer.pkl \
-output ./user_data/result_textrcnncs.csv

########## DPCNN  ##############
############ TRAIN ################
# 5-fold train DPCNN
python ./code/train.py -model DPCNN -epoch 4 -no_word2vec_pretrain -seed 7 \
-output_dir ./user_data/model_data/TextRCNN -lr 5.e-4 -input ./tcdata/train.csv \
-dropout 0.5 -nemb 100 -max_len 100 -num_filters 512 -eda_alpha 0.15 -n_aug 4 \
-lstm_dropout 0.1 -tokenizer_path ./user_data/tokenizer/tokenizer.pkl -batch_size 128
########## PREDICTION #############
python ./code/predict.py -input ./tcdata/testB.csv \
-models TextRCNN/DPCNN_fold1 TextRCNN/DPCNN_fold2 TextRCNN/DPCNN_fold3 TextRCNN/DPCNN_fold4 TextRCNN/DPCNN_fold5 \
-model_path ./user_data/model_data -tokenizer_file ./user_data/tokenizer/tokenizer.pkl \
-output ./user_data/result_dpcnn.csv

########## Text  ##############
############ TRAIN ################
# 5-fold train TextCNN
python ./code/train.py -model TextCNN -epoch 8 -no_word2vec_pretrain -seed 7 \
-output_dir ./user_data/model_data/TextRCNN -lr 5.e-4 -input ./tcdata/train.csv \
-dropout 0.5 -nemb 100 -max_len 100 -num_filters 128 -eda_alpha 0.15 -n_aug 4 \
-lstm_dropout 0.1 -tokenizer_path ./user_data/tokenizer/tokenizer.pkl -batch_size 128

########## PREDICTION #############
python ./code/predict.py -input ./tcdata/testB.csv \
-models TextRCNN/TextCNN_fold1 TextRCNN/TextCNN_fold2 TextRCNN/TextCNN_fold3 TextRCNN/TextCNN_fold4 TextRCNN/TextCNN_fold5 \
-model_path ./user_data/model_data -tokenizer_file ./user_data/tokenizer/tokenizer.pkl \
-output ./user_data/result_textcnn.csv

######## Mix All Result ##########
python ./code/mix_results.py -in_files ./user_data/result_textrcnn.csv ./user_data/result_textrcnncs.csv ./user_data/result_dpcnn.csv ./user_data/result_textcnn.csv \
-in_weights 0.35 0.3 0.25 0.1

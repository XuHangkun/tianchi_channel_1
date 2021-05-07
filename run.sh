#!/bin/bash

# we need to set a enviroment variable here
cd .
export PROJTOP=$(pwd)
cd -

nvidia-smi

########## Tokenizer ##############
python ./code/generate_tokenizer.py -input ./tcdata/train.csv -output ./user_data/tokenizer/tokenizer.pkl 

############ TRAIN ################
# 5-fold train TextRCNN
python ./code/train.py -model TextRCNN -epoch 12 -no_word2vec_pretrain -seed 7 \
-output_dir ./user_data/model_data/TextRCNN -lr 5.e-4 -input ./tcdata/train.csv \
-dropout 0.5 -nemb 100 -max_len 100 -hidden_size 1024 -eda_alpha 0.15 -n_aug 4 \
-lstm_dropout 0.1 -tokenizer_path ./user_data/tokenizer/tokenizer.pkl -batch_size 128

########## PREDICTION #############
python ./code/predict.py -input ./tcdata/testA.csv \
-models TextRCNN/TextRCNN_fold1 TextRCNN/TextRCNN_fold2 TextRCNN/TextRCNN_fold3 TextRCNN/TextRCNN_fold4 TextRCNN/TextRCNN_fold5 \
-model_path ./user_data/model_data -tokenizer_file ./user_data/tokenizer/tokenizer.pkl \
-output ./result.csv

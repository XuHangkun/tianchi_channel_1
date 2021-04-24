#!/bin/bash

# we need to set a enviroment variable here
cd .
export PROJTOP=$(pwd)
cd -

nvidia-smi

############ TRAIN ################
# 5-fold train TextRCNN
python ./code/train.py -model TextRCNN -epoch 100 -no_word2vec_pretrain -seed 7 \
-output_dir ./user_data/model_data/TextRCNN -lr 4.e-5 -input ./tcdata/train.csv \
-dropout 0.5 -nemb 100 -max_len 100 -hidden_size 512

########## PREDICTION #############
python ./code/predict.py -input ./tcdata/testA.csv \
-models TextRCNN/TextRCNN_fold1 TextRCNN/TextRCNN_fold2 TextRCNN/TextRCNN_fold3 TextRCNN/TextRCNN_fold4 TextRCNN/TextRCNN_fold5 \
-model_path ./user_data/model_data \
-output ./result.csv

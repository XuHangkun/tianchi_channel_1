#!/bin/bash

# we need to set a enviroment variable here
cd ..
export PROJTOP=$(pwd)
cd -

# 5-fold train TextRCNN
python train.py -model TextRCNN -epoch 30 -no_word2vec_pretrain \
-output_dir ../user_data/model_data/TextRCNN -lr 1.e-3

# 5-fold train DPCNN
python train.py -model DPCNN -eda_alpha 0.1 -n_aug 0.5 -epoch 8 -no_word2vec_pretrain \
-output_dir ../user_data/model_data/DPCNN -lr 1.e-3 -save_mode all -save_start_epoch 1 \
-save_per_epoch 1

# 5-fold train BERT
python train.py -model BERT_PLoss0.64 -epoch 30 \
-bert_path /hpcfs/juno/junogpu/xuhangkun/ML/MyselfProject/tianchi_channel_1/user_data/bert/checkpoint-15800 \
-save_mode all -save_start_epoch 10 -save_per_epoch 3 -fold_index 5 \
-output_dir /hpcfs/juno/junogpu/xuhangkun/ML/MyselfProject/tianchi_channel_1/user_data/model_data/Bert_new \
-lr 5.e-5
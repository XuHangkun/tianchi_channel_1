#!/bin/bash

# we need to set a enviroment variable here
cd ..
export PROJTOP=$(pwd)
cd -

# python predict.py -input ../tcdata/medical_nlp_round1_data/test.csv \
# -models Bert_new/BERT_fold1_epoch_13 Bert_new/BERT_fold2_epoch_13 \
# Bert_new/BERT_fold3_epoch_13 Bert_new/BERT_fold4_epoch_13 Bert_new/BERT_fold5_epoch_13 \
# TextRCNN/TextRCNN_fold1 TextRCNN/TextRCNN_fold2 TextRCNN/TextRCNN_fold3 TextRCNN/TextRCNN_fold4 TextRCNN/TextRCNN_fold5 \
# DPCNN/DPCNN_fold1_epoch_8 \
# DPCNN/DPCNN_fold2_epoch_8 \
# DPCNN/DPCNN_fold3_epoch_7 \
# DPCNN/DPCNN_fold4_epoch_8 \
# DPCNN/DPCNN_fold5_epoch_8 \
# -model_path ../user_data/model_data \
# -tokenizer_path ../user_data/bert \
# -output ../prediction_result/result.csv

python predict.py -input ../tcdata/test.csv \
-models TextRCNN/TextRCNN_fold1 TextRCNN/TextRCNN_fold2 TextRCNN/TextRCNN_fold3 TextRCNN/TextRCNN_fold4 TextRCNN/TextRCNN_fold5 \
-model_path ../user_data/model_data \
-output ../prediction_result/result.csv





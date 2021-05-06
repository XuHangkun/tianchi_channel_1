#!/bin/bash

# we need to set a enviroment variable here
cd .
export PROJTOP=$(pwd)
cd -

nvidia-smi

########## PREDICTION #############
python ./code/predict.py -input ./tcdata/testA.csv \
-models TextRCNN/TextRCNN_fold1 TextRCNN/TextRCNN_fold2 TextRCNN/TextRCNN_fold3 TextRCNN/TextRCNN_fold4 TextRCNN/TextRCNN_fold5 \
-model_path ./user_data/model_data \
-output ./result.csv

########## PREDICTION #############
#python ./code/predict.py -input ./tcdata/testA.csv \
#-models TextRCNN/TextRCNN_task2_fold1 TextRCNN/TextRCNN_task2_fold2 TextRCNN/TextRCNN_task2_fold3 TextRCNN/TextRCNN_task2_fold4 TextRCNN/TextRCNN_task2_fold5 \
#-model_path ./user_data/model_data \
#-output ./user_data/tmp_data/result_2.csv
#### concat all result
#python code/concat_two_result.py -task1_file ./user_data/tmp_data/result_1.csv -task2_file ./user_data/tmp_data/result_2.csv -output ./user_data/tmp_data/result_rcnn.csv   
#
#
## TEST by Bert
#python ./code/predict.py -input ./tcdata/testA.csv \
#-models Bert_new/BERT_fold1 Bert_new/BERT_fold2 \
#Bert_new/BERT_fold3 Bert_new/BERT_fold4 Bert_new/BERT_fold5 \
#-model_path ./user_data/model_data \
#-bert_tokenizer_dir ./user_data/bert \
#-output ./user_data/tmp_data/result_bert.csv


### concat all result
# python code/mix_results.py -in_files ./user_data/tmp_data/result_bert.csv ./user_data/tmp_data/result_rcnn.csv -output ./result.csv 

#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
    pretrain the word model, word2vec and glove
    ~~~~~~~~~~~~~~~~~~~~~~

    :author: Xu Hangkun (许杭锟),Fu Yangsheng,Huang Zheng
    :copyright: © 2020 Xu Hangkun <xuhangkun@ihep.ac.cn>
    :license: MIT, see LICENSE for more details.
"""
import os
import gensim
import pandas as pd
import torch
import argparse
# from glove import Glove
# from glove import Corpus
from utils.EDA import RandomDelete,RandomSwap
import pickle
import numpy as np

def easy_data_augmentation(texts,eda_alpha=0.1,n_aug=4):
    """
    Data Enhancement, randomly delete partial words or swap the words
    For evergy sentence, we need to change eda_alpha*sentence_len words.
    """
    def concat_words(words):
        sentence = ""
        for word in words:
            sentence += "%s "%(word)
        return sentence

    enhanced_texts = []
    if n_aug == 0:
        return
    for i in range(len(texts)):
        true_aug = 0
        if n_aug >1:
            true_aug = int(n_aug)
        elif n_aug >= 0:
            if np.random.random() < n_aug:
                true_aug = 1
        for j in range(true_aug):
            # randomly delete some words
            enhanced_texts.append(RandomDelete(texts[i],eda_alpha))
            # randomly swap some words
            enhanced_texts.append(RandomSwap(texts[i],eda_alpha))
    texts += enhanced_texts
    # randomly break up the data
    for i in range(len(texts)):
        text_1_index = int(np.random.random()*len(texts))
        text_2_index = int(np.random.random()*len(texts))
        x = texts[text_1_index]
        texts[text_1_index] = texts[text_2_index]
        texts[text_2_index] = x

    return texts

parser = argparse.ArgumentParser()
parser.add_argument('-word_size',default=100,type=int,help="dimension of a word vector")
parser.add_argument('-epoch',type=int,default=20,help="epoch")
parser.add_argument('-batch_size',type=int,default=128,help="epoch")
parser.add_argument('-corpus_dir',default=os.path.join(os.getenv('PROJTOP'),'tcdata'),help="dir of corpus")
parser.add_argument('-output',default=os.path.join(os.getenv('PROJTOP'),'user_data/word_pretrain/word2vector.model'),help="dimension of a word vector")
parser.add_argument('-not_do_eda',action="store_true",help="out dir of tokenizer and pretrained model")
opt = parser.parse_args()

# read csv data
print('read data and preprocess the data')
reports = []
corpus_input = ["track1_round1_train_20210222.csv","track1_round1_testA_20210222.csv","track1_round1_testB.csv","train.csv"]
corpus_input_tag = [0,1,1,0]
for corpus_file,tag in zip(corpus_input,corpus_input_tag):
    if tag:
        train_df = pd.read_csv(os.path.join(opt.corpus_dir,corpus_file),sep="\|,\|",names=["id","report"],index_col=0)
    else:
        train_df = pd.read_csv(os.path.join(opt.corpus_dir,corpus_file),sep="\|,\|",names=["id","report","label"],index_col=0)
    for i in range(len(train_df)):
        reports.append(train_df["report"][i].split())

# Do data angumentation here
if opt.not_do_eda:
    pass
else:
    reports = easy_data_augmentation(reports)
# print(reports)

# Train the word2vec model
print('train the word2vec model')
model = gensim.models.Word2Vec(reports, min_count=1,vector_size=opt.word_size,hs=1)
model.train(reports,epochs=opt.epoch,total_examples=len(reports))
print(model.wv)

# Train Glovec
# print('train the glovec model')
# corpus_model = Corpus()
# corpus_model.fit(reports, window=6)
# glove = Glove(no_components=opt.word_size, learning_rate=0.05)
# glove.fit(corpus_model.matrix, epochs=100,
#           no_threads=1, verbose=True)
# glove.add_dictionary(corpus_model.dictionary)

# cancate the two word size
info = {
        "word_size":opt.word_size,
        "wv":{
            }
        }
# print(model.wv.key_to_index)
for word in model.wv.key_to_index:
    info["wv"][word] = torch.Tensor(model.wv[word])

f = open(opt.output,'wb')
pickle.dump(info,f)

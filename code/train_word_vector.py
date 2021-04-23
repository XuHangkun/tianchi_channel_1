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
import pickle


parser = argparse.ArgumentParser()
parser.add_argument('-word_size',default=300,type=int,help="dimension of a word vector")
parser.add_argument('-epoch',type=int,default=100,help="epoch")
parser.add_argument('-batch_size',type=int,default=128,help="epoch")
parser.add_argument('-corpus_dir',default=os.path.join(os.getenv('PROJTOP'),'tcdata'),help="dir of corpus")
parser.add_argument('-output',default=os.path.join(os.getenv('PROJTOP'),'user_data/word_pretrain/word2vector.model'),help="dimension of a word vector")
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
# print(reports)

# Train the word2vec model
print('train the word2vec model')
model = gensim.models.Word2Vec(reports, min_count=1,size=opt.word_size,hs=1)
model.train(reports,epochs=100,total_examples=len(reports))
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
for word in model.wv.vocab:
    info["wv"][word] = torch.Tensor(model.wv[word])
    print(info["wv"][word])

f = open(opt.output,'wb')
pickle.dump(info,f)

#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
    pretrain the word model, word2vec and glove
    ~~~~~~~~~~~~~~~~~~~~~~

    :author: Xu Hangkun (许杭锟)
    :copyright: © 2020 Xu Hangkun <xuhangkun@ihep.ac.cn>
    :license: MIT, see LICENSE for more details.
"""
import os
import gensim
import pandas as pd
import torch
import argparse
from glove import Glove
from glove import Corpus
import pickle


parser = argparse.ArgumentParser()
parser.add_argument('-word_size',default=250,type=int,help="dimension of a word vector")
parser.add_argument('-output',default=os.path.join(os.getenv('PROJTOP'),'user_data/word_pretrain/word2vector.model'),help="dimension of a word vector")
opt = parser.parse_args()

# read csv data
print('read data and preprocess the data')
train_df = pd.read_csv(os.path.join(os.getenv('PROJTOP'),'tcdata/medical_nlp_round1_data/train.csv'),sep="\|,\|",names=["id","report","label"],index_col=0)
reports = []
for index in range(len(train_df)):
    report = train_df['report'][index].split()
    reports.append(report)

# Train the word2vec model
print('train the word2vec model')
model = gensim.models.Word2Vec(reports, min_count=1,size=opt.word_size,hs=1)
model.train(epochs=100)
print(model.wv)

# Train Glovec
print('train the glovec model')
corpus_model = Corpus()
corpus_model.fit(reports, window=6)
glove = Glove(no_components=opt.word_size, learning_rate=0.05)
glove.fit(corpus_model.matrix, epochs=100,
          no_threads=1, verbose=True)
glove.add_dictionary(corpus_model.dictionary)

# cancate the two word size
info = {
        "word_size":2*opt.word_size,
        "wv":{
            }
        }
for word in model.wv.vocab:
    info["wv"][word] = torch.cat((torch.Tensor(model.wv[word]),torch.Tensor(glove.word_vectors[glove.dictionary[word]])),0)
    print(info["wv"][word].shape)

f = open(opt.output,'wb')
pickle.dump(info,f)

# -*- coding: utf-8 -*-
"""
    tokenizer of reports
    ~~~~~~~~~~~~~~~~~~~~~~

    :author: Xu Hangkun (许杭锟)
    :copyright: © 2020 Xu Hangkun <xuhangkun@ihep.ac.cn>
    :license: MIT, see LICENSE for more details.
"""

import pandas as pd
import numpy as np
from collections import Counter
import pickle

class Tokenizer:

    def __init__(self,high_freq_n=0,mul_words_num=0):
        self.use_vocab = False
        self.mul_words_num = mul_words_num
        self.high_freq_n = high_freq_n
        self.high_freq_words = []
        self.word2index = {}
        self.index2word = []

    def initialize(self,df):
        self.df = df
        print("initialize the vocab")
        self.high_freq_words = self.find_high_freq_word(self.high_freq_n)
        self.word2index = {"<pad>":0}
        self.index2word = ["pad"]
        self.make_vocab(self.mul_words_num)
        self.use_vocab = True

    def normal_initialize(self,df):
        print("initialize the vocab")
        self.index2word = [str(i) for i in range(859)]
        for i in range(858):
            self.word2index[str(i)]=i
        self.word2index["<pad>"]=858


    def padding_idx(self):
        return self.word2index["<pad>"]

    def vocab_num(self):
        return len(self.index2word)

    def save(self,filename):
        p_file = open(filename,"wb")
        all_info = {
                "high_freq_n":self.high_freq_n,
                "mul_words_num":self.mul_words_num,
                "high_freq_words":self.high_freq_words,
                "word2index":self.word2index,
                "index2word":self.index2word
                }
        pickle.dump(all_info,p_file)
        p_file.close()

    def load(self,filename):
        p_file = open(filename,"rb")
        all_info = pickle.load(p_file)
        self.mul_words_num = all_info["mul_words_num"]
        self.high_freq_n = all_info["high_freq_n"]
        self.high_freq_words = all_info["high_freq_words"]
        self.word2index = all_info["word2index"]
        self.index2word = all_info["index2word"]
        p_file.close()
        self.use_vocab = True

    def find_high_freq_word(self,n=4):
        counter = Counter()
        for report in self.df["report"]:
            counter += Counter(report.split())
        return [x[0] for x in counter.most_common(n)]

    def del_high_freq_words(self,report):
        new_report = ""
        for i in report.split():
            if i in self.high_freq_words:
                pass
            else:
                new_report += "%s "%(i)
        return new_report.strip()

    def make_vocab(self,mul_words_num=860):
        print("Make Vocab ...")
        mul_word_counter = Counter()
        for report in self.df["report"]:
            tmp = []
            report = self.del_high_freq_words(report).split()
            i = 0
            while i < len(report):
                if i < (len(report) - 1):
                    tmp.append("%s %s"%(report[i],report[i+1]))
                if i < (len(report) - 2):
                    tmp.append("%s %s %s"%(report[i],report[i+1],report[i+1]))
                if i < (len(report) - 3):
                    tmp.append("%s %s %s %s"%(report[i],report[i+1],report[i+2],report[i+3]))
                i += 1
            mul_word_counter += Counter(tmp)

        for item in mul_word_counter.most_common(mul_words_num):
            self.index2word.append(item[0])
            self.word2index[item[0]] = len(self.index2word) - 1

        single_counter = Counter()
        for report in self.df["report"]:
            single_counter += Counter(report.split())
        for item in dict(single_counter).keys():
            self.index2word.append(item)
            self.word2index[item] = len(self.index2word) - 1

    def index2report(self,indexs):
        report = ""
        for index in indexs:
            report += "%s "%(self.index2word[index])
        return report.strip()

    def __call__(self,report):
        if self.high_freq_n > 0:
            report = self.del_high_freq_words(report).split()
        else:
            report = report.split()
        index  = []
        i = 0
        while i < len(report):
            if self.mul_words_num > 0:
                if i < len(report)-3 and "%s %s %s %s"%(report[i],report[i+1],report[i+2],report[i+3]) in self.word2index.keys():
                    index.append(self.word2index["%s %s %s %s"%(report[i],report[i+1],report[i+2],report[i+3])])
                    i += 3
                elif i < len(report)-2 and "%s %s %s"%(report[i],report[i+1],report[i+2]) in self.word2index.keys():
                    index.append(self.word2index["%s %s %s"%(report[i],report[i+1],report[i+2])])
                    i += 2
                elif i < len(report)-1 and "%s %s"%(report[i],report[i+1]) in self.word2index.keys():
                    index.append(self.word2index["%s %s"%(report[i],report[i+1])])
                    i += 1
                else:
                    index.append(self.word2index[report[i]])
            else:
                index.append(self.word2index[report[i]])
            i += 1
        return index


def test():
    import os
    train_df = pd.read_csv(os.path.join(os.getenv('PROJTOP'),'tcdata/train.csv'),sep="\|,\|",names=["id","report","label"],index_col=0)
    tokens = Tokenizer()
    tokens.initialize(train_df)
    #tokens.normal_initialize(train_df)
    print(tokens.high_freq_words)
    print(tokens.index2word)
    print(tokens.word2index)
    s = "623 355 582 617 265 162 498 289 169 137 405 693 399 842 698 335 266 14 177 415 381 693 48 328 461 478 439 473 851 636 739 374 698 494 504 656 575 754 421 421 791 200 103 718 569"
    print(s)
    index = tokens(s)
    print(index)
    print(tokens.index2report(index))
    tokens.save(os.path.join(os.getenv('PROJTOP'),'user_data/tmp_data/tokenizer.pkl'))


if __name__ == "__main__":
    test()

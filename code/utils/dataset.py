# -*- coding: utf-8 -*-
"""
    create a dataset from dataFrame
    ~~~~~~~~~~~~~~~~~~~~~~

    :author: Xu Hangkun (许杭锟)
    :copyright: © 2020 Xu Hangkun <xuhangkun@ihep.ac.cn>
    :license: MIT, see LICENSE for more details.
"""

import pandas as pd
from torch.utils.data import Dataset
import numpy as np
import torch
import math
from collections import Counter
from utils.EDA import RandomDelete,RandomSwap

class ReportDataset(Dataset):
    """
    Dataset of medical report
    """
    def __init__(self,df,nclass=29,max_len=70,label_smoothing=0,eda_alpha=0,n_aug=0,tf_idf=False,tf_idf_cut=0.006):
        """
        create a dataset from dataFrame, ['id','report','label']
        args:
            -df : dataframe which contain three columns , ['id','report','label']
            -nclass : number of classes
            -max_len : max lenght of report, if a report is longer than max_len, just cut it
            -label_smoothing : None or a value between 0 and 1, which means the possibility of wrong label
        """
        self.max_len = max_len
        self.nclass = nclass
        self.label_smoothing = label_smoothing
        self.eda_alpha = eda_alpha
        self.n_aug = n_aug
        self.tf_idf = tf_idf
        self.tf_idf_cut = tf_idf_cut
        super(ReportDataset,self).__init__()
        # generate texts
        self.texts = df['report'].values
        self.preprocess_text()
        # generate the labels
        self.labels = df['label'].values
        self.preprocess_label()
        self.enhanced_texts = []
        self.enhanced_labels = []
        self.easy_data_augmentation()

    def preprocess_text(self):
        """
        convert the text from string to list of token number
        eg:
            "1 2 4" -> [1,2,4]
        """
        texts = []
        if self.tf_idf:
            dict_idf = {i:0 for i in range(858)}
            for report_ in self.texts:
                for key in dict_idf.keys():
                    if str(key) in report_.split():
                        dict_idf[key]+=1
        for text in self.texts:
            if self.tf_idf:
                words = text.split()
                sentence=[]
                for word in words:
                    sentence.append(int(word))
                dict_sent = Counter(sentence)

                for element in dict_sent.keys():
                    dict_sent[element] = (dict_sent[element]/len(sentence))*math.log10(len(self.texts)/dict_idf[element])

                dict_sent_new = sorted(dict_sent.items(), key=lambda x: x[1], reverse=True)
                remove_report=[]
                '''
                for index in range(len(dict_sent_new)):
                    remove_report.append(dict_sent_new[index][0])
                rm_words_num = int(len(words)*0.05)#Delete words based on the percentage of sentence length
                remove_report = [str(i) for i in remove_report[-rm_words_num:]]# Remove the data with the last two digits of if-idf value
                '''
#########################################
                for key,value in dict_sent.items():
                    if value<self.tf_idf_cut:
                        remove_report.append(str(key))
#########################################
                processed_report = []
                for word in words:
                    if word not in remove_report:
                        processed_report.append(int(word))
                texts.append(processed_report)
            else:
                texts.append(self.tokenizer(str(text)))
        self.texts = texts

    def easy_data_augmentation(self):
        """
        Data Enhancement, randomly delete partial words or swap the words
        For evergy sentence, we need to change eda_alpha*sentence_len words.
        """
        if self.n_aug == 0:
            return
        for i in range(len(self.texts)):
            true_aug = 0
            if self.n_aug >1:
                true_aug = int(self.n_aug)
            elif self.n_aug >= 0:
                if np.random.random() < self.n_aug:
                    true_aug = 1
            for j in range(true_aug):
                # randomly delete some words
                self.enhanced_texts.append(RandomDelete(self.texts[i],self.eda_alpha))
                self.enhanced_labels.append(self.labels[i])
                # randomly swap some words
                self.enhanced_texts.append(RandomSwap(self.texts[i],self.eda_alpha))
                self.enhanced_labels.append(self.labels[i])
        self.texts += self.enhanced_texts
        self.labels += self.enhanced_labels
        # randomly break up the data
        for i in range(3*len(self.texts)):
            text_1_index = int(np.random.random()*len(self.texts))
            text_2_index = int(np.random.random()*len(self.texts))
            x = self.texts[text_1_index]
            self.texts[text_1_index] = self.texts[text_2_index]
            self.texts[text_2_index] = x

            x = self.labels[text_1_index]
            self.labels[text_1_index] = self.labels[text_2_index]
            self.labels[text_2_index] = x

    def preprocess_label(self):
        """
        convert the label to multi-hot tensor: [1,2] --> [0,1,1,0,0....]
        """
        labels = []
        for label in self.labels:
            label_tensor = [0.0 for i in range(self.nclass)]
            label = str(label)
            if "," not in label:
                label += ","
            label_area,label_ill = label.split(',')
            # label area
            if label_area == '' or label_area == 'nan':
                pass
            else:
                label_area = self.tokenizer(label_area)
                for index in label_area:
                    label_tensor[index] = 1.0

            if label_ill == '' or label_ill == 'nan':
                pass
            else:
                label_ill = self.tokenizer(label_ill)
                for index in label_ill:
                    label_tensor[index+17] = 1.0
            labels.append(label_tensor)

        self.labels = labels

    def tokenizer(self,text):
        """
        split the sentence and map the tokens to word index
        """
        rep = [int(x) for x in text.split()]
        #if len(rep) >self.max_len:
        #    rep = rep[:self.max_len]
        return rep


    def __len__(self):
        return len(self.labels)

    def __getitem__(self,idx):
        """
        return array of report and label
        report,label = [1,2,3....],[0,1]
        """
        # do label smoothing
        if self.label_smoothing:
            new_label = [label for label in self.labels[idx]]
            for j in range(len(new_label)):
                if np.random.random() < self.label_smoothing:
                    new_label[j] = 1 - new_label[j]
            return np.array(self.texts[idx]),np.array(new_label)
        else:
            return np.array(self.texts[idx]),np.array(self.labels[idx])

    def getitem(self,idx):
        return np.array(self.texts[idx]),np.array(self.labels[idx])


def test():
    import pandas as pd
    import os
    train_df = pd.read_csv(os.path.join(os.getenv('PROJTOP'),'tcdata/train.csv'),sep="\|,\|",names=["id","report","label"],index_col=0)
    data = ReportDataset(train_df)
    count = 0
    for index in range(3):
        text,label = data[index]
        print('text: ',text)
        print('text: ',type(text))
        print('label: ',label)
        print('')

if __name__ == "__main__":
    test()

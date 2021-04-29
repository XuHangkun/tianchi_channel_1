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
from utils.EDA import RandomDelete,RandomSwap

class ReportDataset(Dataset):
    """
    Dataset of medical report
    """
    def __init__(self,df,nclass=29,max_len=100,label_smoothing=0,eda_alpha=0.1,n_aug=4):
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
        super(ReportDataset,self).__init__()
        # generate texts
        self.texts = list(df['report'].values)
        # self.preprocess_text()
        # generate the labels
        self.labels = list(df['label'].values)
        self.preprocess_label()

        # do data enhancement
        self.enhanced_texts = []
        self.enhanced_labels = []
        self.easy_data_augmentation()

    def preprocess_text(self):
        texts = []
        for text in self.texts:
            texts.append(self.tokenizer(str(text)))
        self.texts = texts

    def easy_data_augmentation(self):
        """
        Data Enhancement
        """
        def concat_words(words):
            sentence = ""
            for word in words:
                sentence += "%s "%(word)
            sentence.strip()
            return sentence

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
                self.enhanced_texts.append(concat_words(RandomDelete(self.texts[i].split(),self.eda_alpha)))
                self.enhanced_labels.append(self.labels[i])
                # randomly swap some words
                self.enhanced_texts.append(concat_words(RandomSwap(self.texts[i].split(),self.eda_alpha)))
                self.enhanced_labels.append(self.labels[i])
        self.texts += self.enhanced_texts
        self.labels += self.enhanced_labels
        # randomly break up the data
        for i in range(len(self.texts)):
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
            label_area,label_ill = str(label).split(',')
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
        rep = [int(x) for x in text.split()]
        #if len(rep) >self.max_len:
        #    rep = rep[:self.max_len]
        return rep


    def __len__(self):
        return len(self.labels)

    def __getitem__(self,idx):
        """
        return array of report and label
        report,label = "1 2 3",[0,1]
        """
        # do label smoothing
        if self.label_smoothing:
            new_label = [label for label in self.labels[idx]]
            for j in range(len(new_label)):
                if np.random.random() < self.label_smoothing:
                    new_label[j] = 1 - new_label[j]
            return self.texts[idx],np.array(new_label)
        else:
            return self.texts[idx],np.array(self.labels[idx])


def test():
    import pandas as pd
    import os
    train_df = pd.read_csv(os.path.join(os.getenv('PROJTOP'),'tcdata/train.csv'),sep="\|,\|",names=["id","report","label"],index_col=0)
    data = ReportDataset(train_df)
    print("Data Number : %d"%(len(data)))
    count = 0
    for index in range(10):
        text,label = data[index]
        print('text: ',text)
        print('label: ',label)
        print('')

if __name__ == "__main__":
    test()

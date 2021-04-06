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
    def __init__(self,df,nclass=17,max_len=70,label_smoothing=0,eda_alpha=0.1,n_aug=2):
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
        for text in self.texts:
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
            label = str(label)
            if label == '' or label == 'nan':
                pass
            else:
                label = self.tokenizer(label)
                for index in label:
                    label_tensor[index] = 1.0

            labels.append(label_tensor)

        self.labels = labels

    def tokenizer(self,text):
        """
        split the sentence and map the tokens to word index
        """
        rep = [int(x) for x in text.split()]
        if len(rep) >self.max_len:
            rep = rep[:self.max_len]
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
    train_df = pd.read_csv(os.path.join(os.getenv('PROJTOP'),'tcdata/medical_nlp_round1_data/train.csv'),sep="\|,\|",names=["id","report","label"],index_col=0)
    data = ReportDataset(train_df)
    count = 0
    for index in range(10):
        text,label = data[index]
        print('text: ',text)
        print('label: ',label)
        print('')

if __name__ == "__main__":
    test()

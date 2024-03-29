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
    def __init__(self,df,tokenizer,nclass=29,max_len=70,label_smoothing=0,eda_alpha=0.1,n_aug=2,pretrain=False):
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
        self.tokenizer = tokenizer
        self.label_smoothing = label_smoothing
        self.eda_alpha = eda_alpha
        self.n_aug = n_aug
        self.pretrain = pretrain
        super(ReportDataset,self).__init__()
        # generate texts
        self.texts = df['report'].values
        self.preprocess_text()
        # generate the labels
        self.labels = df['label'].values
        self.labels_freq = [0 for x in range(self.nclass)]
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
        if self.n_aug == 0 or self.n_aug < 0.1:
            return
        for i in range(len(self.texts)):
            true_aug = 0
            if self.n_aug >1:
                true_aug = int(self.n_aug)
            elif self.n_aug >= 0:
                if np.random.random() < self.n_aug:
                    true_aug = 1
            label_all_zero = True
            for j in range(len(self.labels[i])):
                if self.labels[i][j] > 0.5:
                    label_all_zero = False
            for j in range(true_aug):
                # randomly delete some words
                self.enhanced_texts.append(RandomDelete(self.texts[i],self.eda_alpha))
                self.enhanced_labels.append(self.labels[i])
                # randomly swap some words
                self.enhanced_texts.append(RandomSwap(self.texts[i],self.eda_alpha))
                self.enhanced_labels.append(self.labels[i])
        if self.pretrain:
            self.texts = self.enhanced_texts
            self.labels = self.enhanced_labels
        else:
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
        labels_freq = [0 for x in range(self.nclass)]
        for label in self.labels:
            label_tensor = [0.0 for i in range(self.nclass)]
            label = str(label)
            if "," not in label:
                label += ","
            label_area,label_ill = label.split(',')
            # label area
            if label_area == '' or label_area == 'nan' or label_area == " ":
                pass
            else:
                label_area = [int(x) for x in label_area.split()]
                for index in label_area:
                    label_tensor[index] = 1.0
                    labels_freq[index] += 1./len(self.labels)

            if label_ill == '' or label_ill == 'nan' or label_area == " ":
                pass
            else:
                label_ill = [int(x) for x in label_ill.split()]
                for index in label_ill:
                    label_tensor[index + 17] = 1.0
                    labels_freq[index + 17] += 1./len(self.labels)
            labels.append(label_tensor)

        self.labels = labels
        self.labels_freq = labels_freq

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
                if np.random.random() < self.label_smoothing*self.labels_freq[j]:
                    new_label[j] = 1 - new_label[j]
            return np.array(self.texts[idx]),np.array(new_label)
        else:
            #if len(self.texts[idx]) > self.max_len:
            #    new_seq = self.texts[idx][:self.max_len//2] + self.texts[idx][len(self.texts[idx])-self.max_len//2:len(self.texts[idx])]
            #    return np.array(self.texts[idx]),np.array(self.labels[idx])
            return np.array(self.texts[idx]),np.array(self.labels[idx])

    def getitem(self,idx):
        return np.array(self.texts[idx]),np.array(self.labels[idx])


def test():
    import pandas as pd
    import os
    train_df = pd.read_csv(os.path.join(os.getenv('PROJTOP'),'tcdata/train.csv'),sep="\|,\|",names=["id","report","label"],index_col=0)
    data = ReportDataset(train_df)
    count = 0
    for index in range(10):
        text,label = data[index]
        print('text: ',text)
        print('label: ',label)
        print('')

if __name__ == "__main__":
    test()

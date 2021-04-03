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
import os

class StackingDataset(Dataset):
    """
    Dataset of medical pred
    """
    def __init__(self,dfs):
        """
        create a dataset from dataFrame, ['id','pred','label']
        args:
            -dfs :list of dataframe which contain three columns , ['id','pred','label']

        """
        super(StackingDataset,self).__init__()
        self.dfs = dfs
        self.check_dataframes()
        # generate texts
        self.texts = None
        self.preprocess_text()
        # generate the labels
        self.labels = None
        self.preprocess_label()

    def check_dataframes(self):
        """
        lenght and label should be same
        """
        print("Check the dataframes, they must have the same length and lables")
        for i in range(1,len(self.dfs)):
            assert len(self.dfs[i]) == len(self.dfs[0])
            for j in range(len(self.dfs[i])):
                assert self.dfs[i]["label"][j] == self.dfs[0]["label"][j]
        print("Pass, Check successfully!")


    def preprocess_text(self):
        texts = []
        for i in range(len(self.dfs[0])):
            text = []
            for df in self.dfs:
                text.append(self.tokenizer(df["pred"][i]))
            texts.append(text)
        self.texts = texts

    def preprocess_label(self):
        """
        convert the label to multi-hot tensor: [1,2] --> [0,1,1,0,0....]
        """
        labels = []
        for label in self.dfs[0]["label"]:
            label_tensor = [int(x) for x in label.split()]
            labels.append(label_tensor)
        self.labels = labels

    def tokenizer(self,text):
        rep = [float(x) for x in text.split()]
        return rep

    def __len__(self):
        return len(self.labels)

    def __getitem__(self,idx):
        """
        return array of report and label
        report,label = [1,2,3....],[0,1]
        """
        return torch.Tensor(self.texts[idx]).transpose(0,1),torch.Tensor(self.labels[idx])

def test():
    train_df = pd.read_csv(os.path.join(os.getenv('PROJTOP'),'user_data/tmp_data/TextCNN_pred.csv'),index_col=0)
    print(train_df)
    data = StackingDataset([train_df,train_df])
    count = 0
    for index in range(9990,10000):
        text,label = data[index]
        print('text: ',text.shape)
        print('label: ',label.shape)
        print('')

if __name__ == "__main__":
    test()

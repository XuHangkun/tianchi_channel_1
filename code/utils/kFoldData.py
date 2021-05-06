#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
    k Fold DataLoader
    ~~~~~~~~~~~~~~~~~~~~~~

    :author: Xu Hangkun (许杭锟)
    :copyright: © 2020 Xu Hangkun <xuhangkun@ihep.ac.cn>
    :license: MIT, see LICENSE for more details.
"""
from torch.utils.data import DataLoader
from utils.dataset import ReportDataset
from utils.padCollate import PadCollate
from collections import Counter

class KFoldDataLoader:
    """
    K Fold Data Loader
    """
    def __init__(self,df,tokenizer,batch_size=128,k=5,nclass=29,max_len=100,label_smoothing=0,eda_alpha=0.1,n_aug=4,pretrain=False):
        """
        args:
            k - k Folder, default = 5
            df - dataFrame, should have three column ['id','report','label']
        """
        self.df = df
        self.tokenizer = tokenizer
        # random shuffle
        # self.df = self.df.sample(frac=1).reset_index(drop=True)
        self.pad_idx = self.tokenizer.padding_idx()
        self.num_token = self.tokenizer.vocab_num()
        self.lenght = len(df)
        self.a_fold_length = self.lenght//k
        self.fold_k = k
        self.collect_fn = PadCollate(pad_idx = self.pad_idx)
        self.nclass = nclass
        self.max_len = max_len
        self.batch_size = batch_size
        self.label_smoothing = label_smoothing
        self.eda_alpha = eda_alpha
        self.n_aug = n_aug
        self.pretrain = pretrain

    def get_ith_data(self,i):
        """
        Get i'th fold data
        args:
            i - index of fold, 0 < i < k
        """
        assert 0 <= i and i < self.fold_k
        valid_index = []
        train_index = []
        if i < self.fold_k-1:
            valid_index += range(i*self.a_fold_length,(i+1)*self.a_fold_length)
            train_index += range(0,i*self.a_fold_length)
            train_index += range((i+1)*self.a_fold_length,self.lenght)
        else:
            valid_index += range(i*self.a_fold_length,self.lenght)
            train_index += range(0,i*self.a_fold_length)

        if self.pretrain:
            train_df = self.df
        else:
            train_df = self.df.iloc[train_index]

        train_dataset = ReportDataset(train_df,tokenizer=self.tokenizer,nclass=self.nclass,max_len=self.max_len,label_smoothing=self.label_smoothing,eda_alpha=self.eda_alpha,n_aug=self.n_aug,pretrain = self.pretrain)
        train_dataloader = DataLoader(dataset=train_dataset,batch_size=self.batch_size,collate_fn=self.collect_fn)
        valid_df = self.df.iloc[valid_index]
        valid_dataset = ReportDataset(valid_df,tokenizer = self.tokenizer,nclass=self.nclass,max_len=self.max_len,eda_alpha=0,n_aug=0)
        valid_dataloader = DataLoader(dataset=valid_dataset,batch_size=self.batch_size,collate_fn=self.collect_fn)
        return train_dataloader,valid_dataloader,train_dataset,valid_dataset


def test():
    import pandas as pd
    import os
    df = pd.read_csv(os.path.join(os.getenv('PROJTOP'),'tcdata/medical_nlp_round1_data/train.csv'),sep="\|,\|",names=["id","report","label"])
    print(df.iloc[range(10)])
    kfold_load = KFoldDataLoader(df)
    train_loader,valid_loader,train_dataset,valid_dataset = kfold_load.get_ith_data(0)
    print(len(valid_dataset))

    for i in range(3):
        report,label = valid_dataset[i]
        print(report)
    print(train_loader)
    for X,y in train_loader:
        print(X)
        print(y.shape)
        break

if __name__ == '__main__':
    test()

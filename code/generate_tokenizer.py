# -*- coding: utf-8 -*-
"""
    predict according to trained model
    ~~~~~~~~~~~~~~~~~~~~~~

    :author: Xu Hangkun (许杭锟),Fu Yangsheng,Huang Zheng
    :copyright: © 2020 Xu Hangkun <xuhangkun@ihep.ac.cn>
    :license: MIT, see LICENSE for more details.
"""

import pandas as pd
import numpy as np
from utils.tokenizer import Tokenizer
import argparse
import os


parser = argparse.ArgumentParser(description="Generate tokenizer")
parser.add_argument('-input',default=os.path.join(os.getenv('PROJTOP'),'tcdata/train.csv'))
parser.add_argument('-output',default=os.path.join(os.getenv('PROJTOP'),'user_data/tokenizer.pkl'))
opt = parser.parse_args()
print(opt)

train_df = pd.read_csv(opt.input,sep="\|,\|",names=["id","report","label"],index_col=0)
tokens = Tokenizer(train_df)
tokens.initialize()
tokens.save(opt.output)


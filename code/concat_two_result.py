# -*- coding: utf-8 -*-
"""
    concat two results file info one result
    ~~~~~~~~~~~~~~~~~~~~~~

    :author: Xu Hangkun (许杭锟),Fu Yangsheng,Huang Zheng
    :copyright: © 2020 Xu Hangkun <xuhangkun@ihep.ac.cn>
    :license: MIT, see LICENSE for more details.
"""

import os
import argparse
import pandas as pd


# Define the
parser = argparse.ArgumentParser(description="concat two results file into one result")
parser.add_argument('-task1_file',default="./result_1.csv",help="result file of task one")
parser.add_argument('-task2_file',default="./result_2.csv",help="result file of task two")
parser.add_argument('-output',default="./result.csv",help="result file of task two")
args = parser.parse_args()
print(args)

data_task1 = pd.read_csv(args.task1_file,sep="\|,\|",names=["id","report"],index_col=0)
data_task2 = pd.read_csv(args.task2_file,sep="\|,\|",names=["id","report"],index_col=0)
print(data_task1)
print(data_task2)

assert len(data_task1) == len(data_task2)

file = open(args.output,"w")
for i in range(len(data_task1)):
    task1 = data_task1["report"][i].strip()
    task2 = data_task2["report"][i].strip()
    file.write("%d|,|%s %s\n"%(i,task1,task2))
file.close()

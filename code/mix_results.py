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
parser.add_argument('-in_files',nargs="+",default="./result_1.csv",help="result file of task one")
parser.add_argument('-output',default="./result.csv",help="result file of task two")
args = parser.parse_args()
print(args)

inputs = []
for ifile in args.in_files:
    data = pd.read_csv(ifile,sep="\|,\|",names=["id","label"],index_col=0)
    inputs.append(data)


file = open(args.output,"w")
for i in range(len(inputs[0])):
    res = "%d|,|"%(i)
    prediction = []
    for j in range(len(inputs)):
        prediction.append([float(x) for x in inputs[j]["label"][i].split()])
    for j in range(29):
        tmp = [value[j] for value in prediction]
        res += "%.8f "%(sum(tmp)/len(tmp))
    res += "\n"
    file.write(res)
file.close()

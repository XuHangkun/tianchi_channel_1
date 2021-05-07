# -*- coding: utf-8 -*-
"""
    Eval the model
    ~~~~~~~~~~~~~~~~~~~~~~

    :author: Xu Hangkun (许杭锟)
    :copyright: © 2020 Xu Hangkun <xuhangkun@ihep.ac.cn>
    :license: MIT, see LICENSE for more details.
"""

def cal_accuracy(pred,label,task=0):
    """
    If score of a class in pred is bigger than 0.5, we think it's true, otherwise False
    args :
        pred : prediction [batch_size,num_class]
        label: label      [batch_size,num_class]
    return:
        accuracy
    """
    # shape of pred and label must be same
    # assert pred.shape == label.shape

    pred_zero = (pred <= 0.5)
    label_zero = (label <= 0.5 )
    if task == 0 :
        res = ( pred_zero == label_zero )
    elif task == 1 :
        res = ( pred_zero == label_zero[:,:17] )
    elif task == 2 :
        res = ( pred_zero == label_zero[:,17:] )
    elif task == 3 :
        pred_words = [2,3,5,6,9,10,12,13,14,16,17,18,20,21,22,23,24,25,26,27,28]
        mask = [True]*label_zero.shape[1]
        for index in pred_words:
            mask[index] = False
        res = ( pred_zero == label_zero[:,mask] )
    elif task == 4 :
        pred_words = [0,1,4,7,8,11,15,19]
        mask = [True]*label_zero.shape[1]
        for index in pred_words:
            mask[index] = False
        res = ( pred_zero == label_zero[:,mask] )
    
    positive_count = 0
    total = res.size(0)*res.size(1)
    for i in range(res.size(0)):
        for j in range(res.size(1)):
            if res[i][j]:
                positive_count += 1
    return 1.0*positive_count/total

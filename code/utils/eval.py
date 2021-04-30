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

    positive_count = 0
    total = res.size(0)*res.size(1)
    for i in range(res.size(0)):
        for j in range(res.size(1)):
            if res[i][j]:
                positive_count += 1
    return 1.0*positive_count/total

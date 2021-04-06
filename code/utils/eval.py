# -*- coding: utf-8 -*-
"""
    Eval the model
    ~~~~~~~~~~~~~~~~~~~~~~

    :author: Xu Hangkun (许杭锟)
    :copyright: © 2020 Xu Hangkun <xuhangkun@ihep.ac.cn>
    :license: MIT, see LICENSE for more details.
"""

def cal_accuracy(pred,label):
    """
    If score of a class in pred is bigger than 0.5, we think it's true, otherwise False
    args :
        pred : prediction [batch_size,num_class]
        label: label      [batch_size,num_class]
    return:
        accuracy
    """
    # shape of pred and label must be same
    assert pred.shape == label.shape

    pred_zero = (pred <= 0.5)
    label_zero = (label <= 0.5 )
    res = ( pred_zero == label_zero )
    positive_count = 0
    total = label.size(0)*label.size(1)
    for i in range(label.size(0)):
        for j in range(label.size(1)):
            if res[i][j]:
                positive_count += 1
    return 1.0*positive_count/total
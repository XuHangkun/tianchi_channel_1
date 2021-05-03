# -*- coding: utf-8 -*-
"""
    a variant of callate_fn that pads according to the longest sequence in
    a batch of sequences
    ~~~~~~~~~~~~~~~~~~~~~~

    :author: Xu Hangkun (许杭锟)
    :copyright: © 2020 Xu Hangkun <xuhangkun@ihep.ac.cn>
    :license: MIT, see LICENSE for more details.
"""
import torch
import math
def pad_tensor(vec, pad, dim,pad_idx):
    """
    args:
        vec - tensor to pad
        pad - the size to pad to
        dim - dimension to pad

    return:
        a new tensor padded to 'pad' in dimension 'dim'
    """
    #
    remove_element = torch.tensor([693,328,380,698])
    vec_new = torch.IntTensor([])
    for element in vec:
        if element not in remove_element:
            element = element.unsqueeze(0)
            vec_new = torch.cat((vec_new,element),0)
    vec = vec_new

    pad_size = list(vec.shape)
    pad_size[dim] = pad - vec.shape[dim]
    vec_pad_z = torch.Tensor()
    '''
    if math.ceil(pad/2) <= vec.shape[dim]:
        #vec_pad = vec[0:pad_size[dim]]# Replenish information from front to back
        vec_pad = vec[vec.shape[dim]*2-pad:pad]
    else:
        vec_pad_cat = vec
        index = pad//vec.shape[dim]-1
        for i in range(index):
            vec_pad_cat = torch.cat([vec_pad_cat,vec])
        vec = vec_pad_cat
        pad_remain_size = pad%vec.shape[dim]
        vec_pad = vec[0:pad_remain_size]
    '''
    #return torch.cat([vec,vec_pad], dim=dim)
    return torch.cat([vec, pad_idx*torch.ones(*pad_size,dtype=torch.long)], dim=dim)

class PadCollate:
    """
    a variant of callate_fn that pads according to the longest sequence in
    a batch of sequences
    """

    def __init__(self, dim=0,pad_idx = 858):
        """
        args:
            dim - the dimension to be padded (dimension of time in sequences)
            pad_idx - idx of padding
        """
        self.dim = dim
        self.pad_idx = pad_idx

    def pad_collate(self, batch):
        """
        args:
            batch - list of (tensor, label)

        reutrn:
            xs - a tensor of all examples in 'batch' after padding
            ys - a LongTensor of all labels in batch
        """
        # find longest sequence
        max_len = 100
        #max_len = max(map(lambda x: x[0].shape[self.dim], batch))
        # pad according to max_len
        new_batch = []
        for x,y in batch:
            x = torch.LongTensor(x)
            y = torch.LongTensor(y)
            new_batch.append((pad_tensor(x, pad=max_len, dim=self.dim,pad_idx=self.pad_idx),y))
        batch = new_batch
        # stack all
        xs = torch.stack([x for (x,y) in batch])
        ys = torch.stack([y for (x,y) in batch])
        return xs, ys

    def __call__(self, batch):
        return self.pad_collate(batch)

def test():
    import numpy as np
    pad = PadCollate()
    batch = [(np.array([1.0,2,3,4,5,6,693,11,328,15,380, 53,698, 57, 59, 2, 27, 43, 4, 30, 34, 1, 32, 58, 21, 17, 15, 38, 31, 20, 11, 40, 35, 39, 36, 3, 5, 41, 26, 13, 50, 12, 0, 44, 29, 45, 22, 48, 54, 25, 28, 51, 19, 14, 252, 49, 9, 16, 46, 47, 7,42, 11110]),np.array([1,0,0])),
            (np.array([1,2,3,4,5,6,7]),np.array([1,1,0]))]
    print(pad(batch))

if __name__ == '__main__':
    test()

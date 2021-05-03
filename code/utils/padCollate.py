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
    pad_size = list(vec.shape)
    pad_size[dim] = pad - vec.shape[dim]
    vec_pad_z = torch.Tensor()
    if math.ceil(pad/2) <= vec.shape[dim]:
        vec_pad = vec[0:pad_size[dim]]
    else:
        vec_pad_z = vec
        index = pad//vec.shape[dim]-1
        for i in range(index):
            vec_pad_z = torch.cat([vec_pad_z,vec])
        vec = vec_pad_z
        pad_size_z = pad%vec.shape[dim]
        vec_pad = vec[0:pad_size_z]
    #print(vec.shape[dim])
    return torch.cat([vec,vec_pad], dim=dim)
    #return torch.cat([vec, pad_idx*torch.ones(*pad_size,dtype=torch.long)], dim=dim)

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
    batch = [(np.array([1.0,2,3,4,5,6,39,11,17,15]),np.array([1,0,0])),
            (np.array([1,2,3,4]),np.array([1,1,0]))
            ]
    print(pad(batch))

if __name__ == '__main__':
    test()

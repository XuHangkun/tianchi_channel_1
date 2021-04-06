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
class PadCollate:
    """
    a variant of callate_fn that pads according to the longest sequence in
    a batch of sequences
    """

    def __init__(self,tokenizer):
        """
        args:
            dim - the dimension to be padded (dimension of time in sequences)
            pad_idx - idx of padding
        """
        self.tokenizer = tokenizer

    def pad_collate(self, batch):
        """
        args:
            batch - list of (tensor, label)

        reutrn:
            xs - a tensor of all examples in 'batch' after padding
            ys - a LongTensor of all labels in batch
        """
        new_reports = []
        new_labels = []
        for x,y in batch:
            y = torch.LongTensor(y)
            new_labels.append(y)
            new_reports.append(x)
        # stack all
        xs = self.tokenizer(new_reports,padding=True, truncation=True, return_tensors="pt")
        ys = torch.stack(new_labels)
        return xs, ys

    def __call__(self, batch):
        return self.pad_collate(batch)

def test():
    import numpy as np
    pad = PadCollate()
    batch = [(np.array([1.0,2,3,4,5,6]),np.array([1,0,0])),
            (np.array([1,2,3,4]),np.array([1,1,0]))
            ]
    print(pad(batch))

if __name__ == '__main__':
    test()

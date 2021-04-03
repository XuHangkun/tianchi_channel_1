# coding: UTF-8
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy

class StackingConfig(object):

    """配置参数"""
    def __init__(self,n_model,num_calss=17):
        self.model_name = 'Stacking'
        self.n_model=n_model
        self.n_class = num_calss

class StackingModel(nn.Module):

    def __init__(self,config):
        """
        Model for stack model
        """
        super(StackingModel,self).__init__()
        self.n_class = config.n_class
        self.n_model = config.n_model

        self.feed_forward = nn.Sequential(
            nn.Linear(self.n_model,1),
            nn.Sigmoid()
        )

    def forward(self,x):
        x = self.feed_forward(x)
        x = x.squeeze()
        return x

def test():
    config = StackingConfig(3)
    model = StackingModel(config)
    x = torch.randn([256,17,3])
    print(model(x).shape)

if __name__ == "__main__":
    test()

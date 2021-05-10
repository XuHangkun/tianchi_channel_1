import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import copy
class TextCNNConfig(object):

    """配置参数"""
    def __init__(self,n_vocab=859,embedding=100,
            num_class=29,max_seq_len=100,dropout=0.5,
            num_filters=128,padding_idx=0):
        self.model_name = 'TextCNN'
        self.dropout = dropout                                              # 随机失活
        self.n_vocab = n_vocab                                          # 词表大小在运行时赋值
        self.padding_idx = 0
        self.num_class = num_class                                      # 类别数
        self.embedding = embedding
        self.num_filters = num_filters                                          # 卷积核数量(channels数)
        self.kernel_size = [2,4,6,8,10,12]
        self.Ci = 1
        self.static = False
        self.max_seq_len = max_seq_len


class CALayer(nn.Module):
    # Channel Attention (CA) Layer
    def __init__(self, in_channels, out_channels, reduction=4):
        super(CALayer, self).__init__()
        # feature channel downscale and upscale --> channel weight
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels // reduction, 1, 1, 0),
            nn.Conv2d(out_channels // reduction, out_channels, 1, 1, 0),
            nn.Sigmoid())

    def forward(self, x):
        return x * self.attention(x)


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):

        super(ConvBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, 1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1)
        self.conv3 = nn.Conv2d(out_channels, out_channels, 3, 1, 1)
        self.ca = CALayer(out_channels, out_channels)

        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.bn3 = nn.BatchNorm2d(out_channels)

    def forward(self, input, pool_size=(1, 1), pool_type='none'):

        x = input
        x1 = F.relu(self.bn1(self.conv1(x)), inplace=True)
        x2 = F.relu(self.bn2(self.conv2(x1)), inplace=True)
        x = self.bn3(self.conv3(x2))
        x = self.ca(x) + x1
        x = F.relu(x, inplace=True)

        if pool_type == 'max':
            out = F.max_pool2d(x, kernel_size=pool_size)
        elif pool_type == 'avg':
            out = F.avg_pool2d(x, kernel_size=pool_size)
        elif pool_type == 'avg+max':
            x1 = F.avg_pool2d(x, kernel_size=pool_size)
            x2 = F.max_pool2d(x, kernel_size=pool_size)
            out = x1 + x2
        elif pool_type == 'none':
            out = x
        else:
            raise Exception('Incorrect argument!')

        return out


class TextCNNModel(nn.Module):

    def __init__(self,config):
        super(TextCNNModel, self).__init__()
        self.embed_num = config.n_vocab
        self.embed_dim = config.embedding
        self.max_seq_len = config.max_seq_len
        self.num_class = config.num_class
        self.padding_idx = config.padding_idx
        self.Ci = config.Ci
        self.kernel_num = config.num_filters
        self.Ks = config.kernel_size
        self.dropout = config.dropout

        self.embed = nn.Embedding(self.embed_num, self.embed_dim,padding_idx=self.padding_idx)#词嵌入
        self.convs = nn.ModuleList([nn.Conv2d(self.Ci, self.kernel_num, (K, self.embed_dim), 1, (K//2, 0)) for K in self.Ks])

        self.conv_block1 = ConvBlock(in_channels=self.kernel_num, out_channels=self.kernel_num * 2)
        self.conv_block2 = ConvBlock(in_channels=self.kernel_num * 2, out_channels=self.kernel_num * 2)
        self.conv_block3 = ConvBlock(in_channels=self.kernel_num * 2, out_channels=self.kernel_num)
        self.avg = nn.AdaptiveAvgPool2d(1)

        self.feed_forward = nn.Sequential(
            nn.Linear(self.kernel_num, self.kernel_num),
            nn.Dropout(self.dropout),
            nn.Linear(self.kernel_num, self.num_class),
            nn.Sigmoid()
        )

    def complete_short_sentence(self,x):
        device = x.device
        if x.size(1) > self.max_seq_len:
            x = x[:,:self.max_seq_len]
            #x = torch.Tensor(x[:self.max_seq_len],requires_grad=False,device=device)
        else:
            cat_size = (x.size(0),self.max_seq_len-x.size(1))
            pad_tensor = torch.full(cat_size,self.padding_idx,dtype=torch.long,requires_grad=False,device=device)
            x = torch.cat((x,pad_tensor),1)
        return x

    def forward(self, x):
        x = self.complete_short_sentence(x)
        x = self.embed(x)  # (N, W, D)-batch,单词数量维度
        x = x.unsqueeze(1)  # (N, Ci, W, D)
        x = [F.relu(conv(x)) for conv in self.convs]  # [(N, Co, W), ...]*len(Ks)
        x = torch.cat(x, dim=3)
        x = self.conv_block1(x, pool_size=(2, 1), pool_type='max')
        x = self.conv_block2(x, pool_size=(2, 1), pool_type='max')
        x = self.conv_block3(x, pool_size=(2, 1), pool_type='max')
        x = self.avg(x)
        x = x.view(x.size(0), -1)

        return self.feed_forward(x)

if __name__=="__main__":
    config = TextCNNConfig()
    net=TextCNNModel(config)
    x=torch.LongTensor([[1,2,4,5,2,35,43,113,111,451,455,22,45,55],[14,3,12,9,13,4,51,45,53,17,57,100,156,23]])
    logit=net(x)
    print(logit)

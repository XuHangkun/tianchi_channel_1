# coding: UTF-8
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy

class DPCNNConfig(object):

    """配置参数"""
    def __init__(self,n_vocab=859,embedding=100,padding_idx=0,
            num_class=29,max_seq_len=70,dropout=0.5,
            num_filters=512,reduction=8):
        self.model_name = 'DPCNN'
        self.dropout = dropout                                          # 随机失活
        self.n_vocab = n_vocab                                          # 词表大小，在运行时赋值
        self.padding_idx = padding_idx
        self.num_classes = num_class                                    # 类别数
        self.embedding = embedding                                      # dim of embedding
        self.num_filters =  num_filters                                 # 卷积核数量(channels数)
        self.max_seq_len = max_seq_len
        self.reduction = reduction

class DPCNNModel(nn.Module):
    def __init__(self, config):
        """
        Deep Pyramid Convolutional Neural Networks for Text Categorization
        """
        super(DPCNNModel, self).__init__()
        self.padding_idx = config.padding_idx
        self.embed_num = config.n_vocab
        self.embed_dim = config.embedding
        self.max_seq_len = config.max_seq_len
        self.dropout = config.dropout

        self.embed = nn.Embedding(config.n_vocab,config.embedding, padding_idx=config.padding_idx)
        self.avg_pool = torch.nn.AdaptiveAvgPool2d(1)
        self.att = nn.Sequential(
            nn.Linear(config.num_filters, config.num_filters // config.reduction, bias=False),
            nn.ReLU(inplace = True),
            nn.Linear(config.num_filters // config.reduction, config.num_filters, bias=False),
            nn.ReLU(inplace = True),
            )
        self.conv_region = nn.Conv2d(1, config.num_filters, (3, config.embedding), stride=1)
        self.conv = nn.Conv2d(config.num_filters, config.num_filters, (3, 1), stride=1)
        self.conv_1 = nn.Conv2d(config.num_filters, config.num_filters, (3, 1), stride=1)
        self.conv_2 = nn.Conv2d(config.num_filters, config.num_filters, (3, 1), stride=1)
        self.conv_3 = nn.Conv2d(config.num_filters, config.num_filters, (3, 1), stride=1)
        self.max_pool = nn.MaxPool2d(kernel_size=(3, 1), stride=2)
        self.padding1 = nn.ZeroPad2d((0, 0, 1, 1))  # top bottom
        self.padding2 = nn.ZeroPad2d((0, 0, 0, 1))  # bottom
        self.relu = nn.ReLU()
        self.feed_forward = nn.Sequential(
            nn.Linear(config.num_filters, config.num_filters),
            nn.Dropout(self.dropout),
            nn.ReLU(),
            nn.Linear(config.num_filters, config.num_classes),
            nn.Sigmoid()
        )


    def complete_short_sentence(self,x):
        device = x.device
        if x.size(1) > self.max_seq_len:
            x = x[:,:self.max_seq_len]
        else:
            cat_size = (x.size(0),self.max_seq_len-x.size(1))
            pad_tensor = torch.full(cat_size,self.padding_idx,dtype=torch.long,requires_grad=False,device=device)
            x = torch.cat((x,pad_tensor),1)
        return x

    def forward(self, x):
        x = self.complete_short_sentence(x) # [batch_size,seq_length]
        x = self.embed(x)  # [batch_size,seq_length,emb_size]
        x = x.unsqueeze(1)  # [batch_size, 1, seq_len, emb_size]
        x = self.conv_region(x)  # [batch_size, 250, seq_len-3+1, 1]

        x = self.padding1(x)  # [batch_size, 250, seq_len, emb_size]
        x = self.relu(x)
        x = self.conv_1(x)  # [batch_size, 250, seq_len-3+1, 1]
        x = self.padding1(x)  # [batch_size, 250, seq_len, 1]
        x = self.relu(x)
        x = self.conv_2(x)  # [batch_size, 250, seq_len-3+1, 1]
        x = self.padding1(x)  # [batch_size, 250, seq_len, 1]
        x = self.relu(x)
        x = self.conv_3(x)  # [batch_size, 250, seq_len-3+1, 1]
        while x.size()[2] >= 2:
            x = self._block(x)
        x = x.squeeze()  # [batch_size, num_filters(250)]

        x = self.feed_forward(x)
        return x

    def _block(self, x):
        x = self.padding2(x)
        px = self.max_pool(x)

        x = self.padding1(px)
        x = F.relu(x)
        x = self.conv(x)

        x = self.padding1(x)
        x = F.relu(x)
        x = self.conv(x)
        # add the attention here
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.att(y).view(b, c, 1, 1)
        y = x * y.expand_as(x)

        # Short Cut
        x = x + px + y
        return x

    def use_pretrain_word2vec(self,word2vec_model):
        """
        use pretrain model to init the weight in Embeding layer
        """
        # assert word2vec_model.size == self.embed_dim
        assert word2vec_model["word_size"] == self.embed_dim
        vocab = word2vec_model["wv"].keys()

        for index in range(self.embed_num):
            if str(index) in vocab:
                self.embed.weight.data[index] = copy.deepcopy(word2vec_model["wv"][str(index)])
        return True

def test():
    import numpy as np
    input = torch.LongTensor([range(100)])
    print(input)
    config = DPCNNConfig()
    model = DPCNNModel(config)
    print(model.complete_short_sentence(input))
    output = model(input)
    print(output)

if __name__ == "__main__":
    test()

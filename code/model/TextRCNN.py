# coding: UTF-8
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy

class TextRCNNConfig(object):

    """配置参数"""
    def __init__(self, n_vocab=859, embedding=500,num_class=17):
        self.model_name = 'TextRCNN'

        self.dropout = 0.5                                              # 随机失活
        self.num_classes = num_class                         # 类别数
        self.n_vocab = n_vocab                                                # 词表大小，在运行时赋值
        self.embedding = embedding
        self.pad_size = 70
        self.hidden_size = 256                                          # lstm隐藏层
        self.num_layers = 2                                             # lstm层数


'''Recurrent Convolutional Neural Networks for Text Classification'''


class TextRCNNModel(nn.Module):
    def __init__(self, config):


        super(TextRCNNModel, self).__init__()
        self.embed_num = config.n_vocab
        self.embed_dim = config.embedding
        self.pad_size = config.pad_size
        self.dropout = 0.5                                              # 随机失活
        self.padding_idx = config.padding_idx

        self.embed = nn.Embedding(config.n_vocab, config.embedding, padding_idx=config.n_vocab - 1)
        self.lstm = nn.LSTM(config.embedding, config.hidden_size, config.num_layers,
                            bidirectional=True, batch_first=True, dropout=config.dropout)
        self.maxpool = nn.MaxPool1d(config.pad_size)
        self.fc = nn.Linear(config.hidden_size * 2 + config.embedding, config.num_classes)
        self.sigmoid = nn.Sigmoid()
        #self.feed_forward = nn.Sequential(
        #    nn.Linear(config.hidden_size * 2 + config.embedding, config.num_classes),
        #    nn.Sigmoid()
        #)

    def forward(self, x):
        x = self.complete_short_sentence(x)
        #x, _ = x
        embed = self.embed(x)  # [batch_size, seq_len, embeding]=[64, 32, 64]
        #print(embed)
        #print(embed.size())
        #print('embed')
        out, _ = self.lstm(embed)
        #print(out)
        #print(out.size())
        #print('lstm')
        out = torch.cat((embed, out), 2)
        #print(out)
        #print(out.size())
        #print('cat')
        out = F.relu(out)
        out = out.permute(0, 2, 1)
        #print(out)
        #print(out.size())
        #print('permute')
        out = self.maxpool(out).squeeze()
        #print(out)
        #print(out.size())
        #print('maxpool')
        out = self.fc(out)
        #print(out)
        #print(out.size())
        #print('fc')
        out = self.sigmoid(out)
        return out

    def complete_short_sentence(self,x):
        device = x.device
        if x.size(1) > self.pad_size:
            x = torch.Tensor(x[:self.pad_size],requires_grad=False,device=device)
        else:
            cat_size = (x.size(0),self.pad_size-x.size(1))
            pad_tensor = torch.full(cat_size,self.padding_idx,dtype=torch.long,requires_grad=False,device=device)
            x = torch.cat((x,pad_tensor),1)
        return x

    def use_pretrain_word2vec(self,word2vec_model):
        """
        use pretrain model to init the weight in Embedding layer
        """
        assert word2vec_model["word_size"] == self.embed_dim
        vocab = word2vec_model["wv"].keys()
        for index in range(self.embed_num):
            if str(index) in vocab:
                self.embed.weight.data[index] = copy.deepcopy(word2vec_model["wv"][str(index)])
        return True

def test():
    import numpy as np
    input = torch.LongTensor([range(5),range(5),range(5)])
    print(input)
    config = TextRCNNConfig()
    model = TextRCNNModel(config)
    output = model(input)
    print(output)

if __name__ == "__main__":
    test()
    print('?????????????????????????')

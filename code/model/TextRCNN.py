# coding: UTF-8
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy

class TextRCNNConfig(object):

    """配置参数"""
    def __init__(self, n_vocab=859, embedding=200,max_seq_len=60,num_class=17,dropout=0.5):
        self.model_name = 'TextRCNN'

        self.dropout = dropout                                              # 随机失活
        self.num_classes = num_class                                    # 类别数
        self.n_vocab = n_vocab                                          # 词表大小，在运行时赋值
        self.padding_idx = self.n_vocab - 1
        self.embedding = embedding
        self.hidden_size = 128                                          # lstm隐藏层
        self.num_layers = 2                                             # lstm层数
        self.max_seq_len = max_seq_len


'''Recurrent Convolutional Neural Networks for Text Classification'''


class TextRCNNModel(nn.Module):
    def __init__(self, config):


        super(TextRCNNModel, self).__init__()
        self.embed_num = config.n_vocab
        self.embed_dim = config.embedding
        self.max_seq_len = config.max_seq_len
        self.hidden_size = config.hidden_size
        self.dropout = config.dropout                                              # 随机失活
        self.padding_idx = config.padding_idx

        self.embed = nn.Embedding(config.n_vocab, config.embedding, padding_idx=config.n_vocab - 1)
        self.emb_dropout_layer = nn.Dropout(self.dropout)
        self.lstm = nn.LSTM(config.embedding, config.hidden_size, config.num_layers,
                            bidirectional=True, batch_first=True, dropout=config.dropout)
        self.W2 = nn.Linear(2 * self.hidden_size + self.embed_dim, self.hidden_size * 2)
        self.final_dropout_layer = nn.Dropout(self.dropout)
        self.fc = nn.Linear(config.hidden_size * 2, config.num_classes)
        self.sigmoid = nn.Sigmoid()
        #self.feed_forward = nn.Sequential(
        #    nn.Linear(config.hidden_size * 2 + config.embedding, config.num_classes),
        #    nn.Sigmoid()
        #)

    def forward(self, x):
        x = self.complete_short_sentence(x)
        #x, _ = x
        embed = self.embed(x)  # [batch_size, seq_len, embeding]=[64, 32, 64]
        embed = self.emb_dropout_layer(embed)
        out, _ = self.lstm(embed)
        out = torch.cat((embed, out), 2)
        out =  torch.tanh(self.W2(out))
        out = out.permute(0, 2, 1)
        out = F.max_pool1d(out, out.size()[2]).squeeze(2)
        out = self.final_dropout_layer(out)
        out = self.fc(out)
        out = self.sigmoid(out)
        return out

    def complete_short_sentence(self,x):
        device = x.device
        if x.size(1) > self.max_seq_len:
            x = torch.Tensor(x[:self.max_seq_len],requires_grad=False,device=device)
        else:
            cat_size = (x.size(0),self.max_seq_len-x.size(1))
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

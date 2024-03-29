# coding: UTF-8
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy

class TextMRCNNConfig(object):

    """配置参数"""
    def __init__(self, n_vocab=859,embedding=100,
            max_seq_len=100,num_class=29,dropout=0.5,lstm_layer=2,
            hidden_size=256,lstm_dropout=0.1
            ):
        self.model_name = 'MTextRCNN'

        self.dropout = dropout                                              # 随机失活
        self.num_classes = num_class                                    # 类别数
        self.n_vocab = n_vocab                                          # 词表大小，在运行时赋值
        self.padding_idx = self.n_vocab - 1
        self.embedding = embedding
        self.hidden_size = hidden_size                                          # lstm隐藏层
        self.num_layers = lstm_layer                                             # lstm层数
        self.max_seq_len = max_seq_len
        self.lstm_dropout=lstm_dropout


'''Recurrent Convolutional Neural Networks for Text Classification'''


class TextMRCNNModel(nn.Module):
    def __init__(self, config):


        super(TextMRCNNModel, self).__init__()
        self.embed_num = config.n_vocab
        self.embed_dim = config.embedding
        self.max_seq_len = config.max_seq_len
        self.hidden_size = config.hidden_size
        self.dropout = config.dropout                                              # 随机失活
        self.padding_idx = config.padding_idx
        self.lstm_dropout = config.lstm_dropout

        self.embed_1 = nn.Embedding(config.n_vocab, config.embedding, padding_idx=config.n_vocab - 1)
        self.emb_dropout_layer_1 = nn.Dropout(self.dropout)
        self.embed_2 = nn.Embedding(config.n_vocab, config.embedding, padding_idx=config.n_vocab - 1)
        self.emb_dropout_layer_2 = nn.Dropout(self.dropout)
        self.lstm_1 = nn.LSTM(config.embedding, config.hidden_size, config.num_layers,
                            bidirectional=True, batch_first=True, dropout=self.lstm_dropout)
        self.lstm_2 = nn.LSTM(config.embedding, config.hidden_size, config.num_layers,
                            bidirectional=True, batch_first=True, dropout=self.lstm_dropout)
        self.W2_1 = nn.Linear(2 * self.hidden_size + self.embed_dim, self.hidden_size * 2)
        self.W2_2 = nn.Linear(2 * self.hidden_size + self.embed_dim, self.hidden_size * 2)

        self.final_dropout_layer_1 = nn.Dropout(self.dropout)
        self.fc_1 = nn.Linear(config.hidden_size * 2, 17)
        self.final_dropout_layer_2 = nn.Dropout(self.dropout)
        self.fc_2 = nn.Linear(config.hidden_size * 2, 12)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.complete_short_sentence(x)

        #x, _ = x
        embed_1 = self.embed_1(x)  # [batch_size, seq_len, embeding]=[64, 32, 64]
        embed_1 = self.emb_dropout_layer_1(embed_1)
        out_1, _ = self.lstm_1(embed_1)
        out_1 = torch.cat((embed_1, out_1), 2)
        out_1 =  torch.tanh(self.W2_1(out_1))
        out_1 = out_1.permute(0, 2, 1)
        out_1 = F.max_pool1d(out_1, out_1.size()[2]).squeeze(2)
        out_1 = self.final_dropout_layer_1(out_1)
        out_1 = self.fc_1(out_1)

        # second
        embed_2 = self.embed_2(x)  # [batch_size, seq_len, embeding]=[64, 32, 64]
        embed_2 = self.emb_dropout_layer_2(embed_2)
        out_2, _ = self.lstm_2(embed_2)
        out_2 = torch.cat((embed_2, out_2), 2)
        out_2 =  torch.tanh(self.W2_2(out_2))
        out_2 = out_2.permute(0, 2, 1)
        out_2 = F.max_pool1d(out_2, out_2.size()[2]).squeeze(2)
        out_2 = self.final_dropout_layer_2(out_2)
        out_2 = self.fc_2(out_2)

        out = torch.cat([out_1,out_2],-1)
        out = self.sigmoid(out)
        return out

    def complete_short_sentence(self,x):
        device = x.device
        if x.size(1) > self.max_seq_len:
            x = x[:,:self.max_seq_len]
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
    input = torch.LongTensor([range(100)])
    print(input)
    config = TextMRCNNConfig()
    model = TextMRCNNModel(config)
    output = model(input)
    print(output)

if __name__ == "__main__":
    test()

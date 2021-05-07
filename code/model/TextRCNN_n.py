# coding: UTF-8
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy

class HanAtt(nn.Module):

    def __init__(self, input_size, output_size):
        super(HanAtt, self).__init__()
        self.W = nn.Linear(input_size, output_size)
        self.u = nn.Linear(output_size, 1)

    def forward(self, x):
        u = torch.tanh(self.W(x))
        a = F.softmax(self.u(u), dim=1)
        x = x*a
        return x

class TextRCNNConfig(object):

    """配置参数"""
    def __init__(self, n_vocab=859,embedding=100,
            max_seq_len=100,num_class=29,dropout=0.5,lstm_layer=2,
            hidden_size=256,lstm_dropout=0.1,padding_idx=0,high_level_size=100
            ):
        self.model_name = 'TextRCNN'

        self.dropout = dropout                                          # 随机失活
        self.num_classes = num_class                                    # 类别数
        self.n_vocab = n_vocab                                          # 词表大小，在运行时赋值
        self.padding_idx = padding_idx
        self.embedding = embedding
        self.hidden_size = hidden_size                                  # hidden size
        self.num_layers = lstm_layer                                    # lstm层数
        self.max_seq_len = max_seq_len
        self.lstm_dropout=lstm_dropout
        self.high_level_size = high_level_size


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
        self.lstm_dropout = config.lstm_dropout
        self.high_level_size = config.high_level_size

        self.embed = nn.Embedding(config.n_vocab, config.embedding, padding_idx=self.padding_idx)
        self.emb_dropout_layer = nn.Dropout(self.dropout)
        self.lstm = nn.LSTM(config.embedding, config.hidden_size, config.num_layers,
                            bidirectional=True, batch_first=True, dropout=self.lstm_dropout)
        self.W2 = nn.Linear(2 * self.hidden_size + self.embed_dim, self.hidden_size * 2)


        self.feed_forward_1 = nn.Sequential(
            nn.Dropout(self.dropout),
            nn.Linear(config.hidden_size * 2 , 17),
            nn.Sigmoid()
        )

        self.W3 = nn.Linear(config.hidden_size * 2, self.high_level_size)
        self.feed_forward_2 = nn.Sequential(
            nn.Dropout(self.dropout),
            nn.Linear(self.high_level_size + 17, 12),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.complete_short_sentence(x)
        #x, _ = x
        embed = self.embed(x)  # [batch_size, seq_len, embeding]=[64, 32, 64]
        embed = self.emb_dropout_layer(embed)
        out, _ = self.lstm(embed)

        # Add attention here
        #out = self.han_att(out)

        # Add a attention
        #alpha = F.softmax(torch.matmul(out, self.w), dim=1).unsqueeze(-1)
        out = torch.cat((embed, out), 2)
        #out = out * alpha

        out =  torch.tanh(self.W2(out))
        out = out.permute(0, 2, 1)
        out = F.max_pool1d(out, out.size()[2]).squeeze(2)

        out_1 = self.feed_forward_1(out)
        out_2 = self.W3(out)
        out_2 = torch.cat([out_1,out_2], 1)
        out_2 = self.feed_forward_2(out_2)
        return torch.cat((out_1,out_2),1)

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
    from torchsummary import summary
    input = torch.LongTensor([range(100)])
    print(input)
    config = TextRCNNConfig(hidden_size=1024)
    model = TextRCNNModel(config)
    print(summary(model,input_size=(128,100),batch_size=128,dtypes=torch.long))
    output = model(input)
    print(output)

if __name__ == "__main__":
    test()

# coding: UTF-8
import torch
import torch.nn as nn
import numpy as np
import copy


class TextRNNConfig(object):

    """配置参数"""
    def __init__(self, n_vocab=859, embedding=500,num_class=17):
        self.model_name = 'TextRNN'
        self.dropout = 0.5                                             # 随机失活
        self.num_classes = num_class                         # 类别数
        self.n_vocab = n_vocab                                                # 词表大小，在运行时赋值
        self.embedding = embedding           # 字向量维度, 若使用了预训练词向量，则维度统一
        self.hidden_size = 128                                          # lstm隐藏层
        self.num_layers = 2                                             # lstm层数

'''Recurrent Neural Network for Text Classification with Multi-Task Learning'''

class TextRNNModel(nn.Module):
    def __init__(self, config):


        super(TextRNNModel, self).__init__()
        self.embed_num = config.n_vocab
        self.embed_dim = config.embedding

        self.embed = nn.Embedding(config.n_vocab, config.embedding, padding_idx=config.n_vocab - 1)
        self.lstm = nn.LSTM(config.embedding, config.hidden_size, config.num_layers,
                            bidirectional=True, batch_first=True, dropout=config.dropout)
        self.fc = nn.Linear(config.hidden_size * 2, config.num_classes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        #x, _ = x
        out = self.embed(x)  # [batch_size, seq_len, embeding]=[128, 32, 300]
        out, _ = self.lstm(out)
        out = self.fc(out[:, -1, :])  # 句子最后时刻的 hidden state
        out = self.sigmoid(out)
        return out

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
    input = torch.LongTensor([range(4),range(4),range(4)])
    print(input)
    config = TextRNNConfig()
    model = TextRNNModel(config)
    output = model(input)
    print(output)

if __name__ == "__main__":
    test()
    print('?????????????????????????')


from torch import nn
import torch
import torch.nn.functional as F

class BiLSTMAttConfig(object):

    """配置参数"""
    def __init__(self, n_vocab=859,embedding=100,
            max_seq_len=100,num_class=29,dropout=0.5,lstm_layer=2,
            hidden_size=1024,lstm_dropout=0.1,padding_idx=0
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


# textrnn_att
class BiLSTMAttModel(nn.Module):
    def __init__(self, config):
        super(BiLSTMAttModel, self).__init__()
        self.embed_num = config.n_vocab
        self.embed_dim = config.embedding
        self.max_seq_len = config.max_seq_len
        self.hidden_size = config.hidden_size
        self.dropout = config.dropout                                              # 随机失活
        self.padding_idx = config.padding_idx
        self.lstm_dropout = config.lstm_dropout

        self.embedding = nn.Embedding(self.embed_num,self.embed_dim)
        self.drop_emb = nn.Dropout(self.dropout)

        self.lstm = nn.LSTM(self.embed_dim, self.hidden_size, config.num_layers, bidirectional=True, batch_first=True)
        # self.lstm = nn.GRU(embed_dim, self.hidden_size, num_layers, bidirectional=True, batch_first=True)
        self.tanh1 = nn.Tanh()
        self.w = nn.Parameter(torch.zeros(self.hidden_size * 2), requires_grad=True)
        self.fc_out = nn.Sequential(
            nn.Linear(self.hidden_size * 2, self.hidden_size),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_size, config.num_classes),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.complete_short_sentence(x)
        emb = self.drop_emb(self.embedding(x))
        lstmout, _ = self.lstm(emb)

        M = self.tanh1(lstmout)
        alpha = F.softmax(torch.matmul(M, self.w), dim=1).unsqueeze(-1)
        out = lstmout * alpha
        out = torch.sum(out, dim=1)

        out = self.fc_out(out)
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
    from torchsummary import summary
    input = torch.LongTensor([range(100)])
    print(input)
    config = BiLSTMAttConfig(hidden_size=1024)
    model = BiLSTMAtt(config)
    print(summary(model,input_size=(128,100),batch_size=128,dtypes=torch.long))
    output = model(input)
    print(output)

if __name__ == "__main__":
    test()

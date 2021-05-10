import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import copy
class TextCNNConfig(object):

    """配置参数"""
    def __init__(self,n_vocab=859,embedding=00,num_class=29,max_seq_len=100,dropout=0.5,num_filters=128):
        self.model_name = 'TextCNN'
        self.dropout = dropout                                              # 随机失活
        self.n_vocab = n_vocab                                          # 词表大小，在运行时赋值
        self.padding_idx = n_vocab - 1
        self.num_class = num_class                                      # 类别数
        self.embedding = embedding
        self.num_filters = num_filters                                          # 卷积核数量(channels数)
        self.kernel_size = [2,4,6,8,10,12]
        self.Ci = 1
        self.static = False
        self.max_seq_len = max_seq_len
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
        self.static = config.static

        self.embed = nn.Embedding(self.embed_num, self.embed_dim)#词嵌入
        self.convs = nn.ModuleList([nn.Conv2d(self.Ci, self.kernel_num, (K, self.embed_dim)) for K in self.Ks])
        self.feed_forward = nn.Sequential(
            nn.Dropout(self.dropout),
            nn.Linear(len(self.Ks) * self.kernel_num,len(self.Ks) * self.kernel_num),
            nn.Dropout(self.dropout),
            nn.ReLU(),
            nn.Linear(len(self.Ks) * self.kernel_num, self.num_class),
            nn.Sigmoid()
        )

        if self.static:
            self.embed.weight.requires_grad = False

    def use_pretrain_word2vec(self,word2vec_model):
        """
        use pretrain model to init the weight in Embeding layer
        """
        assert word2vec_model["word_size"] == self.embed_dim
        vocab = word2vec_model["wv"].keys()

        for index in range(self.embed_num):
            if str(index) in vocab:
                self.embed.weight.data[index] = copy.deepcopy(word2vec_model["wv"][str(index)])
        return True

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
        x = self.embed(x)  # (N, W, D)-batch,单词数量，维度
        x = x.unsqueeze(1)  # (N, Ci, W, D)
        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs]  # [(N, Co, W), ...]*len(Ks)
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]  # [(N, Co), ...]*len(Ks)
        x = torch.cat(x, 1)
        # x = self.dropout(x)  # (N, len(Ks)*Co)
        # logit = self.sigmoid(self.fc1(x))  # (N, C)
        return self.feed_forward(x)
if __name__=="__main__":
    config = TextCNNConfig()
    net=TextCNNModel(config)
    x=torch.LongTensor([[1,2,4,5,2,35,43,113,111,451,455,22,45,55],[14,3,12,9,13,4,51,45,53,17,57,100,156,23]])
    logit=net(x)
    print(logit)

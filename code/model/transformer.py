import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy


class TransformerConfig(object):

    """配置参数"""
    def __init__(self,n_vocab=859,embedding=500,num_calss=17,max_length=70,device="cpu"):
        self.model_name = 'Transformer'
        self.n_vocab = n_vocab                                          # 词表大小，在运行时赋值
        self.padding_idx = self.n_vocab - 1
        self.num_classes = num_calss
        self.dropout = 0.1                                              # 随机失活
        self.dim_model = embedding
        self.embed = embedding
        self.hidden = 512
        self.last_hidden = 512
        self.num_head = 8
        self.num_encoder = 6
        self.max_length = max_length
        self.pad_size = self.max_length
        self.device = device

'''Attention Is All You Need'''


class TransformerModel(nn.Module):
    def __init__(self, config):
        super(TransformerModel, self).__init__()

        self.pad_size = config.pad_size
        self.padding_idx = config.padding_idx
        self.embed_dim = config.embed
        self.embed_num = config.n_vocab
        self.device = config.device

        self.embedding = nn.Embedding(config.n_vocab, config.embed, padding_idx=config.padding_idx)

        self.postion_embedding = Positional_Encoding(config.embed, config.max_length, config.dropout,config.device)
        self.encoder = Encoder(config.dim_model, config.num_head, config.hidden, config.dropout)
        self.encoders = nn.ModuleList([
            copy.deepcopy(self.encoder)
            # Encoder(config.dim_model, config.num_head, config.hidden, config.dropout)
            for _ in range(config.num_encoder)])

        self.feed_forward = nn.Sequential(
            # nn.Dropout(0.5),
            # nn.Linear(config.pad_size * config.dim_model,config.pad_size * config.dim_model),
            # nn.Dropout(0.5),
            # nn.ReLU(),
            nn.Linear(config.pad_size * config.dim_model, config.num_classes),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.complete_short_sentence(x)
        out = self.embedding(x)
        out = self.postion_embedding(out)
        for encoder in self.encoders:
            out = encoder(out)
        out = out.view(out.size(0), -1)
        # out = torch.mean(out, 1)
        out = self.feed_forward(out)
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
                self.embedding.weight.data[index] = copy.deepcopy(word2vec_model["wv"][str(index)])
        return True


class Encoder(nn.Module):
    def __init__(self, dim_model, num_head, hidden, dropout):
        super(Encoder, self).__init__()
        self.attention = Multi_Head_Attention(dim_model, num_head, dropout)
        self.feed_forward = Position_wise_Feed_Forward(dim_model, hidden, dropout)

    def forward(self, x):
        out = self.attention(x)
        out = self.feed_forward(out)
        return out


class Positional_Encoding(nn.Module):
    def __init__(self, embed, pad_size, dropout,device):
        super(Positional_Encoding, self).__init__()
        self.device = device
        self.pe = torch.tensor([[pos / (10000.0 ** (i // 2 * 2.0 / embed)) for i in range(embed)] for pos in range(pad_size)])
        self.pe[:, 0::2] = np.sin(self.pe[:, 0::2])
        self.pe[:, 1::2] = np.cos(self.pe[:, 1::2])
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = x + nn.Parameter(self.pe, requires_grad=False).to(self.device)
        out = self.dropout(out)
        return out


class Scaled_Dot_Product_Attention(nn.Module):
    '''Scaled Dot-Product Attention '''
    def __init__(self):
        super(Scaled_Dot_Product_Attention, self).__init__()

    def forward(self, Q, K, V, scale=None):
        '''
        Args:
            Q: [batch_size, len_Q, dim_Q]
            K: [batch_size, len_K, dim_K]
            V: [batch_size, len_V, dim_V]
            scale: 缩放因子 论文为根号dim_K
        Return:
            self-attention后的张量，以及attention张量
        '''
        attention = torch.matmul(Q, K.permute(0, 2, 1))
        if scale:
            attention = attention * scale
        # if mask:  # TODO change this
        #     attention = attention.masked_fill_(mask == 0, -1e9)
        attention = F.softmax(attention, dim=-1)
        context = torch.matmul(attention, V)
        return context


class Multi_Head_Attention(nn.Module):
    def __init__(self, dim_model, num_head, dropout=0.0):
        super(Multi_Head_Attention, self).__init__()
        self.num_head = num_head
        assert dim_model % num_head == 0
        self.dim_head = dim_model // self.num_head
        self.fc_Q = nn.Linear(dim_model, num_head * self.dim_head)
        self.fc_K = nn.Linear(dim_model, num_head * self.dim_head)
        self.fc_V = nn.Linear(dim_model, num_head * self.dim_head)
        self.attention = Scaled_Dot_Product_Attention()
        self.fc = nn.Linear(num_head * self.dim_head, dim_model)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(dim_model)

    def forward(self, x):
        batch_size = x.size(0)
        Q = self.fc_Q(x)
        K = self.fc_K(x)
        V = self.fc_V(x)
        Q = Q.view(batch_size * self.num_head, -1, self.dim_head)
        K = K.view(batch_size * self.num_head, -1, self.dim_head)
        V = V.view(batch_size * self.num_head, -1, self.dim_head)
        # if mask:  # TODO
        #     mask = mask.repeat(self.num_head, 1, 1)  # TODO change this
        scale = K.size(-1) ** -0.5  # 缩放因子
        context = self.attention(Q, K, V, scale)

        context = context.view(batch_size, -1, self.dim_head * self.num_head)
        out = self.fc(context)
        out = self.dropout(out)
        out = out + x  # 残差连接
        out = self.layer_norm(out)
        return out


class Position_wise_Feed_Forward(nn.Module):
    def __init__(self, dim_model, hidden, dropout=0.0):
        super(Position_wise_Feed_Forward, self).__init__()
        self.fc1 = nn.Linear(dim_model, hidden)
        self.fc2 = nn.Linear(hidden, dim_model)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(dim_model)

    def forward(self, x):
        out = self.fc1(x)
        out = F.relu(out)
        out = self.fc2(out)
        out = self.dropout(out)
        out = out + x  # 残差连接
        out = self.layer_norm(out)
        return out

def test():
    import numpy as np
    input = torch.LongTensor([range(4),range(4)])
    print(input)
    config = TransformerConfig()
    model = TransformerModel(config)
    input = model.complete_short_sentence(input)
    output = model(input)
    print(output)

if __name__ == "__main__":
    test()

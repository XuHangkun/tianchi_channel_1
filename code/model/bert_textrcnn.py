import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy
import os
from transformers import BertModel, BertConfig
from transformers import AutoConfig,AutoModel

class BERTConfig:
    def __init__(self,num_class=29,embed_dim=768,frazing_encode=False,dropout=0.5,
        pre_train_path=os.path.join(os.getenv('PROJTOP'),'user_data/bert')):
        self.model_name = 'BERT'
        self.num_classes = num_class                                    # 类别数
        self.embed_dim = embed_dim
        self.pre_train_path = pre_train_path
        self.frazing_encode = frazing_encode
        self.dropout = dropout

class BERTModel(nn.Module):
    def __init__(self, config):
        """
        Deep Pyramid Neural Networks for Text Categorization
        """
        super(BERTModel, self).__init__()
        # par for bert
        self.embed_dim = config.embed_dim
        self.pre_train_path = config.pre_train_path
        self.frazing_encode = config.frazing_encode
        self.dropout = config.dropout

        # par for textrcnn
        self.trcnn_hidden_size = 1024
        self.trcnn_num_layers = 2


        self.bert = AutoModel.from_pretrained(self.pre_train_path)
        if self.frazing_encode:
            for param in self.bert.base_model.parameters():
                param.requires_grad = False


        self.lstm = nn.LSTM(self.embed_dim, self.trcnn_hidden_size, self.trcnn_num_layers,
                            bidirectional=True, batch_first=True, dropout=0.1)
        self.W2 = nn.Linear(2 * self.trcnn_hidden_size , self.trcnn_hidden_size * 2)
        self.final_dropout_layer = nn.Dropout(self.dropout)
        self.fc = nn.Linear(self.trcnn_hidden_size * 2, config.num_classes)
        self.sigmoid = nn.Sigmoid()

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
        input_ids = x.input_ids
        attention_mask = x.attention_mask
        output = self.bert(input_ids=input_ids,attention_mask=attention_mask)
        output = output.last_hidden_state

        out, _ = self.lstm(output)
        out =  torch.tanh(self.W2(out))
        out = out.permute(0, 2, 1)
        out = F.max_pool1d(out, out.size()[2]).squeeze(2)
        out = self.final_dropout_layer(out)
        out = self.fc(out)
        out = self.sigmoid(out)

        return out

def test():
    import numpy as np
    input = torch.LongTensor([range(4),range(4)])
    print(input)
    config = BERTConfig()
    model = BERTModel(config)
    output = model(input)
    print(output)

if __name__ == "__main__":
    test()

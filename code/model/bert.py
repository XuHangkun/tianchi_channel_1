import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy
import os
from transformers import BertModel, BertConfig
from transformers import AutoConfig,AutoModel

#class BERTConfig(BertConfig):
#    def __init__(self,n_vocab=859,num_calss=17):
#        super(BERTConfig,self).__init__(vocab_size=n_vocab,pad_token_id=n_vocab-1)
#        self.model_name = 'BERT'
#        self.n_vocab = n_vocab                                          # 词表大小，在运行时赋值
#        self.padding_idx = n_vocab - 1
#        self.num_classes = num_calss                                    # 类别数
#        self.embedding = self.hidden_size                               # dim of embedding

class BERTConfig:
    def __init__(self,num_class=17,embed_dim=768,
        pre_train_path=os.path.join(os.getenv('PROJTOP'),'user_data/bert')):
        self.model_name = 'BERT'
        self.num_classes = num_class                                    # 类别数
        self.embed_dim = embed_dim
        self.pre_train_path = pre_train_path

class BERTModel(nn.Module):
    def __init__(self, config):
        """
        Deep Pyramid Convolutional Neural Networks for Text Categorization
        """
        super(BERTModel, self).__init__()
        self.embed_dim = config.embed_dim
        self.pre_train_path = config.pre_train_path

        self.bert = AutoModel.from_pretrained(self.pre_train_path)
        self.feed_forward = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(self.embed_dim, config.num_classes),
            nn.Sigmoid()
        )

    def forward(self, x):
        input_ids = x.input_ids
        attention_mask = x.attention_mask
        output = self.bert(input_ids=input_ids,attention_mask=attention_mask)
        output = output.pooler_output
        return self.feed_forward(output)

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

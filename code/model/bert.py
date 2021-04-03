import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy
from transformers import BertModel, BertConfig

class BERTConfig(BertConfig):
    def __init__(self,n_vocab=859,num_calss=17):
        super(BERTConfig,self).__init__(vocab_size=n_vocab,pad_token_id=n_vocab-1)
        self.model_name = 'BERT'
        self.n_vocab = n_vocab                                          # 词表大小，在运行时赋值
        self.padding_idx = n_vocab - 1
        self.num_classes = num_calss                                    # 类别数
        self.embedding = self.hidden_size                                      # dim of embedding

class BERTModel(nn.Module):
    def __init__(self, config):
        """
        Deep Pyramid Convolutional Neural Networks for Text Categorization
        """
        super(BERTModel, self).__init__()
        self.padding_idx = config.pad_token_id
        self.embed_num = config.vocab_size
        self.embed_dim = config.hidden_size

        self.bert = BertModel(config)
        self.feed_forward = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(self.embed_dim, config.num_classes),
            nn.Sigmoid()
        )

    def forward(self, x):
        output = self.bert(x)
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
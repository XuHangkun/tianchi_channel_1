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
        Deep Pyramid Convolutional Neural Networks for Text Categorization
        """
        super(BERTModel, self).__init__()
        self.embed_dim = config.embed_dim
        self.pre_train_path = config.pre_train_path
        self.frazing_encode = config.frazing_encode
        self.dropout = config.dropout

        self.bert = AutoModel.from_pretrained(self.pre_train_path)
        if self.frazing_encode:
            for param in self.bert.base_model.parameters():
                param.requires_grad = False

        self.feed_forward = nn.Sequential(
            nn.Dropout(self.dropout),
            nn.Linear(self.embed_dim, config.num_classes),
            nn.Sigmoid()
        )

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
        output = output.pooler_output
        #print(output)
        return self.feed_forward(output)

def test():
    import numpy as np
    import random
    seed=7
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

    from transformers import RobertaTokenizerFast
    from tokenizers import ByteLevelBPETokenizer
    tokenizer = RobertaTokenizerFast.from_pretrained(os.path.join(os.getenv('PROJTOP'),'user_data/new_bert_768_eda'), max_len=100)
    tokens = [str(i) for i in range(857,-1,-1)]
    tokenizer.add_tokens(tokens)
    #print(tokenizer.vocab)
    vocab_size = len(tokenizer.vocab)
    print("<pad>:%d,<mask>:%d,<s>:%d,</s>:%d,<unk>:%d"%(tokenizer.vocab["<pad>"],
        tokenizer.vocab["<mask>"],
        tokenizer.vocab["<s>"],
        tokenizer.vocab["</s>"],
        tokenizer.vocab["<unk>"]))
    print("total tokens: %d"%(vocab_size))

    input = tokenizer("1 2 4 5 6 7", return_tensors="pt")
    #print(input)
    config = BERTConfig(pre_train_path=os.path.join(os.getenv('PROJTOP'),'user_data/new_bert_768_eda/checkpoint-18000'))
    model = BERTModel(config)
    output = model(input)
    print(output)

if __name__ == "__main__":
    test()

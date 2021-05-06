import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy
import os
import torch.nn.init as init
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


    def reinit_bert(self,n=1):
        num_layers = len(self.bert.encoder.layer)
        parlist = []
        for i in range(num_layers,num_layers-n,-1):
            parlist.append(self.bert.encoder.layer[5].attention.self.query.weight.data),
            parlist.append(self.bert.encoder.layer[5].attention.self.query.bias.data),
        for data in parlist:
            init.normal_(data)

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
        output = output.last_hidden_state[:,0,:]
        # output = output.pooler_output

        return self.feed_forward(output)


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

    input = tokenizer(["623 355 582 617 265 162 498 289 169 137 405 693 399 842 698 335 266 14 177 415 381 693 48 328 461 478 439 473 851 636 739 374 698 494 504 656 575 754 421 421 791 200 103 718 569"],padding=True,truncation=True,return_tensors="pt")
    print(input)
    #print(input)
    config = BERTConfig(pre_train_path=os.path.join(os.getenv('PROJTOP'),'user_data/new_bert_768_eda/checkpoint-18000'))
    model = BERTModel(config)
    model.reinit_bert()
    model_weight_path = os.path.join(os.getenv('PROJTOP'),'user_data/model_data/Bert_new/BERT_fold1.chkpt')
    checkpoint = torch.load(model_weight_path)
    model_setting = checkpoint["settings"]
    model.eval()
    #model.load_state_dict(checkpoint['model'])
    output = model(input)
    print(output)

if __name__ == "__main__":
    test()

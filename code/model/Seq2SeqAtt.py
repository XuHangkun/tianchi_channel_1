# model.py

import torch
from torch import nn
import numpy as np
from torch.nn import functional as F

class Seq2SeqAttConfig(object):

    """配置参数"""
    def __init__(self, n_vocab=859,embedding=100,
            max_seq_len=100,num_class=29,dropout=0.5,lstm_layer=2,
            hidden_size=512,lstm_dropout=0.3,padding_idx=0,high_level_size=100
            ):
        self.model_name = 'Seq2SeqAtt'

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

class Seq2SeqAttModel(nn.Module):
    def __init__(self, config):
        super(Seq2SeqAttModel, self).__init__()
        self.config = config
        self.n_vocab = config.n_vocab

        # Embedding Layer
        self.embeddings = nn.Embedding(self.config.n_vocab, self.config.embedding,padding_idx=self.config.padding_idx)

        # Encoder RNN
        self.lstm = nn.LSTM(input_size =  self.config.embedding,
                            hidden_size = self.config.hidden_size,
                            num_layers = self.config.num_layers,
                            dropout=self.config.lstm_dropout,
                            bidirectional = True
                            )

        # Dropout Layer
        self.dropout = nn.Dropout(self.config.dropout)

        # Fully-Connected Layer
        self.fc = nn.Linear(
            self.config.hidden_size * 2 * 2,
            self.config.num_classes
        )

        # Softmax non-linearity
        self.sigmod = nn.Sigmoid()

    def apply_attention(self, rnn_output, final_hidden_state):
        '''
        Apply Attention on RNN output

        Input:
            rnn_output (batch_size, seq_len, num_directions * hidden_size): tensor representing hidden state for every word in the sentence
            final_hidden_state (batch_size, num_directions * hidden_size): final hidden state of the RNN

        Returns:
            attention_output(batch_size, num_directions * hidden_size): attention output vector for the batch
        '''
        hidden_state = final_hidden_state.unsqueeze(2)
        attention_scores = torch.bmm(rnn_output, hidden_state).squeeze(2)
        soft_attention_weights = F.softmax(attention_scores, 1).unsqueeze(2) #shape = (batch_size, seq_len, 1)
        attention_output = torch.bmm(rnn_output.permute(0,2,1), soft_attention_weights).squeeze(2)
        return attention_output

    def complete_short_sentence(self,x):
        device = x.device
        if x.size(1) > self.config.max_seq_len:
            x = x[:,:self.config.max_seq_len]
        else:
            cat_size = (x.size(0),self.config.max_seq_len-x.size(1))
            pad_tensor = torch.full(cat_size,self.config.padding_idx,dtype=torch.long,requires_grad=False,device=device)
            x = torch.cat((x,pad_tensor),1)
        return x

    def forward(self, x):
        # x.shape = (max_sen_len, batch_size)
        x = self.complete_short_sentence(x)
        x = x.permute(1, 0)
        embedded_sent = self.embeddings(x)
        # embedded_sent.shape = (max_sen_len=20, batch_size=64,embed_size=300)

        ##################################### Encoder #######################################
        lstm_output, (h_n,c_n) = self.lstm(embedded_sent)
        # lstm_output.shape = (seq_len, batch_size, num_directions * hidden_size)

        # Final hidden state of last layer (num_directions, batch_size, hidden_size)
        batch_size = h_n.shape[1]
        h_n_final_layer = h_n.view(self.config.num_layers,
                                   1 + 1,
                                   batch_size,
                                   self.config.hidden_size)[-1,:,:,:]

        ##################################### Attention #####################################
        # Convert input to (batch_size, num_directions * hidden_size) for attention
        final_hidden_state = torch.cat([h_n_final_layer[i,:,:] for i in range(h_n_final_layer.shape[0])], dim=1)

        attention_out = self.apply_attention(lstm_output.permute(1,0,2), final_hidden_state)
        # Attention_out.shape = (batch_size, num_directions * hidden_size)

        #################################### Linear #########################################
        concatenated_vector = torch.cat([final_hidden_state, attention_out], dim=1)
        final_feature_map = self.dropout(concatenated_vector) # shape=(batch_size, num_directions * hidden_size)
        final_out = self.fc(final_feature_map)
        final_out = self.sigmod(final_out)
        return final_out

def test():
    import numpy as np
    from torchsummary import summary
    input = torch.LongTensor([range(100),range(100)])
    print(input)
    config = Seq2SeqAttConfig(hidden_size=512)
    model = Seq2SeqAttModel(config)
    print(summary(model,input_size=(128,100),batch_size=128,dtypes=torch.long))
    output = model(input)
    print(output)

if __name__ == "__main__":
    test()

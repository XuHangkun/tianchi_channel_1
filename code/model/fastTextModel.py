import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class FastTextModel(nn.Module):
    """
    input:
        LongTensor in [batch_size,sequence_length]
    output:
        FloatTensor in  [batch_size,num_class]
    """
    def __init__(self, vocab_size, embed_dim, num_class,dropout=0.5):
        super(FastTextModel, self).__init__()
        self.embedding = nn.EmbeddingBag(vocab_size, embed_dim)
        self.feed_forward = nn.Sequential(
                    nn.Linear(embed_dim, 512),
                    nn.Dropout(dropout),
                    nn.ReLU(),
                    nn.Linear(512, num_class),
                    nn.Sigmoid()
                )
        self.init_weights()

    def init_weights(self):
        initrange = 0.5
        self.embedding.weight.data.uniform_(-initrange, initrange)

    def forward(self, text, offsets = None):
        embedded = self.embedding(text, offsets = offsets)
        return self.feed_forward(embedded)

    def predict(self,text_in_list):
        """
        example of text in list : [1,2,3,4,5]
        output : a list with num_calss feature
        """
        with torch.no_grad():
            text = torch.LongTensor(text_in_list)
            output = self.forward(text,offsets=torch.tensor([0])).squeeze()
            return output

def test():
    """
    test the model
    """
    model = FastTextModel(10,5,3)
    input = torch.LongTensor([[1,2,3],[4,5,6]])
    output = model(input)
    print(output)
    text_in_list = [1,2,3]
    pred = model.predict(text_in_list)
    print(pred)

if __name__ == "__main__":
    test()

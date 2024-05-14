import torch
import torch.nn as nn
from torch.nn import functional as F 
from transformers import GPT2Tokenizer


# hyperparameters

batch_size = 64
ctx_length = 8
max_iter = 1000
learning_rate = 1e-2
device = 'mps'


tokenizer = GPT2Tokenizer.from_pretrained('gpt2')


with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()


# Testing Tokenizer
text_input = text[0:1000]
tokens = tokenizer.encode(text_input)
print(tokens)

data = torch.tensor(tokenizer.encode(text), dtype= torch.long)
n = int(0.9*len(data)) # first 90% will be train, rest val
train_data = data[:n]
val_data = data[n:]

print(len(train_data))

class ZeroLayerModel(nn.Module):

    """

    Zero Layer Transformer Model 

    Takes in a token embedding and outputs logits 
    The Logits are computed by first multiplying 
    with an embedding matrix W_E and then multiplying
    by a decoding matrix W_O.

    y = W_O W_E x 

    """
    
    def __init__(self,d_model,vocab_size):
        super().__init__()
        self.w_e = nn.Embedding(d_model, vocab_size)
        self.w_o = nn.Linear(vocab_size, d_model)

    def forward(self,x):
        x =  self.w_e(x)
        x =  self.w_o(x)
        return x 
    
if __name__ == "__main__":
    d_model = 100
    vocab_size = 50257
    model = ZeroLayerModel(d_model,vocab_size)
    x = torch.tensor([0,2,23,10])
    print(model(x))

    

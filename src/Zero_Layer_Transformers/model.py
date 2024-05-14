import torch.nn as nn
from torch.nn import functional as F 
from Dataset import Tiny_Shakespeare 

# hyperparameters

batch_size = 64
ctx_length = 8
max_iter = 1000
learning_rate = 1e-2
device = 'mps'



class ZeroLayerModel():

    """

    Zero Layer Transformer Model 

    Takes in a token embedding and outputs logits 

    """

    def __init__(self,w_e,w_o):
        self.w_e = w_e 
        self.w_o = w_o

    def forward(self,x):
        y = self.w_o @ self.w_e @ x
        return y 
    

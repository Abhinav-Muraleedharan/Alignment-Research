import torch
import torch.nn  

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
    

import torch
import torch.nn as nn
from torch.nn import functional as F 
from transformers import GPT2Tokenizer
from model import ZeroLayerModel

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')


d_model = 100
vocab_size = 50257
device = "cuda"


# Load the model and weights
model = ZeroLayerModel(d_model, vocab_size).to(device)
model.load_state_dict(torch.load('model_weights.pth'))
model.eval()
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(tokenizer.decode(model.generate(context, max_tokens=500)[0].tolist()))
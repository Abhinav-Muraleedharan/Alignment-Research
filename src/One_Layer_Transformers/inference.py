import torch
import torch.nn as nn
from torch.nn import functional as F 
from transformers import GPT2Tokenizer
from model import OneLayerModel

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')


d_model = 384
vocab_size = 50257
device = "cuda"


# Load the model and weights
model = OneLayerModel(d_model, vocab_size).to(device)
model.load_state_dict(torch.load('model_weights_2.pth'))
model.eval()

text_input = "Would you kill a human?"
context = torch.tensor([tokenizer.encode(text_input)], device=device)

print(tokenizer.decode(model.generate(context, max_tokens=500)[0].tolist()))
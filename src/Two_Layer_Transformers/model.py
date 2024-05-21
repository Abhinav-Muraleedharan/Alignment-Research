import torch
import torch.nn as nn
from torch.nn import functional as F 
from transformers import GPT2Tokenizer


# hyperparameters

batch_size = 8
ctx_length = 32
learning_rate = 3e-4
device = 'cuda'
eval_iters = 200
n_embd = 384
n_head = 6
n_layer = 2
dropout = 0.2

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

def get_batch(split):
    # sample a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data)-ctx_length,(batch_size, ))
    x = torch.stack([data[i:i+ctx_length] for i in ix])
    y = torch.stack([data[i+1:i+ctx_length+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out



class Head(nn.Module):
    """
    
    One Attention Head Layer
    
    """
    def __init__(self,head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias = False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size,bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(ctx_length, ctx_length)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B,T,C = x.shape
        k = self.key(x)
        q = self.query(x)
        # compute attention scores: -
        wei = q @ k.transpose(-2,-1) * k.shape[-1]**-0.5 # (B, T, hs) @ (B, hs, T) -> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
        wei = F.softmax(wei, dim=-1) # (B, T, T)
        wei = self.dropout(wei)
        # perform the weighted aggregation of the values
        v = self.value(x) # (B,T,hs)
        out = wei @ v # (B, T, T) @ (B, T, hs) -> (B, T, hs)
        return out
    

class MultiHeadAttention(nn.Module):
    """

    Multiple Attention Heads in parallel

    """
    def __init__(self,num_heads,head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size*num_heads, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self,x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        return out
    
class Block(nn.Module):
    """
        
    Transformer Block

    """
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)

    def forward(self,x):
        x = x + self.sa(x) # skip Layer Norm
        return x 


class OneLayerModel(nn.Module):

    """

    One Layer Transformer Model 

    Transformer with One Attention Layer. 

    """
    
    def __init__(self,d_model,vocab_size):

        super().__init__()
        self.w_e = nn.Embedding(vocab_size,n_embd)
        self.w_o = nn.Linear(d_model,vocab_size)
        self.position_embedding_table = nn.Embedding(ctx_length, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.lm_head = nn.Linear(n_embd, vocab_size)
        self.apply(self._init_weights)

    def _init_weights(self, module):

        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)



    def forward(self,x,targets=None):
        B,T = x.shape

        x =  self.w_e(x) # embedding Layer
        x =  x + self.position_embedding_table(torch.arange(T, device=device)) # (T,C)
        x = self.blocks(x) # (B,T,C)
        x = self.lm_head(x) # (B,T,vocab_size)

        if targets is None:
            loss = None
        else:
            B,T,C = x.shape
            x = x.view(B*T,C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(x, targets)
        return x, loss 
    
    def generate(self,idx,max_tokens):
        for _ in range(max_tokens):
            idx_cond = idx[:, -ctx_length:]
            logits, loss = self(idx_cond)
            logits = logits[:,-1,:]
            # print(logits)
            probs = F.softmax(logits,dim=-1)
            # nan_mask = torch.isnan(probs)
            # Print the tensor and the NaN mask
            # print("Tensor:", probs)
            # print("NaN Mask:", nan_mask)
            # print("Prob:",probs)
            idx_next = torch.multinomial(probs,num_samples=1)
            idx = torch.cat((idx,idx_next),dim=1)
        return idx
    
    
if __name__ == "__main__":
    d_model = 384
    vocab_size = 50257
    max_iters = 2000
    eval_interval = 100
    model = OneLayerModel(d_model,vocab_size)
    model.load_state_dict(torch.load('model_weights_2.pth'))
    # print the number of parameters in the model
    
    
    m = model.to(device)
    print(sum(p.numel() for p in m.parameters())/1e6, 'M parameters')

    context = torch.zeros((1, 1), dtype=torch.int, device=device)
    print(tokenizer.decode(model.generate(context, max_tokens=50)[0].tolist()))
    # create a PyTorch optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    for iter in range(max_iters):
        # every once in a while evaluate the loss on train and val sets
        if iter % eval_interval == 0:
            losses = estimate_loss()
            print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

        # sample a batch of data
        xb, yb = get_batch('train')

        #evaluate the loss
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    # Save model weights
    torch.save(model.state_dict(), 'model_weights_2.pth')
    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    print(tokenizer.decode(model.generate(context, max_tokens=500)[0].tolist()))

    

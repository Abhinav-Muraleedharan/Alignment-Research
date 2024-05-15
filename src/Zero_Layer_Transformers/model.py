import torch
import torch.nn as nn
from torch.nn import functional as F 
from transformers import GPT2Tokenizer


# hyperparameters

batch_size = 64
ctx_length = 8
learning_rate = 1e-2
device = 'cuda'
eval_iters = 200

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
        self.w_e = nn.Embedding(vocab_size,d_model)
        self.w_o = nn.Linear(d_model,vocab_size)

    def forward(self,x,targets=None):
        x =  self.w_e(x)
        x =  self.w_o(x)
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
            logits, loss = self(idx)
            logits = logits[:,-1,:]
            print(logits)
            probs = F.softmax(logits,dim=-1)
            nan_mask = torch.isnan(probs)
            # Print the tensor and the NaN mask
            #print("Tensor:", probs)
            #print("NaN Mask:", nan_mask)
            
            #print("Prob:",probs)
            idx_next = torch.multinomial(probs,num_samples=1)
            idx = torch.cat((idx,idx_next),dim=1)
<<<<<<< HEAD
        return idx 
=======
            return idx
>>>>>>> af06bf330b56e183e40f23e485d543fc060f35f3
    
    
if __name__ == "__main__":
    d_model = 100
    vocab_size = 50257
    max_iters = 1000
    eval_interval = 100
    model = ZeroLayerModel(d_model,vocab_size)
    
    m = model.to(device)
    context = torch.zeros((1, 1), dtype=torch.long, device=device)
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
    torch.save(model.state_dict(), 'model_weights.pth')
    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    print(tokenizer.decode(model.generate(context, max_tokens=500)[0].tolist()))

    

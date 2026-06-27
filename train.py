import torch
import torch.nn as nn
from torch.nn import functional as F
import math
import sys
import os

batch_size = 64         
gradient_accumulation_steps = 4 
block_size = 256        
max_iters = 15000       
eval_interval = 250   
learning_rate = 3e-4     
min_lr = 3e-5          
warmup_iters = 1000     
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 512            
n_head = 8              
n_layer = 8              
dropout = 0.2           


if not os.path.exists('data.txt'):
    print("Error: 'data.txt' not found! Please create the dataset.")
    sys.exit()

with open('data.txt', 'r', encoding='utf-8') as f:
    text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi.get(c, 0) for c in s]
decode = lambda l: ''.join([itos.get(i, '') for i in l])

data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

def get_batch(split):
    data_set = train_data if split == 'train' else val_data
    ix = torch.randint(len(data_set) - block_size, (batch_size,))
    x = torch.stack([data_set[i:i+block_size] for i in ix])
    y = torch.stack([data_set[i+1:i+block_size+1] for i in ix])
    return x.to(device), y.to(device)

@torch.no_grad()
def estimate_loss(model):
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            with torch.amp.autocast(device_type='cuda' if 'cuda' in device else 'cpu'):
                logits = model(X)
                B, T, C = logits.shape
                loss = F.cross_entropy(logits.view(B*T, C), Y.view(B*T))
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

def get_lr(it):
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    if it > max_iters:
        return min_lr
    decay_ratio = (it - warmup_iters) / (max_iters - warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (learning_rate - min_lr)


class CausalSelfAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.c_attn = nn.Linear(n_embd, 3 * n_embd, bias=False)
        self.c_proj = nn.Linear(n_embd, n_embd, bias=False)
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(n_embd, dim=2)
        k = k.view(B, T, n_head, C // n_head).transpose(1, 2)
        q = q.view(B, T, n_head, C // n_head).transpose(1, 2)
        v = v.view(B, T, n_head, C // n_head).transpose(1, 2)
        
     
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=dropout if self.training else 0, is_causal=True)
        
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.c_proj(y))

class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln_1 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention()
        self.ln_2 = nn.LayerNorm(n_embd)
        self.mlp = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd, bias=False),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd, bias=False),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class SuperGPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block() for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)
       
        self.token_embedding_table.weight = self.lm_head.weight

    def forward(self, idx):
        B, T = idx.shape
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))
        x = self.token_embedding_table(idx) + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        return logits


model = SuperGPT()

if torch.cuda.device_count() > 1:
    print(f"Initializing Distributed DataParallel with {torch.cuda.device_count()} GPUs...")
    model = nn.DataParallel(model)

model = model.to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-1, betas=(0.9, 0.95))
scaler = torch.amp.GradScaler(device_type='cuda' if 'cuda' in device else 'cpu')

print("Starting Advanced GPT Training...")

for iter in range(max_iters):
    lr = get_lr(iter)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
        
    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss(model)
        print(f"Step {iter:05d} | Train Loss: {losses['train']:.4f} | Val Loss: {losses['val']:.4f} | LR: {lr:.2e}")
    
 
    for micro_step in range(gradient_accumulation_steps):
        xb, yb = get_batch('train')
        with torch.amp.autocast(device_type='cuda' if 'cuda' in device else 'cpu'):
            logits = model(xb)
            B, T, C = logits.shape
            loss = F.cross_entropy(logits.view(B*T, C), yb.view(B*T))
            loss = loss / gradient_accumulation_steps
        
        scaler.scale(loss).backward()
    

    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)


model_to_save = model.module if hasattr(model, 'module') else model
torch.save(model_to_save.state_dict(), 'super_gpt.pth')
print("Training Complete. Model saved as 'super_gpt.pth'")

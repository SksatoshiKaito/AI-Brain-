import torch
import torch.nn as nn
from torch.nn import functional as F
import sys
import os

block_size = 256
n_embd = 512
n_head = 8
n_layer = 8
dropout = 0.2
device = 'cuda' if torch.cuda.is_available() else 'cpu'

if not os.path.exists('data.txt'):
    print("Error: 'data.txt' missing. Cannot load vocabulary.")
    sys.exit()

with open('data.txt', 'r', encoding='utf-8') as f:
    text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi.get(c, 0) for c in s]
decode = lambda l: ''.join([itos.get(i, '') for i in l])


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
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=True)
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
        return self.lm_head(x)

model = SuperGPT().to(device)

try:
    model.load_state_dict(torch.load('super_gpt.pth', map_location=device))
    print("Super AI Model Loaded Successfully!")
except Exception as e:
    print(f"Failed to load 'super_gpt.pth'. Did you finish training? Error: {e}")
    sys.exit()

model.eval()

def generate_text(model, context, max_new_tokens, temperature=0.8, top_k=40, top_p=0.9):
    generated = []
    with torch.no_grad():
        for _ in range(max_new_tokens):
            context_cond = context[:, -block_size:]
            logits = model(context_cond)
            logits = logits[:, -1, :] / temperature
            
            # Top-K filtering
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            
            probs = F.softmax(logits, dim=-1)
            
          
            if top_p is not None:
                sorted_probs, sorted_indices = torch.sort(probs, descending=True)
                cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                probs[indices_to_remove] = 0.0
                probs = probs / probs.sum(dim=-1, keepdim=True)
            
            next_idx = torch.multinomial(probs, num_samples=1)
            context = torch.cat((context, next_idx), dim=1)
            generated.append(next_idx.item())
            
    return decode(generated)

print("="*50)
print("SYSTEM: AI is ready. Type 'exit' to terminate the session.")
print("="*50)

while True:
    user_input = input("\nYou: ")
    if user_input.lower() == 'exit':
        print("Terminating session...")
        break
    
    context = torch.tensor([encode(user_input)], dtype=torch.long, device=device)
    
  
    response = generate_text(model, context, max_new_tokens=250, temperature=0.7, top_k=50, top_p=0.85)
    
    print(f"AI: {response}")

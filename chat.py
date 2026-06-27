import torch
import torch.nn as nn
from torch.nn import functional as F

# কনফিগারেশন - তোমার train.py এর সাথে অবশ্যই মিল থাকতে হবে
block_size = 128
n_embd = 192
n_head = 6
n_layer = 6
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# data.txt থেকে ক্যারেক্টার ম্যাপ তৈরি
with open('data.txt', 'r', encoding='utf-8') as f:
    text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)
stoi = {ch:i for i,ch in enumerate(chars)}
itos = {i:ch for i,ch in enumerate(chars)}
encode = lambda s: [stoi.get(c, 0) for c in s]
decode = lambda l: ''.join([itos.get(i, '') for i in l])

# মডেলের গঠন
class HozaiFA_GPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[nn.TransformerEncoderLayer(d_model=n_embd, nhead=n_head, dim_feedforward=4*n_embd, batch_first=True) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx):
        pos_emb = self.position_embedding_table(torch.arange(idx.shape[1], device=device))
        x = self.token_embedding_table(idx) + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        return self.lm_head(x)

# মডেল লোড করা
model = HozaiFA_GPT().to(device)
try:
    model.load_state_dict(torch.load('model_pro.pth', map_location=device))
    print("মডেল সফলভাবে লোড হয়েছে!")
except:
    print("এরর: model_pro.pth ফাইলটি পাওয়া যায়নি। আগে ট্রেইনিং শেষ করো!")
    exit()

model.eval()

print("এআই প্রস্তুত! কথা বলা শুরু করো (বন্ধ করতে 'exit' লেখো):")

# চ্যাট লুপ
while True:
    user_input = input("\nতুমি: ")
    if user_input.lower() == 'exit': break
    
    context = torch.tensor([encode(user_input)], dtype=torch.long, device=device)
    
    generated = []
    # এখানে টেম্পারেচার দিয়ে এআই-এর সৃজনশীলতা নিয়ন্ত্রণ করা হচ্ছে (০.৭ - ০.৯ এর মধ্যে ভালো ফল দেয়)
    temperature = 0.8
    
    for _ in range(120): # উত্তর খুব বড় হবে না কিন্তু অর্থপূর্ণ হবে
        logits = model(context[:, -block_size:])
        logits = logits[:, -1, :] / temperature 
        probs = F.softmax(logits, dim=-1)
        next_idx = torch.multinomial(probs, num_samples=1)
        
        context = torch.cat((context, next_idx), dim=1)
        generated.append(next_idx.item())
    
    print("এআই:", decode(generated))
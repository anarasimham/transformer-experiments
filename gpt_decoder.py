import torch
from torch import nn
import random
import pdb
import time

NUM_BATCHES = 5000
TIMESTEPS = 256
BATCH_SIZE=64
EVAL_ITERS=25
EMB_DIMS=384
NUM_HEADS=6
LEARNING_RATE=1e-3
NUM_BLOCKS=6
DROPOUT_PCT=0.2

#NUM_BATCHES = 5000
#TIMESTEPS = 256
#BATCH_SIZE=64
#EVAL_ITERS=150
#EMB_DIMS=384
#NUM_HEADS=6
#LEARNING_RATE=3e-4
#NUM_BLOCKS=6
#DROPOUT_PCT=0.2

device = 'cuda' if torch.cuda.is_available() else 'cpu'
#device = 'cpu'
print('Running on', device, '\n')

start_time = time.time()

data = open('input.txt', 'r')
lines=data.read()
print(lines[:1000])

uniques = ''.join(sorted(list(set(lines))))
print(len(uniques), 'uniques')

CHANNELS = len(uniques)

charmap = {}
intmap = {}

for idx,c in enumerate(uniques):
    charmap[c] = idx
    intmap[idx] = c

stoi = lambda mystr: [charmap[c] for c in mystr]
itos = lambda myints: [intmap[i] for i in myints]

class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(EMB_DIMS, head_size, bias=False, device=device)
        self.query = nn.Linear(EMB_DIMS, head_size, bias=False, device=device)
        self.value = nn.Linear(EMB_DIMS, head_size, bias=False, device=device)
        self.register_buffer('ones_tril', torch.tril(torch.ones(TIMESTEPS, TIMESTEPS, device=device)))
        self.head_size = head_size
        self.dropout = nn.Dropout(DROPOUT_PCT)

    def forward(self, logits):
        B,T,C = logits.shape
        q = self.query(logits)
        k = self.key(logits)
        weights_mat = q @ k.transpose(-1, -2) * self.head_size ** -0.5
        weights_mat = weights_mat.masked_fill(self.ones_tril == 0, float('-inf'))
        weights_mat = torch.softmax(weights_mat, dim=-1)
        weights_mat = self.dropout(weights_mat)

        v = self.value(logits)
        out = weights_mat @ v
        return out

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(EMB_DIMS, EMB_DIMS, device=device)
        self.dropout = nn.Dropout(DROPOUT_PCT)

    def forward(self, x):
        res = torch.cat([h(x) for h in self.heads], dim=-1)
        return self.dropout(self.proj(res))

class FeedForward(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(EMB_DIMS, EMB_DIMS*4, device=device),
            nn.ReLU(),
            nn.Linear(EMB_DIMS*4, EMB_DIMS, device=device),
            nn.Dropout(DROPOUT_PCT),
        )
    
    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.multi_head = MultiHeadAttention(NUM_HEADS, EMB_DIMS//NUM_HEADS)
        self.ffn = FeedForward()
        self.ln1 = nn.LayerNorm(EMB_DIMS, device=device)
        self.ln2 = nn.LayerNorm(EMB_DIMS, device=device)

    def forward(self, x):
        x = x + self.multi_head(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x

class BigramLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding = nn.Embedding(CHANNELS, EMB_DIMS, device=device)
        self.position_embedding = nn.Embedding(TIMESTEPS, EMB_DIMS, device=device)
        self.blocks = nn.Sequential(*[Block() for _ in range(NUM_BLOCKS)])
        self.ln = nn.LayerNorm(EMB_DIMS, device=device)
        self.lm_head = nn.Linear(EMB_DIMS, CHANNELS, device=device)

    def forward(self, x):
        B,T = x.shape
        tok_emb = self.token_embedding(x)
        pos_emb = self.position_embedding(torch.arange(T, device=device))
        combined = tok_emb + pos_emb
        combined = self.blocks(combined)
        combined = self.ln(combined)
        logits = self.lm_head(combined)
        return logits

def get_batch(dataset, batch_size):
    samples = []
    targets = []
    for i in range(batch_size):
        start = random.randint(0,len(dataset)-(TIMESTEPS+1))
        data = dataset[start:start+TIMESTEPS+1]
        samples.append(torch.tensor(data[:-1], device=device))
        targets.append(torch.tensor(data[1:], device=device))
    sample_t = torch.stack(samples)
    target_t = torch.stack(targets)
    return sample_t, target_t

@torch.no_grad()
def eval_loss():
    losses = []
    for ds in [train, test]:
        ds_loss = []
        for i in range(EVAL_ITERS):
            print(i,'eval')
            samples, targets = get_batch(ds, BATCH_SIZE)
            logits = m(samples)
            B,T,C = logits.shape
            logits = logits.reshape(B*T,C)
            targets = targets.reshape(B*T)
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits, targets)
            ds_loss.append(loss.item())
        losses.append(sum(ds_loss)/len(ds_loss))
    return losses

def print_result():
    inchar = torch.tensor([[0]*TIMESTEPS], device=device)
    outstr = ''
    with torch.no_grad():
        for i in range(1000):
            logits = m(inchar[:,-TIMESTEPS:])
            probs = logits.softmax(dim=2)
            B,T,C = probs.shape
            probs = probs.reshape(B*T,C)
            sample = torch.multinomial(probs,1)
            inchar = torch.cat((inchar,sample[-1].unsqueeze(0)),dim=1)
            outstr += ''.join(itos([sample[-1].item()]))

    return outstr

num_chars = int(len(lines)*0.9)
train = stoi(lines[:num_chars])
test = stoi(lines[num_chars:])

m = BigramLanguageModel()
optim = torch.optim.AdamW(m.parameters(), lr=LEARNING_RATE)

for i in range(NUM_BATCHES):
    if i%500 == 0:
        losses = eval_loss()
        print(i,'train',losses[0],'test',losses[1])
        print(print_result())

    samples, targets = get_batch(train, BATCH_SIZE)
    optim.zero_grad()
    logits = m(samples)

    B,T,C = logits.shape
    logits = logits.reshape((B*T,C))
    targets = targets.reshape((B*T))

    loss_fn = nn.CrossEntropyLoss()
    loss = loss_fn(logits, targets)

    loss.backward()
    optim.step()

    print('iteration',i,'time elapsed',round(time.time()-start_time,3),'seconds','loss',round(loss.item(), 2))


print(print_result())

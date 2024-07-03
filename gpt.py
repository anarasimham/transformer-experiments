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
LEARNING_RATE=3e-4
NUM_BLOCKS=6
DROPOUT_PCT=0.2


sample_text = """
First Citizen:
We are accounted poor citizens, the patricians good.
What authority surfeits on would relieve us: if they
would yield us but the superfluity, while it were
wholesome, we might guess they relieved us humanely;
but they think we are too dear: the leanness that
afflicts us, the object of our misery, is as an
inventory to particularise their abundance; our
sufferance is a gain to them Let us revenge this with
our pikes, ere we become rakes: for the gods know I
speak this in hunger for bread, not in thirst for revenge.

Second Citizen:
Would you proceed especially against Caius Marcius?

All:
Against him first: he's a very dog to the commonalty.

Second Citizen:
Consider you what services he has done for his country?

First Citizen:
Very well; and could be content to give him good
report fort, but that he pays himself with being proud.

Second Citizen:
Nay, but speak not maliciously.

First Citizen:
I say unto you, what he hath done famously, he did
it to that end: though soft-conscienced men can be
content to say it was for his country he did it to
please his mother and to be partly proud; which he
is, even till the altitude of his virtue.

Second Citizen:
What he cannot help in his nature, you account a
vice in him. You must in no way say he is covetous.

First Citizen:
If I must not, I need not be barren of accusations;
he hath faults, with surplus, to tire in repetition.
What shouts are these? The other side o' the city
is risen: why stay we prating here? to the Capitol!
"""

charmap = {}
intmap = {}

stoi = lambda mystr: [charmap[c] for c in mystr]
itos = lambda myints: [intmap[myint] for myint in myints]


class Head(nn.Module):
    def __init__(self, head_size, is_masked=False):
        super().__init__()
        self.key = nn.Linear(EMB_DIMS, head_size, bias=False, device=device)
        self.query = nn.Linear(EMB_DIMS, head_size, bias=False, device=device)
        self.value = nn.Linear(EMB_DIMS, head_size, bias=False, device=device)
        self.register_buffer('ones_tril', torch.tril(torch.ones(TIMESTEPS, TIMESTEPS, device=device)))
        self.head_size = head_size
        self.dropout = nn.Dropout(DROPOUT_PCT)
        self.is_masked = is_masked
        self.features = None

    def set_features(self, features):
        self.features = features

    def forward(self, logits):
        B,T,C = logits.shape
        q = self.query(logits)

        if self.features is not None:
            k = self.key(self.features)
        else:
            k = self.key(logits)

        weights_mat = q @ k.transpose(-1, -2) * self.head_size ** -0.5
        if self.is_masked:
            weights_mat = weights_mat.masked_fill(self.ones_tril == 0, float('-inf'))
        weights_mat = torch.softmax(weights_mat, dim=-1)
        weights_mat = self.dropout(weights_mat)

        if self.features is not None:
            v = self.value(self.features)
        else:
            v = self.value(logits)

        out = weights_mat @ v
        return out

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size, is_masked=False):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size, is_masked) for _ in range(num_heads)])
        self.proj = nn.Linear(EMB_DIMS, EMB_DIMS, device=device)
        self.dropout = nn.Dropout(DROPOUT_PCT)
        self.features = None

    def set_features(self, features):
        self.features = features

    def forward(self, x):
        for h in self.heads:
            h.set_features(self.features)

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

class EncoderBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.multi_head = MultiHeadAttention(NUM_HEADS, EMB_DIMS//NUM_HEADS, is_masked=False)
        self.ffn = FeedForward()
        self.ln1 = nn.LayerNorm(EMB_DIMS, device=device)
        self.ln2 = nn.LayerNorm(EMB_DIMS, device=device)

    def forward(self, x):
        x = x + self.multi_head(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x

class DecoderBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.multi_head_masked = MultiHeadAttention(NUM_HEADS, EMB_DIMS//NUM_HEADS, is_masked=True)
        self.multi_head_unmasked = MultiHeadAttention(NUM_HEADS, EMB_DIMS//NUM_HEADS, is_masked=False)
        self.ffn = FeedForward()
        self.ln1 = nn.LayerNorm(EMB_DIMS, device=device)
        self.ln2 = nn.LayerNorm(EMB_DIMS, device=device)
        self.ln3 = nn.LayerNorm(EMB_DIMS, device=device)
        self.features = None

    def set_features(self, features):
        self.features = features

    def forward(self, x):
        x = x + self.multi_head_masked(self.ln1(x))

        self.multi_head_unmasked.set_features(self.features)
        x = x + self.multi_head_unmasked(self.ln2(x))

        x = x + self.ffn(self.ln3(x))
        return x

class BigramLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding = nn.Embedding(CHANNELS, EMB_DIMS, device=device)
        self.position_embedding = nn.Embedding(TIMESTEPS, EMB_DIMS, device=device)
        self.encoder_blocks = nn.Sequential(*[EncoderBlock() for _ in range(NUM_BLOCKS)])
        self.decoder_blocks = nn.Sequential(*[DecoderBlock() for _ in range(NUM_BLOCKS)])
        self.ln = nn.LayerNorm(EMB_DIMS, device=device)
        self.lm_head = nn.Linear(EMB_DIMS, CHANNELS, device=device)

    def forward(self, ys):
        xs = ys[:,:-1]
        ys = ys[:,1:]
        B,T = ys.shape
        pos_emb = self.position_embedding(torch.arange(T, device=device))

        #xs = [torch.tensor(stoi('|'), device=device).unsqueeze(1).repeat(B,1),ys[:,:-1]]
        #xs = torch.cat(xs, dim=1)

        tok_emb_x = self.token_embedding(xs)
        combined_x = tok_emb_x + pos_emb

        tok_emb_y = self.token_embedding(ys)
        combined_y = tok_emb_y + pos_emb

        combined_x = self.encoder_blocks(combined_x)

        for b in self.decoder_blocks:
            b.set_features(combined_x)
        combined_y = self.decoder_blocks(combined_y)

        combined_y = self.ln(combined_y)

        logits = self.lm_head(combined_y)
        return logits

def get_batch(dataset, batch_size):
    samples = []
    targets = []
    for _ in range(batch_size):
        start = random.randint(0,len(dataset)-(TIMESTEPS+2))
        data = dataset[start:start+TIMESTEPS+2]
        samples.append(torch.tensor(data[:-1], device=device))
        targets.append(torch.tensor(data[1:], device=device))
    sample_t = torch.stack(samples)
    target_t = torch.stack(targets)
    return sample_t, target_t

@torch.no_grad()
def eval_loss(train_ds, test_ds):
    losses = []
    for idx,ds in enumerate([train_ds, test_ds]):
        ds_loss = []
        for _ in range(EVAL_ITERS):
            samples, targets = get_batch(ds, BATCH_SIZE)
            targets = targets[:,1:]
            logits = m(samples)
            B,T,C = logits.shape
            logits = logits.reshape(B*T,C)
            targets = targets.reshape(B*T)
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits, targets)
            ds_loss.append(loss.item())
        losses.append(sum(ds_loss)/len(ds_loss))
    return losses

def generate_sample_output(seed_input):
    converted_input = stoi(seed_input)
    converted_input = [0]*((TIMESTEPS+1)-len(converted_input))+converted_input
    converted_input = converted_input[-(TIMESTEPS+1):]

    inchar = torch.tensor([converted_input], device=device)
    outstr = ''
    with torch.no_grad():
        for _ in range(1000):
            logits = m(inchar[:,-(TIMESTEPS+1):])
            probs = logits.softmax(dim=2)
            B,T,C = probs.shape
            probs = probs.reshape(B*T,C)
            sample = torch.multinomial(probs,1)
            inchar = torch.cat((inchar,sample[-1].unsqueeze(0)),dim=1)
            outstr += ''.join(itos([sample[-1].item()]))
        print_frequency(outstr)

    return outstr

def print_eval(iteration):
    losses = eval_loss(train, test)
    print(iteration,'train',losses[0],'test',losses[1])
    print(generate_sample_output(sample_text))


def print_frequency(chars):
    char_to_count = {}
    for c in chars:
        if c not in char_to_count:
            char_to_count[c] = 1
        else:
            char_to_count[c] = char_to_count[c]+1

    count_to_char = []
    for k in char_to_count.keys():
        count_to_char.append((char_to_count[k],k))

    count_to_char = sorted(count_to_char)

    for item in count_to_char:
        print(item)



if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Running on', device, '\n')

    start_time = time.time()

    data = open('input.txt', 'r')
    lines=data.read()
    print(lines[:1000])

    print_frequency(lines)

    uniques = ''.join(sorted(list(set(lines))))
    print(len(uniques), 'uniques')

    #uniques += '|'

    CHANNELS = len(uniques)

    print(uniques)

    for idx,c in enumerate(uniques):
        charmap[c] = idx
        intmap[idx] = c

    num_chars = int(len(lines)*0.9)
    train = stoi(lines[:num_chars])
    test = stoi(lines[num_chars:])

    m = BigramLanguageModel()
    optim = torch.optim.AdamW(m.parameters(), lr=LEARNING_RATE)

    for i in range(NUM_BATCHES):
        if i%500 == 0:
            print_eval(i)

        samples, targets = get_batch(train, BATCH_SIZE)
        targets = targets[:,1:]
        optim.zero_grad()
        logits = m(samples)

        B,T,C = logits.shape
        logits = logits.reshape((B*T,C))
        targets = targets.reshape((B*T))

        loss_fn = nn.CrossEntropyLoss()
        loss = loss_fn(logits, targets)

        loss.backward()
        optim.step()

        if i%100 == 0:
            print('iter',i,'elapsed:',round(time.time()-start_time,3),'s','loss', round(loss.item(),4))

    print_eval(i)

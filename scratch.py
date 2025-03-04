import time

import torch
from torch.optim import AdamW
from tqdm import tqdm
from sampler import SparseGraphSampler
from tokenizer import Tokenizer
from pretrain_bert import BS
from models.bert import BERT

g = torch.load('data/lanl_tr.pt', weights_only=False)
g = SparseGraphSampler(g, batch_size=BS, neighbors=25)
num_tokens = g.x.max().long() + 3 + 1
t = Tokenizer(num_tokens, 3)

# BERT Mini
model = BERT(
    t.vocab_size,
    device=3,
    layers=4,
    hidden_size=256,
)

opt = AdamW(
    model.parameters(), lr=3e-4,
    betas=(0.9, 0.95), eps=1e-8, weight_decay=0.1
)

def mean(ls): return sum(ls) / len(ls)

samp = []
toke = []
total = []
fwd = []
bwd = []
opt_t = []
i = 0
samples = torch.randperm(g.data.edge_index.size(1)).split(BS)

# Everything: ~9s
# Everything but masking: ~8s
# With mp: 4

for i in range(10):
    sample = g.data.edge_index[:, samples[i]].T
    batch = g.sample(sample)

    st = time.time()
    args = t.tokenize_and_mask(batch)
    t_time = time.time() - st

    toke.append(t_time)

    print(f'Time: {mean(toke)}')
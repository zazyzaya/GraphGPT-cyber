from argparse import ArgumentParser
from copy import deepcopy
import time
from types import SimpleNamespace

import numpy as np
from joblib import Parallel, delayed
from sklearn.metrics import (
    roc_auc_score as auc_score,
    average_precision_score as ap_score
)
import torch
from torch.optim.adamw import AdamW
from torch.optim.lr_scheduler import LRScheduler, CosineAnnealingLR
from transformers import BertConfig
from tqdm import tqdm

from fast_auc import fast_auc, fast_ap
from models.gnn_bert import RWBert, GNNEmbedding
from rw_sampler import TRWSampler as TRW, RWSampler as RW
from eval_kg import Evaluator

DEVICE = 0
WARMUP_E = 9.6  # Epochs
EPOCHS = 32    # Epochs
LR = 3e-4

WALK_LEN = 4
NUM_EVAL_ITERS = 1
MINI_BS = 512
BS = 1024
EVAL_BS = 1024
EVAL_EVERY = 250
T_MAX = 100_000 # From alibaba source code

class Scheduler(LRScheduler):
    def __init__(self, optimizer, warmup_stop, total_steps, last_epoch=-1, verbose="deprecated"):
        self.warmup_stop = warmup_stop
        self.total_steps = total_steps
        self.cosine = CosineAnnealingLR(
            optimizer, T_MAX, last_epoch=-1
        )

        super().__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        # Warmup period
        if self.last_epoch < self.warmup_stop:
            return [group['initial_lr'] * (self.last_epoch / self.warmup_stop)
                    for group in self.optimizer.param_groups]
        else:
            return [group['initial_lr'] * (1 - ((self.last_epoch-self.warmup_stop)/(self.total_steps-self.warmup_stop)))
                    for group in self.optimizer.param_groups]

def sample_uni(tr, src,dst,ts, walk_len, edge_features=None):
    if walk_len > 0:
        rw = tr.rw(src, reverse=True, trim_missing=False)
    else:
        rw = src.unsqueeze(-1)

    if edge_features is not None:
        # Predict next node AND edge features (generally worse performance)
        #mask = torch.tensor([GNNEmbedding.MASK], device=DEVICE).repeat(edge_features.size())
        #rw = torch.cat([rw, mask], dim=1)
        #dst = torch.cat([edge_features, dst.unsqueeze(-1)], dim=1).flatten()
        rw = torch.cat([rw, edge_features], dim=1)

    masks = torch.tensor([[GNNEmbedding.MASK]], device=DEVICE).repeat(rw.size(0),1)
    rw = torch.cat([rw,masks], dim=1)
    attn_mask = rw != GNNEmbedding.PAD

    return rw, rw==GNNEmbedding.MASK, dst, attn_mask

def sample_bi(tr, src,dst,ts, walk_len, edge_features=None):
    if walk_len > 0:
        src_rw = tr.rw(src, reverse=True, trim_missing=False)
        dst_rw = tr.rw(dst, reverse=False, trim_missing=False)

        if edge_features is None:
            rw = torch.cat([src_rw, dst_rw], dim=1)
            feat_dim = 0
        else:
            rw = torch.cat([src_rw, edge_features, dst_rw], dim=1)
            feat_dim = edge_features.size(1)

    else:
        return sample_uni(tr, src,dst,ts, walk_len, edge_features)

    mask_col = src_rw.size(1) + feat_dim
    rw[:, mask_col] = GNNEmbedding.MASK
    attn_mask = rw != GNNEmbedding.PAD

    return rw, rw==GNNEmbedding.MASK, dst, attn_mask


def train(tr,va,te, model: RWBert):
    opt = AdamW(
        model.parameters(), lr=LR,
        betas=(0.9, 0.99), eps=1e-10, weight_decay=0.02
    )

    updates_per_epoch = tr.col.size(0) / BS
    warmup_stop = int(updates_per_epoch * WARMUP_E)
    total_steps = int(updates_per_epoch * EPOCHS)

    model.eval()
    best, best_te = te_stats, va_stats = evaluator.get_metrics(tr,va,te, model)
    sched = Scheduler(opt, warmup_stop, total_steps)

    with open(f'{HOME}/{DATASET}/snapshot-ft_results_{FNAME}_{SIZE}_wl{WALK_LEN}.txt', 'w+') as f:
            f.write(f'epoch,updates,te_mmr,te_hits@1,te_hits@5,te_hits@10,va_mmr,va_hits@1,va_hits@5va_hits@10\n')
            f.write(f'0,0,{",".join([str(s) for s in te_stats])},{",".join([str(s) for s in va_stats])}\n')

    updates = 0
    opt.zero_grad()
    st = time.time()
    steps = 0

    e = 0
    for e in range(EPOCHS):
        for samp in tr.edge_iter():
            if tr.edge_features:
                src,dst,ts,ef = samp
            else:
                src,dst,ts = samp
                ef = None

            model.train()
            walk,mask,tgt,attn_mask = sample(tr, src,dst,ts, WALK_LEN, edge_features=ef)
            loss = model.modified_fwd(walk, mask, tgt, attn_mask)
            loss.backward()

            steps += 1
            if steps*MINI_BS == BS:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
                opt.step()
                sched.step()
                en = time.time()

                lr = sched.get_last_lr()[0]
                print(f'[{updates}-{e}] {loss} (lr: {lr:0.2e}, {en-st:0.2f}s)')

                updates += 1
                steps = 0
                opt.zero_grad()
                st = time.time()

            if updates and updates % EVAL_EVERY == 0:
                model.eval()
                te_stats,va_stats = evaluator.get_metrics(tr,va,te, model)

                if va_stats[0] > best[0]:
                    best = va_stats
                    best_te = te_stats 

                with open(f'{HOME}/{DATASET}/snapshot-ft_results_{FNAME}_{SIZE}_wl{WALK_LEN}.txt', 'a') as f:
                    f.write(f'{e+1},{updates},{",".join([str(s) for s in te_stats])},{",".join([str(s) for s in va_stats])}\n')

        model.eval()
        te_stats,va_stats = evaluator.get_metrics(tr,va,te, model)

        if va_stats[0] > best[0]:
            best = va_stats
            best_te = te_stats 

        with open(f'{HOME}/{DATASET}/snapshot-ft_results_{FNAME}_{SIZE}_wl{WALK_LEN}.txt', 'a') as f:
            f.write(f'{e+1},{updates},{",".join([str(s) for s in te_stats])},{",".join([str(s) for s in va_stats])}\n')



if __name__ == '__main__':
    arg = ArgumentParser()
    arg.add_argument('--size', default='tiny')
    arg.add_argument('--device', type=int, default=0)
    arg.add_argument('--walk-len', type=int, default=4)
    arg.add_argument('--fb15', action='store_true')
    args = arg.parse_args()
    print(args)

    SIZE = args.size
    DEVICE = args.device if args.device >= 0 else 'cpu'
    WALK_LEN = args.walk_len
    DATASET = 'fb15'
    WORKERS = 1
    EVAL_EVERY = 1000

    HOME = f'results/rw/'

    sample = sample_uni

    edge_features = True

    params = {
        'tiny': SimpleNamespace(H=128, L=2, MINI_BS=1024),
        'mini': SimpleNamespace(H=256, L=4, MINI_BS=1024),
        'med': SimpleNamespace(H=512, L=8, MINI_BS=1024),
        'baseline': SimpleNamespace(H=768, L=12, MINI_BS=512)
    }[SIZE]
    MINI_BS = params.MINI_BS

    print(DATASET)

    sd = torch.load(f'pretrained/rw_sampling/{DATASET}/rw_bert_{DATASET}_{SIZE}-best.pt', weights_only=True)

    FNAME = f'snapshot_bert_static'

    TRWSampler = RW

    tr = torch.load(f'data/{DATASET}_tgraph_tr.pt', weights_only=False)
    tr = TRWSampler(tr, device=DEVICE, walk_len=WALK_LEN, batch_size=MINI_BS, edge_features=edge_features)

    va = torch.load(f'data/{DATASET}_tgraph_va.pt', weights_only=False)
    va = TRWSampler(va, walk_len=WALK_LEN, batch_size=EVAL_BS, edge_features=edge_features)

    te = torch.load(f'data/{DATASET}_tgraph_te.pt', weights_only=False)
    te = TRWSampler(te, walk_len=WALK_LEN, batch_size=EVAL_BS, edge_features=edge_features)

    if DATASET == 'fb15':
        MINI_BS = 256
    
    else:
        print(f"Unrecognized dataset: {DATASET}")

    config = BertConfig(
        tr.num_tokens + GNNEmbedding.OFFSET,
        hidden_size=         params.H,
        num_hidden_layers=   params.L,
        num_attention_heads= params.H // 64,
        intermediate_size=   params.H * 4,
        num_nodes = tr.num_tokens
    )
    model = RWBert(config)
    model.load_state_dict(sd)
    model = model.to(DEVICE)
    #model.bert.requires_grad = False

    evaluator = Evaluator(
        walk_len=WALK_LEN, 
        num_eval_iters=NUM_EVAL_ITERS,
        workers=WORKERS, 
        device=DEVICE, 
        dataset=DATASET
    )

    train(tr,va,te, model)


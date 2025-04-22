from argparse import ArgumentParser
from random import choice, shuffle
import time
from types import SimpleNamespace

import torch
from torch.optim.adamw import AdamW
from torch.optim.lr_scheduler import LRScheduler
from transformers import BertConfig

from eval_trw import Evaluator
from models.gnn_bert import RWBert as BERT, GNNEmbedding
from rw_sampler import RWSampler
from trw_sampler import TRWSampler
from tokenizer import RWTokenizer
from snapshot_finetune import get_metrics

WARMUP_T = 10 ** 8 # Tokens (originally 10**9)
TOTAL_T = 10 ** 9           #(originally 10**10)
FIXED_SMTP_RATE = 0.7

WALK_LEN = 64
BS = 1024
EVAL_BS = 1024
EVAL_EVERY = 1

class Scheduler(LRScheduler):
    def get_lr(self):
        # Warmup period of 10 ** 8 tokens
        if self.last_epoch < WARMUP_T:
            return [group['initial_lr'] * (self.last_epoch / WARMUP_T)
                    for group in self.optimizer.param_groups]
        # Linear decay after that
        else:
            return [group['initial_lr'] * (1 - ((self.last_epoch-WARMUP_T)/(TOTAL_T-WARMUP_T)))
                    for group in self.optimizer.param_groups]

def minibatch(mb, model: BERT):
    walks,masks,targets,attn_mask = t.mask(mb)
    token_count = (walks != GNNEmbedding.PAD).sum()

    loss = model.modified_fwd(walks, masks, targets, attn_mask)
    loss.backward()

    return loss, token_count

def train(g: TRWSampler, model: BERT):
    opt = AdamW(
        model.parameters(), lr=3e-4,
        betas=(0.9, 0.95), eps=1e-8, weight_decay=0.1
    )
    sched = Scheduler(opt)

    with open(f'{OUT_F}log_{SIZE}.txt', 'w+') as f:
        pass

    #model.eval()
    te_auc, te_ap, va_auc, va_ap = evaluator.get_metrics(g,va,te,model)
    with open(f'{OUT_F}eval_{SIZE}.csv', 'w+') as f:
        f.write('e,te_auc,te_ap,va_auc,va_ap\n')
        #f.write(f'0,{te_auc},{te_ap},{va_auc},{va_ap}\n')

    updates = 1
    opt.zero_grad()
    st = time.time()
    steps = 0
    processed_tokens = 0
    best = 0

    e = 0
    while processed_tokens < TOTAL_T:
        shuffle(SNAPSHOTS)
        for snapshot in SNAPSHOTS:
            g.min_ts = snapshot     * DELTA
            g.max_ts = (snapshot+1) * DELTA

            for i,mb in enumerate(g):
                if mb.size(0) == 0:
                    continue
                model.train()
                loss, tokens = minibatch(mb, model)
                steps += 1

                if steps * MINI_BS >= BS:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
                    opt.step()
                    sched.step()

                    processed_tokens += tokens
                    t.set_mask_rate(min(1, (processed_tokens / WARMUP_T)))

                    updates += 1
                    lr = sched.get_last_lr()[0]
                    sched.last_epoch = processed_tokens
                    opt.zero_grad()
                    steps = 0

                    en = time.time()

                    if updates % CHECKPOINT == CHECKPOINT-1:
                        torch.save(
                            (model.state_dict()),
                            f'{OUT_F}_{SIZE}.pt'
                        )

                    # Log update
                    with open(f'{OUT_F}log_{SIZE}.txt', 'a') as f:
                        f.write(f'{loss},{updates},{processed_tokens},{en-st}\n')

                    print(f'[{updates}-{e}] {loss:0.6f} (lr: {lr:0.2e}, mask rate {t.mask_rate:0.4f} tokens: {processed_tokens:0.2e}, seq len: {tokens/MINI_BS:0.2f} {en-st:0.2f}s)')


                st = time.time()


                if processed_tokens >= TOTAL_T:
                    break

        e += 1
        with torch.no_grad():
            if e % EVAL_EVERY == 0: 
                torch.cuda.empty_cache()
                model.eval()
                te_auc, te_ap, va_auc, va_ap = evaluator.get_metrics(g, va, te, model)

                if va_auc > best:
                    torch.save(
                        model.state_dict(),
                        f'{OUT_F}_{SIZE}-best.pt'
                    )
                    best = va_auc

                with open(f'{OUT_F}eval_{SIZE}.csv', 'a') as f:
                    f.write(f'{updates},{te_auc},{te_ap},{va_auc},{va_ap}\n')

    torch.save(
        model.state_dict(),
        f'{OUT_F}_{SIZE}.pt'
    )



if __name__ == '__main__':
    arg = ArgumentParser()
    arg.add_argument('--size', default='tiny')
    arg.add_argument('--device', type=int, default=0)
    arg.add_argument('--optc', action='store_true')
    arg.add_argument('--unsw', action='store_true')
    arg.add_argument('--delta', type=int, default=-1)
    args = arg.parse_args()

    SIZE = args.size
    DEVICE = args.device if args.device >= 0 else 'cpu'
    params = {
        'tiny': SimpleNamespace(H=128, L=2, MINI_BS=1024),
        'mini': SimpleNamespace(H=256, L=4, MINI_BS=512),
        'med': SimpleNamespace(H=512, L=8, MINI_BS=512),
        'baseline': SimpleNamespace(H=768, L=12, MINI_BS=256)
    }[SIZE]

    DATASET = 'optc' if args.optc else 'unsw' if args.unsw else 'lanl'
    MINI_BS = params.MINI_BS
    edge_features = args.unsw #or args.optc

    if DATASET == 'optc': 
        MINI_BS = 1035

    tr = torch.load(f'data/{DATASET}_tgraph_tr.pt', weights_only=False)
    g = TRWSampler(tr, device=DEVICE, walk_len=WALK_LEN, batch_size=MINI_BS, edge_features=edge_features)

    va = torch.load(f'data/{DATASET}_tgraph_va.pt', weights_only=False)
    va = TRWSampler(va, device=DEVICE, walk_len=WALK_LEN, batch_size=EVAL_BS, edge_features=edge_features)
    va.label = torch.zeros_like(va.col)

    te = torch.load(f'data/{DATASET}_tgraph_te.pt', weights_only=False)
    label = te.label
    te = TRWSampler(te, device=DEVICE, walk_len=WALK_LEN, batch_size=EVAL_BS, edge_features=edge_features)
    te.label = label

    CHECKPOINT = 1000
    WORKERS = 1
    EVAL_BS = 512
    DOWNSAMPLE = False 

    if DATASET == 'lanl':
        WARMUP_T = 10 ** 7 # Tokens (originally 10**9)
        TOTAL_T = 10 ** 8           #(originally 10**10)
        DELTA = 60*60*24 # 1 day
        SNAPSHOTS = list(range(59))
        WORKERS = 2

    elif DATASET == 'unsw':
        WARMUP_T = 10 ** 7 # Tokens (originally 10**9)
        TOTAL_T = 10 ** 8           #(originally 10**10)
        DELTA = 0
        SNAPSHOTS = tr.ts.unique().tolist()
        if SIZE == 'tiny':
            g.n_walks = 20 # Get up to about 1024 samples per update
        elif SIZE == 'mini':
            g.n_walks = 10

    elif DATASET == 'optc':
        WARMUP_T = 10 ** 7 # Tokens (originally 10**9)
        TOTAL_T = 10 ** 8
        
        DELTA = 60*60*24 if args.delta == -1 else args.delta
        SNAPSHOTS = (tr.ts // DELTA).unique(sorted=True).tolist()#[:5]
        CHECKPOINT = len(SNAPSHOTS)
        WORKERS = 16
        EVAL_BS = 2048*2
        EVAL_EVERY = 10 

        if SIZE == 'med': 
            EVAL_BS = 2048
        #    WORKERS = 1
        #DOWNSAMPLE = 0.1

    else:
        print(f"Unrecognized dataset: {DATASET}")

    OUT_F = 'trw_bert'

    t = RWTokenizer(g.x)
    t.set_mask_rate(0)

    config = BertConfig(
        g.num_tokens + GNNEmbedding.OFFSET,
        hidden_size=         params.H,
        num_hidden_layers=   params.L,
        num_attention_heads= params.H // 64,
        intermediate_size=   params.H * 4,
        num_nodes = g.num_tokens
    )

    evaluator = Evaluator(
        1,
        dataset=DATASET, device=DEVICE,
        delta=DELTA, workers=WORKERS,
        eval_bs=EVAL_BS, downsample=DOWNSAMPLE
    )
    model = BERT(config).to(DEVICE)
    train(g,model)

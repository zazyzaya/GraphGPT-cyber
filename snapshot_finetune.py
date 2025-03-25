from argparse import ArgumentParser
from copy import deepcopy
import time
from types import SimpleNamespace

from sklearn.metrics import (
    roc_auc_score as auc_score,
    average_precision_score as ap_score
)
import torch
from torch.optim.adamw import AdamW
from torch.optim.lr_scheduler import LRScheduler, CosineAnnealingLR
from transformers import BertConfig
from tqdm import tqdm

from models.gnn_bert import RWBert, GNNEmbedding
from trw_sampler import TRWSampler
from rw_sampler import RWSampler

DEVICE = 0
WARMUP_E = 9.6  # Epochs
EPOCHS = 32    # Epochs

HOME = 'results/rw/'

WHITELIST = 10949 # 'C15244' whitelisted

WALK_LEN = 4
NUM_EVAL_ITERS = 1
MINI_BS = 512
BS = 1024
EVAL_BS = 1024*2
T_MAX = 100_000 # From alibaba source code
DELTA = 60*60 # 1hr

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

def sample(tr, src,dst,ts, walk_len):
    if walk_len > 1:
        rw = tr.rw(src, max_ts=ts, min_ts=(ts-DELTA).clamp(0), reverse=True, trim_missing=False)
    else:
        rw = src.unsqueeze(-1)

    masks = torch.tensor([[GNNEmbedding.MASK]], device=DEVICE).repeat(rw.size(0),1)
    rw = torch.cat([rw,masks], dim=1)
    attn_mask = rw != GNNEmbedding.PAD

    return rw, rw==GNNEmbedding.MASK, dst, attn_mask

@torch.no_grad()
def eval(model, tr: TRWSampler, te: TRWSampler):
    preds = torch.zeros(te.col.size(0))

    prog = tqdm(desc='Eval', total=(va.col.size(0) // EVAL_BS)*NUM_EVAL_ITERS)
    for i in range(NUM_EVAL_ITERS):
        for (src,dst,ts,idx) in te.edge_iter(shuffle=False, return_index=True):
            walk, mask, tgt, attn_mask = sample(tr, src,dst,ts, WALK_LEN)

            out = model.modified_fwd(walk, mask, tgt, attn_mask, return_loss=False).logits
            # Sigmoid on logits to prevent squishing high scores on high-dim vector
            out = 1 - torch.sigmoid(out)

            pred = out[torch.arange(out.size(0)), -1, tgt]
            preds[idx.cpu()] += pred.squeeze().to('cpu')
            prog.update()

            del out
            del pred

    prog.close()

    preds /= NUM_EVAL_ITERS
    labels = te.label

    auc = auc_score(labels, preds)
    ap = ap_score(labels, preds)

    return auc,ap

@torch.no_grad()
def get_metrics(tr,va,te, model):
    te_auc, te_ap = eval(model, tr, te)
    print('#'*20)
    print(f'TEST SCORES')
    print('#'*20)
    print(f"AUC: {te_auc:0.4f}, AP:  {te_ap:0.4f}")

    va_auc, va_ap = validate(model, tr, va)
    print('#'*20)
    print(f'VAL SCORES')
    print('#'*20)
    print(f"AUC: {va_auc:0.4f}, AP:  {va_ap:0.4f}")

    return te_auc, te_ap, va_auc, va_ap

def train(tr,va,te, model: RWBert):
    opt = AdamW(
        model.parameters(), lr=3e-4,
        betas=(0.9, 0.99), eps=1e-10, weight_decay=0.02
    )

    updates_per_epoch = tr.col.size(0) / BS
    warmup_stop = int(updates_per_epoch * WARMUP_E)
    total_steps = int(updates_per_epoch * EPOCHS)

    model.eval()
    best_te = (te_auc, te_ap, va_auc, va_ap) = get_metrics(tr,va,te, model)
    best = va_ap
    sched = Scheduler(opt, warmup_stop, total_steps)

    with open(f'{HOME}/{DATASET}/snapshot-ft_results_{FNAME}_{SIZE}_wl{WALK_LEN}.txt', 'w+') as f:
            f.write(f'epoch,updates,auc,ap,val_auc,val_ap\n')
            f.write(f'0,0,{te_auc},{te_ap},{va_auc},{va_ap}\n')

    updates = 0
    opt.zero_grad()
    st = time.time()
    steps = 0

    e = 0
    for e in range(EPOCHS):
        for (src,dst,ts) in tr.edge_iter():
            model.train()
            walk,mask,tgt,attn_mask = sample(tr, src,dst,ts, WALK_LEN)
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

            if updates and updates % 100 == 0:
                model.eval()
                te_auc, te_ap, va_auc, va_ap = get_metrics(tr,va,te, model)

                if va_ap > best:
                    best = va_ap
                    best_te = (te_auc, te_ap, va_auc, va_ap)

                with open(f'{HOME}/{DATASET}/snapshot-ft_results_{FNAME}_{SIZE}_wl{WALK_LEN}.txt', 'a') as f:
                    f.write(f'{e+1},{updates},{te_auc},{te_ap},{va_auc},{va_ap}\n')

                auc, ap, va_auc, va_ap = best_te
                print('#'*20)
                print(f'BEST SCORES')
                print('#'*20)
                print(f"VAL:  AUC: {va_auc:0.4f}, AP:  {va_ap:0.4f}")
                print(f"TEST: AUC: {auc:0.4f}, AP:  {ap:0.4f}")

@torch.no_grad()
def validate(model, tr,va, percent=0.01):
    tns = torch.zeros(va.col.size(0))
    tps = torch.zeros(int(va.col.size(0) * percent))

    prog = tqdm(desc='TNs', total=(va.col.size(0) // EVAL_BS)*NUM_EVAL_ITERS)
    for i in range(NUM_EVAL_ITERS):
        for (src,dst,ts,idx) in va.edge_iter(shuffle=False, return_index=True):
            walk, mask, tgt, attn_mask = sample(tr, src,dst,ts, WALK_LEN)

            out = model.modified_fwd(walk, mask, tgt, attn_mask, return_loss=False).logits
            # Sigmoid on logits to prevent squishing high scores on high-dim vector
            out = 1 - torch.sigmoid(out)

            pred = out[torch.arange(out.size(0)), -1, tgt]
            tns[idx.cpu()] += pred.squeeze().to('cpu')
            prog.update()

    prog.close()

    prog = tqdm(desc='TPs', total=(tps.size(0) // EVAL_BS)*NUM_EVAL_ITERS)
    src = torch.randint_like(tr.col, 0, tr.num_nodes)[:tps.size(0)]
    dst = torch.randint_like(tr.col, 0, tr.num_nodes)[:tps.size(0)]
    ts = torch.randint_like(tr.col, 0, tr.ts.max())[:tps.size(0)]
    for i in range(NUM_EVAL_ITERS):
        batches = torch.arange(src.size(0)).split(EVAL_BS)
        for b in batches:
            walk, mask, tgt, attn_mask = sample(tr, src[b],dst[b],ts[b], WALK_LEN)

            out = model.modified_fwd(walk, mask, tgt, attn_mask, return_loss=False).logits
            # Sigmoid on logits to prevent squishing high scores on high-dim vector
            out = 1 - torch.sigmoid(out)

            pred = out[torch.arange(out.size(0)), -1, tgt]
            tps[b] += pred.squeeze().to('cpu')
            prog.update()

    prog.close()

    tps /= NUM_EVAL_ITERS
    tns /= NUM_EVAL_ITERS

    preds = torch.cat([tps, tns])
    labels = torch.zeros(preds.size())
    labels[:tps.size(0)] = 1

    auc = auc_score(labels, preds)
    ap = ap_score(labels, preds)

    return auc,ap



if __name__ == '__main__':
    arg = ArgumentParser()
    arg.add_argument('--size', default='tiny')
    arg.add_argument('--device', type=int, default=0)
    arg.add_argument('--walk-len', type=int, default=4)
    arg.add_argument('--optc', action='store_true')
    args = arg.parse_args()

    SIZE = args.size
    DEVICE = args.device if args.device >= 0 else 'cpu'
    WALK_LEN = args.walk_len
    DATASET = 'lanl' if not args.optc else 'optc'

    params = {
        'tiny': SimpleNamespace(H=128, L=2, MINI_BS=1024),
        'mini': SimpleNamespace(H=256, L=4, MINI_BS=1024),
        'med': SimpleNamespace(H=512, L=8, MINI_BS=512),
        'baseline': SimpleNamespace(H=768, L=12, MINI_BS=512)
    }[SIZE]
    MINI_BS = params.MINI_BS


    tr = torch.load(f'data/{DATASET}_tgraph_tr.pt', weights_only=False)
    tr = TRWSampler(tr, device=DEVICE, walk_len=WALK_LEN, batch_size=MINI_BS)
    sd = torch.load(f'pretrained/snapshot_rw/{DATASET}/trw_bert_{SIZE}.pt', weights_only=True)
    FNAME = 'snapshot_bert'

    if DATASET == 'lanl':
        va = torch.load('data/lanl_tgraph_va.pt', weights_only=False)
        va = TRWSampler(va, device=DEVICE, walk_len=WALK_LEN, batch_size=EVAL_BS)
        va.label = torch.zeros_like(va.col)
        te = torch.load('data/lanl_tgraph_te.pt', weights_only=False)
        label = te.label
        te = TRWSampler(te, device=DEVICE, walk_len=WALK_LEN, batch_size=EVAL_BS)
        te.label = label
    else:
        va = torch.load('data/optc_tgraph_va.pt', weights_only=False)
        te = torch.load('data/optc_tgraph_te.pt', weights_only=False)

    config = BertConfig(
        tr.x.size(0) + GNNEmbedding.OFFSET,
        hidden_size=         params.H,
        num_hidden_layers=   params.L,
        num_attention_heads= params.H // 64,
        intermediate_size=   params.H * 4,
        num_nodes = tr.x.size(0)
    )
    model = RWBert(config)
    model.load_state_dict(sd)
    model = model.to(DEVICE)

    train(tr,va,te, model)


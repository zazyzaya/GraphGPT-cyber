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
        rw = tr.rw(src, max_ts=ts, min_ts=(ts-DELTA).clamp(0), reverse=True, trim_missing=False)
    else:
        rw = src.unsqueeze(-1)

    if edge_features is not None:
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
        src_rw = tr.rw(src, max_ts=ts, min_ts=(ts-DELTA).clamp(0), reverse=True, trim_missing=False)
        dst_rw = tr.rw(dst, max_ts=(ts+DELTA), min_ts=ts, reverse=False, trim_missing=False)

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

@torch.no_grad()
def parallel_eval(model, tr, te, workers=1):
    preds = np.zeros(te.col.size(0))
    prog = tqdm(desc='Eval', total=(te.col.size(0) // EVAL_BS)*NUM_EVAL_ITERS)

    def thread_job(pid, idx):
        nonlocal preds

        # Try to spread the jobs use of the GPU evenly
        if pid < workers:
            time.sleep(0.1 * pid)

        samp = te._single_iter(idx, shuffled=False)
        if te.edge_features:
            src,dst,ts,ef = samp
        else:
            src,dst,ts = samp
            ef = None

        walk, mask, tgt, attn_mask = sample(tr, src,dst,ts, WALK_LEN, edge_features=ef)
        out = model.modified_fwd(walk, mask, tgt, attn_mask, return_loss=False).logits

        # Sigmoid on logits to prevent squishing high scores on high-dim vector
        out = 1 - torch.sigmoid(out)
        out = out[mask]
        pred = out[torch.arange(out.size(0)), tgt]

        # Use prob of edge features as part of probability
        if ef is not None:
            pred = pred.view(walk.size(0), -1)
            pred = pred.prod(dim=1)

        preds[idx.cpu()] += pred.squeeze().detach().to('cpu').numpy()
        prog.update()

        del out
        del pred
        torch.cuda.empty_cache()

    for i in range(NUM_EVAL_ITERS):
        Parallel(n_jobs=workers, prefer='threads')(
            delayed(thread_job)(i,b)
            for i,b in enumerate(torch.arange(te.col.size(0)).split(EVAL_BS))
        )


    prog.close()

    preds /= NUM_EVAL_ITERS
    labels = te.label.numpy()

    auc = fast_auc(labels, preds)
    ap = fast_ap(labels, preds)

    return auc,ap

@torch.no_grad()
def parallel_validate(model, tr, va, workers=16, percent=1):
    tns = np.zeros(va.col.size(0))
    tps = np.zeros(int(va.col.size(0) * percent))

    prog = tqdm(desc='TNs', total=(va.col.size(0) // EVAL_BS)*NUM_EVAL_ITERS)

    def thread_job_tn(pid, idx):
        # Try to spread the jobs use of the GPU evenly
        if pid < workers:
            time.sleep(0.1 * pid)

        samp = va._single_iter(idx, shuffled=False)
        if va.edge_features:
            src,dst,ts,ef = samp
        else:
            src,dst,ts = samp
            ef = None

        walk, mask, tgt, attn_mask = sample(tr, src,dst,ts, WALK_LEN, edge_features=ef)
        out = model.modified_fwd(walk, mask, tgt, attn_mask, return_loss=False).logits

        # Sigmoid on logits to prevent squishing high scores on high-dim vector
        out = 1 - torch.sigmoid(out)
        out = out[mask]
        pred = out[torch.arange(out.size(0)), tgt]

        # Use prob of edge features as part of probability
        if ef is not None:
            pred = pred.view(walk.size(0), -1)
            pred = pred.prod(dim=1)

        tns[idx.cpu()] += pred.squeeze().detach().to('cpu').numpy()
        prog.update()

        del out
        del pred
        torch.cuda.empty_cache()

    for i in range(NUM_EVAL_ITERS):
        Parallel(n_jobs=workers, prefer='threads')(
            delayed(thread_job_tn)(i,b)
            for i,b in enumerate(torch.arange(va.col.size(0)).split(EVAL_BS))
        )

    prog.close()

    prog = tqdm(desc='TPs', total=(tps.shape[0] // EVAL_BS)*NUM_EVAL_ITERS)
    def thread_job_tp(pid, src,dst,ts,ef,b):
        # Try to spread the jobs use of the GPU evenly
        if pid < workers:
            time.sleep(0.1 * pid)

        walk, mask, tgt, attn_mask = sample(tr, src,dst,ts, WALK_LEN, edge_features=ef)
        out = model.modified_fwd(walk, mask, tgt, attn_mask, return_loss=False).logits

        # Sigmoid on logits to prevent squishing high scores on high-dim vector
        out = 1 - torch.sigmoid(out)
        out = out[mask]
        pred = out[torch.arange(out.size(0)), tgt]

        # Use prob of edge features as part of probability
        if ef is not None:
            pred = pred.view(walk.size(0), -1)
            pred = pred.prod(dim=1)

        tps[b.cpu()] += pred.squeeze().detach().to('cpu').numpy()
        prog.update()

        del out
        del pred
        torch.cuda.empty_cache()


    src = torch.randint_like(va.col, tr.num_nodes)[:tps.shape[0]]
    dst = torch.randint_like(va.col, tr.num_nodes)[:tps.shape[0]]
    ts = torch.randint_like(va.col, tr.ts.max())[:tps.shape[0]]

    if DATASET == 'unsw':
        # Edges only have very specific timecodes
        ts = tr.ts.unique()
        idx = torch.randint_like(src, ts.size(0))
        ts = ts[idx]

    batches = torch.arange(src.size(0)).split(EVAL_BS)

    for i in range(NUM_EVAL_ITERS):
        if not va.edge_features:
            Parallel(n_jobs=workers, prefer='threads')(
                delayed(thread_job_tp)(i,src[b],dst[b],ts[b],None, b)
                for i,b in enumerate(batches)
            )
        else:
            efs = torch.randint(0, tr.edge_attr.max()+1, (src.size(0), 1), device=src.device)
            efs += tr.num_nodes
            Parallel(n_jobs=workers, prefer='threads')(
                delayed(thread_job_tp)(i,src[b],dst[b],ts[b],efs[b], b)
                for i,b in enumerate(batches)
            )


    prog.close()

    tps /= NUM_EVAL_ITERS
    tns /= NUM_EVAL_ITERS

    preds = np.concatenate([tps, tns])
    labels = np.zeros(preds.shape[0])
    labels[:tps.shape[0]] = 1

    auc = fast_auc(labels, preds)
    ap = fast_ap(labels, preds)

    return auc,ap


@torch.no_grad()
def get_metrics(tr,va,te, model):
    te_auc, te_ap = parallel_eval(model, tr, te, workers=WORKERS)
    print('#'*20)
    print(f'TEST SCORES')
    print('#'*20)
    print(f"AUC: {te_auc:0.4f}, AP:  {te_ap:0.4f}")
    print('#'*20)
    print()

    va_auc, va_ap = parallel_validate(model, tr, va, workers=WORKERS)
    print('#'*20)
    print(f'VAL SCORES')
    print('#'*20)
    print(f"AUC: {va_auc:0.4f}, AP:  {va_ap:0.4f}")

    return te_auc, te_ap, va_auc, va_ap

def train(tr,va,te, model: RWBert):
    opt = AdamW(
        model.parameters(), lr=LR,
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



if __name__ == '__main__':
    arg = ArgumentParser()
    arg.add_argument('--size', default='tiny')
    arg.add_argument('--device', type=int, default=0)
    arg.add_argument('--walk-len', type=int, default=4)
    arg.add_argument('--optc', action='store_true')
    arg.add_argument('--unsw', action='store_true')
    arg.add_argument('--bi', action='store_true')
    arg.add_argument('--static', action='store_true')
    args = arg.parse_args()
    print(args)

    SIZE = args.size
    DEVICE = args.device if args.device >= 0 else 'cpu'
    WALK_LEN = args.walk_len
    DATASET = 'optc' if args.optc else 'unsw' if args.unsw else 'lanl'
    WORKERS = 16
    EVAL_EVERY = 1000 
    
    HOME = f'results/{"rw" if args.static else "trw"}/'

    sample = sample_bi if args.bi else sample_uni 

    edge_features = args.unsw #or args.optc

    params = {
        'tiny': SimpleNamespace(H=128, L=2, MINI_BS=1024),
        'mini': SimpleNamespace(H=256, L=4, MINI_BS=1024),
        'med': SimpleNamespace(H=512, L=8, MINI_BS=1024),
        'baseline': SimpleNamespace(H=768, L=12, MINI_BS=512)
    }[SIZE]
    MINI_BS = params.MINI_BS

    print(DATASET)

    if not args.static: 
        sd = torch.load(f'pretrained/snapshot_rw/{DATASET}/trw_bert_{SIZE}-best.pt', weights_only=True)
    else: 
        sd = torch.load(f'pretrained/rw_sampling/{DATASET}/rw_bert_{DATASET}_{SIZE}-best.pt', weights_only=True)

    FNAME = f'{"bi_" if args.bi else ""}snapshot_bert{"_static" if args.static else ""}'

    TRWSampler = RW if args.static else TRW

    tr = torch.load(f'data/{DATASET}_tgraph_tr.pt', weights_only=False)
    tr = TRWSampler(tr, device=DEVICE, walk_len=WALK_LEN, batch_size=MINI_BS, edge_features=edge_features)

    va = torch.load(f'data/{DATASET}_tgraph_va.pt', weights_only=False)
    va = TRWSampler(va, device=DEVICE, walk_len=WALK_LEN, batch_size=EVAL_BS, edge_features=edge_features)
    va.label = torch.zeros_like(va.col)

    te = torch.load(f'data/{DATASET}_tgraph_te.pt', weights_only=False)
    label = te.label
    te = TRWSampler(te, device=DEVICE, walk_len=WALK_LEN, batch_size=EVAL_BS, edge_features=edge_features)
    te.label = label

    if DATASET == 'lanl':
        DELTA = 60*60*24 # 1 day
        SNAPSHOTS = list(range(59))

    elif DATASET == 'unsw':
        WORKERS = 8
        DELTA = 0
        SNAPSHOTS = tr.ts.unique().tolist()
        EVAL_EVERY = 500

    elif DATASET == 'optc':
        DELTA = 60*60*24
        SNAPSHOTS = (tr.ts // DELTA).unique().tolist()[:5]
        WORKERS = 1
        EVAL_BS = 2048*2

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

    train(tr,va,te, model)


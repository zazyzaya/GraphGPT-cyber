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

from models.hugging_bert import GraphBertFT
from sampler import SparseGraphSampler
from tokenizer import Tokenizer

DEVICE = 0
WARMUP_E = 9.6  # Epochs
EPOCHS = 32    # Epochs

MINI_BS = 512
BS = 1024 // 2 # Adding equal number of neg samples
EVAL_BS = 1024
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
        # Cosine decay after that
        # Linear decay after that
        else:
            return [group['initial_lr'] * (1 - ((self.last_epoch-self.warmup_stop)/(self.total_steps-self.warmup_stop)))
                    for group in self.optimizer.param_groups]

def simple_sample(x,edges):
    walks = (x[edges] + t.feat_offset).reshape(edges.size(0), -1).long()
    walks = torch.cat([walks, torch.full((walks.size(0),1), t.MASK)], dim=1)
    masks = torch.zeros(walks.size(), dtype=torch.bool)
    masks[:,-1] = True

    return walks,masks

@torch.no_grad()
def simple_eval(model, tr, to_eval):
    can_eval = to_eval.edge_index < tr.x.size(0) # Some new nodes we don't know what to do with
    can_eval = can_eval.prod(dim=0).nonzero().squeeze()
    idxs = can_eval.split(EVAL_BS)

    preds = torch.zeros(to_eval.edge_index.size(1))

    for idx in tqdm(idxs, desc='Evaluating'):
        edges = to_eval.edge_index[:, idx].T
        walks,masks = simple_sample(tr.x, edges)
        pred = model.predict(walks, masks).to('cpu')
        preds[idx] = pred.squeeze()

    labels = to_eval.label
    weights = to_eval.edge_attr

    auc = auc_score(
        labels, preds, sample_weight=weights
    )
    ap = ap_score(
        labels, preds, sample_weight=weights
    )

    return auc,ap

@torch.no_grad()
def evaluate(model, tr, to_eval):
    can_eval = to_eval.edge_index < tr.x.size(0) # Some new nodes we don't know what to do with
    can_eval = can_eval.prod(dim=0).nonzero().squeeze()
    idxs = can_eval.split(EVAL_BS)

    preds = torch.zeros(to_eval.edge_index.size(1))
    for idx in tqdm(idxs, desc='Evaluating'):
        edges = to_eval.edge_index[:, idx]
        data = tr.sample(edges.T)
        walks,masks = t.lp_tokenize(data)

        pred = model.predict(walks, masks).to('cpu')
        preds[idx] = pred.squeeze()

    labels = to_eval.label
    weights = to_eval.edge_attr

    auc = auc_score(
        labels, preds, sample_weight=weights
    )
    ap = ap_score(
        labels, preds, sample_weight=weights
    )

    return auc,ap


def simple_train(tr,va,te, model: GraphBertFT):
    opt = AdamW(
        model.parameters(), lr=3e-4,
        betas=(0.9, 0.99), eps=1e-10, weight_decay=0.02
    )

    updates_per_epoch = len(tr) / BS
    warmup_stop = int(updates_per_epoch * WARMUP_E)
    total_steps = int(updates_per_epoch * EPOCHS)

    print(updates_per_epoch)

    sched = Scheduler(opt, warmup_stop, total_steps)

    with open('ft_results.txt', 'w+') as f:
            f.write(f'epoch,auc,ap\n')

    updates = 0
    opt.zero_grad()
    st = time.time()
    steps = 0

    e = 0
    best = 0
    best_te = None
    for e in range(EPOCHS):
        ei = tr.data.edge_index[:, torch.randperm(tr.data.edge_index.size(1))]
        for pos in ei.split(MINI_BS, dim=1):
            pos = pos.T

            # Negative sample non-edges
            neg = torch.randint(0, tr.x.size(0), (pos.size(0),2))
            labels = torch.zeros(pos.size(0)+neg.size(0), 1)
            labels[pos.size(0):] = 1

            edges = torch.cat([pos,neg])
            walks,masks = simple_sample(tr.x, edges)
            loss = model.forward(walks, masks, labels)
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


        add_fake_data(va)
        auc, ap = simple_eval(model, tr, va)
        print('#'*20)
        print(f'VAL SCORES')
        print('#'*20)
        print(f"AUC: {auc:0.4f}, AP:  {ap:0.4f}")

        store_best = False
        if ap > best:
            best = ap
            store_best = True

        va_auc = auc
        va_ap = ap

        auc, ap = simple_eval(model, tr, te)
        print('#'*20)
        print(f'TEST SCORES')
        print('#'*20)
        print(f"AUC: {auc:0.4f}, AP:  {ap:0.4f}")

        if store_best:
            best_te = (auc, ap, va_auc, va_ap)

        with open('ft_results.txt', 'a') as f:
            f.write(f'{e+1},{auc},{ap},{va_auc},{va_ap}\n')

        torch.save(
            model.state_dict(),
            'finetune.pt'
        )

        auc, ap, va_auc, va_ap = best_te
        print('#'*20)
        print(f'BEST SCORES')
        print('#'*20)
        print(f"VAL:  AUC: {va_auc:0.4f}, AP:  {va_ap:0.4f}")
        print(f"TEST: AUC: {auc:0.4f}, AP:  {ap:0.4f}")

def train(tr,va,te, model: GraphBertFT):
    opt = AdamW(
        model.parameters(), lr=1e-3,
        betas=(0.9, 0.99), eps=1e-10, weight_decay=0.02
    )

    updates_per_epoch = len(tr) / BS
    warmup_stop = int(updates_per_epoch * WARMUP_E)
    total_steps = int(updates_per_epoch * EPOCHS)

    print(updates_per_epoch)

    sched = Scheduler(opt, warmup_stop, total_steps)

    with open(f'ft_results_{SIZE}.txt', 'w+') as f:
            f.write(f'epoch,auc,ap\n')

    updates = 0
    opt.zero_grad()
    st = time.time()
    steps = 0

    e = 0
    best = 0
    best_te = None
    for e in range(EPOCHS):
        for pos_mb in tr:
            # Negative sample non-edges
            neg_mb = torch.randint(0, tr.x.size(0), (len(pos_mb),2))
            labels = torch.zeros(len(pos_mb)+neg_mb.size(0), 1)
            labels[len(pos_mb):] = 1
            mb = pos_mb + tr.sample(neg_mb)

            walks,masks = t.lp_tokenize(mb)
            loss = model.forward(walks, masks, labels)
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



        add_fake_data(va)
        auc, ap = evaluate(model, tr, va)
        print('#'*20)
        print(f"VAL: AUC: {auc:0.4f}, AP: {ap:0.4f}")

        store_best = False
        if ap > best:
            best = ap
            store_best = True

        val_auc = auc
        val_ap = ap

        auc, ap = evaluate(model, tr, te)
        print('#'*20)
        print(f"TEST: AUC: {auc:0.4f}, AP: {ap:0.4f}")

        if store_best:
            best_te = (auc, ap, val_auc, val_ap)

        with open(f'ft_results_{SIZE}.txt', 'a') as f:
            f.write(f'{e+1},{auc},{ap},{val_auc},{val_ap},{loss}\n')

        torch.save(
            model.state_dict(),
            'finetune.pt'
        )

        auc, ap, val_auc,val_ap = best_te
        print('#'*20)
        print(f'BEST SCORES')
        print('#'*20)
        print(f"VAL: AUC: {val_auc:0.4f}, AP: {val_ap:0.4f}")
        print(f"TEST: AUC: {auc:0.4f}, AP: {ap:0.4f}")


def add_fake_data(data, percent=1):
    real = data.label == 0
    to_fake = int(real.sum() * percent)
    neg = torch.randint(0, data.x.size(0), (2,to_fake))

    data.edge_index = torch.cat([
        data.edge_index[:, real], neg
    ], dim=1)
    data.edge_attr = torch.cat([
        data.edge_attr[real], torch.ones(neg.size(1))
    ])
    data.label = torch.ones(data.edge_attr.size())
    data.label[:real.sum()] = 0



if __name__ == '__main__':
    tr = torch.load('data/lanl_tr.pt', weights_only=False)
    va = torch.load('data/lanl_va.pt', weights_only=False)
    te = torch.load('data/lanl_te.pt', weights_only=False)

    idx = torch.randperm(tr.edge_index.size(1))
    n_tr = int(idx.size(0) * 0.9)

    va.label = torch.zeros(va.edge_attr.size())
    add_fake_data(va)

    arg = ArgumentParser()
    arg.add_argument('--size', default='NO INPUT')
    arg.add_argument('--device', type=int, default=0)
    args = arg.parse_args()

    SIZE = args.size
    DEVICE = args.device

    params = {
        'tiny': SimpleNamespace(H=128, L=2, MINI_BS=512),
        'mini': SimpleNamespace(H=256, L=4, MINI_BS=512),
        'med': SimpleNamespace(H=512, L=8, MINI_BS=512)
    }[SIZE]
    MINI_BS = params.MINI_BS

    tr = SparseGraphSampler(tr, neighbors=15, batch_size=MINI_BS, mode='finetune')
    t = Tokenizer(tr.x)

    config = BertConfig(
        t.vocab_size,
        hidden_size=         params.H,
        num_hidden_layers=   params.L,
        num_attention_heads= params.H // 64,
        intermediate_size=   params.H * 4,
        padding_idx = t.PAD
    )

    model = GraphBertFT(config, f'pretrained/bert_{SIZE}.pt', device=DEVICE)
    train(tr,va,te,model)

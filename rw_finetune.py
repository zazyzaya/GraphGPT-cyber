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

def sample(tr, batch, walk_len):
    starts = batch[0]

    if walk_len > 1:
        rw = tr.rw(starts, walk_len, reverse=True)
    else:
        rw = starts.unsqueeze(-1)

    masks = torch.tensor([[GNNEmbedding.MASK]], device=DEVICE).repeat(rw.size(0),1)
    rw = torch.cat([rw,masks], dim=1)
    attn_mask = rw != GNNEmbedding.PAD

    return rw, rw==GNNEmbedding.MASK, batch[1], attn_mask

@torch.no_grad()
def eval(model, tr, to_eval):
    preds = torch.zeros(to_eval.edge_index.size(1))

    prog = tqdm(desc='Evaluating', total=(to_eval.edge_index.size(1) // EVAL_BS)*NUM_EVAL_ITERS)
    for i in range(NUM_EVAL_ITERS):
        idxs = torch.arange(to_eval.edge_index.size(1)).split(EVAL_BS)

        for idx in idxs:
            walk, mask, tgt, attn_mask = sample(tr, to_eval.edge_index[:, idx].to(DEVICE), WALK_LEN)

            out = model.modified_fwd(walk, mask, tgt, attn_mask, return_loss=False).logits
            # Sigmoid on logits to prevent squishing high scores on high-dim vector
            out = 1 - torch.sigmoid(out)

            pred = out[torch.arange(out.size(0)), -1, tgt]
            preds[idx] += pred.squeeze().to('cpu')
            prog.update()

    prog.close()

    preds /= NUM_EVAL_ITERS

    wl = (to_eval.edge_index == WHITELIST).sum(dim=0, dtype=torch.bool)
    preds[wl] = 0

    labels = to_eval.label
    weights = to_eval.edge_attr

    auc = auc_score(
        labels, preds, sample_weight=weights
    )
    ap = ap_score(
        labels, preds, sample_weight=weights
    )

    return auc,ap


def train(tr,va,te, model: RWBert):
    opt = AdamW(
        model.parameters(), lr=3e-4,
        betas=(0.9, 0.99), eps=1e-10, weight_decay=0.02
    )

    updates_per_epoch = tr.edge_index.size(1) / BS
    warmup_stop = int(updates_per_epoch * WARMUP_E)
    total_steps = int(updates_per_epoch * EPOCHS)

    print(updates_per_epoch)

    sched = Scheduler(opt, warmup_stop, total_steps)

    with open(f'{HOME}/ft_results_{FNAME}_{SIZE}_wl{WALK_LEN}.txt', 'w+') as f:
            f.write(f'epoch,auc,ap\n')

    updates = 0
    opt.zero_grad()
    st = time.time()
    steps = 0

    e = 0
    best = 0
    best_te = None
    for e in range(EPOCHS):
        idxs = torch.randperm(tr.edge_index.size(1)).split(MINI_BS)
        for idx in idxs:
            model.train()
            walk,mask,tgt,attn_mask = sample(tr, tr.edge_index[:,idx], WALK_LEN)
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

        model.eval()
        add_fake_data(va)
        auc, ap = eval(model, tr, va)
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

        auc, ap = eval(model, tr, te)
        print('#'*20)
        print(f'TEST SCORES')
        print('#'*20)
        print(f"AUC: {auc:0.4f}, AP:  {ap:0.4f}")

        if store_best:
            best_te = (auc, ap, va_auc, va_ap)

        with open(f'{HOME}/ft_results_{FNAME}_{SIZE}_wl{WALK_LEN}.txt', 'a') as f:
            f.write(f'{e+1},{auc},{ap},{va_auc},{va_ap}\n')

        auc, ap, va_auc, va_ap = best_te
        print('#'*20)
        print(f'BEST SCORES')
        print('#'*20)
        print(f"VAL:  AUC: {va_auc:0.4f}, AP:  {va_ap:0.4f}")
        print(f"TEST: AUC: {auc:0.4f}, AP:  {ap:0.4f}")


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
    va = torch.load('data/lanl_sgraph_va.pt', weights_only=False)
    te = torch.load('data/lanl_sgraph_te.pt', weights_only=False)

    va.label = torch.zeros(va.edge_attr.size())
    add_fake_data(va)

    arg = ArgumentParser()
    arg.add_argument('--size', default='NO INPUT')
    arg.add_argument('--device', type=int, default=0)
    arg.add_argument('--walk-len', type=int, default=4)
    arg.add_argument('--temporal', action='store_true')
    args = arg.parse_args()

    SIZE = args.size
    DEVICE = args.device
    WALK_LEN = args.walk_len

    params = {
        'tiny': SimpleNamespace(H=128, L=2, MINI_BS=512),
        'mini': SimpleNamespace(H=256, L=4, MINI_BS=512),
        'med': SimpleNamespace(H=512, L=8, MINI_BS=512)
    }[SIZE]
    MINI_BS = params.MINI_BS

    if args.temporal:
        tr = torch.load('data/lanl_tgraph_tr.pt', weights_only=False)
        tr = TRWSampler(tr, device=DEVICE, n_walks=1, walk_len=WALK_LEN)
        tr.add_edge_index()
        sd = torch.load(f'trw_bert_{SIZE}.pt', weights_only=True)
        FNAME = 'trw_bert'

    else:
        tr = torch.load('data/lanl_sgraph_tr.pt', weights_only=False)
        tr = RWSampler(tr, device=DEVICE, n_walks=1, walk_len=WALK_LEN)
        sd = torch.load(f'rw_bert_{SIZE}.pt', weights_only=True)
        FNAME = 'rw_bert'

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


from copy import deepcopy
import time

from sklearn.metrics import (
    roc_auc_score as auc_score,
    average_precision_score as ap_score
)
import torch
from torch.optim.adamw import AdamW
from torch.optim.lr_scheduler import LRScheduler, CosineAnnealingLR
from tqdm import tqdm

from models.ft_bert import FT_BERT
from sampler import SparseGraphSampler
from tokenizer import Tokenizer

DEVICE = 3
WARMUP_E = 9.6  # Epochs
EPOCHS = 32    # Epochs

MINI_BS = 128
BS = 1024
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



@torch.no_grad()
def evaluate(model, tr, to_eval):
    can_eval = to_eval.edge_index < tr.x.size(0) # Some new nodes we don't know what to do with
    can_eval = can_eval.prod(dim=0).nonzero().squeeze()
    idxs = can_eval.split(EVAL_BS)

    preds = torch.zeros(to_eval.edge_index.size(1))
    preds_one = torch.ones(to_eval.edge_index.size(1))

    for idx in tqdm(idxs, desc='Evaluating'):
        edges = to_eval.edge_index[:, idx]
        #data = tr.sample(edges.T)
        #walks,masks = t.lp_tokenize(data)
        walks = t.simple_lp_tokenize(tr.x, edges)

        pred = model.predict(walks).to('cpu')
        preds[idx] = pred.squeeze()
        preds_one[idx] = pred.squeeze()

    labels = to_eval.label
    weights = to_eval.edge_attr

    auc = auc_score(
        labels, preds, sample_weight=weights
    )
    ap = ap_score(
        labels, preds, sample_weight=weights
    )
    auc_sus = auc_score(
        labels, preds_one, sample_weight=weights
    )
    ap_sus = ap_score(
        labels, preds_one, sample_weight=weights
    )
    auc_trunc = auc_score(
        labels[can_eval], preds[can_eval], sample_weight=weights[can_eval]
    )
    ap_trunc = ap_score(
        labels[can_eval], preds[can_eval], sample_weight=weights[can_eval]
    )

    return auc,ap, auc_trunc,ap_trunc, auc_sus,ap_sus

def simple_train(tr,va,te, model: FT_BERT):
    opt = AdamW(
        model.parameters(), lr=1e-3,
        betas=(0.9, 0.99), eps=1e-10, weight_decay=0.02
    )

    updates_per_epoch = len(tr) / BS
    warmup_stop = int(updates_per_epoch * WARMUP_E)
    total_steps = int(updates_per_epoch * EPOCHS)

    print(updates_per_epoch)

    sched = Scheduler(opt, warmup_stop, total_steps)

    with open('ft_log.txt', 'w+') as f:
        pass

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
        for pos_mb in ei.split(BS, dim=1):
            # Negative sample non-edges
            neg_mb = torch.randint(0, tr.x.size(0), (2, BS))
            labels = torch.zeros(pos_mb.size(1)+neg_mb.size(1), 1)
            labels[len(pos_mb):] = 1
            mb = torch.cat([pos_mb, neg_mb], dim=1)

            st = time.time()
            opt.zero_grad()
            walks = t.simple_lp_tokenize(tr.x, mb)
            loss = model.forward(walks, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            opt.step()
            sched.step()
            en = time.time()

            # Log epoch
            with open('ft_log.txt', 'a') as f:
                f.write(f'{loss},{updates}\n')

            lr = sched.get_last_lr()[0]
            print(f'[{updates}-{e}] {loss} (lr: {lr:0.2e}, {en-st:0.2f}s)')

            updates += 1
            steps = 0

            if updates % 100 == 0:
                torch.save(
                        model.state_dict(),
                    'finetune.pt'
                )

        add_fake_data(va)
        auc, ap, auc_trunc, ap_trunc, auc_sus, ap_sus = evaluate(model, tr, va)
        print('#'*20)
        print(f'VAL SCORES')
        print('#'*20)
        print(f"AUC (full dataset):     {auc:0.4f}, AP: {ap:0.4f}")
        print(f"AUC (ignore missing):   {auc_trunc:0.4f}, AP: {ap_trunc:0.4f}")
        print(f"AUC (missing are anom): {auc_sus:0.4f}, AP: {ap_sus:0.4f}")

        store_best = False
        if ap > best:
            best = ap
            store_best = True

        auc, ap, auc_trunc, ap_trunc, auc_sus, ap_sus = evaluate(model, tr, te)
        print('#'*20)
        print(f'TEST SCORES')
        print('#'*20)
        print(f"AUC (full dataset):     {auc:0.4f}, AP: {ap:0.4f}")
        print(f"AUC (ignore missing):   {auc_trunc:0.4f}, AP: {ap_trunc:0.4f}")
        print(f"AUC (missing are anom): {auc_sus:0.4f}, AP: {ap_sus:0.4f}")

        if store_best:
            best_te = (auc, ap, auc_trunc, ap_trunc, auc_sus, ap_sus)

        with open('ft_results.txt', 'a') as f:
            f.write(f'{e+1},{auc},{ap},{auc_trunc},{ap_trunc},{auc_sus},{ap_sus},{loss}\n')

        torch.save(
            model.state_dict(),
            'finetune.pt'
        )

        auc, ap, auc_trunc, ap_trunc, auc_sus, ap_sus = best_te
        print('#'*20)
        print(f'BEST SCORES')
        print('#'*20)
        print(f"AUC (full dataset):     {auc:0.4f}, AP: {ap:0.4f}")
        print(f"AUC (ignore missing):   {auc_trunc:0.4f}, AP: {ap_trunc:0.4f}")
        print(f"AUC (missing are anom): {auc_sus:0.4f}, AP: {ap_sus:0.4f}")


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
    te = torch.load('data/lanl_te.pt', weights_only=False)
    va = deepcopy(tr)

    idx = torch.randperm(tr.edge_index.size(1))
    n_tr = int(idx.size(0) * 0.9)

    # Partition training data into train and val
    tr.edge_index = tr.edge_index[:, idx[:n_tr]]
    tr.edge_attr = tr.edge_attr[idx[:n_tr]]
    va.edge_index = va.edge_index[:, idx[n_tr:]]
    va.edge_attr = va.edge_attr[idx[n_tr:]]
    va.label = torch.zeros(va.edge_attr.size())
    add_fake_data(va)

    tr = SparseGraphSampler(tr, batch_size=MINI_BS, mode='finetune')

    num_tokens = tr.x.max().long() + 3 + 1
    t = Tokenizer(num_tokens, 3)

    model = FT_BERT('bert.pt', device=DEVICE)
    simple_train(tr,va,te,model)

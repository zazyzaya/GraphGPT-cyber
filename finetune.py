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

from models.ft_gpt import GraphGPT
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
        else:
            return self.cosine.get_lr()

def minibatch(mb, model: GraphGPT, labels):
    walks,masks = t.lp_tokenize(mb)
    loss = model(walks, masks, labels)
    loss.backward()

    return loss

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

        pred = model.predict(walks,masks).to('cpu')
        preds[idx] = pred.squeeze()

    labels = to_eval.label
    weights = to_eval.edge_attr

    auc = auc_score(
        labels, preds, sample_weight=weights
    )
    ap = ap_score(
        labels, preds, sample_weight=weights
    )
    auc_trunc = auc_score(
        labels[can_eval], preds[can_eval], sample_weight=weights[can_eval]
    )
    ap_trunc = ap_score(
        labels[can_eval], preds[can_eval], sample_weight=weights[can_eval]
    )

    return auc,ap, auc_trunc,ap_trunc

def train(tr,va,te, model: GraphGPT):
    opt = AdamW(
        model.parameters(), lr=6e-4,
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
    for e in range(EPOCHS):
        for pos_mb in tr:
            # Negative sample non-edges
            neg_mb = tr.sample(torch.randint(0, tr.x.size(0), (MINI_BS, 2)))
            labels = torch.zeros((len(pos_mb) + len(neg_mb), 1))
            labels[len(pos_mb):] = 1
            mb = pos_mb + neg_mb

            loss = minibatch(mb, model, labels)
            steps += 1

            if steps * MINI_BS >= BS:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
                opt.step()
                sched.step()
                opt.zero_grad()

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

                st = time.time()

        t_auc, t_ap, t_auc_trunc, t_ap_trunc = evaluate(model, tr, te)
        print('#'*20)
        print(f'TEST SCORES')
        print('#'*20)
        print(f"AUC (full dataset):   {t_auc:0.4f}, AP: {t_ap:0.4f}")
        print(f"AUC (ignore missing): {t_auc_trunc:0.4f}, AP: {t_ap_trunc:0.4f}")

        with open('ft_results.txt', 'a') as f:
            f.write(f'{e+1},{t_auc},{t_ap},{t_auc_trunc},{t_ap_trunc}\n')

        torch.save(
            model.state_dict(),
            'finetune.pt'
        )



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

    tr = SparseGraphSampler(tr, batch_size=MINI_BS, mode='finetune')

    num_tokens = tr.x.max().long() + 3 + 1
    t = Tokenizer(num_tokens, 3)

    model = GraphGPT('pretrained/mini_15_neighbors.pt', device=DEVICE)
    train(tr,va,te,model)

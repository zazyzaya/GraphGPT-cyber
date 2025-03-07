from argparse import ArgumentParser
from copy import deepcopy
from types import SimpleNamespace

from sklearn.metrics import (
    roc_auc_score as auc_score,
    average_precision_score as ap_score
)
import torch
from torch import nn
from torch.optim.adam import Adam
from torch_geometric.nn.models import GCN

from tokenizer import Tokenizer

DEVICE=0

@torch.no_grad()
def evaluate(z, to_eval):
    can_eval = to_eval.edge_index < tr.x.size(0) # Some new nodes we don't know what to do with
    idx = can_eval.prod(dim=0).nonzero().squeeze()

    preds = torch.zeros(to_eval.edge_index.size(1))
    preds_one = torch.ones(to_eval.edge_index.size(1))

    edges = to_eval.edge_index[:, idx]
    pred = 1 - torch.sigmoid((z[edges[0]] * z[edges[1]]).sum(dim=1))
    preds[idx] = pred.squeeze().cpu()
    preds_one[idx] = pred.squeeze().cpu()

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
        labels[idx], preds[idx], sample_weight=weights[idx]
    )
    ap_trunc = ap_score(
        labels[idx], preds[idx], sample_weight=weights[idx]
    )

    return auc,ap, auc_trunc,ap_trunc, auc_sus,ap_sus

def train(tr,va,te):
    model = GCN(tr.x.size(1), tr.x.size(1)*2, num_layers=2).to(DEVICE)
    opt = Adam(model.parameters(), lr=0.001)
    bce = nn.BCEWithLogitsLoss()

    # Test no training
    add_fake_data(va)
    auc, ap, auc_trunc, ap_trunc, auc_sus, ap_sus = evaluate(tr.x, va)
    print('#'*20)
    print(f'VAL SCORES')
    print('#'*20)
    print(f"AUC (full dataset):     {auc:0.4f}, AP: {ap:0.4f}")
    print(f"AUC (ignore missing):   {auc_trunc:0.4f}, AP: {ap_trunc:0.4f}")
    print(f"AUC (missing are anom): {auc_sus:0.4f}, AP: {ap_sus:0.4f}")

    auc, ap, auc_trunc, ap_trunc, auc_sus, ap_sus = evaluate(tr.x, te)
    print('#'*20)
    print(f'TEST SCORES')
    print('#'*20)
    print(f"AUC (full dataset):     {auc:0.4f}, AP: {ap:0.4f}")
    print(f"AUC (ignore missing):   {auc_trunc:0.4f}, AP: {ap_trunc:0.4f}")
    print(f"AUC (missing are anom): {auc_sus:0.4f}, AP: {ap_sus:0.4f}")


    best = 0
    best_te = None
    for e in range(1000):
        opt.zero_grad()
        model.train()
        z = model(tr.x, tr.edge_index)

        targets = torch.cat([
            tr.edge_index,
            torch.randint(0, tr.x.size(0), tr.edge_index.size(), device=DEVICE)
        ], dim=1)
        labels = torch.zeros((targets.size(1),), device=DEVICE)
        labels[:tr.edge_index.size(1)] = 1

        preds = (z[targets[0]] * z[targets[1]]).sum(dim=1)
        loss = bce(preds, labels)
        loss.backward()
        opt.step()

        print(f'[{e}] {loss}')

        model.eval()
        with torch.no_grad():
            z = model(tr.x, tr.edge_index)

        add_fake_data(va)
        auc, ap, auc_trunc, ap_trunc, auc_sus, ap_sus = evaluate(z, va)
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

        auc, ap, auc_trunc, ap_trunc, auc_sus, ap_sus = evaluate(z, te)
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

        auc, ap, auc_trunc, ap_trunc, auc_sus, ap_sus = best_te
        print('#'*20)
        print(f'BEST SCORES')
        print('#'*20)
        print(f"AUC (full dataset):     {auc:0.4f}, AP: {ap:0.4f}")
        print(f"AUC (ignore missing):   {auc_trunc:0.4f}, AP: {ap_trunc:0.4f}")
        print(f"AUC (missing are anom): {auc_sus:0.4f}, AP: {ap_sus:0.4f}")

def set_x(graph_x, bert_x):
    t = Tokenizer(graph_x)
    x = graph_x + t.feat_offset
    x = bert_x[x.long()].reshape(x.size(0), -1)
    return x

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

    arg = ArgumentParser()
    arg.add_argument('--size', default='NO INPUT')
    arg.add_argument('--device', type=int, default=0)
    args = arg.parse_args()

    SIZE = 'mini' #args.size
    DEVICE = 0 #args.device

    params = {
        'mini': SimpleNamespace(H=256, L=4, MINI_BS=512),
        'med': SimpleNamespace(H=512, L=8, MINI_BS=128)
    }[SIZE]
    MINI_BS = params.MINI_BS

    weights = torch.load(f'bert_{SIZE}.pt', weights_only=True, map_location='cpu')
    x = weights['cls.predictions.decoder.weight']
    x = set_x(tr.x, x)

    tr.x = x.to(DEVICE)
    tr.edge_index = tr.edge_index.to(DEVICE)
    train(tr,va,te)
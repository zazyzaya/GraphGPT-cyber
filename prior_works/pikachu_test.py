from collections import defaultdict 
import json 
import os 

import pandas as pd 
import torch 
from torch.optim import Adam, SGD
from torch_geometric.data import Data 
from tqdm import tqdm 
from sklearn.metrics import roc_auc_score as auc_score, average_precision_score as ap_score

from pikachu_models import CTDNE, Autoencoder, AnomalyDetector 

DEVICE = 0 

# Defaults from Word2Vec and Pikachu code 
W2V_EPOCHS = 5 
W2V_EMB = 128 
WALK_LEN = 64 # Default is 500, but graphs are much smaller
CTXT_SIZE = 5 
AE_EPOCHS = 50 
AE_LR = 0.001 # Default keras lr 
AE_HIDDEN = 128
AE_LATENT = 64
ANOM_LR = 0.001
ANOM_EPOCHS = 10
TRAIN_WIN = 5 # How many snapshots per batch for anomaly detection

def preprocess(g): 
    '''
    Slice input CSR graph into discrete time units
    '''
    ts = g.ts.unique()
    snapshots = []

    for t in tqdm(ts, desc='Preprocessing'): 
        csr = defaultdict(list)    
        src = g.src[g.ts == t]
        dst = g.col[g.ts == t]

        for i in range(src.size(0)): 
            csr[src[i].item()].append(dst[i].item())

        ptr = [0]
        col = []
        for i in range(g.x.size(0)): 
            col_n = csr[i] 
            ptr.append(ptr[-1] + len(col_n))
            col += col_n 

        if 'label' in g.keys(): 
            y = g.label[g.ts == t]
        else: 
            y = None 

        g_t = Data(
            x = g.x, 
            edge_index = torch.stack([src,dst]), 
            rowptr = torch.tensor(ptr), 
            col = torch.tensor(col),
            label = y 
        )
        snapshots.append(g_t)

    return snapshots 

def get_node_embeddings(tr): 
    embs = []

    for tr_t in tqdm(tr, desc='Running n2v'): 
        model = CTDNE(W2V_EMB, WALK_LEN, CTXT_SIZE, tr_t.x.size(0), device=DEVICE)
        opt = Adam(model.parameters(), lr=0.025)

        for e in range(1, W2V_EPOCHS+1): 
            opt.zero_grad()
            pos, neg = model.sample(torch.arange(tr_t.x.size(0), device=DEVICE), tr_t.to(DEVICE))
            loss = model.loss(pos, neg)
            loss.backward()
            opt.step()
        
        with torch.no_grad():
            model.eval()
            embs.append(model.forward().detach())

    return embs 

def train_ae(emb_list): 
    embs = torch.stack(emb_list, dim=1).to(DEVICE)
    model = Autoencoder(embs.size(-1), AE_HIDDEN, AE_LATENT, device=DEVICE)
    opt = Adam(model.parameters(), lr=AE_LR)

    print("Training GRU Autoencoder")
    for e in range(AE_EPOCHS):  
        opt.zero_grad() 
        model.train()
        loss = model.forward(embs)
        loss.backward()
        opt.step()

        print(f'\t[{e}] {loss.item()}')

    with torch.no_grad(): 
        model.eval()
        z = model.enc(embs).detach()

    return z 

def train_anom(z, tr, va, te): 
    '''
    z: N x T x d matrix of node embs
    '''
    model = AnomalyDetector(AE_LATENT, z.size(0), device=DEVICE)
    
    # Pikachu src code does gradient descent by hand using SGD 
    opt = SGD(model.parameters(), lr=ANOM_LR) 
    
    def evaluate(g): 
        model.eval()
        y = torch.cat([g[i].label for i in range(len(g))])
        
        y_hat = []
        for i in range(z.size(1)): 
            y_hat.append(model.predict(z[:, i], g.edge_index, g.rowptr, g.col).detach())
        y_hat = torch.cat(y_hat).cpu()

        auc = auc_score(y, y_hat)
        ap = ap_score(y, y_hat)
        
        return auc,ap 

    best = (0,0,0)
    snooped = (0,0)

    print("Training anomaly detector")
    for e in range(1,ANOM_EPOCHS): 
        opt.zero_grad()
        model.train()
        for t in range(z.size(1)): 
            loss = model.forward(z[:, t], tr[t].edge_index, tr[t].rowptr, tr[t].col)
            loss.backward() 
            print(f'\t[{e}-{t}] {loss.item()}')
        opt.step()

        with torch.no_grad(): 
            va_auc,va_ap = evaluate(va)
            print(f'\tVa: AUC {va_ap:0.4f}, AP {va_ap:0.4f}')
            te_auc,te_ap = evaluate(te) 
            print(f'\tTe: AUC {te_ap:0.4f}, AP {te_ap:0.4f}')
        
        if va_auc > best[0]: 
            best = (va_auc, te_auc, te_ap)
        if te_auc > snooped[0]: 
            snooped = (te_auc, te_ap)

    return {
        'best-auc': best[1], 
        'best-ap': best[2], 
        'snooped-auc': snooped[0],
        'snooped-ap': snooped[1], 
        'last-auc': te_auc, 
        'last-ap': te_ap
    }

def train_full(tr,va,te): 
    embs = get_node_embeddings(tr) 
    embs = train_ae(embs)
    stats = train_anom(embs, tr,va,te)

    print(json.dumps(stats, indent=1))
    return stats 

if __name__ == '__main__': 
    if not os.path.exists('tmp/pikachu_tr.pt'): 
        tr = preprocess(torch.load('../data/unsw_tgraph_tr.pt', weights_only=False))
        va = preprocess(torch.load('../data/unsw_tgraph_va.pt', weights_only=False))
        te = preprocess(torch.load('../data/unsw_tgraph_te.pt', weights_only=False))

        torch.save(tr, 'tmp/pikachu_tr.pt')
        torch.save(va, 'tmp/pikachu_va.pt')
        torch.save(te, 'tmp/pikachu_te.pt')
    else: 
        tr = torch.load('tmp/pikachu_tr.pt', weights_only=False)
        va = torch.load('tmp/pikachu_va.pt', weights_only=False)
        te = torch.load('tmp/pikachu_te.pt', weights_only=False)

    stats = [
        train_full(tr,va,te)
        for _ in range(10)
    ]
    df = pd.DataFrame(stats)
    df.loc['mean'] = df.mean()
    df.loc['sem'] = df.sem()
    
    df.to_csv('pikachu_results.csv')
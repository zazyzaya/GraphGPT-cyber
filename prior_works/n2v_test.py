from math import ceil
import pandas as pd 
import torch 
from torch import nn 
from torch.optim import Adam 
from torch_geometric.nn.models import Node2Vec
from sklearn.metrics import roc_auc_score as auc_score, average_precision_score as ap_score 

EMB_DIM = 128
WL = 500
WINDOW = 5 
N2V_EPOCHS = 25 
ANOM_EPOCHS = 200 
DEVICE = 0 

def preprocess(dataset): 
    eis = []
    num_nodes = 0 
    for split in ['tr', 'va', 'te']: 
        g = torch.load(f'../data/{dataset}_tgraph_{split}.pt')
        ei = torch.stack([g.src, g.col])
        eis.append(ei.to(DEVICE))

        if split == 'te': 
            y = g.label
        
        num_nodes = max(ei.max(), num_nodes)

    return eis, y.to(DEVICE), num_nodes+1

class AnomDetector(nn.Module):
    def __init__(self, z_dim): 
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(z_dim, z_dim//2, device=DEVICE), 
            nn.ReLU(), 
            nn.Linear(z_dim//2, 1, device=DEVICE)
        )
        self.criterion = nn.BCEWithLogitsLoss()

    def predict(self, z,ei): 
        z = z[ei[0]] * z[ei[1]]
        return self.net(z) 
    
    def forward(self, z,ei): 
        pos = self.predict(z,ei)
        neg = self.predict(z,torch.randint(0,z.size(0),ei.size(), device=DEVICE))    
        
        labels = torch.zeros(pos.size(0)*2, 1, device=DEVICE)
        labels[pos.size(0):] = 1
        loss = self.criterion(
            torch.cat([pos,neg]),
            labels
        )

        return loss 
    
def train(tr,va,te,y,num_nodes): 
    n2v = Node2Vec(tr, EMB_DIM, WL, WINDOW, num_nodes=num_nodes).to(DEVICE)
    n2v_opt = Adam(n2v.parameters(), lr=0.01)

    def validate(model,z,ei):
        neg = torch.randint(0, ei.max(),ei.size(), device=DEVICE)
        ei = torch.cat([ei,neg], dim=1)
        y = torch.zeros(ei.size(1),1).to(DEVICE)
        y[neg.size(1):] = 1

        return evaluate(model,z,ei,y)

    def evaluate(model,z,ei,y): 
        pred = model.predict(z,ei).cpu()
        y = y.cpu()

        auc = auc_score(y,pred)
        ap = ap_score(y,pred)

        return auc,ap 

    
    anom = AnomDetector(EMB_DIM)
    anom_opt = Adam(anom.parameters(), lr=0.01)

    print("n2v")
    MBS = 2048
    for e in range(N2V_EPOCHS): 
        n2v_opt.zero_grad()
        
        batches = torch.randperm(tr.max()+1)
        for i in range(ceil(batches.size(0) / MBS)): 
            pos,neg = n2v.sample(batches[i*MBS: (i+1)*MBS])
            loss = n2v.loss(pos.to(DEVICE),neg.to(DEVICE))
            print(f'[{e}-{i}] Loss: {loss.item()}')
            loss.backward()
        
        n2v_opt.step()

    z = n2v(torch.arange(num_nodes)).detach()
    anom_best = (0,0,0,0)
    impatience = 0 
    ea = 0 
    while True: 
        anom_opt.zero_grad()
        loss = anom(z,tr)
        print(f'[{ea}] {loss}')
        loss.backward() 
        anom_opt.step() 

        with torch.no_grad(): 
            t_auc, t_ap = validate(anom,z,tr)
            print(f'\tTr:  AUC {t_auc:0.4f}, AP {t_ap:0.4f}')
            v_auc, v_ap = validate(anom,z,va)
            print(f'\tVa:  AUC {v_auc:0.4f}, AP {v_ap:0.4f}', end='')
            
            if v_auc > anom_best[0]: 
                impatience = 0 
                end = '*'
            else: 
                impatience += 1
                end = ''
                if impatience >= 10: 
                    print()
                    break  
            print(end) 

            auc, ap = evaluate(anom,z,te,y)
            print(f'\tTe:  AUC {auc:0.4f}, AP {ap:0.4f}')

            if impatience == 0: 
                anom_best = (v_auc,v_ap,auc,ap)

        ea += 1

    return {
        'val_auc': anom_best[0],
        'val_ap': anom_best[1], 
        'te_auc': anom_best[2],
        'te_ap': anom_best[3]
    }

for dataset in ['lanl14argus', 'optc', 'unsw']:
    ei,y,nodes = preprocess(dataset)
    df = pd.DataFrame([train(*ei,y,nodes) for _ in range(10)])
    df.loc['mean'] = df.mean()
    df.loc['sem'] = df.sem()
    df.to_csv(f'n2v_results_{dataset}.csv')
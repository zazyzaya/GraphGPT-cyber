from collections import defaultdict

import torch 
from torch_geometric.data import Data 
from tqdm import tqdm 

INPUT_DIR = '/mnt/raid10/kg_datasets/FB15k-237'

def get_filter(): 
    tr = torch.load('data/fb15_tgraph_tr.pt', weights_only=False)
    va = torch.load('data/fb15_tgraph_va.pt', weights_only=False)
    te = torch.load('data/fb15_tgraph_te.pt', weights_only=False)

    idx = [0]
    col = torch.tensor([])
    rel = torch.tensor([])
    for i in range(tr.x.size(0)): 
        new_col = torch.cat([
            tr.col[tr.idxptr[i]:tr.idxptr[i+1]],
            va.col[va.idxptr[i]:va.idxptr[i+1]],
            te.col[te.idxptr[i]:te.idxptr[i+1]]
        ])

        new_rel = torch.cat([
            tr.edge_attr[tr.idxptr[i]:tr.idxptr[i+1]].squeeze(-1),
            va.edge_attr[va.idxptr[i]:va.idxptr[i+1]].squeeze(-1),
            te.edge_attr[te.idxptr[i]:te.idxptr[i+1]].squeeze(-1)
        ])

        col = torch.cat([col, new_col])
        rel = torch.cat([rel, new_rel])

        idx.append(idx[-1] + col.size(0))

    tr.filter_ptr = torch.tensor(idx)
    tr.filter_col = col.long()
    tr.filter_rel = rel 

    torch.save(tr, 'data/fb15_tgraph_tr.pt')

def to_torch(csr, num_nodes, fold): 
    rowptr = [0]
    src = []
    rel = []
    col = []
    for s in range(num_nodes):
        d,r = csr[s]
        
        rowptr.append(rowptr[-1] + len(d))
        src += [s] * len(d)
        col += d 
        rel += r 

    x = torch.zeros((num_nodes,1))
    ts = torch.zeros(len(col))
    torch.save(
        Data(
            x=x, 
            idxptr=torch.tensor(rowptr),
            col=torch.tensor(col),
            src=torch.tensor(src),
            ts=ts,
            edge_attr=torch.tensor(rel)
        ),
        # Not actually a tgraph but so old training scripts will work
        f'data/fb15_tgraph_{fold}.pt' 
    )

def build_graphs(): 
    n_ids = dict() 
    e_ids = dict()

    def fmt_line(line): 
        head,rel,tail = line.split('\t')
        tail = tail[:-1] # Trailing newline
        return head,rel,tail
    
    def get_or_add_n(n): 
        if (nid := n_ids.get(n)) is None: 
            nid = len(n_ids)
            n_ids[n] = nid 
        return nid 
    
    def get_or_add_e(e): 
        if (eid := e_ids.get(e)) is None: 
            eid = len(e_ids)
            e_ids[e] = eid 
        return eid 

    # Build training set
    f = open(f'{INPUT_DIR}/train.txt', 'r')
    line = f.readline() 
    
    csr = defaultdict(lambda : [[],[]])
    prog = tqdm(desc='Train')
    while line: 
        h,r,t = fmt_line(line)
        
        h = get_or_add_n(h)
        r = [get_or_add_e(r)]
        t = get_or_add_n(t)

        csr[h][0].append(t)
        csr[h][1].append(r)

        prog.update()
        line = f.readline() 
    
    prog.close()
    f.close()
    to_torch(csr, len(n_ids), 'tr')

    # Build val set
    f = open(f'{INPUT_DIR}/valid.txt', 'r')
    line = f.readline() 
    
    csr = defaultdict(lambda : [[],[]])
    prog = tqdm(desc='Val')
    while line: 
        h,r,t = fmt_line(line)
        
        h = n_ids.get(h)
        r = [e_ids.get(r)]
        t = n_ids.get(t)

        # Skips 29 out of 20k edges in test set 
        if h is None or t is None: 
            prog.update()
            line = f.readline() 
            continue 

        csr[h][0].append(t)
        csr[h][1].append(r)

        prog.update()
        line = f.readline() 
    
    prog.close()
    f.close()
    to_torch(csr, len(n_ids), 'va')

    # Build test set
    f = open(f'{INPUT_DIR}/test.txt', 'r')
    line = f.readline() 
    
    csr = defaultdict(lambda : [[],[]])
    prog = tqdm(desc='Test')
    while line: 
        h,r,t = fmt_line(line)
        
        h = n_ids.get(h)
        r = [e_ids.get(r)]
        t = n_ids.get(t)

        # Skips 29 out of 20k edges in test set 
        if h is None or t is None: 
            prog.update()
            line = f.readline() 
            continue 

        csr[h][0].append(t)
        csr[h][1].append(r)

        prog.update()
        line = f.readline() 
    
    prog.close()
    to_torch(csr, len(n_ids), 'te')

if __name__ == '__main__': 
    build_graphs()
    get_filter()
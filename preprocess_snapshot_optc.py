from collections import defaultdict

from tqdm import tqdm
import torch
from torch_geometric.data import Data

from joblib import Parallel, delayed

HOME_DIR = '/mnt/raid10/cyber_datasets/OpTC'
T0 = 1568676405.628
ATTACK_STARTS = 1569240001
N_FILES = 12672

def load_one(fid):
    try: 
        f = open(f'{HOME_DIR}/flow_start/{fid}.csv', 'r')
    except FileNotFoundError: 
        return set() 
    
    line = f.readline()
    edge_set = set()
    while line:
        ts,src,dst,y = line.split(',')
        ts = int(ts.split('.')[0])
        src = int(src)
        dst = int(dst)

        ts -= T0

        # I don't think they do, but just in case
        if y == '1': 
            print("THEY HAVE LABELS!")

        # Bi-directional
        edge_set.add((src,dst))
        edge_set.add((dst,src))

        line = f.readline()

    f.close()
    return edge_set

def load_tgraph_full(): 
    edge_sets = Parallel(n_jobs=32, prefer='processes')(
        delayed(load_one)(i) for i in tqdm(range(N_FILES))
    )

    csr = defaultdict(lambda : [[],[]])
    for t,e in tqdm(enumerate(edge_sets)): 
        while e: 
            src,dst = e.pop()
            csr[src][0].append(dst)
            csr[src][1].append(t)

    x = torch.zeros(1000, 1)

    idxptr = [0]
    col = []
    ts = []
    for i in tqdm(range(x.size(0))):
        neighbors,t = csr[i]

        col += neighbors
        ts += t

        idxptr.append(len(neighbors) + idxptr[-1])
        del csr[i]

    torch.save(
        Data(
            x = x,
            idxptr = torch.tensor(idxptr),
            col = torch.tensor(col),
            ts = torch.tensor(ts)
        ),
        'data/optc_tgraph_csr.pt'
    )

def partition_tgraph():
    g = torch.load('data/optc_tgraph_csr.pt', weights_only=False)

    idxs = torch.randperm(g.col.size(0))
    tr_end = int(idxs.size(0) * 0.9)

    tr = torch.zeros(g.col.size(0), dtype=torch.bool)
    va = torch.zeros_like(tr)

    tr[idxs[:tr_end]] = True
    va[idxs[tr_end:]] = True

    # Generate new index pointer for subset of column that was selected
    def reindex(idxptr, subset_mask):
        new_ptr = [0]
        for i in range(1, idxptr.size(0)):
            st = idxptr[i-1]; en = idxptr[i]
            selected = subset_mask[st:en].sum().item()
            new_ptr.append(new_ptr[-1] + selected)

        return torch.tensor(new_ptr)

    for mask,name in [(tr, 'tr'), (va, 'va')]:
        new_ptr = reindex(g.idxptr, mask)
        data = Data(
            x = g.x,
            idxptr = new_ptr,
            col = g.col[mask],
            ts = g.ts[mask]
        )

        torch.save(data, f'data/optc_tgraph_{name}.pt')


if __name__ == '__main__': 
    #load_tgraph_full()
    partition_tgraph()
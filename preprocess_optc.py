from collections import defaultdict

import torch 
from torch_geometric.data import Data 
from tqdm import tqdm 

HOME_DIR = '/mnt/raid10/cyber_datasets/OpTC'
NUM_NODES = 1034

def full_to_tgraph():
    csr = defaultdict(lambda : [[],[],[],[]])
    first = True
    t0 = None

    f = open(f'{HOME_DIR}/full_graph.csv', 'r')
    line = f.readline()
    prog = tqdm()

    edge_feats = dict()
    def get_or_add(ef, is_port=False):
        if is_port and int(ef) > 1024:
            ef = 'EPHEMERAL'
             
        if (eid := edge_feats.get(ef)) is None: 
            eid = len(edge_feats)
            edge_feats[ef] = eid 

        return eid 

    while line:
        ts,src,dst,port,img,usr,label = line.split(',')
        ts = int(ts.split('.')[0])
        
        src = int(src)
        dst = int(dst)
        port = get_or_add(port, is_port=True) 
        usr = get_or_add(usr) 
        label = int(label)

        if first:
            t0 = ts
            first = False

        ts -= t0

        # Unidentified host
        if src >= 1100 or dst >= 1100: 
            line = f.readline() 
            prog.update()
            continue 

        csr[src][0].append(dst)
        csr[src][1].append(ts)
        csr[src][2].append([port,usr])
        csr[src][3].append(label)

        # Bi-directional
        csr[dst][0].append(src)
        csr[dst][1].append(ts)
        csr[dst][2].append([port,usr])
        csr[dst][3].append(label)

        line = f.readline()
        prog.update()

    prog.close()
    f.close()

    # Do this at the end so all sections of the graph
    # agree on node mappings
    x = torch.zeros(NUM_NODES, 1)

    idxptr = [0]
    col = []
    ts = []
    efs = []
    labels = []
    for i in tqdm(range(x.size(0))):
        neighbors,t,ef,label = csr[i]

        col += neighbors
        ts += t
        efs += ef 
        labels += label 

        idxptr.append(len(neighbors) + idxptr[-1])
        del csr[i]

    # Makes file larger but speeds up inference
    idxptr = torch.tensor(idxptr, dtype=torch.long)
    src = torch.arange(x.size(0))
    deg = idxptr[1:] - idxptr[:-1]
    src = src.repeat_interleave(deg)

    torch.save(
        Data(
            x = x,
            idxptr = idxptr,
            col = torch.tensor(col),
            src = src,
            ts = torch.tensor(ts), 
            edge_attr=torch.tensor(efs),
            label = torch.tensor(labels)
        ),
        'data/optc_tgraph_csr.pt'
    )

def partition_tgraph():
    torch.manual_seed(0)

    g = torch.load('data/optc_tgraph_csr.pt', weights_only=False)

    idxs = torch.randperm(g.col.size(0))
    tr_end = int(idxs.size(0) * 0.8)
    va_end = int(idxs.size(0) * 0.9)

    tr = torch.zeros(g.col.size(0), dtype=torch.bool)
    va = torch.zeros_like(tr)
    te = torch.zeros_like(tr)

    tr[idxs[:tr_end]] = True
    va[idxs[tr_end:va_end]] = True
    te[idxs[va_end:]] = True

    # Mask out anomalies so they're all in test set
    is_mal = g.label == 1
    tr[is_mal] = False
    va[is_mal] = False
    te[is_mal] = True

    # Generate new index pointer for subset of column that was selected
    def reindex(idxptr, subset_mask):
        new_ptr = [0]
        for i in range(1, idxptr.size(0)):
            st = idxptr[i-1]; en = idxptr[i]
            selected = subset_mask[st:en].sum().item()
            new_ptr.append(new_ptr[-1] + selected)

        return torch.tensor(new_ptr)

    for mask,name in [(tr, 'tr'), (va, 'va'), (te, 'te')]:
        new_ptr = reindex(g.idxptr, mask)
        data = Data(
            x = g.x,
            idxptr = new_ptr,
            col = g.col[mask],
            src = g.src[mask],
            edge_attr = g.edge_attr[mask],
            ts = g.ts[mask]
        )

        if name == 'te':
            data.label = g.label[mask]

        torch.save(data, f'data/optc_tgraph_{name}.pt')

if __name__ == '__main__':
    full_to_tgraph()
    partition_tgraph()
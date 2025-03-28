from collections import defaultdict

import torch
from torch_geometric.data import Data
from tqdm import tqdm


HOME_DIR = '/mnt/raid10/cyber_datasets/unsw_nb15'
SRC_IP = 0
SRC_PORT = 1
DST_IP = 2
DST_PORT = 3
PROTO = 4
SERVICE = 13
TS = 28
LABEL = 48

def argsort(seq): return sorted(range(len(seq)), key=seq.__getitem__)

def build_tgraph():
    '''
    TODO there are a lot of edge features in this dataset
    It may be useful to add them in at a later point
    '''
    t0 = float('inf')
    node_map = dict()
    edge_map = dict()
    csr = defaultdict(lambda : [[],[],[],[]])

    def get_or_add_n(n):
        if (nid := node_map.get(n)) is None:
            nid = len(node_map)
            node_map[n] = nid
        return nid

    def get_or_add_e(e):
        if (eid := edge_map.get(e)) is None:
            eid = len(edge_map)
            edge_map[e] = eid
        return eid

    def add_port(p):
        if p.startswith('0x'):
            p = int(p, base=16)
        elif p == '-':
            return get_or_add_e('UNKNOWN')
        else:
            p = int(p)

        if p < 1024:
            return get_or_add_e(p)
        elif p < 49152:
            return get_or_add_e('REGISTERED')
        else:
            return get_or_add_e('EPHEMERAL')


    for file in range(1,5):
        f = open(f'{HOME_DIR}/UNSW-NB15_{file}.csv')
        line = f.readline()
        prog = tqdm(desc=str(file), total=700_000)

        while line:
            tokens = line.split(',')

            src = get_or_add_n(tokens[SRC_IP])
            dst = get_or_add_n(tokens[DST_IP])
            e = [
                add_port(tokens[SRC_PORT]),
                get_or_add_e(tokens[PROTO]),
                get_or_add_e(tokens[SERVICE]),
                add_port(tokens[DST_PORT])
            ]
            ts = float(tokens[TS])
            label = int(tokens[LABEL])

            t0 = min(t0, ts)

            csr[src][0].append(dst)
            csr[src][1].append(ts)
            csr[src][2].append(label)
            csr[src][3].append(e)

            csr[dst][0].append(src)
            csr[dst][1].append(ts)
            csr[dst][2].append(label)
            csr[dst][3].append(e)

            line = f.readline()
            prog.update()

        prog.close()
        f.close()

    print(f"{len(edge_map)} unique edge types")

    # Do this at the end so all sections of the graph
    # agree on node mappings
    x = torch.zeros(len(node_map), 1)

    idxptr = [0]
    col = []
    ts = []
    labels = []
    edge_attr = []
    for i in tqdm(range(x.size(0))):
        neighbors,t,label,ea = csr[i]
        sort_idx = argsort(t)

        neighbors = [neighbors[i] for i in sort_idx]
        t = [t[i] for i in sort_idx]
        label = [label[i] for i in sort_idx]
        ea = [ea[i] for i in sort_idx]

        col += neighbors
        ts += t
        labels += label
        edge_attr += ea

        idxptr.append(len(neighbors) + idxptr[-1])
        del csr[i]

    # Makes file larger but speeds up inference
    idxptr = torch.tensor(idxptr, dtype=torch.long)
    src = torch.arange(x.size(0))
    deg = idxptr[1:] - idxptr[:-1]
    src = src.repeat_interleave(deg)

    data = Data(
        x = x,
        idxptr = idxptr,
        src = src,
        col = torch.tensor(col, dtype=torch.long),
        ts = (torch.tensor(ts) - t0).long(),
        edge_attr = torch.tensor(edge_attr),
        label = torch.tensor(labels)
    )
    torch.save(data, f'data/unsw_tgraph_csr.pt')

def partition_tgraph():
    torch.manual_seed(0)

    g = torch.load('data/unsw_tgraph_csr.pt', weights_only=False)

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

        torch.save(data, f'data/unsw_tgraph_{name}.pt')

if __name__ == '__main__':
    build_tgraph()
    partition_tgraph()
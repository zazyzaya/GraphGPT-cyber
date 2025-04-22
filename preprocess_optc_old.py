from collections import defaultdict

from tqdm import tqdm
import torch
from torch_geometric.data import Data

HOME_DIR = '/mnt/raid10/cyber_datasets/OpTC'
N_FILES = 12672 # Originally split for Euler paper into hour-long chunks
NUM_NODES = 1000

def argsort(seq): return sorted(range(len(seq)), key=seq.__getitem__)

def full_to_tgraph():
    csr = defaultdict(lambda : [[],[]])
    first = True
    t0 = None

    f = open(f'{HOME_DIR}/benign_compressed.csv', 'r')
    line = f.readline()
    prog = tqdm(total=274682407)

    while line:
        src,dst,_,ts = line.split(',')
        ts = int(ts.split('.')[0])
        src = int(src)
        dst = int(dst)

        if first:
            t0 = ts
            first = False

        ts -= t0

        # Bi-directional
        csr[src][0].append(dst)
        csr[src][1].append(ts)
        csr[dst][0].append(src)
        csr[dst][1].append(ts)

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
    for i in tqdm(range(x.size(0))):
        neighbors,t = csr[i]
        sort_idx = argsort(t)

        neighbors = [neighbors[i] for i in sort_idx]
        t = [t[i] for i in sort_idx]

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

        if name == 'va': 
            src = torch.arange(1000)
            deg = new_ptr[1:] - new_ptr[:-1]
            src = src.repeat_interleave(deg)
        else: 
            src = None 

        data = Data(
            x = g.x,
            idxptr = new_ptr,
            col = g.col[mask],
            ts = g.ts[mask],
            src = src 
        )

        torch.save(data, f'data/optc_tgraph_{name}.pt')

def build_malgraphs(day):
    f = open(f'{HOME_DIR}/attack_day-{day}.csv')

    edges = defaultdict(lambda : 0)
    mal = set()

    line = f.readline()
    prog = tqdm(desc=f'Day {day}')
    while line:
        src,dst,ts,is_mal = line.split(',')
        src = int(src); dst = int(dst)
        is_mal = int(is_mal)

        edges[(src,dst)] += 1
        if is_mal:
            mal.add((src,dst))

        line = f.readline()
        prog.update()

    src,dst = [],[]
    attr,label = [],[]
    for (s,d),cnt in edges.items():
        src.append(s)
        dst.append(d)
        attr.append(cnt)

        if (s,d) in mal:
            label.append(1)
        else:
            label.append(0)

    data = Data(
        x=torch.zeros(NUM_NODES),
        edge_index=torch.tensor([src,dst]),
        edge_attr=torch.tensor(attr),
        label=torch.tensor(label)
    )
    torch.save(data, f'data/optc_attack_day-{day}.pt')

def build_mal_tgraph():
    csr = defaultdict(lambda : [[],[],[]])
    first = True
    t0 = float('inf')

    for day in range(1,4):
        f = open(f'{HOME_DIR}/attack_day-{day}.csv')

        line = f.readline()
        prog = tqdm(desc=f'Day {day}')
        while line:
            src,dst,wgt,ts,is_mal = line.split(',')
            src = int(src); dst = int(dst)
            ts = int(ts.split('.')[0]); is_mal = int(is_mal)

            # Directed edges for testing
            csr[src][0].append(dst)
            csr[src][1].append(ts)
            csr[src][2].append(is_mal)

            t0 = min(ts, t0)
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
    labels = []
    for i in tqdm(range(x.size(0))):
        neighbors,t,label = csr[i]
        sort_idx = argsort(t)

        neighbors = [neighbors[i] for i in sort_idx]
        t = [t[i] for i in sort_idx]
        label = [label[i] for i in sort_idx]

        col += neighbors
        ts += t
        labels += label

        idxptr.append(len(neighbors) + idxptr[-1])
        del csr[i]

    idxptr = torch.tensor(idxptr, dtype=torch.long)
    src = torch.arange(1000)
    deg = idxptr[1:] - idxptr[:-1]
    src = src.repeat_interleave(deg)

    data = Data(
        x = x,
        idxptr = idxptr, 
        col = torch.tensor(col, dtype=torch.long),
        ts = (torch.tensor(ts) - t0).long(),
        label = torch.tensor(labels),
        src = src 
    )
    torch.save(data, f'data/optc_tgraph_te.pt')

def build_split_mal_tgraph():
    for day in range(1,4):
        csr = defaultdict(lambda : [[],[],[]])
        t0 = float('inf')

        f = open(f'{HOME_DIR}/attack_day-{day}_compressed.csv')

        line = f.readline()
        prog = tqdm(desc=f'Day {day}')
        while line:
            src,dst,ts,_,is_mal = line.split(',')
            src = int(src); dst = int(dst)
            ts = int(ts.split('.')[0]); is_mal = int(is_mal)

            # Directed edges for testing
            csr[src][0].append(dst)
            csr[src][1].append(ts)
            csr[src][2].append(is_mal)

            t0 = min(ts, t0)
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
        labels = []
        for i in tqdm(range(x.size(0))):
            neighbors,t,label = csr[i]
            sort_idx = argsort(t)

            neighbors = [neighbors[i] for i in sort_idx]
            t = [t[i] for i in sort_idx]
            label = [label[i] for i in sort_idx]

            col += neighbors
            ts += t
            labels += label

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
            label = torch.tensor(labels)
        )
        torch.save(data, f'data/optc_attack_day-{day}.pt')

def to_static():
    # Connect all attack graphs into single data unit to make testing easier
    gs = [torch.load(f'data/optc_attack_day-{i}.pt', weights_only=False) for i in range(1,4)]
    g = Data(
        x = gs[0].x,
        edge_index=torch.cat([g.edge_index for g in gs], dim=1),
        edge_attr=torch.cat([g.edge_attr for g in gs]),
        label = torch.cat([g.label for g in gs])
    )
    torch.save(g, 'data/optc_sgraph_te.pt')

    # Convert val graph into static graph
    g = torch.load('data/optc_tgraph_va.pt', weights_only=False)
    edges = defaultdict(lambda : 0)
    for src in tqdm(range(g.idxptr.size(0)-1)):
        st = g.idxptr[src]; en = g.idxptr[src+1]
        for i in range(st,en):
            dst = g.col[i].item()
            edges[(src, dst)] += 1

    src,dst,cnt = [],[],[]
    for (s,d),c in edges.items():
        src.append(s)
        dst.append(d)
        cnt.append(c)

    data = Data(
        x=g.x,
        edge_index = torch.tensor([src,dst]),
        edge_attr = torch.tensor(cnt)
    )
    torch.save(data, 'data/optc_sgraph_va.pt')

def compress_tr_ei():
    g = torch.load('data/optc_tgraph_tr.pt', weights_only=False)
    edges = set()
    for src in tqdm(range(g.idxptr.size(0)-1)):
        st = g.idxptr[src]; en = g.idxptr[src+1]
        for i in range(st,en):
            dst = g.col[i].item()
            edges.add((src,dst))

    src,dst = [],[]
    for (s,d) in edges:
        src.append(s); dst.append(d)
    g.edge_index = torch.tensor([src,dst])
    torch.save(g, 'data/optc_tgraph_tr.pt')


if __name__ == '__main__':
    full_to_tgraph()
    partition_tgraph()
    #build_malgraphs(1)
    #build_malgraphs(2)
    #build_malgraphs(3)
    #to_static()
    #compress_tr_ei()
    build_mal_tgraph()
    #build_split_mal_tgraph()
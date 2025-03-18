from collections import defaultdict

from tqdm import tqdm
import torch
from torch_geometric.data import Data

HOME_DIR = '/mnt/raid10/cyber_datasets/OpTC'
N_FILES = 12672 # Originally split for Euler paper into hour-long chunks

def full_to_tgraph():
    csr = defaultdict(lambda : [[],[],[]])
    first = True
    t0 = None

    max_nid = 0

    for i in tqdm(range(N_FILES)):
        try:
            f = open(f'{HOME_DIR}/{i}.csv', 'r')
        except FileNotFoundError:
            # Didn't record data outside 8-6 but file numbers
            # correspond to timecodes so they continue increasing
            # as if those files exist
            continue

        line = f.readline()
        sub_csr = defaultdict(lambda : set())

        # Squish edges that happen within single file
        while line:
            ts,src,dst,label = line.split(',')
            ts = int(ts.split('.')[0])
            src = int(src)
            dst = int(dst)
            label = int(label)

            if first:
                t0 = ts
                first = False

            ts -= t0

            # Bi-directional
            sub_csr[src].add((dst,label))
            sub_csr[dst].add((src,label))

            line = f.readline()

        f.close()

        for src,d_l in sub_csr.items():
            for dst,label in d_l:
                csr[src][0].append(dst)
                csr[src][1].append(i)

                if label:
                    csr[src][2].append(len(csr[src][0])-1)

    # Do this at the end so all sections of the graph
    # agree on node mappings
    x = torch.zeros(len(csr), 1)

    idxptr = [0]
    col = []
    ts = []
    is_mal = []
    for i in tqdm(range(x.size(0))):
        neighbors,t,label = csr[i]
        col += neighbors
        ts += t

        if label:
            is_mal += [l + idxptr[-1] for l in label]

        idxptr.append(len(neighbors) + idxptr[-1])
        del csr[i]

    torch.save(
        Data(
            x = x,
            idxptr = torch.tensor(idxptr),
            col = torch.tensor(col),
            ts = torch.tensor(ts),
            is_mal = torch.tensor(is_mal),
        ),
        'data/optc_tgraph_csr.pt'
    )

def partition_tgraph():
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
    tr[g.is_mal] = False
    va[g.is_mal] = False
    te[g.is_mal] = True

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
            ts = g.ts[mask]
        )

        if name == 'te':
            label = torch.zeros(mask.size(0))
            label[g.is_mal] = 1
            label = label[mask]
            data.label = label

        torch.save(data, f'data/optc_tgraph_{name}.pt')

def tgraph_to_static(partition='va'):
    g = torch.load(f'data/optc_tgraph_{partition}.pt', weights_only=False)

    edges = defaultdict(lambda : 0)
    is_mal = set()
    prog = tqdm(total=g.col.size(0), desc=partition)
    for src in range(g.idxptr.size(0)-1):
        st = g.idxptr[src]; en = g.idxptr[src+1]
        for j in range(st,en):
            dst = g.col[j].item()
            edges[(src,dst)] += 1

            if partition == 'te' and g.label[j]:
                is_mal.add((src,dst))

            prog.update()
    prog.close()

    src,dst,cnt,label = [],[],[],[]
    for (s,d),weight in tqdm(edges.items()):
        src.append(s)
        dst.append(d)
        cnt.append(weight)
        if (s,d) in is_mal:
            label.append(1)
        else:
            label.append(0)

    data = Data(
        g.x, edge_index=torch.tensor([src,dst]),
        edge_attr=torch.tensor(cnt),
        label = torch.tensor(label)
    )
    torch.save(data, f'data/optc_sgraph_{partition}.pt')


if __name__ == '__main__':
    full_to_tgraph()
    partition_tgraph()
    tgraph_to_static('tr')
    tgraph_to_static('va')
    tgraph_to_static('te')
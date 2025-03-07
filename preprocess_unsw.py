from collections import defaultdict
from copy import deepcopy

import torch
from torch_geometric.data import Data
from tqdm import tqdm

from joblib import Parallel, delayed

HOME = '/mnt/raid10/cyber_datasets/unsw_nb15/'

SRC_IP = 0
DST_IP = 2
LABEL  = -1

def read_one(file):
    f = open(f'{HOME}/UNSW-NB15_{file}.csv', 'r')
    line = f.readline()

    edges = defaultdict(lambda : 0)
    mal = set()
    prog = tqdm()

    while line:
        line = line.split(',')
        src = line[SRC_IP]
        dst = line[DST_IP]
        y = int(line[LABEL][:-1])

        edges[(src,dst)] += 1
        if y:
            mal.add((src,dst))

        line = f.readline()
        prog.update()

    f.close()
    prog.close()

    return edges,mal

def combine_files(edges, mals):
    mals = set().union(*mals)

    all_edges = defaultdict(lambda : 0)
    node_map = dict()

    def get_or_add(n):
        n = n.replace('\ufeff', '')
        if (nid := node_map.get(n)) is None:
            nid = len(node_map)
            node_map[n] = nid

        return nid

    for e in edges:
        for k,v in e.items():
            s = get_or_add(k[0])
            d = get_or_add(k[1])
            all_edges[(s,d)] += v

    feats = []
    for ip in node_map.keys():
        octets = [int(i) for i in ip.split('.')]
        feats.append(octets)
    feats = torch.tensor(feats)

    ei,ew,y = [],[],[]
    for e,v in all_edges.items():
        ei.append(list(e))
        ew.append(v)

        if e in mals:
            y.append(1)
        else:
            y.append(0)

    ei = torch.tensor(ei).T
    ew = torch.tensor(ew)
    y = torch.tensor(y)

    return Data(
        x=feats, num_nodes=feats.size(0),
        edge_index=ei, edge_attr=ew, label=y
    )

def partition(g):
    mal = g.label == 1
    te = g.edge_index[:, mal]
    clean = (~te).nonzero().squeeze()

    ei = g.edge_index

    # 80 / 10 / 10 split
    idxs = torch.randperm(clean.size(0))
    tr = ei[:, idxs[:int(clean.size(0) * 0.8)]]
    va = ei[:, idxs[int(clean.size(0) * 0.8) : int(clean.size(0) * 0.9)]]
    te_clean = ei[:, idxs[int(clean.size(0) * 0.9):]]
    te = torch.cat([te, te_clean], dim=1)

    tr_g = deepcopy(g)
    tr_g.edge_index = tr

    va_g = deepcopy(g)
    va_g.edge_index = va

    te_g = deepcopy(g)
    te_g.edge_index = te

    return tr_g, va_g, te_g

if __name__ == '__main__':
    out = Parallel(n_jobs=1, prefer='processes')(
        delayed(read_one)(i) for i in range(1,5)
    )

    edges,mals = zip(*out)
    g = combine_files(edges,mals)

    tr,va,te = partition(g)
    [
        torch.save(f'data/unsw_{f}.pt',fname)
        for f,fname in
        [(tr,'tr'), (va,'va'), (te,'te')]
    ]
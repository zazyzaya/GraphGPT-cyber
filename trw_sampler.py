import torch
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected, remove_self_loops
from temporal_walks import temporal_rw
from joblib import Parallel, delayed

from models.gnn_bert import GNNEmbedding

class TRWSampler():
    def __init__(self, data: Data, walk_len=64, batch_size=64, device='cpu'):
        self.x = data.x
        self.rowptr = data.idxptr.to(device)
        self.col = data.col.to(device)
        self.ts = data.ts.to(device)
        self.data = data

        self.num_nodes = data.x.size(0)
        self.walk_len = walk_len
        self.batch_size = batch_size
        self.device = device

        self.min_ts = None
        self.max_ts = None

    def rw(self, batch, n_walks=1, min_ts=None, max_ts=None, reverse=False, trim_missing=True):
        batch = batch.repeat(n_walks)

        walks,eids = temporal_rw(
            self.rowptr, self.col, self.ts, batch.to(self.device),
            self.walk_len, min_ts=min_ts, max_ts=max_ts, reverse=reverse, return_edge_indices=True
        )

        pad = eids == -1
        whole_col = ~torch.prod(pad, dim=0, dtype=torch.bool)
        whole_row = ~torch.prod(pad, dim=1, dtype=torch.bool)

        walks[:, 1:][pad] = GNNEmbedding.PAD
        whole_col = torch.cat([torch.tensor([True], device=whole_col.device), whole_col])

        # If no walks went to full walk_len, trim them down to save mem
        if trim_missing:
            walks = walks[:, whole_col]
            walks = walks[whole_row]

        if reverse:
            walks = walks.flip(1)

        return walks

    def __iter__(self):
        batches = torch.randperm(self.num_nodes).split(self.batch_size)
        for b in batches:
            yield self.rw(b, min_ts=self.min_ts, max_ts=self.max_ts)

    def edge_iter(self, shuffle=True, return_index=False):
        if shuffle:
            batches = torch.randperm(self.col.size(0), device=self.device).split(self.batch_size)
        else:
            batches = torch.arange(self.col.size(0), device=self.device).split(self.batch_size)

        for b in batches:
            # Keep in ascending order so ts and idxptr are still in proper order
            # should be relatively fast since batch-size is fairly low
            if shuffle:
                b = b.sort().values

            dst = self.col[b]
            ts = self.ts[b]

            # Using binary search is O(n log n), perfect parallelism gets to O(log n)
            # Maybe overhead will be too high?
            '''
            src = Parallel(n_jobs=16, prefer='processes')(
                delayed(find_src)(b_, self.rowptr) for b_ in b
            )
            '''
            # Overhead is too high. Goes to abt 15 mins w/ threads, unclear how long w procs (too much memcopy)

            # Worst case O(|V| + |b|) -> O(n)
            src = []
            cur_src = 0
            cur_max = self.rowptr[cur_src+1]
            for b_ in b:
                while cur_max < b_:
                    cur_src += 1
                    cur_max  = self.rowptr[cur_src+1]
                src.append(cur_src)

            src = torch.tensor(src, device=self.device)

            if return_index:
                yield src,dst,ts,b
            else:
                yield src,dst,ts

    def add_edge_index(self):
        if not hasattr(self.data, 'edge_index'):
            src = torch.arange(self.rowptr.size(0)-1, device=self.col.device)
            deg = self.rowptr[1:] - self.rowptr[:-1]
            src = src.repeat_interleave(deg)

            ei = torch.stack([src, self.col])
            ei,ew = ei.unique(dim=1, return_counts=True)
            self.edge_index = ei
        else:
            self.edge_index = self.data.edge_index.to(self.col.device)

def find_src(col_idx, idxptr):
    st = 0
    en = idxptr.size(0)-1
    while (en-st > 1):
        mid = st + ((en-st) // 2)
        if idxptr[mid] > col_idx:
            en = mid
        else:
            st = mid
    if idxptr[st] > col_idx:
        return st-1
    else:
        return st


class RWSampler(TRWSampler):
    def rw(self, batch, p=1, q=1, reverse=False):
        batch = batch.repeat(self.n_walks)
        walks,eids = torch.ops.torch_cluster.random_walk(
            self.rowptr, self.col, batch.to(self.device),
            self.walk_len, p, q
        )

        pad = eids == -1
        pad[:, 0] = False
        walks[:, 1:][pad] = GNNEmbedding.PAD

        # If no walks went to full walk_len, trim them down to save mem
        whole_col = ~torch.prod(pad, dim=0, dtype=torch.bool)
        whole_col = torch.cat([torch.tensor([True], device=whole_col.device), whole_col])
        walks = walks[:, whole_col]

        if reverse:
            walks = walks.flip(1)

        return walks
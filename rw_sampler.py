import torch
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected, remove_self_loops
from torch_cluster import random_walk

from models.gnn_bert import GNNEmbedding

class RWSampler():
    def __init__(self, data: Data, walk_len=64, n_walks=4, batch_size=64, device='cpu'):
        self.data = data
        self.x = data.x
        self.edge_index = data.edge_index

        self.num_nodes = data.x.size(0)
        self.walk_len = walk_len
        self.n_walks = n_walks
        self.batch_size = batch_size
        self.device = device

        self._preprocess_edges()

    def _preprocess_edges(self):
        ei = self.data.edge_index
        ei = remove_self_loops(ei)[0]
        ei = to_undirected(ei)
        self._set_csr(ei)

    def _set_csr(self, ei):
        # Essentially the sorting that happens in torch_cluster.random_walk:rw
        # but only do it one time when object is initiated to save time
        row,col = ei
        perm = torch.argsort(row * self.num_nodes + col)
        row, col = row[perm], col[perm]

        deg = row.new_zeros(self.num_nodes)
        deg.scatter_add_(0, row, torch.ones_like(row))
        rowptr = row.new_zeros(self.num_nodes + 1)
        torch.cumsum(deg, 0, out=rowptr[1:])

        self.col = col.to(self.device)
        self.rowptr = rowptr.to(self.device)

    def rw(self, batch, p=1, q=1):
        batch = batch.repeat(self.n_walks).to(self.device)
        walks,eids = torch.ops.torch_cluster.random_walk(
            self.rowptr, self.col, batch,
            self.walk_len, p, q
        )

        pad = eids == -1
        pad[:, 0] = False
        walks[:, 1:][pad] = GNNEmbedding.PAD

        # If no walks went to full walk_len, trim them down to save mem
        whole_col = ~torch.prod(pad, dim=0, dtype=torch.bool)
        walks[:, 1:] = walks[:, 1:][:, whole_col]

        return walks

    def __iter__(self):
        batches = torch.randperm(self.num_nodes)
        batches = batches.split(self.batch_size // self.n_walks)

        for b in batches:
            yield self.rw(b)
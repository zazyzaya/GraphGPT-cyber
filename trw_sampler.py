import torch
import torch_cluster # Use local version to include temporal rw in torch.ops
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected, remove_self_loops

from models.gnn_bert import GNNEmbedding

class TRWSampler():
    def __init__(self, data: Data, walk_len=64, n_walks=4, batch_size=64, device='cpu'):
        self.x = data.x
        self.rowptr = data.idxptr.to(device)
        self.col = data.col.to(device)
        self.ts = data.ts.to(device)

        self.num_nodes = data.x.size(0)
        self.walk_len = walk_len
        self.n_walks = n_walks
        self.batch_size = batch_size
        self.device = device

    def rw(self, batch, p=1, q=1):
        batch = batch.repeat(self.n_walks)
        walks,eids = torch.ops.torch_cluster.temporal_random_walk(
            self.rowptr, self.col, self.ts, batch.to(self.device),
            self.walk_len, p, q
        )

        pad = eids == -1
        pad[:, 0] = False
        walks[:, 1:][pad] = GNNEmbedding.PAD

        # If no walks went to full walk_len, trim them down to save mem
        whole_col = ~torch.prod(pad, dim=0, dtype=torch.bool)
        whole_col = torch.cat([torch.tensor([True], device=whole_col.device), whole_col])
        walks = walks[:, whole_col]

        return walks

    def __iter__(self):
        batches = torch.randperm(self.num_nodes)
        batches = batches.split(self.batch_size // self.n_walks)

        for b in batches:
            yield self.rw(b)
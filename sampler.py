import torch
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.typing import SparseTensor

class SparseGraphSampler():
    '''
    Stripped down version of ShaDowKHopSeqMapDataset from
    https://github.com/alibaba/graph-gpt/blob/main/src/data/dataset_map.py#L214
    '''
    def __init__(self, data: Data, adj_t: SparseTensor=None, depth=1, neighbors=15, reindex=False, batch_size=64, mode='pretrain'):
        self.data = data
        self.x = data.x

        assert hasattr(data, "edge_index")
        if adj_t is None:
            row, col = data.edge_index.cpu()

            if hasattr(data, 'edge_attr'):
                value = data.edge_attr
            else:
                value=torch.arange(col.size(0)),

            adj_t = SparseTensor(
                row=row,
                col=col,
                value=value,
                sparse_sizes=(data.num_nodes, data.num_nodes),
            ).t()

        self.adj_t = adj_t
        self.depth = depth
        self.neighbors = neighbors
        self.reindex = reindex
        self.batch_size = batch_size
        self.is_finetuning = mode == 'finetune'

    def __getitem__(self, index):
        rowptr, col, _ = self.adj_t.csr()
        out = torch.ops.torch_sparse.ego_k_hop_sample_adj(
            rowptr,
            col,
            index,
            self.depth,
            self.neighbors,
            False
        )

        n_id = torch.cat([index, out[2]])
        n_id_unique = torch.unique(n_id)  # The output tensor is always sorted!

        # Reindex so ei is between 0 and M
        adj, e_id = self.adj_t.saint_subgraph(n_id_unique)
        row, col, _ = adj.t().coo()
        edge_index = torch.vstack([row, col])

        root_n_id_src = (n_id_unique == index[0]).nonzero(as_tuple=True)[0]
        root_n_id_dst = (n_id_unique == index[1]).nonzero(as_tuple=True)[0]
        root_n_id = torch.tensor([root_n_id_src, root_n_id_dst], dtype=torch.int64)

        if self.is_finetuning:
            edge_index = self._remove_target_edge(root_n_id_src, root_n_id_dst, edge_index)

        data = Data(num_nodes=n_id_unique.numel())
        data.root_n_id = root_n_id
        data.seed_node = index
        data.edge_index = edge_index
        data.nids = n_id_unique
        data.x = self.x[n_id_unique]

        return data

    def sample(self, eidx):
        '''
        Expects B x 2 batch of edges
        '''
        datas = []
        for e in eidx:
            data = self.__getitem__(e)
            data.edge_idx = e
            datas.append(data)

        return datas

    def __iter__(self):
        idx = torch.randperm(self.data.edge_index.size(1))
        idx = idx.split(self.batch_size)

        for batch in idx:
            eidx = self.data.edge_index[:, batch].T
            data = self.sample(eidx)
            yield data

    def __len__(self):
        return self.data.edge_index.size(1)

    def _remove_target_edge(self, src, dst, ei):
        to_remove = (ei[0] == src).logical_and(ei[1] == dst)

        # Bidirectional
        to_remove = to_remove.logical_or(
            (ei[1] == src).logical_and(ei[0] == dst)
        )

        return ei[:, ~to_remove]
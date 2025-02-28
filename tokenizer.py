from random import choice

import torch
import networkx as nx
from torch_geometric.utils import to_networkx

class Tokenizer():
    def __init__(self, num_nodes, num_nodetypes):
        self.num_nodetypes = num_nodetypes
        self.num_nodes = num_nodes

        self.SOS = num_nodes + num_nodetypes
        self.EOS = self.SOS + 1
        self.PAD = self.EOS + 1
        self.MASK = self.PAD + 1
        self.vocab_size = self.MASK+1

    def _tokenize_one(self, data):
        '''
        Expects data object of subgraph with attrs
            x: N x 2 matrix of nodetype and node id (uq to the type)
            ei: edge index
        '''
        g = to_networkx(data, to_undirected=True)
        if not nx.is_eulerian(g):
            ccs = [g.subgraph(g_).copy() for g_ in nx.connected_components(g)]
            g = choice(ccs)
            g = nx.eulerize(g)

        path_ = [p for p in nx.eulerian_path(g)]
        path = [p[0] for p in path_] + [path_[-1][1]]

        # TODO get rid of repeated attrs
        seq = data.x[path]
        seq[:, 1] += self.num_nodetypes
        seq = seq.view(-1).long()
        seq = torch.cat([torch.tensor([self.SOS]), seq, torch.tensor([self.EOS])])
        return seq

    def tokenize(self, subgraphs):
        seq_ls = [self._tokenize_one(sg) for sg in subgraphs]
        seqs = torch.full(
            (len(seq_ls), max([s.size(0) for s in seq_ls])),
            self.PAD
        )

        targets = torch.zeros(seqs.size(), dtype=torch.bool)
        for i, seq in enumerate(seq_ls):
            seqs[i, :seq.size(0)] = seq
            targets[i, :seq.size(0)] = True

        return seqs.T, targets.T
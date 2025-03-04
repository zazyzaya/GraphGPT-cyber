from random import choice

import torch
import networkx as nx
from torch_geometric.utils import to_networkx

class Tokenizer():
    def __init__(self, num_nodes, num_nodetypes):
        self.num_nodetypes = num_nodetypes
        self.num_nodes = num_nodes

        self.SOS = num_nodes + num_nodetypes + 1
        self.EOS = self.SOS + 1
        self.PAD = self.EOS + 1
        self.NEW_PATH = self.PAD + 1
        self.MASK = self.NEW_PATH + 1
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
            ccs = [nx.eulerize(cc) for cc in ccs]
        else:
            ccs = [g]

        seqs = []
        for cc in ccs:
            path_ = [p for p in nx.eulerian_path(cc)]
            path = [p[0] for p in path_] + [path_[-1][1]]

            # TODO get rid of repeated attrs
            seq = data.x[path]
            seq[:, 1] += self.num_nodetypes
            seq = seq.view(-1).long()
            seqs.append(seq)

        joined_seqs = [torch.tensor([self.SOS])]
        for seq in seqs:
            joined_seqs.append(seq)
            joined_seqs.append(torch.tensor([self.NEW_PATH]))

        joined_seqs = joined_seqs[:-1]
        joined_seqs.append(torch.tensor([self.EOS]))

        seq = torch.cat(joined_seqs)
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

    def _lp_tokenize_one(self, data):
        g = to_networkx(data, to_undirected=True)
        if not nx.is_eulerian(g):
            ccs = [g.subgraph(g_).copy() for g_ in nx.connected_components(g)]
            ccs = [nx.eulerize(cc) for cc in ccs]
        else:
            ccs = [g]

        seqs = []
        for cc in ccs:
            # Isolated node
            if len(cc) == 1:
                path = [n for n in cc.nodes]
            else:
                path_ = [p for p in nx.eulerian_path(cc)]
                path = [p[0] for p in path_] + [path_[-1][1]]

            # TODO get rid of repeated attrs
            seq = data.x[path]
            seq[:, 1] += self.num_nodetypes
            seq = seq.view(-1).long()
            seqs.append(seq)

        joined_seqs = [torch.tensor([self.SOS])]
        for seq in seqs:
            joined_seqs.append(seq)
            joined_seqs.append(torch.tensor([self.NEW_PATH]))

        joined_seqs = joined_seqs[:-1]
        joined_seqs.append(torch.tensor([self.EOS]))

        # Last 4 tokens are node identity embeddings of target
        # [ntype], [nid], [ntype], [nid]
        target = data.x[data.root_n_id].view(-1)
        joined_seqs.append(target)

        seq = torch.cat(joined_seqs)
        return seq

    def lp_tokenize(self, subgraphs):
        seq_ls = [self._lp_tokenize_one(sg) for sg in subgraphs]
        seqs = torch.full(
            (len(seq_ls), max([s.size(0) for s in seq_ls])),
            self.PAD
        )

        targets = []
        for i, seq in enumerate(seq_ls):
            seqs[i, :seq.size(0)] = seq
            targets.append([j + (seq.size(0)-4) for j in range(4)])

        targets = torch.tensor(targets)
        return seqs.T, targets.T
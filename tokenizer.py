from collections import defaultdict
from joblib import Parallel, delayed
import math
from random import randint

import torch
import networkx as nx
from torch_geometric.utils import to_networkx

class Tokenizer():
    def __init__(self, x, max_path_len=256):
        self.max_path_len = max_path_len

        max_feats = x.max(dim=0).values.long()
        feat_offset = [0]
        for i in range(max_feats.size(0)-1):
            feat_offset.append(feat_offset[-1] + max_feats[i] + 1)

        self.feat_offset = torch.tensor(feat_offset)
        self.max_feat_id = max_feats[-1] + feat_offset[-1]

        self.mask_rate = 0.15

        self.IDX = self.max_feat_id+1
        self.PAD = self.IDX+max_path_len + 1
        self.SOS = self.PAD + 1
        self.EOS = self.SOS + 1
        self.JUMP = self.EOS + 1
        self.MASK = self.JUMP +1

        self.vocab_size = self.MASK + 1

    def set_mask_rate(self, percent_done, fixed_rate=0.7, decay_type='poly'):
        # Use polynomial annealing as in GraphGPT PPA settings
        if decay_type == 'cos':
            anneal = math.cos( percent_done * (math.pi / 2) )
        elif decay_type == 'poly':
            anneal = 1 - (percent_done ** 2)

        self.mask_rate = anneal * fixed_rate

    def _mask_nodes(self, nids, seq):
        to_mask = (seq[:,0] == nids).sum(dim=0).bool()
        to_mask = to_mask.unsqueeze(-1).repeat(1, seq.size(1))
        to_mask[:, -1] = False # Don't mask special tokens like EOS and JUMP
        to_mask = to_mask.logical_and(seq != -1) # Dont mask ignored parts that won't be included

        tgt = seq[to_mask]
        predict_idx = to_mask.clone()

        # Don't actually mask 20%
        dont_mask = (to_mask == True).nonzero()
        idx = torch.randperm(dont_mask.size(0))[int(dont_mask.size(0) * 0.8):]
        dont_mask = dont_mask[idx]
        to_mask[dont_mask[:, 0], dont_mask[:, 1]] = False

        seq[to_mask] = self.MASK
        return seq,tgt,predict_idx

    def _tokenize_one(self, data, mask):
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
        feats = []
        for cc in ccs:
            path_ = [p for p in nx.eulerian_path(cc)]
            path = [p[0] for p in path_] + [path_[-1][1]]
            seqs.append(torch.tensor(path))

            x = data.x[path] + self.feat_offset
            feat = torch.cat([x, torch.full((len(path),1), -1)], dim=1)
            feat[-1, -1] = self.JUMP
            feats.append(feat)


        seq_join = torch.cat(seqs)
        feat_join = torch.cat(feats)
        tokens = torch.cat([seq_join.unsqueeze(-1), feat_join], dim=1).long()
        uq,cnt = seq_join.unique(return_counts=True)
        gets_features_at = [randint(0, c-1) for c in cnt] # At which occurrence do we append features

        appearences = defaultdict(lambda : 0)
        for i in range(tokens.size(0)):
            idx = tokens[i,0].item()

            if appearences[idx] != gets_features_at[idx]:
                tokens[i, 1:-1] = -1

            appearences[idx] += 1

        # Use path indexes
        offset = randint(0,self.max_path_len)
        tokens[:, 0] += offset
        tokens[:, 0] %= self.max_path_len
        tokens[:, 0] += self.IDX

        if mask:
            nids = tokens[:,0].unique()
            to_mask = torch.rand(nids.size()) < self.mask_rate
            nids = nids[to_mask].unsqueeze(-1)
            seq,tgt,tgt_idx = self._mask_nodes(nids, tokens)

        seq = tokens.view(-1)
        remove_dupes = seq != -1
        seq = seq[remove_dupes]
        seq[-1] = self.EOS

        if mask:
            return seq, tgt, tgt_idx.view(-1)[remove_dupes]
        return seq

    def tokenize_and_mask(self, subgraphs):
        out = Parallel(prefer='processes', n_jobs=16)(
            delayed(self._tokenize_one)(sg, True) for sg in subgraphs
        )

        seqs,tgts,tgt_idx = zip(*out)
        out_seqs = torch.full(
            (len(seqs), max([s.size(0) for s in seqs])),
            self.PAD
        )
        mask = torch.zeros(out_seqs.size(), dtype=torch.bool)

        for i, seq in enumerate(seqs):
            out_seqs[i, :seq.size(0)] = seq
            idx = tgt_idx[i]
            mask[i, :idx.size(0)] = idx

        out_seqs = out_seqs
        mask = mask
        return out_seqs, mask, torch.cat(tgts)



if __name__ == '__main__':
    from sampler import SparseGraphSampler
    g = torch.load('data/lanl_tr.pt', weights_only=False)
    g = SparseGraphSampler(g)
    samp = g.sample(torch.randint(0, g.x.size(0), (64,2)))
    t = Tokenizer(g.x)

    t.set_mask_rate(0)
    print(t.tokenize_and_mask(samp)[2].size())
    t.set_mask_rate(0.001)
    print(t.tokenize_and_mask(samp)[2].size())
    t.set_mask_rate(0.25)
    print(t.tokenize_and_mask(samp)[2].size())
    t.set_mask_rate(0.5)
    print(t.tokenize_and_mask(samp)[2].size())
    t.set_mask_rate(0.75)
    print(t.tokenize_and_mask(samp)[2].size())
    t.set_mask_rate(0.999)
    print(t.tokenize_and_mask(samp)[2].size())
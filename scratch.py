import torch
from trw_sampler import TRWSampler

g = torch.load('data/unsw_tgraph_csr.pt', weights_only=False)

g = TRWSampler(g, edge_features=True)
g.rw(torch.arange(50))
import torch
from types import SimpleNamespace
from transformers import BertConfig
from models.gnn_bert import RWBert, RWBertFT, GNNEmbedding

tr = torch.load(f'data/lanl_tgraph_tr.pt', weights_only=False)
params = SimpleNamespace(H=128, L=2, MINI_BS=512)
config = BertConfig(
    tr.x.size(0) + GNNEmbedding.OFFSET,
    hidden_size=         params.H,
    num_hidden_layers=   params.L,
    num_attention_heads= params.H // 64,
    intermediate_size=   params.H * 4,
    num_nodes = tr.x.size(0)
)

bert = RWBert(config).to(0)

walks = torch.randint(0,5, (10,6))
masks = torch.zeros(walks.size(), dtype=torch.bool)
masks[:, 2:4] = True 

print(
    bert.modified_fwd(walks, masks, None, None, skip_cls=True, return_loss=False).size()
)
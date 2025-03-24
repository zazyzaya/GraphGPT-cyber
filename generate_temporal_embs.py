from types import SimpleNamespace

import torch
from torch_geometric.data import Data
from torch_scatter import scatter_sum
from transformers import BertConfig
from tqdm import tqdm

from models.gnn_bert import RWBert, GNNEmbedding
from trw_sampler import TRWSampler, RWSampler

DEVICE = 0

def get_period_graph(idx, delta, tr):
    st = idx*delta
    en = (idx+1)*(delta)

    mask = (tr.ts >= st).logical_and(tr.ts < en)
    col = tr.col[mask]

    new_ptr = [0]
    for i in range(1, tr.idxptr.size(0)):
        st = tr.idxptr[i-1]; en = tr.idxptr[i]
        selected = mask[st:en].sum().item()
        new_ptr.append(new_ptr[-1] + selected)

    new_ptr = torch.tensor(new_ptr)
    data = Data(tr.x, idxptr=new_ptr, col=col, ts=tr.ts[mask])

    return data

@torch.no_grad()
def get_embs(batch, gt, model, out):
    samp = RWSampler(gt, n_walks=1, device=DEVICE)
    rws = samp.rw(batch)

    attn_mask = rws == -2
    zs = model.modified_fwd(rws, attn_mask, None, attn_mask, return_loss=False, skip_cls=True)
    zs = zs[:, 0].cpu()

    emb = scatter_sum(zs, batch.unsqueeze(-1), dim=0, out=out)
    return emb

if __name__ == '__main__':
    tr = torch.load('data/lanl_tgraph_tr.pt', weights_only=False)
    sd = torch.load(f'pretrained/rw_sampling/lanl/trw_bert_tiny.pt', weights_only=True)

    params = SimpleNamespace(H=128, L=2, MINI_BS=512)
    config = BertConfig(
        tr.x.size(0) + GNNEmbedding.OFFSET,
        hidden_size=         params.H,
        num_hidden_layers=   params.L,
        num_attention_heads= params.H // 64,
        intermediate_size=   params.H * 4,
        num_nodes = tr.x.size(0)
    )
    model = RWBert(config)
    model.load_state_dict(sd)
    model = model.to(DEVICE)
    model.eval()

    for i in range(58*24):
        g = get_period_graph(i, 60*60, tr)
        batch = torch.arange(tr.x.size(0)).repeat(128)
        emb = torch.zeros(tr.x.size(0), params.H)

        for b in tqdm(batch.split(128**2), desc=f'{i+1}/{58*24}'):
            get_embs(b, g, model, emb)

        torch.save(emb / 128, f'data/lanl_per_hour/{i}.pt')
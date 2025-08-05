import time

from joblib import Parallel, delayed
import numpy as np
import torch
from tqdm import tqdm

from fast_auc import fast_auc, fast_ap
from models.gnn_bert import RWBert, GNNEmbedding
from rw_sampler import TRWSampler

'''
Kind of ugly, but wanted to be able to use this in parts of the code other than snapshot_finetune.py
'''

class Evaluator():
    def __init__(self, walk_len, num_eval_iters=1, workers=16, eval_bs=2048, downsample=False, device='cpu', dataset='unsw', cls=False):
        self.WALK_LEN = walk_len
        self.NUM_EVAL_ITERS = num_eval_iters
        self.EVAL_BS = eval_bs
        self.DEVICE = device
        self.workers = workers
        self.DATASET = dataset
        self.cls = cls 

        self.downsample = downsample
        self.sample = self.sample_rw if not self.cls else self.sample_cls

    def sample_rw(self, tr, src,dst,ts, walk_len, edge_features=None):
        if walk_len > 1:
            rw = tr.rw(src, reverse=True, trim_missing=False)
        else:
            rw = src.unsqueeze(-1)

        if edge_features is not None:
            #mask = torch.tensor([GNNEmbedding.MASK], device=self.DEVICE).repeat(edge_features.size())
            #rw = torch.cat([rw, mask], dim=1)
            #dst = torch.cat([edge_features, dst.unsqueeze(-1)], dim=1).flatten()
            rw = torch.cat([rw, edge_features], dim=1)

        masks = torch.tensor([[GNNEmbedding.MASK]], device=self.DEVICE).repeat(rw.size(0),1)
        rw = torch.cat([rw,masks], dim=1)
        attn_mask = rw != GNNEmbedding.PAD

        return rw, rw==GNNEmbedding.MASK, dst, attn_mask
    
    def sample_cls(self, tr, src,dst,ts, walk_len, edge_features): 
        if walk_len > 0:
            rw = tr.rw(src, reverse=True, trim_missing=False)
        else:
            rw = src.unsqueeze(-1)

        if edge_features is not None:
            #mask = torch.tensor([GNNEmbedding.MASK], device=DEVICE).repeat(edge_features.size())
            #rw = torch.cat([rw, mask], dim=1)
            #dst = torch.cat([edge_features, dst.unsqueeze(-1)], dim=1).flatten()
            rw = torch.cat([rw, edge_features], dim=1)

        mask = torch.full((rw.size(0), 1), GNNEmbedding.MASK, device=rw.device)
        rw = torch.cat([rw,dst.unsqueeze(-1),mask], dim=1)
        attn_mask = rw != GNNEmbedding.PAD

        return rw, attn_mask, rw == GNNEmbedding.MASK

    @torch.no_grad()
    def parallel_eval(self, model, tr: TRWSampler, va: TRWSampler, percent=1):
        prog = tqdm(desc='TNs', total=(va.col.size(0) // self.EVAL_BS)*self.NUM_EVAL_ITERS)
        ranks = torch.zeros_like(va.col).float()

        def thread_job(pid, idx):
            # Try to spread the jobs use of the GPU evenly
            if pid < self.workers:
                time.sleep(0.1 * pid)

            samp = va._single_iter(idx, shuffled=False)
            if va.edge_features:
                src,dst,ts,ef = samp
            else:
                src,dst,ts = samp
                ef = None

            walk, mask, tgt, attn_mask = self.sample(tr, src,dst,ts, self.WALK_LEN, edge_features=ef)
            out = model.modified_fwd(walk, mask, tgt, attn_mask, return_loss=False).logits

            out = out[mask] # B x |V| 
            pred = (out[torch.arange(out.size(0)), tgt]).unsqueeze(-1)

            # Filter out other TPs 
            for i,s in enumerate(src): 
                neighbors = tr.filter_col[tr.filter_ptr[s]:tr.filter_ptr[s+1]]
                rels = tr.filter_rel[tr.filter_ptr[s]:tr.filter_ptr[s+1]]
                
                to_filter = neighbors[rels == ef[i].cpu()]
                out[i][to_filter] = float('-inf')

            rank = 1+(out > pred).sum(dim=1)
            ranks[idx] += rank
            
            prog.update()

            del out
            torch.cuda.empty_cache()

        idxs = torch.arange(va.col.size(0)).split(self.EVAL_BS)
        for _ in range(self.NUM_EVAL_ITERS):
            Parallel(n_jobs=self.workers, prefer='threads')(
                delayed(thread_job)(i,b)
                for i,b in enumerate(idxs)
            )

        prog.close()

        ranks /= self.NUM_EVAL_ITERS

        mmr = (1 / ranks).mean()
        hits_at_10 = (ranks <= 10).float().mean()
        hits_at_5 = (ranks <= 5).float().mean()
        hits_at_1 = (ranks == 1).float().mean()

        return mmr.item(), hits_at_10.item(), hits_at_5.item(), hits_at_1.item()


    @torch.no_grad()
    def get_metrics(self, tr,va,te, model):
        te.to(tr.device)
        mmr, ha10, ha5, ha1 = self.parallel_eval(model, tr, te)
        print('#'*20)
        print(f'TEST SCORES')
        print('#'*20)
        print(f"MMR: {mmr:0.4f}")
        print("Hits @ ")
        print(f"\t1:  {ha1:0.4f}")
        print(f"\t5:  {ha5:0.4f}")
        print(f"\t10: {ha10:0.4f}")
        print('#'*20)
        print()
        te.to('cpu')

        va.to(tr.device)
        vmmr, vha10, vha5, vha1 = self.parallel_eval(model, tr, va)
        print('#'*20)
        print(f'VAL SCORES')
        print('#'*20)
        print(f"MMR: {vmmr:0.4f}")
        print("Hits @ ")
        print(f"\t1:  {vha1:0.4f}")
        print(f"\t5:  {vha5:0.4f}")
        print(f"\t10: {vha10:0.4f}")
        va.to('cpu')

        return (vmmr, vha1, vha5, vha10), (mmr, ha1, ha5, ha10)
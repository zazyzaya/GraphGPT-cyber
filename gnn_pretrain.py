from argparse import ArgumentParser
import time
from types import SimpleNamespace

import torch
from torch.optim.adamw import AdamW
from torch.optim.lr_scheduler import LRScheduler
from transformers import BertConfig

from models.gnn_bert import GNNBert as BERT, GNNEmbedding
from rw_sampler import RWSampler
from tokenizer import RWTokenizer

WARMUP_T = 10 ** 6 # Tokens (originally 10**9)
TOTAL_T = 10 ** 8           #(originally 10**10)
FIXED_SMTP_RATE = 0.7

WALK_LEN = 64
N_WALKS = 16
MINI_BS = 256
BS = 1024

class Scheduler(LRScheduler):
    def get_lr(self):
        # Warmup period of 10 ** 8 tokens
        if self.last_epoch < WARMUP_T:
            return [group['initial_lr'] * (self.last_epoch / WARMUP_T)
                    for group in self.optimizer.param_groups]
        # Linear decay after that
        else:
            return [group['initial_lr'] * (1 - ((self.last_epoch-WARMUP_T)/(TOTAL_T-WARMUP_T)))
                    for group in self.optimizer.param_groups]

def minibatch(mb, model: BERT):
    walks,masks,targets,attn_mask = t.mask(mb)
    loss = model.modified_fwd(g.edge_index, walks, masks, targets, attn_mask)
    loss.backward()

    token_count = (walks != GNNEmbedding.PAD).sum()
    return loss, token_count

def train(g: RWSampler, model: BERT):
    opt = AdamW(
        model.parameters(), lr=3e-4,
        betas=(0.9, 0.95), eps=1e-8, weight_decay=0.1
    )
    sched = Scheduler(opt)
    next_checkpoint = 1

    with open(f'gnn_bertlog_{SIZE}.txt', 'w+') as f:
        pass

    updates = 1
    opt.zero_grad()
    st = time.time()
    steps = 0
    processed_tokens = 0

    e = 0
    while processed_tokens < TOTAL_T:
        for i,mb in enumerate(g):
            loss, tokens = minibatch(mb, model)
            steps += 1

            if steps * MINI_BS >= BS:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
                opt.step()
                sched.step()

                processed_tokens += tokens
                t.set_mask_rate(processed_tokens / TOTAL_T)

                updates += 1
                lr = sched.get_last_lr()[0]
                sched.last_epoch = processed_tokens
                opt.zero_grad()
                steps = 0

                en = time.time()

                # Log epoch
                with open(f'gnn_bertlog_{SIZE}.txt', 'a') as f:
                    f.write(f'{loss},{updates},{processed_tokens},{en-st}\n')

                print(f'[{updates}-{e}] {loss:0.6f} (lr: {lr:0.2e}, mask rate {t.mask_rate:0.4f} tokens: {processed_tokens:0.2e}, {en-st:0.2f}s)')


                st = time.time()


            if processed_tokens >= TOTAL_T:
                break

            if updates % 100 == 99:
                torch.save(
                    (model.state_dict()),
                    f'gnn_bert_{SIZE}.pt'
                )

            if processed_tokens > WARMUP_T*10*next_checkpoint:
                torch.save(
                    (model.state_dict()),
                    f'gnn_bert-{next_checkpoint}_{SIZE}.pt'
                )
                next_checkpoint += 1


        e += 1

    torch.save(
        model.state_dict(),
        f'gnn_bert_{SIZE}.pt'
    )



if __name__ == '__main__':
    arg = ArgumentParser()
    arg.add_argument('--size', default='NO INPUT')
    arg.add_argument('--device', type=int, default=0)
    args = arg.parse_args()

    SIZE = args.size
    DEVICE = args.device
    params = {
        'tiny': SimpleNamespace(H=128, L=2, MINI_BS=1024),
        'mini': SimpleNamespace(H=256, L=4, MINI_BS=1024),
        'med': SimpleNamespace(H=512, L=8, MINI_BS=256)
    }[SIZE]

    MINI_BS = params.MINI_BS

    g = torch.load('data/lanl_tr.pt', weights_only=False)
    g = RWSampler(g, walk_len=WALK_LEN, n_walks=N_WALKS, batch_size=MINI_BS)
    t = RWTokenizer(g.x)
    t.set_mask_rate(0)

    config = BertConfig(
        g.x.size(0) + GNNEmbedding.OFFSET,
        hidden_size=         params.H,
        num_hidden_layers=   params.L,
        num_attention_heads= params.H // 64,
        intermediate_size=   params.H * 4,
        num_nodes = g.x.size(0),
        gnn_layers = 2,
        device = DEVICE
    )

    model = BERT(config).to(DEVICE)
    train(g,model)

from argparse import ArgumentParser
import time
from types import SimpleNamespace

import torch
from torch.optim.adamw import AdamW
from torch.optim.lr_scheduler import LRScheduler
from transformers import BertConfig

from models.hugging_bert import GraphBertForMaskedLM as BERT
from sampler import SparseGraphSampler
from tokenizer import Tokenizer

WARMUP_T = 10 ** 9 # Tokens
TOTAL_T = 10 ** 10
FIXED_SMTP_RATE = 0.7

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
    walks,masks,targets = t.tokenize_and_mask(mb)
    attn_mask = (walks != t.PAD).float()
    loss = model.modified_fwd(walks, masks, targets, attn_mask)
    loss.backward()

    token_count = (walks != t.PAD).sum()
    return loss, token_count

def train(g: SparseGraphSampler, model: BERT):
    opt = AdamW(
        model.parameters(), lr=3e-4,
        betas=(0.9, 0.95), eps=1e-8, weight_decay=0.1
    )
    sched = Scheduler(opt)
    next_checkpoint = 1

    with open(f'bertlog_{SIZE}.txt', 'w+') as f:
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
                en = time.time()

                processed_tokens += tokens
                t.set_mask_rate(processed_tokens / TOTAL_T)

                updates += 1

                # Log epoch
                with open(f'bertlog_{SIZE}.txt', 'a') as f:
                    f.write(f'{loss},{updates},{processed_tokens},{en-st}\n')

                lr = sched.get_last_lr()[0]
                sched.last_epoch = processed_tokens

                print(f'[{updates}-{e}] {loss:0.6f} (lr: {lr:0.2e}, mask rate {t.mask_rate:0.4f} tokens: {processed_tokens:0.2e}, {en-st:0.2f}s)')
                del loss
                steps = 0

                st = time.time()
                opt.zero_grad()

            if processed_tokens >= TOTAL_T:
                break

            if updates % 100 == 99:
                torch.save(
                    (model.state_dict()),
                    f'bert_{SIZE}.pt'
                )

            if processed_tokens > WARMUP_T*next_checkpoint:
                torch.save(
                    (model.state_dict()),
                    f'bert-{next_checkpoint}_{SIZE}.pt'
                )
                next_checkpoint += 1


        e += 1

    torch.save(
        (model.args, model.kwargs, model.state_dict()),
        f'bert_{SIZE}.pt'
    )



if __name__ == '__main__':
    arg = ArgumentParser()
    arg.add_argument('--size', default='NO INPUT')
    arg.add_argument('--device', type=int, default=0)
    args = arg.parse_args()

    SIZE = args.size
    DEVICE = args.device

    g = torch.load('data/lanl_all_tr.pt', weights_only=False)
    g = SparseGraphSampler(g, batch_size=MINI_BS, neighbors=15)
    num_tokens = g.x.max().long() + 1
    t = Tokenizer(g.x)
    t.set_mask_rate(0)


    params = {
        'mini': SimpleNamespace(H=256, L=4, MINI_BS=512),
        'med': SimpleNamespace(H=512, L=8, MINI_BS=256)
    }[SIZE]

    MINI_BS = params.MINI_BS

    config = BertConfig(
        t.vocab_size,
        hidden_size=         params.H,
        num_hidden_layers=   params.L,
        num_attention_heads= params.H // 64,
        intermediate_size=   params.H * 4
    )

    model = BERT(config).to(DEVICE)
    train(g,model)

import time
import torch
from torch.optim.adamw import AdamW
from torch.optim.lr_scheduler import LRScheduler

from models.bert import BERT
from sampler import SparseGraphSampler
from tokenizer import Tokenizer

DEVICE = 2
WARMUP_T = 10 ** 8 # Tokens
TOTAL_T = 10 ** 9

MINI_BS =256
BS = 1024

class Scheduler(LRScheduler):
    def get_lr(self):
        # Warmup period of 10k steps
        if self.last_epoch < WARMUP_T:
            return [group['initial_lr'] * (self.last_epoch / WARMUP_T)
                    for group in self.optimizer.param_groups]
        # Linear decay after that
        else:
            return [group['initial_lr'] * (1 - ((self.last_epoch-WARMUP_T)/(TOTAL_T-WARMUP_T)))
                    for group in self.optimizer.param_groups]

def minibatch(mb, model: BERT):
    walks,masks,targets = t.tokenize_and_mask(mb)
    loss = model(walks, masks, targets)
    loss.backward()

    return loss, walks.numel()

def train(g: SparseGraphSampler, model: BERT):
    opt = AdamW(
        model.parameters(), lr=3e-4,
        betas=(0.9, 0.95), eps=1e-8, weight_decay=0.1
    )
    sched = Scheduler(opt)

    with open('bertlog.txt', 'w+') as f:
        pass

    updates = 0
    opt.zero_grad()
    st = time.time()
    steps = 0
    processed_tokens = 0

    e = 0
    while processed_tokens < TOTAL_T:
        for i,mb in enumerate(g):
            loss, tokens = minibatch(mb, model)
            processed_tokens += tokens
            steps += 1

            if steps * MINI_BS >= BS:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
                opt.step()
                sched.step()
                opt.zero_grad()

                en = time.time()

                # Log epoch
                with open('bertlog.txt', 'a') as f:
                    f.write(f'{loss},{updates},{processed_tokens}\n')

                lr = sched.get_last_lr()[0]
                sched.last_epoch = processed_tokens

                print(f'[{updates}-{e}] {loss} (lr: {lr:0.2e}, tokens: {processed_tokens:0.2e}, {en-st:0.2f}s)')

                updates += 1
                steps = 0

                if processed_tokens >= TOTAL_T:
                    break

                if updates % 100 == 0:
                    torch.save(
                        (model.args, model.kwargs, model.state_dict()),
                        'bert.pt'
                    )

                if updates % 10_000 == 0:
                    torch.save(
                        (model.args, model.kwargs, model.state_dict()),
                        f'bert-{updates//10_000}.pt'
                    )

                st = time.time()
        e += 1

    torch.save(
        (model.args, model.kwargs, model.state_dict()),
        'bert.pt'
    )



if __name__ == '__main__':
    g = torch.load('data/lanl_tr.pt', weights_only=False)
    g = SparseGraphSampler(g, batch_size=MINI_BS, neighbors=25)
    num_tokens = g.x.max().long() + 3 + 1
    t = Tokenizer(num_tokens, 3)

    # BERT Mini
    model = BERT(
        t.vocab_size,
        device=DEVICE,
        layers=4,
        hidden_size=256,
    )

    train(g,model)

import pandas as pd
import torch
from torch import nn
from torch.optim import Adam
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, NNConv
from torch_geometric.utils import add_remaining_self_loops
from sklearn.metrics import \
    roc_auc_score as auc_score, \
    average_precision_score as ap_score

EPOCHS = 100
DEVICE = 'cpu'

def squared_loss(margin, t): return (margin - t)** 2
def squared_hinge_loss(margin, t): return torch.max(margin - t, torch.zeros_like(t)) ** 2
def logistic_loss(margin, t): return torch.log(1+torch.log(-margin*t))

def _check_tensor_shape(inputs, shape=(-1, 1)):
    input_shape = inputs.shape
    target_shape = shape
    if len(input_shape) != len(target_shape):
        inputs = inputs.reshape(target_shape)
    return inputs

def _get_surrogate_loss(backend='squared_hinge'):
    if backend == 'squared_hinge':
       surr_loss = squared_hinge_loss
    elif backend == 'squared':
       surr_loss = squared_loss
    elif backend == 'logistic':
       surr_loss = logistic_loss
    else:
        raise ValueError('Out of options!')
    return surr_loss


class APLoss(torch.nn.Module):
    """AP Loss with squared-hinge function: a novel loss function to directly optimize AUPRC.

        Args:
            margin: margin for squred hinge loss, e.g., m in [0, 1]
            gamma: factors for moving average

        Return:
            loss value (scalar)

        Reference:
                Stochastic Optimization of Areas Under Precision-Recall Curves with Provable Convergence},
                Qi, Qi and Luo, Youzhi and Xu, Zhao and Ji, Shuiwang and Yang, Tianbao,
                Advances in Neural Information Processing Systems 2021.

        Notes: This version of AP loss reduces redundant computation for the original implementation by Qi Qi.
        In addition, it fixed a potential memory leaking issue related to 'u_all' and 'u_pos'. This version is contributed by Gang Li.
    """

    def __init__(self, pos_len=None, margin=1.0, gamma=0.99, surrogate_loss='squared_hinge', device=None):
        super(APLoss, self).__init__()
        if not device:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            # self.device = torch.device('cpu')

        else:
            self.device = device
        self.u_all = torch.tensor([0.0]*pos_len).reshape(-1, 1).to(self.device)#.cpu()
        self.u_pos = torch.tensor([0.0]*pos_len).reshape(-1, 1).to(self.device)#.cpu()
        self.margin = margin
        self.gamma = gamma
        self.surrogate_loss = _get_surrogate_loss(surrogate_loss)

    def forward(self, y_pred, y_true, index_p):
        CHECKMEORY=False
        y_pred = _check_tensor_shape(y_pred, (-1, 1))
        y_true = _check_tensor_shape(y_true, (-1, 1))
        index_p = _check_tensor_shape(index_p, (-1,))
        index_p = index_p[index_p>=0] # only need indices from positive samples

        pos_mask = (y_true == 1).flatten()
        f_ps = y_pred[pos_mask]
        mat_data = y_pred.flatten().repeat(len(f_ps), 1)

        sur_loss = self.surrogate_loss(self.margin, (f_ps - mat_data))#.detach() # memory leak here
        if CHECKMEORY: print(torch.cuda.max_memory_allocated())

        pos_sur_loss = sur_loss * pos_mask
        if CHECKMEORY: print(torch.cuda.max_memory_allocated())

        self.u_all[index_p] = (1 - self.gamma) * self.u_all[index_p] + self.gamma * (sur_loss.mean(1, keepdim=True)).detach() # memory leak here
        if CHECKMEORY: print(torch.cuda.max_memory_allocated())

        self.u_pos[index_p] = (1 - self.gamma) * self.u_pos[index_p] + self.gamma * (pos_sur_loss.mean(1, keepdim=True)).detach()
        if CHECKMEORY: print(torch.cuda.max_memory_allocated())

        p = (self.u_pos[index_p] - (self.u_all[index_p]) * pos_mask) / (self.u_all[index_p] ** 2) # size of p: len(f_ps)* len(y_pred)
        # p.detach_()
        loss = torch.mean(p * sur_loss)
        # loss = (p * sur_loss).mean()
        if CHECKMEORY: print(torch.cuda.max_memory_allocated())

        # del sur_loss, pos_sur_loss, p
        return loss

class GRU(nn.Module):
    def __init__(self, x_dim, h_dim, z_dim, hidden_units=1):
        super(GRU, self).__init__()

        self.rnn = nn.GRU(
            x_dim, h_dim, num_layers=hidden_units
        )

        self.drop = nn.Dropout(0.25)
        self.lin = nn.Linear(h_dim, z_dim)

        self.z_dim = z_dim

    def forward(self, xs, h0, include_h=False):
        xs = self.drop(xs)

        if isinstance(h0, type(None)):
            xs, h = self.rnn(xs)
        else:
            xs, h = self.rnn(xs, h0)

        if not include_h:
            return self.lin(xs)

        return self.lin(xs), h

class Argus(nn.Module):
    def __init__(self, in_dim, edge_dim, h_dim, z_dim, device):
        super().__init__()

        self.c1 = GCNConv(in_dim, h_dim).to(device)
        self.relu = nn.ReLU()
        self.c2 = GCNConv(h_dim, h_dim).to(device)
        self.drop = nn.Dropout(0.1)
        self.ac = nn.Tanh()
        self.c3 = GCNConv(h_dim, h_dim).to(device)
        nn4 = nn.Sequential(nn.Linear(edge_dim, 8, device=device), nn.ReLU(), # lanl: 3 or 10; optc: 5
                            nn.Linear(8, h_dim * h_dim, device=device))
        
        self.c4 = NNConv(h_dim, h_dim, nn4, aggr='mean').to(device)
        self.rnn = GRU(h_dim, h_dim, z_dim).to(device)

        self.device = device
        self.ap_loss = APLoss(pos_len=1238452, margin=0.8, gamma=0.1, surrogate_loss='squared', device=device)


    def forward(self, x, eis, eas):
        eis = [ei.to(self.device) for ei in eis]
        eas = [ea.to(self.device) for ea in eas]
        zs = []

        for i in range(len(eis)):
            ei = eis[i]; ea = eas[i]
            ei_self_loops = add_remaining_self_loops(ei)[0]

            z = self.c1(x.to(self.device), ei_self_loops)
            z = self.c2(z, ei_self_loops)
            z = self.relu(z)
            z = self.drop(z)
            z = self.c3(z, ei_self_loops)
            z = self.relu(z)
            z = self.drop(z)
            z = self.c4(z, ei, edge_attr=ea)
            z = self.ac(z)

            zs.append(z)

        zs = torch.stack(zs, dim=1)
        out = self.rnn(zs, None)

        return out 
    
    #Adjust the edge prediction score based on other edges sharing the same src (excluding the edge)
    def get_src_score(self, src, dst, preds):
        src_dict = {}
        for i in range(0, len(src)):
            k = int(src[i])
            if not k in src_dict:
                src_dict[k] = [float(preds[i])]
            else:
                src_dict[k].append(float(preds[i]))
        preds_src = []
        #weights for edge score and neighborhood score
        lambda1 = 0.5
        lambda2 = 0.5
        for i in range(0, len(src)):
            k = int(src[i])
            preds_src.append(lambda1 * float(preds[i]) + lambda2 * np.mean(src_dict[k]))
        return torch.tensor(preds_src)
    

    def decode(self, src, dst, z):
        return torch.sigmoid(
            (z[src] * z[dst]).sum(dim=1)
        )

    def calc_loss_argus(self, z, partition, nratio, device):
        tot_loss = torch.zeros(1).to(device)
        ns = self.module.data.get_negative_edges(partition, nratio)
        for i in range(len(z)):
            ps = self.module.data.ei_masked(partition, i)
            if ps.size(1) == 0:
                continue
            t_index = torch.arange(0, ps.shape[1], dtype=torch.int64).to(device).detach()
            pos_pred = self.decode(ps, z[i], False)
            neg_pred = self.decode(ns[i], z[i], False)

            tot_loss += self.ap_loss(
                torch.cat((pos_pred, neg_pred), 0),
                torch.cat((torch.ones(pos_pred.shape[0]),torch.zeros(neg_pred.shape[0])), 0).to(self.module.device).detach(),
                t_index)
        return tot_loss.true_divide(len(z))


def to_snapshots(g, ts=None):
    # Assumes graph has src and col already
    ei = torch.stack([g.src, g.col])
    
    # Normalize 
    g.raw_edge_attr = g.raw_edge_attr / g.raw_edge_attr.max(dim=0).values

    eis = []
    eas = []
    y = []
    for t in ts:
        mask = g.ts == t
        ei_t = ei[:, mask]
        ea_t = g.raw_edge_attr[mask]

        eis.append(ei_t)
        eas.append(ea_t)

        if 'label' in g.keys():
            label = g.label[mask]
            y.append(label)

    x = torch.eye(g.x.size(0))
    return Data(x=x, edge_index=eis, label=y, eas=eas)

def train(tr,va,te):
    model = Argus(tr.x.size(0), tr.eas[0].size(1), 128, 64, DEVICE)
    opt = Adam(model.parameters(), lr=0.01)
    bce = nn.BCEWithLogitsLoss()

    def calc_loss(zs, eis, grad=True):
        tot_loss = 0
        for i in range(len(eis)):
            z = zs[i]; ei = eis[i]

            if ei.size(1) == 0:
                continue

            pos = (z[ei[0]] * z[ei[1]]).sum(dim=1)
            neg = (
                z[torch.randint(ei[0].min(), ei[0].max(), (pos.size(0),), device=DEVICE)] *
                z[torch.randint(ei[1].min(), ei[1].max(), (pos.size(0),), device=DEVICE)]
            ).sum(dim=1)

            labels = torch.zeros(pos.size(0)*2, device=DEVICE)
            labels[pos.size(0):] = 1

            loss = bce.forward(
                torch.cat([pos,neg]),
                labels
            )
            tot_loss += loss

        return tot_loss / len(eis)

    best = (100,0,0)
    for e in range(EPOCHS):
        model.train()
        opt.zero_grad()

        zs = model.forward(tr.x, tr.edge_index, tr.eas)
        loss = calc_loss(zs, tr.edge_index)
        loss.backward()
        opt.step()

        print(f'[{e}] Loss: {loss.item():0.4f}')

        with torch.no_grad():
            model.eval()
            zs = model.forward(tr.x, tr.edge_index, tr.eas)
            va_loss = calc_loss(zs, va.edge_index, grad=False)
            print(f'\tVal loss: {va_loss:0.4f}')

            preds = []
            ys = []
            for i in range(len(te.edge_index)):
                pred = (
                    zs[i][te.edge_index[i][0]] *
                    zs[i][te.edge_index[i][1]]
                ).sum(dim=1)
                preds.append(pred)

            preds = torch.sigmoid(torch.cat(preds))
            y = torch.cat(te.label).clamp(0,1)

            auc = auc_score(y, preds.cpu())
            ap = ap_score(y, preds.cpu())
            print(f'\tTe AUC: {auc:0.4f}, AP: {ap:0.4f}', end='', flush=True)

            if va_loss < best[0]:
                best = (va_loss, auc, ap)
                print('*')
            else:
                print()

    print(f"Best: AUC {best[1]:0.4f}, AP {best[2]:0.4f}")
    return {'auc': best[1], 'ap': best[2]}

if __name__ == '__main__':
    tr = torch.load('data/unsw_tgraph_tr_raw.pt', weights_only=False)
    va = torch.load('data/unsw_tgraph_va_raw.pt', weights_only=False)
    te = torch.load('data/unsw_tgraph_te_raw.pt', weights_only=False)

    ts = tr.ts.unique()

    tr = to_snapshots(tr, ts)
    va = to_snapshots(va, ts)
    te = to_snapshots(te, ts)

    best = [train(tr,va,te) for _ in range(10)]
    df = pd.DataFrame(best)
    print(df.mean())
    print(df.sem())
    df.to_csv('argus_results.csv')
import gzip
from collections import defaultdict

from tqdm import tqdm
import torch
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected

TEST_START = 147_600
ATTACK_START = 150_885
FOURTEEN_DAYS = (60*60*24) * 14
TOTAL_EVENTS = 239_558_591
HOME_DIR = '/mnt/raid10/cyber_datasets/lanl'

def parse_auth():
    f = gzip.open(f'{HOME_DIR}/auth.txt.gz', 'rt')
    line = f.readline()

    TS = 0
    SRC_U = 1
    DST_U = 2
    SRC_C = 3
    DST_C = 4
    AUTH_TYPE = 5
    SUCCESS = 8
    # Other two feats are always the same

    def parse_src(src):
        usr,_ = src.split('@')
        usr = usr.replace('$', '')
        return usr

    redlog = gzip.open(f'{HOME_DIR}/redteam.txt.gz', 'rt')
    next_red = redlog.readline().split(',')

    out_f = open(f'{HOME_DIR}/processed/auth_all_tr.txt', 'w+')
    opened_test = False
    prog = tqdm(total=TOTAL_EVENTS)

    while line:
        tokens = line.split(',')
        if tokens[AUTH_TYPE] != 'NTLM':
            line = f.readline()
            continue

        if tokens[TS] == str(TEST_START) and not opened_test:
            opened_test = True
            out_f.close()
            out_f = open(f'{HOME_DIR}/processed/auth_all_te.txt', 'w+')

        src_u = parse_src(tokens[SRC_U])
        src_c = tokens[SRC_C]
        dst = tokens[DST_C]

        if not opened_test:
            out_f.write(
                ','.join([
                    src_u, src_c,
                    tokens[TS], tokens[SUCCESS][0]
                ]) + '\n' + ','.join([
                    src_c, dst,
                    tokens[TS], tokens[SUCCESS][0]
                ]) + '\n'
            )
        else:
            is_mal = '0'
            if next_red[0] == tokens[TS]:
                if (tokens[SRC_U] == next_red[1] and
                    tokens[SRC_C] == next_red[2] and
                    tokens[DST_C] == next_red[3][:-1]):

                    is_mal = '1'
                    next_red = redlog.readline().split(',')

            out_f.write(
                ','.join([
                    src_u, src_c,
                    tokens[TS], tokens[SUCCESS][0],
                    is_mal
                ]) + '\n' + ','.join([
                    src_c, dst,
                    tokens[TS], tokens[SUCCESS][0],
                    is_mal
                ]) + '\n'
            )

        line = f.readline()
        prog.update()

    out_f.close()
    prog.close()

def to_torch(partition='tr', dicts=None):
    f = open(f'{HOME_DIR}/processed/auth_all_{partition}.txt', 'r')

    if dicts is None:
        nid = dict()
        users = dict(); computers = dict(); other = dict()
    else:
        nid,users,computers,other = dicts

    edges = defaultdict(lambda : 0)
    labels = defaultdict(lambda : 0)

    def get_or_add(v, d):
        if (nid := d.get(v)) is None:
            nid = len(d)
            d[v] = nid
        return nid

    def sort_node(n):
        if n.startswith('U'):
            get_or_add(n, users)
        elif n.startswith('C'):
            get_or_add(n, computers)
        else:
            get_or_add(n, other)

        return get_or_add(n, nid)


    line = f.readline()
    prog = tqdm()
    while line:
        if partition == 'tr':
            src,dst,_ = line.split(',', 2)
        else:
            tokens = line.split(',')
            src = tokens[0]; dst = tokens[1]
            label = int(tokens[-1])

        # Ignore anonymous login
        if src.startswith("ANON"):
            line = f.readline()
            continue

        src = sort_node(src)
        dst = sort_node(dst)

        edges[(src,dst)] += 1

        if partition != 'tr':
            labels[(src,dst)] = max(labels[(src,dst)], label)

        prog.update()
        line = f.readline()

    prog.close()
    f.close()

    x = torch.zeros(len(nid), 2)
    for k,v in tqdm(nid.items()):
        if k.startswith('U'):
            x[v] = torch.tensor([0, users[k]])
        elif k.startswith('C'):
            x[v] = torch.tensor([1, computers[k]])
        else:
            x[v] = torch.tensor([2, other[k]])

    src,dst,weight,label = [],[],[],[]
    for (s,d),v in tqdm(edges.items()):
        src.append(s)
        dst.append(d)
        weight.append(v)

        if partition != 'tr':
            label.append(labels[(s,d)])

    ei = torch.tensor([src,dst], dtype=torch.long)
    ew = torch.tensor(weight)

    if partition == 'tr':
        ei, ew = to_undirected(ei, ew)
    else:
        label = torch.tensor(label)


    names = [k for k in nid.keys()]
    return Data(
        x=x, edge_index=ei,
        edge_attr=ew, num_nodes=x.size(0),
        label=label, names=names
    ), nid, users, computers, other

def full_to_tgraph(delta=60*60):
    nid = dict(); users = dict(); computers = dict(); other = dict()
    csr = defaultdict(lambda : [[],[],[]])

    def get_or_add(v, d):
        if (nid := d.get(v)) is None:
            nid = len(d)
            d[v] = nid
        return nid

    def sort_node(n):
        if n.startswith('U'):
            get_or_add(n, users)
        elif n.startswith('C'):
            get_or_add(n, computers)
        else:
            get_or_add(n, other)

        return get_or_add(n, nid)

    f = open(f'{HOME_DIR}/processed/auth_all_tr.txt', 'r')
    line = f.readline()
    prog = tqdm(desc='Train', total=2211245)
    while line:
        tokens = line.split(',')
        src = tokens[0]; dst = tokens[1]; ts = int(tokens[2])

        # Ignore anonymous login
        if src.startswith("ANON"):
            line = f.readline()
            continue

        src = sort_node(src)
        dst = sort_node(dst)

        # Needs to be bi-directional otherwise RW doesn't work
        # bc it's a bipartite graph of U -> C
        csr[src][0].append(dst)
        csr[src][1].append(ts)
        csr[dst][0].append(src)
        csr[dst][1].append(ts)

        prog.update()
        line = f.readline()

    prog.close()
    f.close()

    f = open(f'{HOME_DIR}/processed/auth_all_te.txt', 'r')
    line = f.readline()
    prog = tqdm(desc='Test', total=75657132)
    while line:
        tokens = line.split(',')
        src = tokens[0]; dst = tokens[1]; ts = int(tokens[2])
        label = int(tokens[-1])

        # Ignore anonymous login
        if src.startswith("ANON"):
            line = f.readline()
            continue

        src = sort_node(src)
        dst = sort_node(dst)

        csr[src][0].append(dst)
        csr[src][1].append(ts)
        csr[dst][0].append(src)
        csr[dst][1].append(ts)

        # Only store index of anomalous edges to save space
        if label:
            idx = len(csr[src][0])-1
            csr[src][2].append(idx)

        prog.update()
        line = f.readline()

    prog.close()
    f.close()

    # Do this at the end so all sections of the graph
    # agree on node mappings
    x = torch.zeros(len(nid), 2)
    for k,v in tqdm(nid.items(), desc='Features'):
        if k.startswith('U'):
            x[v] = torch.tensor([0, users[k]])
        elif k.startswith('C'):
            x[v] = torch.tensor([1, computers[k]])
        else:
            x[v] = torch.tensor([2, other[k]])

     # String repr of nodes (e.g. C123)
    names = [k for k in nid.keys()]

    idxptr = [0]
    col = []
    ts = []
    is_mal = []
    for i in tqdm(range(x.size(0))):
        neighbors,t,label = csr[i]
        col += neighbors
        ts += t

        if label:
            is_mal += [t_ + idxptr[-1] for t_ in t]

        idxptr.append(len(neighbors) + idxptr[-1])
        del csr[i]

    torch.save(
        Data(
            x = x,
            idxptr = torch.tensor(idxptr),
            col = torch.tensor(col),
            ts = torch.tensor(ts),
            is_mal = torch.tensor(is_mal),
            names = names
        ),
        'data/lanl_tgraph_csr.pt'
    )

def partition_tgraph():
    g = torch.load('data/lanl_tgraph_csr.pt', weights_only=False)

    idxs = torch.randperm(g.col.size(0))
    tr_end = int(idxs.size(0) * 0.8)
    va_end = int(idxs.size(0) * 0.9)

    tr = torch.zeros(g.col.size(0), dtype=torch.bool)
    va = torch.zeros_like(tr)
    te = torch.zeros_like(tr)

    tr[idxs[:tr_end]] = True
    va[idxs[tr_end:va_end]] = True
    te[idxs[va_end:]] = True

    # Mask out anomalies so they're all in test set
    tr[g.is_mal] = False
    va[g.is_mal] = False
    te[g.is_mal] = True

    # Generate new index pointer for subset of column that was selected
    def reindex(idxptr, subset_mask):
        new_ptr = [0]
        for i in range(1, idxptr.size(0)):
            st = idxptr[i-1]; en = idxptr[i]
            selected = subset_mask[st:en].sum().item()
            new_ptr.append(selected)

        return torch.tensor(new_ptr)

    for mask,name in [(tr, 'tr'), (va, 'va'), (te, 'te')]:
        new_ptr = reindex(g.idxptr, tr)
        data = Data(
            x = g.x,
            idxptr = new_ptr,
            col = g.col[mask],
            ts = g.ts[mask]
        )

        if name == 'te':
            label = torch.zeros(mask.size(0))
            label[g.is_mal] = 1
            label = label[mask]
            data.label = label

        torch.save(data, f'data/lanl_tgraph_{name}.pt')

def full_to_torch():
    nid = dict(); users = dict(); computers = dict(); other = dict()
    edges = defaultdict(lambda : 0); labels = defaultdict(lambda : 0)

    def get_or_add(v, d):
        if (nid := d.get(v)) is None:
            nid = len(d)
            d[v] = nid
        return nid

    def sort_node(n):
        if n.startswith('U'):
            get_or_add(n, users)
        elif n.startswith('C'):
            get_or_add(n, computers)
        else:
            get_or_add(n, other)

        return get_or_add(n, nid)

    f = open(f'{HOME_DIR}/processed/auth_all_tr.txt', 'r')
    line = f.readline()
    prog = tqdm(desc='Train')
    while line:
        src,dst,_ = line.split(',', 2)

        # Ignore anonymous login
        if src.startswith("ANON"):
            line = f.readline()
            continue

        src = sort_node(src)
        dst = sort_node(dst)

        edges[(src,dst)] += 1

        prog.update()
        line = f.readline()

    prog.close()
    f.close()

    f = open(f'{HOME_DIR}/processed/auth_all_te.txt', 'r')
    line = f.readline()
    prog = tqdm(desc='Test')
    while line:
        tokens = line.split(',')
        src = tokens[0]; dst = tokens[1]
        label = int(tokens[-1])

        # Ignore anonymous login
        if src.startswith("ANON"):
            line = f.readline()
            continue

        src = sort_node(src)
        dst = sort_node(dst)

        edges[(src,dst)] += 1
        labels[(src,dst)] = max(labels[(src,dst)], label)

        prog.update()
        line = f.readline()

    prog.close()
    f.close()

    x = torch.zeros(len(nid), 2)
    for k,v in tqdm(nid.items()):
        if k.startswith('U'):
            x[v] = torch.tensor([0, users[k]])
        elif k.startswith('C'):
            x[v] = torch.tensor([1, computers[k]])
        else:
            x[v] = torch.tensor([2, other[k]])

    src,dst,weight,label = [],[],[],[]
    for (s,d),v in tqdm(edges.items()):
        src.append(s)
        dst.append(d)
        weight.append(v)
        label.append(labels[(s,d)])

    ei = torch.tensor([src,dst], dtype=torch.long)
    ew = torch.tensor(weight)
    label = torch.tensor(label)

    names = [k for k in nid.keys()]
    return Data(
        x=x, edge_index=ei,
        edge_attr=ew, num_nodes=x.size(0),
        label=label, names=names
    )

def partition_full():
    torch.manual_seed(0)
    g = torch.load('data/lanl_full.pt', weights_only=False)

    anoms = g.label == 1
    anom_edges = g.edge_index[:, anoms]

    rest = g.edge_index[:, ~anoms]
    ew_rest = g.edge_attr[~anoms]
    idx = torch.randperm(rest.size(1))
    tr_idx = idx[:int(0.8 * idx.size(0))]
    va_idx = idx[int(0.8 * idx.size(0)):int(0.9 * idx.size(0))]
    te_idx = idx[int(0.9 * idx.size(0)):]

    labels = torch.cat([torch.zeros(te_idx.size(0)), torch.ones(anom_edges.size(1))])

    tr = Data(x=g.x, edge_index=rest[:, tr_idx], edge_attr=ew_rest[tr_idx])
    va = Data(x=g.x, edge_index=rest[:, va_idx], edge_attr=ew_rest[va_idx])
    te = Data(
        x=g.x,
        edge_index=torch.cat([rest[:, te_idx], anom_edges], dim=1),
        edge_attr=torch.cat([ew_rest[te_idx], g.edge_attr[anoms]]),
        label = labels
    )

    torch.save(tr, 'data/lanl_tr.pt')
    torch.save(va, 'data/lanl_va.pt')
    torch.save(te, 'data/lanl_te.pt')


def load_full_tr():
    full = torch.load('data/lanl_full.pt', weights_only=False)
    mapping = {n:i for i,n in enumerate(full.names)}
    x = full.x

    f = open(f'{HOME_DIR}/processed/auth_all_tr.txt', 'r')
    srcs,dsts,ts = [],[],[]

    line = f.readline()
    prog = tqdm(total=2211245)
    while line:
        src,dst,t,_ = line.split(',')

        # Ignore anonymous login
        if src.startswith("ANON"):
            line = f.readline()
            continue

        src = mapping[src]
        dst = mapping[dst]
        t = int(t)

        srcs.append(src)
        dsts.append(dst)
        ts.append(t)

        prog.update()
        line = f.readline()

    prog.close()
    f.close()

    g = Data(
        x = x,
        edge_index = torch.tensor([src,dst]),
        edge_attr = torch.tensor(ts)
    )

    torch.save(g, 'data/lanl_continuous_tgraph_tr.pt')

if __name__ == '__main__':
    full_to_tgraph()
    partition_tgraph()
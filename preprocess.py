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
HOME_DIR = '/mnt/raid10/lanl'

def parse_auth():
    f = gzip.open(f'{HOME_DIR}/auth.txt.gz', 'rt')
    line = f.readline()

    TS = 0
    SRC = 3
    DST = 4
    AUTH_TYPE = 5
    SUCCESS = 8
    # Other two feats are always the same

    def parse_src(src):
        usr,_ = src.split('@')
        usr = usr.replace('$', '')
        return usr

    redlog = gzip.open(f'{HOME_DIR}/redteam.txt.gz', 'rt')
    next_red = redlog.readline().split(',')

    out_f = open(f'{HOME_DIR}/processed/auth_cc_tr.txt', 'w+')
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
            out_f = open(f'{HOME_DIR}/processed/auth_cc_te.txt', 'w+')

        src = tokens[SRC]

        if not opened_test:
            out_f.write(
                ','.join([
                    src, tokens[DST],
                    tokens[TS], tokens[SUCCESS][0]]
                ) + '\n'
            )
        else:
            is_mal = 0
            if next_red[0] == tokens[TS]:
                if (tokens[SRC] == next_red[1] and
                    tokens[DST-1] == next_red[2] and
                    tokens[DST] == next_red[3][:-1]):

                    is_mal = 1
                    next_red = redlog.readline().split(',')

            out_f.write(
                ','.join([
                    src, tokens[DST],
                    tokens[TS], tokens[SUCCESS][0],
                    str(is_mal)
                ]) + '\n'
            )

        line = f.readline()
        prog.update()

    out_f.close()
    prog.close()

def to_torch(partition='tr'):
    f = open(f'{HOME_DIR}/processed/auth_cc_{partition}.txt', 'r')
    nid = dict()
    users = dict(); computers = dict(); other = dict()
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

    return Data(
        x=x, edge_index=ei,
        edge_attr=ew, num_nodes=x.size(0),
        label=label
    )

if __name__ == '__main__':
    parse_auth()
    g = to_torch('tr')
    torch.save(g, 'data/lanl_cc_tr.pt')
    g = to_torch('te')
    torch.save(g, 'data/lanl_cc_te.pt')
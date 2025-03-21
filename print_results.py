import glob
import pandas as pd

from argparse import ArgumentParser
ap = ArgumentParser()
ap.add_argument('--ap', action='store_true')
args = ap.parse_args()

if args.ap:
    IDX = 2
else:
    IDX = 1

files = glob.glob('results/rw/ft_results_trw_*.txt')

ap_results = {
    k: [0] * 11
    for k in ['tiny', 'mini', 'med', 'baseline']
}
auc_results = {
    k: [0] *11
    for k in ['tiny', 'mini', 'med', 'baseline']
}
no_ft_results = {
    k: [0] *11
    for k in ['tiny', 'mini', 'med', 'baseline']
}

for f in files:
    tokens = f.split('_')
    size = tokens[4]
    wl = int(tokens[-1].replace('.txt','').replace('wl',''))

    with open(f, 'r') as f_:
        data = f_.read()

    lines = data.split('\n')
    lines = [
        [float(d) for d in line.split(',')]
        for line in lines[1:-1]
    ]

    no_ft_results[size][wl] = lines[0][IDX]

    best_auc = [0]; best_ap = [0]
    for line in lines:
        if line[3] > best_auc[0]:
            best_auc = (line[3], line[1], line[2])

        if line[4] > best_ap[0]:
            best_ap = (line[4], line[1], line[2])

    auc_results[size][wl] = best_auc[IDX]
    ap_results[size][wl] = best_ap[IDX]

nft = pd.DataFrame(no_ft_results)
auc = pd.DataFrame(auc_results)
ap = pd.DataFrame(ap_results)

print(nft.to_csv())
print(auc.to_csv())
print(ap.to_csv())

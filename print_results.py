import glob
import pandas as pd

from argparse import ArgumentParser
ap = ArgumentParser()
ap.add_argument('--ap', action='store_true')
args = ap.parse_args()

if args.ap:
    IDX = 3
else:
    IDX = 2

files = glob.glob('results/rw/unsw/snapshot-ft_results_snapshot_bert_*.txt')

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
    size = tokens[-2]
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
        if line[1] % 500 == 0:
            continue
        if line[4] > best_auc[0]:
            best_auc = (line[4], line[2], line[3])

        if line[5] > best_ap[0]:
            best_ap = (line[5], line[2], line[3])

    auc_results[size][wl] = best_auc[IDX-1]
    ap_results[size][wl] = best_ap[IDX-1]

nft = pd.DataFrame(no_ft_results)
auc = pd.DataFrame(auc_results)
ap = pd.DataFrame(ap_results)

print("No fine tuning")
print(nft.to_csv())

print("\nBest Val AUC")
print(auc.to_csv())

print("\nBest Val AP")
print(ap.to_csv())

import glob
import pandas as pd

from argparse import ArgumentParser
ap = ArgumentParser()
ap.add_argument('--ap', action='store_true')
ap.add_argument('--ft', action='store_true')
ap.add_argument('--dataset', default='optc')
args = ap.parse_args()

AUC = 2
AP = 3

folder = 'ft' if args.ft else 'trw'
files = glob.glob(f'results/trw/lanl14/snapshot-ft_results_bi_snapshot_bert*.txt')

keys = ['tiny', 'mini', 'med', 'baseline']
all_keys = []
for k in keys:
    all_keys.append(k + '_auc')
    all_keys.append(k + '_ap')

ap_results = {
    k: dict()
    for k in all_keys
}
auc_results = {
    k: dict()
    for k in all_keys
}
no_ft_results = {
    k: dict()
    for k in all_keys
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

    no_ft_results[size + '_auc'][wl] = lines[0][AUC]
    no_ft_results[size + '_ap'][wl] = lines[0][AP]

    best_auc = (0,0,0); best_ap = (0,0,0)
    for line in lines:
        if line[1] % 500 == 0:
            continue
        if line[4] > best_auc[0]:
            best_auc = (line[4], line[2], line[3])

        if line[5] > best_ap[0]:
            best_ap = (line[5], line[2], line[3])

    auc_results[size + '_auc'][wl] = best_auc[1]
    ap_results[size + '_auc'][wl] = best_ap[1]
    auc_results[size + '_ap'][wl] = best_auc[2]
    ap_results[size + '_ap'][wl] = best_ap[2]

nft = pd.DataFrame(no_ft_results)
auc = pd.DataFrame(auc_results)
ap = pd.DataFrame(ap_results)

print("No fine tuning")
print(nft.to_csv())

print("\nBest Val AUC")
print(auc.to_csv())

print("\nBest Val AP")
print(ap.to_csv())

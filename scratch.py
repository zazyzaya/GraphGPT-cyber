import torch

def find_src(col_idx, idxptr):
    st = 0
    en = idxptr.size(0)-1
    while (en-st > 1):
        mid = st + ((en-st) // 2)
        if idxptr[mid] > col_idx:
            en = mid
        else:
            st = mid
    if idxptr[st] > col_idx:
        return st-1
    else:
        return st

idx = torch.tensor([ 0,  3,  7, 20, 20, 21, 22])
dst = 21
src = find_src(dst, idx)
print(dst, ' in [', idx[src], '-', idx[src+1], ']')
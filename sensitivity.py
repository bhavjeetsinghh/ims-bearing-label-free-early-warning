import sys, json, numpy as np, pandas as pd
from scipy import stats
s = pd.read_csv(sys.argv[1], parse_dates=['t']); R = json.load(open(sys.argv[2]))
mu = R['base_rms_mean']; n, p = R['n_base'], 4
ucl = lambda a: p*(n+1)*(n-1)/(n*(n-p))*stats.f.ppf(1-a, p, n-p)
def persist(f, k):
    f = np.asarray(f, bool); c = 0; out = np.zeros(len(f), bool)
    for i, x in enumerate(f): c = c+1 if x else 0; out[i] = c >= k
    return out
mon = (s.h >= 72).values
def ev(score, thr, k, amb):
    al = persist(score > thr, k) & mon
    ref = (s.h >= 72) & (s.ttg > amb); rising = al & ~np.r_[False, al[:-1]]
    ep = int((rising & ref.values).sum()); hrs = s.h[ref].iloc[-1] - s.h[ref].iloc[0]
    idx = np.where(al)[0]; j = len(idx)-1
    while j > 0 and idx[j]-idx[j-1] <= 6: j -= 1
    return ep, round(1000*ep/hrs, 1), round(float(s.ttg[idx[j]]), 1)
out = {}
for amb in (72, 168, 240):
    for k in (1, 3, 5):
        out[f'amb{amb}_k{k}'] = {'D1_2x': ev(s.rms.values, 2*mu, k, amb), 'D1_1.2x': ev(s.rms.values, 1.2*mu, k, amb),
                                 'D2': ev(s.rms.values, mu+3*R['base_rms_sd'], k, amb), 'T2_1e-4': ev(s.T2.values, ucl(1e-4), k, amb),
                                 'T2_1e-3': ev(s.T2.values, ucl(1e-3), k, amb)}
for k, v in out.items(): print(k, v)
json.dump(out, open(sys.argv[3], 'w'), indent=1)

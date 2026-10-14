"""Label-free evaluation across all documented IMS failures.
Usage: python multi.py test1.csv test2.csv test3.csv out_dir
Inputs are long-format feature tables from extract_features.py.
Clock: operating hours = elapsed time with each recording pause (>15 min) counted as one nominal 10-min interval,
because the IMS documentation states that longer gaps mark resumption of the experiment on the next working day.
All protocol settings are those fixed on Test 3 (run-in 24 h, baseline 24-72 h, ambiguous final 168 h, k = 3).
"""
import sys, json, numpy as np, pandas as pd
from scipy import stats

f1, f2, f3, out = sys.argv[1:5]
RUNIN, BASE_END, AMBIG, K, MIN_REF_H = 24.0, 72.0, 168.0, 3, 100.0
FAILED = {('T1', 3): 'inner race', ('T1', 4): 'rolling element', ('T2', 1): 'outer race', ('T3', 3): 'outer race'}

def load(path, run):
    d = pd.read_csv(path, parse_dates=['timestamp']).sort_values(['timestamp', 'channel'])
    # invalid trailing snapshots: every channel below 10% of that channel's median RMS
    med = d.groupby('channel').rms.median()
    d['dead'] = d.rms < 0.1 * d.channel.map(med)
    dead_files = d.groupby('filename').dead.all()
    bad = []
    for fn in sorted(dead_files.index, reverse=True):          # strip from the end only
        if dead_files[fn]: bad.append(fn)
        else: break
    d = d[~d.filename.isin(bad)].copy()
    ts = pd.Series(sorted(d.timestamp.unique()))
    dt = ts.diff().dt.total_seconds().fillna(0) / 3600
    op = np.where(dt > 0.25, 10 / 60, dt).cumsum()
    el = (ts - ts.iloc[0]).dt.total_seconds() / 3600
    clock = pd.DataFrame({'timestamp': ts, 'h': op, 'h_el': el})
    d = d.merge(clock, on='timestamp')
    d['ttg'] = d.h.max() - d.h; d['ttg_el'] = d.h_el.max() - d.h_el
    d['crest'] = (d.peak2peak / 2) / d.rms
    info = dict(run=run, files=int(ts.size), excluded_dead=bad, op_hours=float(op[-1]), elapsed_hours=float(el.iloc[-1]),
                pauses=int((dt > 0.25).sum()), pause_hours=float(dt[dt > 0.25].sum()))
    return d, info

def persist(f, k=K):
    f = np.asarray(f, bool); c = 0; o = np.zeros(len(f), bool)
    for i, x in enumerate(f):
        c = c + 1 if x else 0; o[i] = c >= k
    return o

def evaluate(s, score, thr):
    mon = (s.h >= BASE_END).values
    al = persist(score > thr) & mon
    ref = ((s.h >= BASE_END) & (s.ttg > AMBIG)).values
    ref_h = float(s.h[ref].max() - s.h[ref].min()) if ref.any() else 0.0
    rising = al & ~np.r_[False, al[:-1]]
    ep = int((rising & ref).sum())
    fin = np.where(al & (s.ttg <= AMBIG).values)[0]
    idx = np.where(al)[0]; sus = None; sus_el = None
    if len(idx) and idx[-1] >= len(s) - 3:
        j = len(idx) - 1
        while j > 0 and s.h.iloc[idx[j]] - s.h.iloc[idx[j - 1]] <= 1.0 + 1e-9: j -= 1
        sus = float(s.ttg.iloc[idx[j]]); sus_el = float(s.ttg_el.iloc[idx[j]])
    return dict(ref_hours=ref_h, fa_assessable=ref_h >= MIN_REF_H, fa_episodes=ep if ref_h >= MIN_REF_H else None,
                fa_per_1000h=(1000 * ep / ref_h) if ref_h >= MIN_REF_H else None,
                first_final=float(s.ttg.iloc[fin[0]]) if len(fin) else None,
                first_final_time=str(s.timestamp.iloc[fin[0]]) if len(fin) else None,
                sustained=sus, sustained_elapsed=sus_el,
                sustained_start=str(s.timestamp.iloc[idx[j]]) if sus is not None else None)

runs = {}; res = []
for run, path in [('T1', f1), ('T2', f2), ('T3', f3)]:
    d, info = load(path, run); runs[run] = info
    for ch in sorted(d.channel.unique()):
        s = d[d.channel == ch].reset_index(drop=True)
        b = int(s.bearing.iloc[0]); base = ((s.h >= RUNIN) & (s.h < BASE_END)).values
        mu, sd = s.rms[base].mean(), s.rms[base].std()
        X = np.c_[np.log(s.rms), s.kurtosis_pearson, s.skewness, s.crest]
        Xb = X[base]; m0 = Xb.mean(0); Ci = np.linalg.inv(np.cov(Xb, rowvar=False)); n, p = Xb.shape
        T2 = np.einsum('ij,jk,ik->i', X - m0, Ci, X - m0)
        ucl = lambda a: p * (n + 1) * (n - 1) / (n * (n - p)) * stats.f.ppf(1 - a, p, n - p)
        row = dict(run=run, channel=int(ch), bearing=b, failed=FAILED.get((run, b)), n_base=int(base.sum()),
                   base_rms=float(mu), base_sd=float(sd), end_ratio=float(s.rms.iloc[-1] / mu),
                   max_ratio_final=float((s.rms / mu)[s.ttg <= AMBIG].max()))
        row['D1_2x'] = evaluate(s, s.rms.values, 2 * mu)
        row['D1_1.2x'] = evaluate(s, s.rms.values, 1.2 * mu)
        row['D2'] = evaluate(s, s.rms.values, mu + 3 * sd)
        row['D3'] = evaluate(s, s.kurtosis_pearson.values, 6.0)
        row['D4_1e-4'] = evaluate(s, T2, ucl(1e-4))
        row['D4_1e-3'] = evaluate(s, T2, ucl(1e-3))
        res.append(row)
        if run == 'T3' and ch == 3:
            s[['timestamp', 'h', 'h_el', 'ttg', 'ttg_el', 'rms', 'kurtosis_pearson']].assign(T2=T2).to_csv(f'{out}/t3b3_series.csv', index=False)
        if (run, b) in FAILED or run == 'T1':
            s[['timestamp', 'h', 'ttg', 'rms', 'kurtosis_pearson']].assign(T2=T2, ratio=s.rms / mu).to_csv(f'{out}/series_{run}_ch{ch}.csv', index=False)

# detection summary for failed bearings (Test 1 bearings: detected if either channel detects)
MIN_LEAD = 4.0
det = {}
for dn in ['D1_2x', 'D1_1.2x', 'D2', 'D3', 'D4_1e-4', 'D4_1e-3']:
    cases = []
    for (run, b), mode in FAILED.items():
        rows = [r for r in res if r['run'] == run and r['bearing'] == b]
        leads = [r[dn]['sustained'] for r in rows if r[dn]['sustained'] is not None]
        lead = max(leads) if leads else None
        cases.append(dict(run=run, bearing=b, mode=mode, lead=lead, detected=bool(lead is not None and lead >= MIN_LEAD)))
    k = sum(c['detected'] for c in cases); nn = len(cases)
    lo = stats.beta.ppf(0.025, k, nn - k + 1) if k > 0 else 0.0
    hi = stats.beta.ppf(0.975, k + 1, nn - k) if k < nn else 1.0
    det[dn] = dict(cases=cases, detected=k, n=nn, ci95=[float(lo), float(hi)])
json.dump(dict(runs=runs, results=res, detection=det, settings=dict(runin=RUNIN, base_end=BASE_END, ambig=AMBIG, k=K, min_lead=MIN_LEAD, min_ref_h=MIN_REF_H)),
          open(f'{out}/multi.json', 'w'), indent=1, default=str)
print(json.dumps(runs, indent=1, default=str))
fmt = lambda v: '-' if v is None else f'{v:.1f}'
for r in res:
    print(f"{r['run']} ch{r['channel']} B{r['bearing']} {('FAILED '+r['failed']) if r['failed'] else '':22s} base n={r['n_base']:3d} end×{r['end_ratio']:.2f} | " +
          ' | '.join(f"{dn}: FA {r[dn]['fa_episodes'] if r[dn]['fa_assessable'] else 'n/a'} first {fmt(r[dn]['first_final'])} sus {fmt(r[dn]['sustained'])}" for dn in ['D1_2x', 'D1_1.2x', 'D2', 'D3', 'D4_1e-4']))
for dn, v in det.items():
    print(dn, f"{v['detected']}/{v['n']} CI {v['ci95'][0]:.2f}-{v['ci95'][1]:.2f}", [(c['run'], c['bearing'], fmt(c['lead'])) for c in v['cases']])

"""Stage-1 re-analysis of NASA IMS Test 3 (Bearing 3).
Usage: python analysis.py features.csv out_dir
All thresholds are fitted on a causal baseline window only (hours 24-72).
"""
import sys, json, numpy as np, pandas as pd
from scipy import stats, integrate, optimize
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

csv, out = sys.argv[1], sys.argv[2]
R = {}
df = pd.read_csv(csv)
df['t'] = pd.to_datetime(df.filename, format='%Y.%m.%d.%H.%M.%S')
R['n_files'] = len(df)
dead = df.iloc[-1]
R['excluded_file'] = dead.filename
R['excluded_rms'] = float(dead.rms)
v = df.iloc[:-1].reset_index(drop=True).copy()
v['h'] = (v.t - v.t.iloc[0]).dt.total_seconds() / 3600
END = v.t.iloc[-1]
v['ttg'] = (END - v.t).dt.total_seconds() / 3600          # hours to end of recording
v['crest'] = (v.peak2peak / 2) / v.rms                     # crest-factor proxy (half peak-to-peak / RMS)
R['n_valid'] = len(v)
R['start'] = str(v.t.iloc[0]); R['end'] = str(END)
R['run_hours'] = float(v.h.iloc[-1])
gap = v.t.diff().dt.total_seconds() / 60
gaps = v.loc[gap > 15, ['filename']].assign(gap_h=(gap[gap > 15] / 60).round(2))
R['gaps'] = gaps.to_dict('records'); R['gap_total_h'] = float(((gap[gap > 15] - 10) / 60).sum())

# ---- feature redundancy -----------------------------------------------------
F = ['rms', 'kurtosis_pearson', 'skewness', 'variance', 'peak2peak']
R['pearson_with_rms'] = v[F].corr()['rms'].round(3).to_dict()
R['spearman_var_rms'] = float(stats.spearmanr(v.variance, v.rms)[0])
R['max_rel_diff_var_rms2'] = float(((v.variance - v.rms ** 2).abs() / v.rms ** 2).max())
R['ranges_valid'] = v[F].agg(['min', 'max']).round(4).to_dict()

# ---- windows -----------------------------------------------------------------
RUNIN_H, BASE_END_H, AMBIG_H = 24.0, 72.0, 168.0
base = (v.h >= RUNIN_H) & (v.h < BASE_END_H)
ref = (v.h >= BASE_END_H) & (v.ttg > AMBIG_H)              # healthy-reference zone (any alarm = false alarm)
ambig = v.ttg <= AMBIG_H
MON = (v.h >= BASE_END_H).values                           # monitoring starts once the baseline is complete
R['n_base'] = int(base.sum()); R['n_ref'] = int(ref.sum())
R['ref_hours'] = float(v.h[ref].iloc[-1] - v.h[ref].iloc[0])
R['day0_vs_base'] = {c: [float(v.loc[v.h < 24, c].mean()), float(v.loc[base, c].mean())] for c in ['rms', 'kurtosis_pearson', 'peak2peak']}
R['rms_drift_pct'] = float(100 * (v.loc[(v.ttg > 168) & (v.ttg <= 216), 'rms'].mean() / v.loc[base, 'rms'].mean() - 1))

def persist_runs(flag, k=3):
    """Return boolean array 'alarm active' = k consecutive exceedances (alarm raised on k-th)."""
    f = np.asarray(flag, bool); run = np.zeros(len(f), int); c = 0
    for i, x in enumerate(f):
        c = c + 1 if x else 0; run[i] = c
    return run >= k

def episodes(alarm, mask):
    """Count alarm episodes (rising edges) inside mask."""
    a = np.asarray(alarm, bool); m = np.asarray(mask, bool)
    rising = a & ~np.r_[False, a[:-1]]
    return int((rising & m).sum())

def evaluate(score, thr, k=3):
    al = persist_runs(score > thr, k) & MON
    ep_ref = episodes(al, ref)
    ep_amb = episodes(al, ambig)
    # lead time: start of the final continuous alarm block that persists (allowing <=1 h dropouts) to the end
    idx = np.where(al)[0]
    lead = None; first_any = None
    if len(idx):
        first_any = float(v.ttg[idx[0]])
        # final sustained alarm: walk back from end while gaps between alarm samples <= 6 samples
        j = len(idx) - 1
        while j > 0 and idx[j] - idx[j - 1] <= 6: j -= 1
        lead = float(v.ttg[idx[j]]) if idx[-1] >= len(v) - 3 else None
    fin = np.where(al & ambig.values)[0]
    first_final = float(v.ttg[fin[0]]) if len(fin) else None
    return dict(first_alarm_final7d_ttg=first_final, fa_episodes_ref=ep_ref, fa_per_1000h=1000 * ep_ref / R['ref_hours'],
                episodes_final_7d=ep_amb, first_alarm_ttg=first_any, sustained_lead_h=lead,
                first_alarm_time=None if first_any is None else str(v.t[idx[0]]))

mu, sd = v.loc[base, 'rms'].mean(), v.loc[base, 'rms'].std()
R['base_rms_mean'] = float(mu); R['base_rms_sd'] = float(sd)
det = {}
det['D1 RMS > 2x baseline'] = evaluate(v.rms.values, 2 * mu)
det['D2 RMS > mean+3sd'] = evaluate(v.rms.values, mu + 3 * sd)
det['D3 Kurtosis > 6'] = evaluate(v.kurtosis_pearson.values, 6.0)

# Hotelling T2 on standardized [log RMS, kurtosis, skewness, crest] (variance & p2p omitted: redundant with RMS)
X = np.c_[np.log(v.rms), v.kurtosis_pearson, v.skewness, v.crest]
Xb = X[base.values]; m0 = Xb.mean(0); C = np.cov(Xb, rowvar=False); Ci = np.linalg.inv(C)
d = X - m0; T2 = np.einsum('ij,jk,ik->i', d, Ci, d)
n, p = Xb.shape
ucl = p * (n + 1) * (n - 1) / (n * (n - p)) * stats.f.ppf(0.999, p, n - p)   # Phase-II UCL, alpha=0.001
R['T2_ucl'] = float(ucl)
det['D5 Hotelling T2 (alpha=0.001)'] = evaluate(T2, ucl)
v['T2'] = T2
R['detectors'] = det

# ---- threshold sweep: lead time vs false-alarm rate --------------------------
sweep = []
for r in np.round(np.arange(1.10, 3.01, 0.05), 2):
    e = evaluate(v.rms.values, r * mu); sweep.append(dict(detector='RMS ratio', thr=float(r), **e))
for q in [0.99, 0.995, 0.999, 0.9995, 0.9999, 0.99999]:
    u = p * (n + 1) * (n - 1) / (n * (n - p)) * stats.f.ppf(q, p, n - p)
    e = evaluate(T2, u); sweep.append(dict(detector='T2', thr=float(q), ucl=float(u), **e))
R['sweep'] = sweep

# ---- cross-bearing specificity with each bearing's own causal baseline -------
xb = {}
for b in ['b1_rms', 'b2_rms', 'b3_rms', 'b4_rms']:
    mb = v.loc[base, b].mean(); e = evaluate(v[b].values, 2 * mb); xb[b] = dict(base_mean=float(mb), **e)
R['cross_bearing'] = xb

# ---- leakage demonstration (reproduces the original design) ------------------
mu_o = v.rms.iloc[:5933].mean(); a = persist_runs(v.rms > 2 * mu_o); i1 = int(np.argmax(a)) - 2
y = np.zeros(len(v), int); y[i1:] = 1
tr = np.r_[0:5933, i1:i1 + 50]; te = np.r_[5933:i1, i1 + 50:len(v)]
leak = {}; preds = {}
for s, cols in {'A': F, 'B': ['skewness', 'variance', 'peak2peak']}.items():
    for mn, mdl in [('LR', LogisticRegression(class_weight='balanced', max_iter=2000)),
                    ('RF', RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=0))]:
        mdl.fit(v[cols].values[tr], y[tr]); pr = mdl.predict(v[cols].values[te]); preds[s + mn] = pr
        fp = (pr == 1) & (y[te] == 0)
        leak[s + mn] = dict(fp=int(fp.sum()), fn=int(((pr == 0) & (y[te] == 1)).sum()),
                            fp_first=str(v.t.iloc[te][fp].min()), fp_last=str(v.t.iloc[te][fp].max()))
leak['all_four_identical'] = bool(all((preds[k] == preds['ALR']).all() for k in preds))
lr = LogisticRegression(class_weight='balanced', max_iter=2000).fit(v[['rms']].values[tr], y[tr])
leak['implied_rms_threshold_g'] = float(-lr.intercept_[0] / lr.coef_[0][0])
leak['label_onset'] = str(v.t[i1]); leak['label_ttg'] = float(v.ttg[i1])
leak['test_healthy_from'] = str(v.t[5933])
R['leakage'] = leak

# ---- cost model: renewal-reward long-run cost rate ---------------------------
Cp, Cf, Cfa = 1500 + 5000 + 4 * 2000, 40000 + 24 * 2000, 2500
R['costs'] = dict(Cp=Cp, Cf=Cf, Cfa=Cfa)
from math import gamma
def eta_of(beta, mean): return mean / gamma(1 + 1 / beta)
def age_rate(Ta, beta, eta):
    Rl = np.exp(-(Ta / eta) ** beta)
    EL = integrate.quad(lambda t: np.exp(-(t / eta) ** beta), 0, Ta)[0]
    return (Cp * Rl + Cf * (1 - Rl)) / EL
def age_opt(beta, eta):
    res = optimize.minimize_scalar(lambda x: age_rate(x, beta, eta), bounds=(1, 5 * eta), method='bounded')
    return res.x, res.fun
def cbm_rate(beta, eta, pd_, lead, lam, cm=0.0):
    ET = eta * gamma(1 + 1 / beta); EL = ET - pd_ * lead
    return (pd_ * Cp + (1 - pd_) * Cf) / EL + Cfa * lam + cm
MEAN = R['run_hours']
cost = {}
for beta in [1.5, 2.0, 3.0]:
    eta = eta_of(beta, MEAN); Ta, ga = age_opt(beta, eta)
    rtf = Cf / MEAN
    cost[str(beta)] = dict(eta=eta, rtf_per_1000h=1000 * rtf, age_opt_T=Ta, age_opt_frac_mean=Ta / MEAN, age_per_1000h=1000 * ga)
    for name in ['D1 RMS > 2x baseline', 'D5 Hotelling T2 (alpha=0.001)']:
        dd = det[name]; lam = dd['fa_per_1000h'] / 1000; lead = dd['sustained_lead_h'] or 0
        cost[str(beta)][name] = {str(pdv): 1000 * cbm_rate(beta, eta, pdv, lead, lam) for pdv in [1.0, 0.95, 0.9, 0.8]}
    # break-even false-alarm rate (per 1000 h) vs detection probability, lead 24 h
    be = {}
    for pdv in [1.0, 0.95, 0.9, 0.85, 0.8, 0.75]:
        base_cbm = cbm_rate(beta, eta, pdv, 24.0, 0.0)
        be[str(pdv)] = 1000 * max(0.0, (ga - base_cbm) / Cfa)
    cost[str(beta)]['breakeven_fa_per_1000h_lead24'] = be
    cost[str(beta)]['max_monitoring_cost_per_h_D1_pd1'] = ga - cbm_rate(beta, eta, 1.0, det['D1 RMS > 2x baseline']['sustained_lead_h'], 0.0)
    # break-even p_d with zero false alarms
    f = lambda pdv: cbm_rate(beta, eta, pdv, 24.0, 0.0) - ga
    cost[str(beta)]['breakeven_pd_zeroFA'] = float(optimize.brentq(f, 0.0, 1.0)) if f(0) * f(1) < 0 else None
R['cost'] = cost

v[['filename', 't', 'h', 'ttg', 'rms', 'kurtosis_pearson', 'skewness', 'peak2peak', 'crest', 'b1_rms', 'b2_rms', 'b3_rms', 'b4_rms', 'T2']].to_csv(f'{out}/series.csv', index=False)
json.dump(R, open(f'{out}/results.json', 'w'), indent=1, default=str)
print(json.dumps({k: R[k] for k in R if k not in ('sweep',)}, indent=1, default=str))

"""Combine four-failure detection evidence with the renewal-reward cost model.
Usage: python decision.py results/results.json results/multi.json results/decision.json
"""
import sys, json, numpy as np
from math import gamma
from scipy import integrate, optimize
R = json.load(open(sys.argv[1])); M = json.load(open(sys.argv[2]))
Cp, Cf, Cfa = R['costs']['Cp'], R['costs']['Cf'], R['costs']['Cfa']
MEAN = R['run_hours']                     # operating-hour scale (Test 3)
def eta_of(b): return MEAN / gamma(1 + 1 / b)
def age_opt(b):
    eta = eta_of(b)
    f = lambda T: (Cp * np.exp(-(T / eta) ** b) + Cf * (1 - np.exp(-(T / eta) ** b))) / integrate.quad(lambda t: np.exp(-(t / eta) ** b), 0, T)[0]
    r = optimize.minimize_scalar(f, bounds=(1, 5 * eta), method='bounded'); return r.fun
def cbm(pd_, lead, lam): return (pd_ * Cp + (1 - pd_) * Cf) / (MEAN - pd_ * lead) + lam * Cfa
res = M['results']; out = {}
for dn in ['D1_2x', 'D1_1.2x', 'D2', 'D3', 'D4_1e-4']:
    d = M['detection'][dn]
    ep = sum(r[dn]['fa_episodes'] for r in res if r[dn]['fa_assessable'])
    hrs = sum(r[dn]['ref_hours'] for r in res if r[dn]['fa_assessable'])
    lam = ep / hrs
    leads = [c['lead'] for c in d['cases'] if c['detected']]
    lead_min = min(leads) if leads else 0.0; lead_med = float(np.median(leads)) if leads else 0.0
    nf = [r for r in res if not r['failed']]
    nf_alarm = sum(1 for r in nf if r[dn]['sustained'] is not None and r[dn]['sustained'] >= M['settings']['min_lead'])
    phat = d['detected'] / d['n']; lo = d['ci95'][0]
    row = dict(detected=d['detected'], n=d['n'], p_hat=phat, ci95=d['ci95'], lead_min=lead_min, lead_median=lead_med,
               fa_episodes=ep, ref_channel_hours=hrs, fa_per_1000h=1000 * lam, nonfailed_alarms=nf_alarm, nonfailed_n=len(nf),
               cost_per_1000h={})
    for b in (1.5, 2.0, 3.0):
        ga = age_opt(b)
        row['cost_per_1000h'][str(b)] = dict(age_opt=1000 * ga, cbm_at_phat=1000 * cbm(phat, lead_min, lam), cbm_at_lower=1000 * cbm(lo, lead_min, lam))
    out[dn] = row
json.dump(out, open(sys.argv[3], 'w'), indent=1)
for dn, r in out.items():
    c = r['cost_per_1000h']['2.0']
    print(f"{dn:8s} {r['detected']}/{r['n']} p̂={r['p_hat']:.2f} CI {r['ci95'][0]:.2f}-{r['ci95'][1]:.2f} lead min/med {r['lead_min']:.1f}/{r['lead_median']:.1f} FA {r['fa_episodes']} ({r['fa_per_1000h']:.1f}/1000h over {r['ref_channel_hours']:.0f} ch-h) nonfailed {r['nonfailed_alarms']}/{r['nonfailed_n']} | beta2 age {c['age_opt']:,.0f} cbm p̂ {c['cbm_at_phat']:,.0f} cbm low {c['cbm_at_lower']:,.0f}")

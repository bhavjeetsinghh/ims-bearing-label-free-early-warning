"""Revision analyses: bearing-level counts of warnings on non-failed bearings, run-level detection interval,
Poisson interval for the false-alarm rate, and cost scenarios for warnings on non-failed bearings.
Usage: python src/uncertainty_scenarios.py results/multi.json results/decision.json results/uncertainty.json"""
import sys, json
from scipy.stats import beta, chi2
MJ, DJ, OUT = sys.argv[1:4]
M = json.load(open(MJ)); D = json.load(open(DJ))
Cp, Cf, Cfa, MEAN = 14500.0, 88000.0, 2500.0, 1053.2
AGE = {b: D['D1_1.2x']['cost_per_1000h'][b]['age_opt'] for b in ('1.5', '2.0', '3.0')}
R = {}
# 1. non-failed bearings with a sustained warning of >= 4 h (any channel), per detector
nf = {}
for det in ['D1_2x', 'D1_1.2x', 'D2', 'D3', 'D4_1e-4']:
    bear, ch = set(), 0
    for r in M['results']:
        if r['failed'] is None and r[det]['sustained'] is not None and r[det]['sustained'] >= 4:
            bear.add((r['run'], r['bearing'])); ch += 1
    nf[det] = dict(bearings=len(bear), of_bearings=len({(r['run'], r['bearing']) for r in M['results'] if r['failed'] is None}), channels=ch,
                   of_channels=sum(r['failed'] is None for r in M['results']), list=sorted(map(list, bear)))
R['nonfailed'] = nf
# timing: failing bearing warning precedes non-failed warnings in each run (D1 1.2x)
fail = {c['run']: c['lead'] for c in M['detection']['D1_1.2x']['cases']}
R['timing_1.2x'] = {run: dict(failed_max_lead=max(c['lead'] for c in M['detection']['D1_1.2x']['cases'] if c['run'] == run),
                              nonfailed_max_lead=max([r['D1_1.2x']['sustained'] for r in M['results'] if r['run'] == run and r['failed'] is None and r['D1_1.2x']['sustained']] or [0]))
                    for run in fail}
# 2. run-level detection (3 runs, all detected) exact interval
R['run_level'] = dict(detected=3, n=3, ci95=[float(beta.ppf(0.025, 3, 1)), 1.0])
# 3. Poisson interval for the 1.2x false-alarm rate
x = D['D1_1.2x']; ep = x['fa_episodes']; hrs = x['ref_channel_hours']
lo, hi = chi2.ppf(0.025, 2 * ep) / 2, chi2.ppf(0.975, 2 * ep + 2) / 2
R['fa_rate_1.2x'] = dict(episodes=ep, channel_hours=hrs, per1000=1000 * ep / hrs, ci95_per1000=[1000 * lo / hrs, 1000 * hi / hrs])
def g(pd, l, lam, X=0.0): return 1000 * ((pd * Cp + (1 - pd) * Cf + X) / (MEAN - pd * l) + lam * Cfa)
l = x['lead_min']; ph = x['p_hat']; pl = x['ci95'][0]
R['cost_over_fa_ci'] = {k: dict(at_phat=g(ph, l, lam / 1000), at_lower=g(pl, l, lam / 1000)) for k, lam in zip(('lo', 'hi'), R['fa_rate_1.2x']['ci95_per1000'])}
# 4. cost scenarios for warnings on non-failed bearings (per failure cycle)
lam = x['fa_per_1000h'] / 1000; n_nf = nf['D1_1.2x']['bearings'] / x['n']
R['scenarios_1.2x'] = dict(n_nf_per_failure=n_nf, base=dict(at_phat=g(ph, l, lam), at_lower=g(pl, l, lam)),
                           inspection=dict(at_phat=g(ph, l, lam, n_nf * Cfa), at_lower=g(pl, l, lam, n_nf * Cfa)),
                           replacement=dict(at_phat=g(ph, l, lam, n_nf * Cp), at_lower=g(pl, l, lam, n_nf * Cp)),
                           max_unnecessary_replacements_per_failure={b: dict(at_phat=((a / 1000 - lam * Cfa) * (MEAN - ph * l) - (ph * Cp + (1 - ph) * Cf)) / Cp,
                                                                            at_lower=((a / 1000 - lam * Cfa) * (MEAN - pl * l) - (pl * Cp + (1 - pl) * Cf)) / Cp) for b, a in AGE.items()})
json.dump(R, open(OUT, 'w'), indent=1)
print(json.dumps(R, indent=1))

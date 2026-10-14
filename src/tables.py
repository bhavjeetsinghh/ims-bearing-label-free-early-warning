"""Print Tables 1-4 of the paper from the JSON outputs in results/."""
import json, sys
d = sys.argv[1] if len(sys.argv) > 1 else 'results'
R = json.load(open(f'{d}/results.json')); M = json.load(open(f'{d}/multi.json')); D = json.load(open(f'{d}/decision.json'))
f = lambda v: '-' if v is None else f'{v:.1f}'
print('Table 1. Runs'); [print(f"{k}: operating {v['op_hours']:.1f} h (elapsed {v['elapsed_hours']:.1f}), snapshots used {v['files']} (excluded {len(v['excluded_dead'])}), pauses {v['pauses']}") for k, v in M['runs'].items()]
mu = R['base_rms_mean']; sw = {(x['detector'], x['thr']): x for x in R['sweep']}; De = R['detectors']
print('\nTable 2. Test 3 detectors (FA episodes (per 1000 h), first alarm final 7 d, sustained)')
for n, d_ in [('D1 c=2.0', De['D1 RMS > 2x baseline']), ('D1 c=1.2 (post hoc)', sw[('RMS ratio', 1.2)]), ('D2', De['D2 RMS > mean+3sd']),
              ('D3', De['D3 Kurtosis > 6']), ('D4 a=1e-3', De['D5 Hotelling T2 (alpha=0.001)']), ('D4 a=1e-4', sw[('T2', 0.9999)])]:
    print(f"{n:22s} {d_['fa_episodes_ref']} ({f(d_['fa_per_1000h'])})  {f(d_['first_alarm_final7d_ttg'])}  {f(d_['sustained_lead_h'])}")
print('\nTable 3. Four failures (sustained warning h; detected; FA per 1000 ch-h; non-failed alarmed)')
for dn in ['D1_2x', 'D1_1.2x', 'D2', 'D3', 'D4_1e-4']:
    leads = [f(c['lead']) if (c['lead'] or 0) >= 0.15 else '-' for c in M['detection'][dn]['cases']]
    r = D[dn]; print(f"{dn:8s} {' '.join(f'{x:>6s}' for x in leads)}  {r['detected']}/{r['n']} ({r['ci95'][0]:.2f}-{r['ci95'][1]:.2f})  {r['fa_per_1000h']:.1f}  {r['nonfailed_alarms']}/{r['nonfailed_n']}")
print('\nTable 4. Cost per 1000 operating h')
for b, c in R['cost'].items():
    print(f"beta={b}: RTF {c['rtf_per_1000h']:,.0f} | age {c['age_opt_T']:.0f} h / {c['age_per_1000h']:,.0f} | break-even pd {c['breakeven_pd_zeroFA']:.2f} | "
          f"D1 2x {D['D1_2x']['cost_per_1000h'][b]['cbm_at_phat']:,.0f} | D1 1.2x {D['D1_1.2x']['cost_per_1000h'][b]['cbm_at_phat']:,.0f} / {D['D1_1.2x']['cost_per_1000h'][b]['cbm_at_lower']:,.0f}")

import sys, json, numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt, matplotlib.dates as mdates
from math import gamma
from scipy import integrate, optimize

out = sys.argv[1]
s = pd.read_csv(f'{out}/series.csv', parse_dates=['t']); R = json.load(open(f'{out}/results.json'))
BLUE, ORANGE, AQUA, YELLOW = '#2a78d6', '#eb6834', '#1baf7a', '#eda100'
INK, MUTED, GRID = '#222222', '#6b6b6b', '#e4e4e0'
plt.rcParams.update({'font.family': 'serif', 'font.serif': ['Times New Roman', 'DejaVu Serif', 'Liberation Serif'],
                     'font.size': 8, 'axes.titlesize': 8, 'axes.labelsize': 8, 'xtick.labelsize': 7, 'ytick.labelsize': 7,
                     'legend.fontsize': 7, 'axes.edgecolor': MUTED, 'axes.linewidth': 0.6, 'xtick.color': MUTED,
                     'ytick.color': MUTED, 'axes.labelcolor': INK, 'axes.grid': True, 'grid.color': GRID,
                     'grid.linewidth': 0.5, 'lines.linewidth': 1.0, 'savefig.dpi': 600, 'figure.dpi': 150,
                     'axes.spines.top': False, 'axes.spines.right': False, 'legend.frameon': False})
W1, W2 = 3.4, 7.0
mu = R['base_rms_mean']

def gapped(df, col):
    """insert NaN where recording gap > 15 min so lines are not drawn across gaps"""
    d = df[['t', col]].copy(); g = d.t.diff().dt.total_seconds() > 900
    ins = d[g].copy(); ins['t'] = ins.t - pd.Timedelta(minutes=5); ins[col] = np.nan
    return pd.concat([d, ins]).sort_values('t')

def save(fig, name):
    fig.savefig(f'{out}/{name}.png', bbox_inches='tight'); fig.savefig(f'{out}/{name}.pdf', bbox_inches='tight'); plt.close(fig)

t0 = s.t.iloc[0]; tb = t0 + pd.Timedelta(hours=72); tr = t0 + pd.Timedelta(hours=24); ta = s.t[s.ttg <= 168].iloc[0]

# ---------- Fig 1: full run ----------
fig, ax = plt.subplots(2, 1, figsize=(W2, 3.2), sharex=True)
s['ratio'] = s.rms / mu
for a, col, lab in [(ax[0], 'ratio', 'RMS / baseline mean'), (ax[1], 'kurtosis_pearson', 'Kurtosis (Pearson)')]:
    d = gapped(s, col); a.plot(d.t, d[col], color=BLUE, lw=0.7)
    for (x0, x1, txt) in [(t0, tr, 'run-in'), (tr, tb, 'baseline'), (ta, s.t.iloc[-1], 'final 7 d')]:
        a.axvspan(x0, x1, color='#000000', alpha=0.06 if txt != 'baseline' else 0.12, lw=0)
    a.set_ylabel(lab)
ax[0].set_yscale('log'); ax[0].set_yticks([1, 2, 5, 10]); ax[0].set_yticklabels(['1', '2', '5', '10'])
ax[1].set_ylim(2.5, 20)
ymax = ax[0].get_ylim()[1]
for x, txt in [(t0 + pd.Timedelta(hours=2), 'run-in &\nbaseline'), (t0 + pd.Timedelta(hours=450), f"healthy-reference zone ({R['ref_hours']:.0f} h)"), (ta + pd.Timedelta(hours=20), 'final\n7 d')]:
    ax[0].text(x, 8.5, txt, fontsize=6.5, color=MUTED, va='top')
ax[1].xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
ax[1].set_xlabel('Date (2004)')
save(fig, 'Figure1_full_run')

# ---------- Fig 2: final 7 days, RMS ratio + T2 ----------
f7 = s[s.ttg <= 168]
fig, ax = plt.subplots(2, 1, figsize=(W2, 3.4), sharex=True)
d = gapped(f7, 'ratio'); ax[0].plot(d.t, d.ratio, color=BLUE, lw=0.9)
for th, ls, lab in [(2.0, '--', 'Rule 1: 2.0x baseline'), (1.2, ':', '1.2x baseline')]:
    ax[0].axhline(th, color=INK, ls=ls, lw=0.7); ax[0].text(f7.t.iloc[5], th * 1.04, lab, fontsize=6.5, color=INK)
ax[0].axhline(R_max := 1.0831865593972567, color=MUTED, lw=0.6)
ax[0].set_yscale('log'); ax[0].set_yticks([1, 1.2, 2, 5, 10]); ax[0].set_yticklabels(['1', '1.2', '2', '5', '10'])
ax[0].set_ylabel('RMS / baseline mean')
d = gapped(f7, 'T2'); ax[1].plot(d.t, d.T2, color=ORANGE, lw=0.9)
from scipy import stats
n, p = R['n_base'], 4
u4 = p * (n + 1) * (n - 1) / (n * (n - p)) * stats.f.ppf(0.9999, p, n - p)
ax[1].axhline(u4, color=INK, ls='--', lw=0.7); ax[1].text(f7.t.iloc[5], u4 * 1.15, r'UCL ($\alpha$ = 10$^{-4}$)', fontsize=6.5)
ax[1].set_yscale('log'); ax[1].set_ylabel(r'Hotelling $T^2$')
for a in ax:
    a.axvline(pd.Timestamp('2004-04-15 15:22:55'), color=MUTED, lw=0.6, ls='-.')
    a.axvline(pd.Timestamp('2004-04-17 01:52:55'), color=MUTED, lw=0.6, ls='-.')
ax[0].text(pd.Timestamp('2004-04-15 16:30'), 6, 'A: Apr 15 15:22\n(59.2 h)', fontsize=6.5, color=MUTED, bbox=dict(fc='white', ec='none', pad=1))
ax[0].text(pd.Timestamp('2004-04-17 01:00'), 3.0, 'B: Apr 17 01:52\n(24.7 h)', fontsize=6.5, color=MUTED, ha='right', bbox=dict(fc='white', ec='none', pad=1))
ax[1].xaxis.set_major_formatter(mdates.DateFormatter('%b %d\n%H:%M')); ax[1].set_xlabel('Date/time (2004)')
save(fig, 'Figure3_final_week')

# ---------- Fig 3: operating characteristic ----------
sw = pd.DataFrame([x for x in R['sweep'] if x['detector'] == 'RMS ratio'])
fig, ax = plt.subplots(figsize=(W1, 2.3))
ax.plot(sw.thr, sw.first_alarm_final7d_ttg, color=BLUE, marker='o', ms=3, lw=1, label='first alarm')
ax.plot(sw.thr, sw.sustained_lead_h.fillna(0), color=ORANGE, marker='s', ms=3, lw=1, label='sustained alarm')
ax.axvline(1.083, color=MUTED, lw=0.6); ax.text(1.12, 100, 'largest persistent\nratio while healthy\n(1.08)', fontsize=6.5, color=MUTED)
ax.set_xlabel('RMS threshold (multiple of baseline mean)'); ax.set_ylabel('Warning time before end (h)')
ax.set_ylim(0, 160); ax.legend(loc='upper right')
ax.text(1.9, 60, 'zero false-alarm episodes\nin the healthy-reference\nzone at every threshold', fontsize=6.5, color=MUTED)
save(fig, 'Figure2_threshold_sweep')

# ---------- Fig 4: leakage demonstration ----------
L = R['leakage']; f3 = s[s.ttg <= 75]
fig, ax = plt.subplots(figsize=(W1, 2.3))
d = gapped(f3, 'rms'); ax.plot(d.t, d.rms, color=BLUE, lw=0.9)
mu_o = 0.0678
ax.axhline(2 * mu_o, color=INK, ls='--', lw=0.7); ax.text(f3.t.iloc[2], 2 * mu_o * 1.05, 'labelling threshold 0.136 g', fontsize=6.5)
ax.axhline(L['implied_rms_threshold_g'], color=ORANGE, ls=':', lw=0.9); ax.text(f3.t.iloc[2], L['implied_rms_threshold_g'] * 1.04, 'learned threshold 0.109 g', fontsize=6.5, color=INK)
ax.axvspan(pd.Timestamp(L['ALR']['fp_first']), pd.Timestamp(L['label_onset']), color=ORANGE, alpha=0.25, lw=0)
ax.annotate('13 "false positives"\n= the extra 2.2 h', xy=(pd.Timestamp('2004-04-17 00:40'), 0.13), xytext=(pd.Timestamp('2004-04-15 20:00'), 0.30),
            fontsize=6.5, arrowprops=dict(arrowstyle='-', color=MUTED, lw=0.6))
ax.set_yscale('log'); ax.set_ylabel('RMS (g)')
ax.set_yticks([0.07, 0.1, 0.2, 0.4, 0.7]); ax.set_yticklabels(['0.07', '0.10', '0.20', '0.40', '0.70']); ax.minorticks_off()
ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d\n%H:%M'))
save(fig, 'Figure5_threshold_labels')

# ---------- Fig 5: cost ----------
Cp, Cf, Cfa = R['costs']['Cp'], R['costs']['Cf'], R['costs']['Cfa']; MEAN = R['run_hours']
lead = R['detectors']['D1 RMS > 2x baseline']['sustained_lead_h']
pds = np.linspace(0, 1, 101)
fig, ax = plt.subplots(1, 2, figsize=(W2, 2.5)); fig.subplots_adjust(wspace=0.35)
cols = {'1.5': BLUE, '2.0': ORANGE, '3.0': AQUA}
cbm = 1000 * ((pds * Cp + (1 - pds) * Cf) / (MEAN - pds * lead))
ax[0].plot(pds, cbm, color=INK, lw=1.4, label='condition-based, no false alarms')
ax[0].axhline(1000 * Cf / MEAN, color=MUTED, lw=0.8, ls='--', label='run-to-failure')
for b, c in cols.items():
    ga = R['cost'][b]['age_per_1000h']; ax[0].axhline(ga, color=c, lw=0.9, label=rf'optimal age replacement, $\beta$={b}')
ax[0].set_xlabel(r'Detection probability $p_d$'); ax[0].set_ylabel('Cost rate (USD per 1,000 h)'); ax[0].set_ylim(0, 130000); ax[0].legend(loc='upper right', fontsize=6.3)
ax[0].yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, _: f'{x/1000:.0f}k'))
ax[0].set_title('(a)', loc='left')
for b, c in cols.items():
    be = R['cost'][b]['breakeven_fa_per_1000h']; x = np.array([float(k) for k in be]); y = np.array(list(be.values()))
    o = np.argsort(x); ax[1].plot(x[o], y[o], color=c, marker='o', ms=3, label=rf'$\beta$={b}')
if len(sys.argv) > 2:
    _fa = json.load(open(f'{sys.argv[2]}/decision.json'))['D1_1.2x']['fa_per_1000h']
    ax[1].axhline(_fa, color=MUTED, ls=':', lw=0.8); ax[1].text(0.755, _fa + 0.5, f'D1 1.2x observed: {_fa:.1f}', fontsize=6.5, color=MUTED)
ax[1].set_ylim(0, None)
ax[1].set_xlabel(r'Detection probability $p_d$'); ax[1].set_ylabel('Break-even false alarms per 1,000 h')
ax[1].legend(loc='upper left'); ax[1].set_title('(b)', loc='left')
if len(sys.argv) > 2:
    Dd = json.load(open(f'{sys.argv[2]}/decision.json'))
    for (dn, lab, mk), yb in zip([('D1_2x', 'D1 2x: 2/4 detected', 's'), ('D1_1.2x', 'D1 1.2x: 4/4 detected', 'o')], (16000, 5000)):
        r = Dd[dn]; c = r['cost_per_1000h']['2.0']
        ax[0].plot(r['p_hat'], c['cbm_at_phat'], marker=mk, mfc='white', mec=INK, mew=1.0, ms=6, ls='none', label=lab)
        ax[0].plot(r['ci95'], [yb, yb], color=INK, lw=1.0, marker='|', ms=6)
        ax[0].text(r['ci95'][0], yb + 2200, f"{lab.split(':')[0]} 95% CI for $p_d$: {r['ci95'][0]:.2f}-{r['ci95'][1]:.2f}", fontsize=6)
    ax[0].legend(loc='upper right', fontsize=6.0)
save(fig, 'Figure6_cost')


print('ok')

# ---------- Fig: four documented failures (operating hours before end) ----------
import os
M2 = sys.argv[2] if len(sys.argv) > 2 else None
if M2:
    cases = [('T1', [5, 6], 'Test 1, Bearing 3 (inner race)'), ('T1', [7, 8], 'Test 1, Bearing 4 (rolling element)'),
             ('T2', [1], 'Test 2, Bearing 1 (outer race)'), ('T3', [3], 'Test 3, Bearing 3 (outer race)')]
    fig, axs = plt.subplots(2, 2, figsize=(W2, 4.0), sharey=True)
    for ax, (run, chs, title) in zip(axs.flat, cases):
        for ci, ch in enumerate(chs):
            s4 = pd.read_csv(f'{M2}/series_{run}_ch{ch}.csv')
            s4 = s4[s4.ttg <= 200]
            ax.plot(-s4.ttg, s4.ratio, color=BLUE if ci == 0 else ORANGE, lw=0.8, label=('x-axis' if ci == 0 else 'y-axis') if len(chs) > 1 else None)
        ax.axhline(2.0, color=INK, ls='--', lw=0.6); ax.axhline(1.2, color=INK, ls=':', lw=0.7)
        ax.set_yscale('log'); ax.set_yticks([1, 1.2, 2, 5, 10]); ax.set_yticklabels(['1', '1.2', '2', '5', '10']); ax.minorticks_off()
        ax.set_title(title, loc='left'); ax.set_xlim(-200, 2)
        if len(chs) > 1: ax.legend(loc='upper left')
    for ax in axs[1]: ax.set_xlabel('Operating hours before end of recording')
    for ax in axs[:, 0]: ax.set_ylabel('RMS / baseline mean')
    fig.tight_layout(); save(fig, 'Figure4_four_failures')
print('ok2')

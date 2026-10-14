"""Extract per-snapshot time-domain features from raw NASA IMS bearing files.

Usage:
    python extract_features.py <folder with raw snapshot files> <output.csv> [--channels 4|8]

The IMS raw files are tab-separated text with one column per accelerometer channel
(Test 1: 8 channels, two per bearing; Tests 2 and 3: 4 channels, one per bearing) and
20,480 rows per snapshot. File names encode the timestamp (YYYY.MM.DD.hh.mm.ss).

Output: one row per (snapshot, channel) with RMS, Pearson kurtosis, skewness, variance
and peak-to-peak amplitude, plus the bearing number that channel belongs to.
"""
import sys, os, argparse
import numpy as np, pandas as pd
from scipy.stats import kurtosis, skew

ap = argparse.ArgumentParser()
ap.add_argument('folder'); ap.add_argument('out')
ap.add_argument('--channels', type=int, default=None, help='4 or 8; inferred from the first file if omitted')
a = ap.parse_args()

files = sorted(f for f in os.listdir(a.folder) if f.count('.') == 5 and not f.startswith('.'))
if not files:
    sys.exit('No IMS snapshot files (named like 2004.03.04.09.27.46) found in ' + a.folder)
rows = []
for i, fn in enumerate(files):
    x = np.loadtxt(os.path.join(a.folder, fn), dtype=float)
    if x.ndim == 1: x = x[:, None]
    nch = a.channels or x.shape[1]
    for ch in range(nch):
        s = x[:, ch]
        rows.append(dict(filename=fn, channel=ch + 1, bearing=(ch // 2 + 1) if nch == 8 else ch + 1,
                         n_rows=len(s), rms=float(np.sqrt(np.mean(s ** 2))),
                         kurtosis_pearson=float(kurtosis(s, fisher=False)), skewness=float(skew(s)),
                         variance=float(np.var(s)), peak2peak=float(s.max() - s.min())))
    if (i + 1) % 250 == 0: print(f'{i + 1}/{len(files)} files')
df = pd.DataFrame(rows)
df['timestamp'] = pd.to_datetime(df.filename, format='%Y.%m.%d.%H.%M.%S')
df.to_csv(a.out, index=False)
print(f'Wrote {len(df)} rows ({len(files)} snapshots x {df.channel.nunique()} channels) to {a.out}')

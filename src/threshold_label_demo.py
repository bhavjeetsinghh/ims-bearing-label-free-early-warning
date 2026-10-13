import pandas as pd, numpy as np, sys
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, balanced_accuracy_score
from scipy.stats import spearmanr

df = pd.read_csv(sys.argv[1])
df['t'] = pd.to_datetime(df.filename, format='%Y.%m.%d.%H.%M.%S')
print('all rows', len(df), df.t.min(), df.t.max())
dt = df.t.diff().dt.total_seconds()/60
print('gap >15 min count', (dt > 15).sum()); print(df.loc[dt > 15, ['file_index','filename']].assign(gap_min=dt[dt>15]).to_string())
v = df.iloc[:-1].copy()           # drop final anomalous file
print('valid', len(v), 'run hours', (v.t.iloc[-1]-v.t.iloc[0]).total_seconds()/3600)

F = ['rms','kurtosis_pearson','skewness','variance','peak2peak']
print('\nRanges incl. excluded file:'); print(df[F].agg(['min','max']).T)
print('Ranges valid only:'); print(v[F].agg(['min','max']).T)
print('\nPearson corr with rms:', v[F].corr()['rms'].round(3).to_dict())
print('Spearman var~rms', spearmanr(v.variance, v.rms)[0], ' var vs rms^2 max rel diff', ((v.variance - v.rms**2).abs()/v.rms**2).max())

def persist(mask, k=3):
    m = mask.values.astype(int); r = np.convolve(m, np.ones(k, int), 'valid') == k
    idx = np.where(r)[0]
    return None if len(idx)==0 else idx[0]          # start index of first k-run

end = v.t.iloc[-1]
nb = 5933
base = v.rms.iloc[:nb]; mu, sd = base.mean(), base.std()
print(f'\nbaseline n={nb} mean={mu:.4f} sd={sd:.4f}')
lead = lambda i: (end - v.t.iloc[i]).total_seconds()/3600
for name, m in [('R1', v.rms > 2*mu), ('R2', v.rms > mu+3*sd), ('R3', v.kurtosis_pearson > 6)]:
    i = persist(m); n_single_healthy = m.iloc[:nb].sum()
    print(name, v.t.iloc[i] if i is not None else None, round(lead(i),2) if i is not None else None, 'exceedances in baseline period:', int(n_single_healthy))
# R1: exceedances anywhere before first persistent warning?
m1 = v.rms > 2*mu; i1 = persist(m1)
print('R1 single exceedances before first persistent:', int(m1.iloc[:i1].sum()))
# cross bearing
for b in ['b1_rms','b2_rms','b3_rms','b4_rms']:
    mb = v[b].iloc[:nb].mean(); i = persist(v[b] > 2*mb)
    print(b, None if i is None else (v.t.iloc[i], round(lead(i),2)))

# labels
y = np.zeros(len(v), int); y[i1:] = 1
print('\nlabels healthy', (y==0).sum(), 'degraded', (y==1).sum())
tr = np.r_[0:nb, i1:i1+50]; te = np.r_[nb:i1, i1+50:len(v)]
print('train', len(tr), 'test', len(te), 'test healthy', (y[te]==0).sum(), 'test degraded', (y[te]==1).sum())
print('test healthy span', v.t.iloc[nb], '->', v.t.iloc[i1-1], '; train end', v.t.iloc[i1+49])
sets = {'A':F, 'B':['skewness','variance','peak2peak'], 'C':['skewness']}
for s, cols in sets.items():
    for mn, mdl in [('LR', LogisticRegression(class_weight='balanced', max_iter=2000)),
                    ('RF', RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=42))]:
        mdl.fit(v[cols].values[tr], y[tr]); p = mdl.predict(v[cols].values[te])
        yt = y[te]; far = ((p==1)&(yt==0)).sum()/(yt==0).sum(); mar = ((p==0)&(yt==1)).sum()/(yt==1).sum()
        tt = v.t.iloc[te].reset_index(drop=True); j = persist(pd.Series(p==1))
        fp_idx = np.where((p==1)&(yt==0))[0]
        print(f'{s} {mn} P={precision_score(yt,p):.3f} R={recall_score(yt,p):.3f} F1={f1_score(yt,p):.3f} BA={balanced_accuracy_score(yt,p):.3f} FAR={far:.3f} MAR={mar:.3f}',
              'firstwarn', None if j is None else (tt[j], round((end-tt[j]).total_seconds()/3600,2)),
              'FP times', (tt[fp_idx[0]], tt[fp_idx[-1]]) if len(fp_idx) else None, 'nFP', len(fp_idx))
# Same models, but FAR across the whole healthy history (in-sample for baseline -- shown only to reveal threshold)
# Forward test: train only on data before test with simple RMS rule equivalence
lr = LogisticRegression(class_weight='balanced', max_iter=2000).fit(v[['rms']].values[tr], y[tr])
print('\nLR on RMS alone implied threshold (g):', -lr.intercept_[0]/lr.coef_[0][0], ' vs 2*mu', 2*mu)
lrb = LogisticRegression(class_weight='balanced', max_iter=2000).fit(v[['variance']].values[tr], y[tr])
thr = -lrb.intercept_[0]/lrb.coef_[0][0]; print('LR on variance alone implied threshold', thr, '-> sqrt', np.sqrt(thr))

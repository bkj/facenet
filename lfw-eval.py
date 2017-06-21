#!/usr/bin/env python

import warnings
warnings.filterwarnings('ignore')

import os
import sys
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.model_selection import KFold

# --
# Load feats

inpath = sys.argv[1]
print "lfw-eval.py\t %s" % inpath

df = pd.read_csv(inpath, header=None, sep='\t')

meta  = pd.DataFrame(df[0])

meta['id']  = meta[0].apply(lambda x: '_'.join(os.path.basename(x).split('_')[:-1]))
meta['img'] = meta[0].apply(lambda x: int(os.path.basename(x).split('_')[-1].split('.')[0]))
meta['uid'] = meta[['id', 'img']].apply(lambda x: '_'.join(map(str, x)), 1)
meta['row'] = np.arange(meta.shape[0])
meta = meta[['id', 'img', 'uid', 'row']].set_index('uid')

# --
# Load pairs

pairs = map(lambda x: x.strip().split('\t'), open('./data/pairs.txt').read().splitlines())[1:]
for i,p in enumerate(pairs):
    if len(p) == 3:
        p = '_'.join((p[0], p[1])), '_'.join((p[0], p[2])), True
    else:
        p = '_'.join((p[0], p[1])), '_'.join((p[2], p[3])), False
    
    pairs[i] = p

pairs = pd.DataFrame(pairs)
pairs.columns = ('a', 'b', 'lab')

# --
# Subset pairs

pairs = pairs[pairs.a.isin(meta.index) & pairs.b.isin(meta.index)]
sel_a = np.array(meta.row.loc[pairs.a])
sel_b = np.array(meta.row.loc[pairs.b])
assert(sel_a.shape == sel_b.shape)

print "n_pairs\t\t %d" % sel_a.shape[0]

# --
# Compute performance

feats = np.array(df[range(1, df.shape[1])])
ds = np.sqrt(((feats[sel_a] - feats[sel_b]) ** 2).sum(axis=1))
y = np.array(pairs.lab)
print "avg_dist\t %f" % ds.mean()

res = np.zeros(y.shape) - 1
ts = np.percentile(ds, np.arange(0, 100, 0.1))
good_ts = []
for train, test in KFold(10).split(range(y.shape[0])):
    accs = [(y[train] == (ds[train] < t)).mean() for t in ts]
    t = ts[np.argmax(accs)]
    res[test] = y[test] == (ds[test] < t)
    good_ts.append(t)

print 'tuned accuracy\t %f @ %f' % (res.mean(), np.mean(good_ts))


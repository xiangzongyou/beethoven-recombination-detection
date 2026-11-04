"""
Mean transition features for real, randomly spliced and selectively spliced bar pairs
(Beethoven only). Shows how each splicing strategy differs from real transitions.

Usage: python transition_stats.py <work dir> <output csv> [--pairs 20000] [--seed 0]
"""
import argparse
import numpy as np
import pandas as pd
from splice_detector import (TARGET, COMP, PERF, trans_features, smart_pick, other_title_pick)

ap = argparse.ArgumentParser()
ap.add_argument("workdir"); ap.add_argument("out")
ap.add_argument("--pairs", type=int, default=20000); ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()

d = np.load(f"{a.workdir}/features.npz"); e = np.load(f"{a.workdir}/edges.npz")
meta = pd.read_csv(f"{a.workdir}/meta.csv")
X, pid, pc, sc = d["X"], d["piece_id"], e["pc"], e["sc"]
pos = np.zeros(len(X), int)
for p in np.unique(pid):
    idx = np.where(pid == p)[0]; pos[idx] = np.arange(len(idx))
plen = np.bincount(pid)[pid]
isB = meta.canonical_composer.values[pid] == TARGET
title = pd.factorize(meta.canonical_title)[0][pid]
snd = X[:, 0] > 0
okA = isB & snd & (pos >= 1) & (pos < plen - 2) & np.roll(snd, -1)

rng = np.random.default_rng(a.seed)
pool = np.where(okA)[0]
A = rng.choice(pool, a.pairs)
rows = {}
for name, B in (("real", A + 1),
                ("random_splice", other_title_pick(rng, A, pool, title)),
                ("selected_splice", smart_pick(rng, A, pool, pc, sc, title))):
    rows[name] = trans_features(X, pc, sc, A, B).mean(0)
df = pd.DataFrame(rows, index=COMP + PERF).round(3)
df.to_csv(a.out)
print(df.to_string())

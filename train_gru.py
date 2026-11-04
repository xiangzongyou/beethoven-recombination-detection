"""
Train and test the "Beethoven or not" classifier (after Galajda & Hua, MIPR 2024).

Model: 2-layer GRU (512 units) -> batch norm -> dense layers -> dropout 0.4 -> 1 output.
Input: one segment = 32 bars x 27 features.

Split modes:
  random  : shuffle all segments, split 80/20
  grouped : split by canonical_title, so all performances of a work are on one side
  year    : hold out two recording years; drop training works that also appear in test

Extra test:
  shuffle : shuffle the bar order of Beethoven test segments and predict again

Usage: python train_gru.py <work dir> <output dir> [--runs 5] [--epochs 12]
       [--modes random,grouped,year] [--drop-perf]
"""
import argparse, json, os, time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, f1_score, balanced_accuracy_score, roc_auc_score

SEG = 32  # bars per segment (STRIDE = 32 in the paper)
TARGET = "Ludwig van Beethoven"


def make_segments(X, piece_id, meta, seg=SEG, hop=SEG):
    """Cut each performance into segments of seg bars; hop=seg means no overlap."""
    segs, pid = [], []
    for i in np.unique(piece_id):
        F = X[piece_id == i]
        for s in range(0, len(F) - seg + 1, hop):
            segs.append(F[s:s + seg]); pid.append(i)
    segs, pid = np.stack(segs), np.array(pid)
    y = (meta.canonical_composer.values[pid] == TARGET).astype(np.float32)
    title = meta.canonical_title.values[pid]
    return segs, y, pid, title


class GRUClassifier(nn.Module):
    def __init__(self, n_feat=27, hidden=512):
        super().__init__()
        self.gru = nn.GRU(n_feat, hidden, num_layers=2, batch_first=True)
        self.head = nn.Sequential(
            nn.BatchNorm1d(hidden),
            nn.Linear(hidden, 128), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(128, 32), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(32, 1))

    def forward(self, x):
        _, h = self.gru(x)          # h: [layers, batch, hidden]; use the last layer
        return self.head(h[-1]).squeeze(-1)


def metrics(y, p, thr=0.5):
    yh = (p >= thr).astype(int)
    return dict(precision=precision_score(y, yh, zero_division=0),
                recall=recall_score(y, yh, zero_division=0),
                f1=f1_score(y, yh, zero_division=0),
                balanced_acc=balanced_accuracy_score(y, yh),
                auc=roc_auc_score(y, p))


def split(y, pid, title, mode, rng, test_frac=0.2):
    n = len(y)
    if mode == "random":
        perm = rng.permutation(n)
        test = np.zeros(n, bool); test[perm[:int(n * test_frac)]] = True
        return ~test, test
    if mode == "year":
        # hold out two years; remove training works that also appear in the test years
        years = np.unique(YEAR)
        test_years = rng.choice(years, 2, replace=False)
        test = np.isin(YEAR, test_years)
        train = ~test & ~np.isin(title, np.unique(title[test]))
        return train, test
    # grouped: hold out 20% of Beethoven titles and 20% of other titles
    test = np.zeros(n, bool)
    for cls in (0, 1):
        titles = np.unique(title[y == cls])
        pick = rng.choice(titles, int(round(len(titles) * test_frac)), replace=False)
        test |= np.isin(title, pick) & (y == cls)
    return ~test, test


def train_gru(Xtr, ytr, epochs, seed, device="cpu"):
    torch.manual_seed(seed)
    model = GRUClassifier(Xtr.shape[2]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    pos_weight = torch.tensor((1 - ytr.mean()) / ytr.mean())  # weight the minority class
    lossf = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    Xt, yt = torch.tensor(Xtr), torch.tensor(ytr)
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(Xt))
        tot = 0
        for b in range(0, len(Xt), 128):
            idx = perm[b:b + 128]
            if len(idx) < 2: continue
            opt.zero_grad()
            loss = lossf(model(Xt[idx]), yt[idx])
            loss.backward(); opt.step()
            tot += loss.item() * len(idx)
    return model


@torch.no_grad()
def predict(model, X):
    model.eval()
    return torch.sigmoid(torch.cat([model(torch.tensor(X[b:b + 512])) for b in range(0, len(X), 512)])).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("featdir"); ap.add_argument("outdir")
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--modes", default="random,grouped")
    ap.add_argument("--drop-perf", action="store_true",
                    help="drop velocity, pulse-strength and duration features")
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    global YEAR
    torch.set_num_threads(os.cpu_count())

    d = np.load(os.path.join(a.featdir, "features.npz"))
    meta = pd.read_csv(os.path.join(a.featdir, "meta.csv"))
    # remove composers with only one work, as in the paper
    counts = meta.canonical_composer.value_counts()
    keep_composers = counts[counts > 1].index
    print("composers:", meta.canonical_composer.nunique(), "-> kept", len(keep_composers))

    S, y, pid, title = make_segments(d["X"], d["piece_id"], meta)
    keep = np.isin(meta.canonical_composer.values[pid], keep_composers)
    S, y, pid, title = S[keep], y[keep], pid[keep], title[keep]
    YEAR = meta.year.values[pid]
    if a.drop_perf:
        names = list(d["names"])
        perf = ["avg_velocity", "std_velocity", "pulse_strength", "avg_duration", "std_duration"]
        cols = [i for i, nm in enumerate(names) if nm not in perf]
        S = S[:, :, cols]
        print("dropped:", perf, "-> features left:", len(cols))
    NF = S.shape[2]
    print(f"segments: {len(S)}, Beethoven: {int(y.sum())} ({y.mean():.1%})")

    results = []
    for mode in a.modes.split(","):
        for r in range(a.runs):
            rng = np.random.default_rng(r)
            tr, te = split(y, pid, title, mode, rng)
            # standardise with training statistics only
            mu = S[tr].reshape(-1, NF).mean(0); sd = S[tr].reshape(-1, NF).std(0) + 1e-6
            Z = ((S - mu) / sd).astype(np.float32)

            # baseline: logistic regression on the segment mean (ignores bar order)
            lr = LogisticRegression(max_iter=2000, class_weight="balanced")
            lr.fit(Z[tr].mean(1), y[tr])
            p_lr = lr.predict_proba(Z[te].mean(1))[:, 1]

            t = time.time()
            model = train_gru(Z[tr], y[tr], a.epochs, seed=r)
            p = predict(model, Z[te])

            # shuffle test: Beethoven test segments with bar order shuffled
            b = te & (y == 1)
            Zs = Z[b].copy()
            for k in range(len(Zs)):
                Zs[k] = Zs[k][rng.permutation(SEG)]
            p_shuf = predict(model, Zs)

            row = dict(mode=mode, run=r, n_train=int(tr.sum()), n_test=int(te.sum()),
                       n_test_beethoven=int(b.sum()), train_sec=round(time.time() - t))
            row.update({f"gru_{k}": v for k, v in metrics(y[te], p).items()})
            row.update({f"lr_{k}": v for k, v in metrics(y[te], p_lr).items()})
            row["gru_beethoven_called_beethoven"] = float((p[y[te] == 1] >= 0.5).mean())
            row["gru_shuffled_called_beethoven"] = float((p_shuf >= 0.5).mean())
            results.append(row)
            print(json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items()}), flush=True)
            pd.DataFrame(results).to_csv(os.path.join(a.outdir, "results_runs.csv"), index=False)

    df = pd.DataFrame(results)
    summ = df.drop(columns="run").groupby("mode").agg(["mean", "std"]).round(3)
    summ.to_csv(os.path.join(a.outdir, "results_summary.csv"))
    print(summ.T.to_string())


if __name__ == "__main__":
    main()

"""
Score-based replication of the recombination experiments, with real barlines.

  classify   Experiment A on scores: train the "Beethoven or not" GRU on score bars
             (split by sonata), then test spliced imitations built from held-out works.
  detect     Experiment B: transition detector (original vs. spliced bar pairs), on any
             bar dataset made by build_bars.py: score, perf_bars or perf_2s.
             Running it on perf_bars and perf_2s compares real barlines with
             2-second windows on the same performances.

Usage:
  python score_experiments.py classify <bars dir> <output dir> [--runs 10] [--epochs 30] [--drop-perf]
  python score_experiments.py detect   <bars dir> <output dir> --data score|perf_bars|perf_2s [--runs 10]

Progress is printed and also written to <output dir>/log.txt.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# shared code (features, GRU, transition features) lives in the repository root
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from train_gru import GRUClassifier, metrics
from splice_detector import COMP, PERF, PITCH, cos, trans_features

SEG = 32
TARGET = "Beethoven"
CLASSICAL = ("Haydn", "Mozart")
PERF_FEATURES = ["avg_velocity", "std_velocity", "pulse_strength", "avg_duration", "std_duration"]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class Log:
    def __init__(self, path):
        self.f = open(path, "a", encoding="utf-8")

    def __call__(self, *a):
        s = " ".join(str(x) for x in a)
        print(s, flush=True)
        self.f.write(s + "\n"); self.f.flush()


def load(bars_dir, name):
    d = dict(np.load(os.path.join(bars_dir, f"{name}.npz")))
    pieces = pd.read_csv(os.path.join(bars_dir, f"{name}_pieces.csv"))
    pid = d["piece_id"]
    n = len(pid)
    pos = np.zeros(n, int)
    for p in np.unique(pid):
        idx = np.where(pid == p)[0]; pos[idx] = np.arange(len(idx))
    d["pos"] = pos
    d["plen"] = np.bincount(pid)[pid]
    d["group"] = pd.factorize(pieces.group)[0][pid]
    d["composer"] = pieces.composer.values[pid]
    return d, pieces


# ---------------------------------------------------------------- Experiment A

def segments(Z, pid, starts):
    return np.stack([Z[s:s + SEG] for s in starts])


def seg_starts(pid, pos, plen, mask, hop):
    """Start indices of SEG-bar segments inside one piece, every `hop` bars."""
    ok = mask & (pos % hop == 0) & (pos + SEG <= plen)
    return np.where(ok)[0]


def train(Ztr, ytr, epochs, seed):
    torch.manual_seed(seed)
    model = GRUClassifier(Ztr.shape[2]).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    pw = torch.tensor((1 - ytr.mean()) / ytr.mean(), device=DEVICE)
    lossf = nn.BCEWithLogitsLoss(pos_weight=pw)
    X = torch.tensor(Ztr, device=DEVICE); y = torch.tensor(ytr, device=DEVICE)
    for _ in range(epochs):
        model.train()
        perm = torch.randperm(len(X), device=DEVICE)
        for b in range(0, len(X), 64):
            idx = perm[b:b + 64]
            if len(idx) < 2:
                continue
            opt.zero_grad(); lossf(model(X[idx]), y[idx]).backward(); opt.step()
    return model


@torch.no_grad()
def predict(model, Z):
    model.eval()
    out = [torch.sigmoid(model(torch.tensor(Z[b:b + 512], device=DEVICE))) for b in range(0, len(Z), 512)]
    return torch.cat(out).cpu().numpy()


def chunk_sampler(rng, d, pieces_mask, k, aligned=False):
    """Return a function that draws (start index) of k consecutive bars from allowed pieces."""
    ok = pieces_mask & (d["pos"] + k <= d["plen"])
    if aligned:
        ok &= d["pos"] % 4 == 0
    cand = np.where(ok)[0]
    return lambda: cand[rng.integers(len(cand))], cand


def build_splice(rng, d, Z, k, poolB, poolO=None, p_b=1.0, aligned=False, same_meter=False, smart=False):
    """One 32-bar imitation from k-bar chunks; neighbouring chunks come from different groups."""
    out, last = [], None
    while sum(len(c) for c in out) < SEG:
        kk = min(k, SEG - sum(len(c) for c in out))
        use = poolB if (poolO is None or rng.random() < p_b) else poolO
        cand = use[kk] if not aligned else use[("a", kk)]
        for _ in range(200):
            if smart and last is not None:
                C = cand[rng.integers(len(cand), size=50)]
                C = C[d["group"][C] != d["group"][last]]
                if same_meter:
                    C = C[d["timesig"][C] == d["timesig"][last]]
                if len(C) == 0:
                    continue
                cost = (1 - cos(d["pc"][last][None, :], d["pc"][C])) + \
                    np.abs(np.nan_to_num(d["sc"][C, 0] - d["sc"][last, 1])) / 12
                s = C[np.argmin(cost)]
            else:
                s = cand[rng.integers(len(cand))]
                if last is not None and d["group"][s] == d["group"][last]:
                    continue
                if same_meter and last is not None and d["timesig"][s] != d["timesig"][last]:
                    continue
            break
        out.append(Z[s:s + kk]); last = s + kk - 1
    return np.concatenate(out)[:SEG]


def run_classify(a, log):
    d, pieces = load(a.bars, "score")
    X = d["X"]
    cols = list(range(27))
    if a.drop_perf:
        from real_bars import FEATURE_NAMES
        cols = [i for i, n in enumerate(FEATURE_NAMES) if n not in PERF_FEATURES]
        log("dropping:", PERF_FEATURES)
    isB = d["composer"] == TARGET
    groupsB = np.unique(d["group"][isB]); groupsO = np.unique(d["group"][~isB])
    log(f"score bars {len(X)}, Beethoven {isB.sum()}, Beethoven sonatas {len(groupsB)}, other groups {len(groupsO)}, device {DEVICE}")
    rows = []
    for r in range(a.runs):
        t = time.time()
        rng = np.random.default_rng(r)
        teB = rng.choice(groupsB, max(1, round(len(groupsB) * 0.2)), replace=False)
        teO = rng.choice(groupsO, max(1, round(len(groupsO) * 0.2)), replace=False)
        test = np.isin(d["group"], np.r_[teB, teO])
        mu = X[~test][:, cols].mean(0); sd = X[~test][:, cols].std(0) + 1e-6
        Z = ((X[:, cols] - mu) / sd).astype(np.float32)
        tr_s = seg_starts(d["piece_id"], d["pos"], d["plen"], ~test, a.hop_train)
        te_s = seg_starts(d["piece_id"], d["pos"], d["plen"], test, SEG)
        Ztr, ytr = segments(Z, None, tr_s), isB[tr_s].astype(np.float32)
        Zte, yte = segments(Z, None, te_s), isB[te_s].astype(np.float32)

        lr = LogisticRegression(max_iter=3000, class_weight="balanced").fit(Ztr.mean(1), ytr)
        p_lr = lr.predict_proba(Zte.mean(1))[:, 1]
        model = train(Ztr, ytr, a.epochs, r)
        p = predict(model, Zte)
        row = dict(run=r, n_train=len(Ztr), n_test=len(Zte), n_test_beethoven=int(yte.sum()),
                   test_sonatas=" ".join(pieces.group.unique()[g].split("/")[-1] for g in teB))
        row.update({f"gru_{k}": v for k, v in metrics(yte, p).items()})
        row.update({f"lr_{k}": v for k, v in metrics(yte, p_lr).items()})
        cl = np.isin(d["composer"][te_s], CLASSICAL)
        row["classical_test_segments"] = int(cl.sum())
        row["classical_called_beethoven"] = float((p[cl] >= 0.5).mean()) if cl.any() else np.nan
        rows.append(row)

        # imitations from held-out works
        poolB, poolO = {}, {}
        for k in (1, 2, 4, 8):
            poolB[k] = chunk_sampler(rng, d, test & isB, k)[1]
            poolO[k] = chunk_sampler(rng, d, test & ~isB, k)[1]
        poolB[("a", 4)] = chunk_sampler(rng, d, test & isB, 4, aligned=True)[1]
        realB = Zte[yte == 1]; realO = Zte[yte == 0]
        N = a.n_imit
        conds = {
            "real_beethoven": realB,
            "shuffle_within": np.stack([s[rng.permutation(SEG)] for s in realB]),
            "real_other": realO,
            "real_haydn_mozart": Zte[cl] if cl.any() else None,
        }
        for k in (1, 2, 4, 8):
            conds[f"splice_random_{k}"] = np.stack([build_splice(rng, d, Z, k, poolB) for _ in range(N)])
        conds["splice_random_4_same_meter"] = np.stack([build_splice(rng, d, Z, 4, poolB, same_meter=True) for _ in range(N)])
        conds["splice_random_4_aligned"] = np.stack([build_splice(rng, d, Z, 4, poolB, aligned=True) for _ in range(N)])
        conds["splice_selected_4_same_meter"] = np.stack([build_splice(rng, d, Z, 4, poolB, same_meter=True, smart=True) for _ in range(N)])
        for pb in (0.75, 0.5, 0.25, 0.0):
            conds[f"mix_{pb}"] = np.stack([build_splice(rng, d, Z, 4, poolB, poolO, p_b=pb) for _ in range(N)])
        conds["splice_other_4"] = np.stack([build_splice(rng, d, Z, 4, poolO) for _ in range(N)])
        for name, Zc in conds.items():
            if Zc is None or len(Zc) == 0:
                continue
            pc_ = predict(model, Zc)
            rows.append(dict(run=r, condition=name, n=len(Zc), mean_prob=float(pc_.mean()),
                             called_beethoven=float((pc_ >= 0.5).mean())))
        log(f"[classify run {r}] {time.time() - t:.0f}s  AUC {row['gru_auc']:.3f}  "
            f"P {row['gru_precision']:.2f} R {row['gru_recall']:.2f} | called Beethoven: "
            + "  ".join(f"{x['condition']} {x['called_beethoven']:.2f}" for x in rows
                        if x.get("run") == r and x.get("condition") in ("real_beethoven", "splice_random_4", "real_other")))
        df = pd.DataFrame(rows)
        df.to_csv(os.path.join(a.out, "classify_runs.csv"), index=False)

    df = pd.DataFrame(rows)
    main = df[df.condition.isna()]
    rec = df[df.condition.notna()]
    log("\n== classifier (held-out sonatas), mean +/- sd ==")
    for c in ["gru_auc", "gru_precision", "gru_recall", "gru_f1", "lr_auc", "classical_called_beethoven"]:
        log(f"{c:28s} {main[c].mean():.3f} +/- {main[c].std():.3f}")
    log("\n== imitations: share called Beethoven, mean +/- sd ==")
    order = list(dict.fromkeys(rec.condition))
    s = rec.groupby("condition").called_beethoven.agg(["mean", "std"]).loc[order]
    for k, v in s.iterrows():
        log(f"{k:30s} {v['mean']:.3f} +/- {v['std']:.3f}")


# ---------------------------------------------------------------- Experiment B

def pick_random(rng, A, pool, group):
    B = pool[rng.integers(0, len(pool), len(A))]
    for _ in range(50):
        bad = group[B] == group[A]
        if not bad.any():
            break
        B[bad] = pool[rng.integers(0, len(pool), bad.sum())]
    return B


def pick_selected(rng, A, pool, d, n_cand=50, same_meter=False):
    C = pool[rng.integers(0, len(pool), (len(A), n_cand))]
    cost = (1 - cos(d["pc"][A][:, None, :], d["pc"][C])) + \
        np.abs(np.nan_to_num(d["sc"][C, 0] - d["sc"][A, 1][:, None])) / 12
    cost[d["group"][C] == d["group"][A][:, None]] = np.inf
    if same_meter:
        cost[d["timesig"][C] != d["timesig"][A][:, None]] = np.inf
    return C[np.arange(len(A)), cost.argmin(1)]


def run_detect(a, log):
    d, pieces = load(a.bars, a.data)
    X, pc, sc = d["X"], d["pc"], d["sc"]
    isB = d["composer"] == TARGET
    sounding = X[:, 0] > 0
    okA = isB & sounding & (d["pos"] >= 1) & (d["pos"] < d["plen"] - 2) & np.roll(sounding, -1)
    okA_al = okA & ((d["pos"] + 1) % 4 == 0)     # a ends a 4-bar unit
    okB_al = isB & sounding & (d["pos"] % 4 == 0) & (d["pos"] >= 1) & (d["pos"] < d["plen"] - 2)
    groupsB = np.unique(d["group"][isB])
    log(f"{a.data}: {len(X)} bars, {len(groupsB)} Beethoven sonatas, device n/a (sklearn)")
    meter = a.data != "perf_2s"
    rows = []
    for r in range(a.runs):
        t = time.time()
        rng = np.random.default_rng(r)
        test_g = rng.choice(groupsB, max(1, round(len(groupsB) * 0.3)), replace=False)
        in_test = np.isin(d["group"], test_g)
        data = {}
        for part, m in (("train", ~in_test), ("test", in_test)):
            pool = np.where(okA & m)[0]
            A = rng.choice(pool, a.pairs)
            pos = dict(real=trans_features(X, pc, sc, A, A + 1),
                       rand=trans_features(X, pc, sc, A, pick_random(rng, A, pool, d["group"])),
                       sel=trans_features(X, pc, sc, A, pick_selected(rng, A, pool, d, same_meter=meter)))
            if meter:
                Aa = rng.choice(np.where(okA_al & m)[0], a.pairs)
                Ba_pool = np.where(okB_al & m)[0]
                pos["real_aligned"] = trans_features(X, pc, sc, Aa, Aa + 1)
                pos["sel_aligned"] = trans_features(X, pc, sc, Aa, pick_selected(rng, Aa, Ba_pool, d, same_meter=True))
            pos["pool"] = pool
            data[part] = pos

        for featset, cols in (("all", COMP + PERF), ("composition", COMP), ("pitch", PITCH)):
            ci = [(COMP + PERF).index(c) for c in cols]
            h = a.pairs // 2
            Xtr = np.concatenate([data["train"]["real"][:, ci],
                                  data["train"]["rand"][:h, ci], data["train"]["sel"][h:, ci]])
            ytr = np.r_[np.zeros(a.pairs), np.ones(a.pairs)]
            gb = HistGradientBoostingClassifier(max_iter=300, random_state=r).fit(Xtr, ytr)
            lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000)).fit(Xtr, ytr)
            tests = [("random", "real", "rand"), ("selected", "real", "sel")]
            if meter:
                tests.append(("selected_aligned", "real_aligned", "sel_aligned"))
            for name, neg, posk in tests:
                Xte = np.concatenate([data["test"][neg][:, ci], data["test"][posk][:, ci]])
                yte = np.r_[np.zeros(a.pairs), np.ones(a.pairs)]
                rows.append(dict(run=r, features=featset, test=f"pair_{name}",
                                 auc_boosting=roc_auc_score(yte, gb.predict_proba(Xte)[:, 1]),
                                 auc_logreg=roc_auc_score(yte, lr.predict_proba(Xte)[:, 1])))
            if featset != "all":
                # 32-bar segments: real vs. spliced every 4 bars; score = mean of the 7 highest transitions
                T = data["test"]["pool"]
                starts = np.where(okA & in_test & (d["pos"] + SEG <= d["plen"]))[0]
                for kind in ("random", "selected"):
                    sc_real, sc_sp = [], []
                    for _ in range(300):
                        s0 = rng.choice(starts)
                        bars = np.arange(s0, s0 + SEG)
                        sc_real.append(np.sort(gb.predict_proba(trans_features(X, pc, sc, np.maximum(bars[:-1], 1), bars[1:])[:, ci])[:, 1])[-7:].mean())
                        bars = []
                        st = rng.choice(T)
                        while len(bars) < SEG:
                            if st + 3 >= len(X) or d["piece_id"][st] != d["piece_id"][st + 3]:
                                st = rng.choice(T); continue
                            bars += list(range(st, st + 4))
                            last = np.array([bars[-1]])
                            st = (pick_selected(rng, last, T, d, same_meter=meter)[0] if kind == "selected"
                                  else pick_random(rng, last, T, d["group"])[0])
                        bars = np.array(bars[:SEG])
                        sc_sp.append(np.sort(gb.predict_proba(trans_features(X, pc, sc, np.maximum(bars[:-1], 1), bars[1:])[:, ci])[:, 1])[-7:].mean())
                    rows.append(dict(run=r, features=featset, test=f"segment_{kind}",
                                     auc_boosting=roc_auc_score(np.r_[np.zeros(300), np.ones(300)], np.r_[sc_real, sc_sp])))
        means = {k: v for k, v in zip(["real", "rand", "sel"],
                 [data["test"][k][:, COMP.index("mel_jump")].mean() for k in ("real", "rand", "sel")])}
        rows.append(dict(run=r, features="stat", test="mean_melodic_leap",
                         real=means["real"], random=means["rand"], selected=means["sel"]))
        df = pd.DataFrame(rows)
        df.to_csv(os.path.join(a.out, f"detect_{a.data}_runs.csv"), index=False)
        cur = df[(df.run == r) & (df.features == "pitch")].set_index("test").auc_boosting
        log(f"[detect {a.data} run {r}] {time.time() - t:.0f}s  pitch-only: " +
            "  ".join(f"{k} {v:.3f}" for k, v in cur.items()))

    df = pd.DataFrame(rows)
    log(f"\n== transition detector on {a.data} (gradient boosting AUC, mean +/- sd over {a.runs} runs) ==")
    s = df[df.features != "stat"].groupby(["features", "test"]).auc_boosting.agg(["mean", "std"])
    for (f, tst), v in s.iterrows():
        log(f"{f:12s} {tst:24s} {v['mean']:.3f} +/- {v['std']:.3f}")
    st = df[df.features == "stat"][["real", "random", "selected"]].mean()
    log(f"mean melodic leap (semitones): real {st.real:.1f}, random {st.random:.1f}, selected {st.selected:.1f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("task", choices=["classify", "detect"])
    ap.add_argument("bars"); ap.add_argument("out")
    ap.add_argument("--data", default="score", choices=["score", "perf_bars", "perf_2s"])
    ap.add_argument("--runs", type=int, default=10)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--hop-train", type=int, default=8)
    ap.add_argument("--n-imit", type=int, default=300)
    ap.add_argument("--pairs", type=int, default=20000)
    ap.add_argument("--drop-perf", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    log = Log(os.path.join(a.out, "log.txt"))
    log(f"=== {' '.join(sys.argv[1:])}  ({time.strftime('%Y-%m-%d %H:%M')}) ===")
    (run_classify if a.task == "classify" else run_detect)(a, log)


if __name__ == "__main__":
    main()

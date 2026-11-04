"""
Experiment B: tell original bar transitions from spliced ones.

Data (Beethoven only):
  real     : adjacent bars (i, i+1) of the same performance
  random   : bar a followed by a random bar b from another work
  selected : b is the best of 50 candidates from other works (closest key + smallest melodic leap)
Train and test are split by work title; 5 runs.

Transition features for a -> b:
  pitch       : key_sim, key_sim_ctx (one extra bar on each side), mel_jump, bass_jump, register_jump
  timing      : density_jump, rhythm_change, held_over, first_onset, held_x_onset
  performance : vel_edge_jump, vel_mean_jump, pulse_change, poly_change (control only)

Models: logistic regression and gradient boosting, trained on random, selected or both.
Segment level: real 32-bar segments vs. segments spliced every 4 bars;
segment score = mean of the 7 highest transition scores.

Usage: python splice_detector.py <work dir> <output dir> [--runs 5] [--pairs 20000]
"""
import argparse, os, json
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score

TARGET = "Ludwig van Beethoven"
COMP = ["key_sim", "key_sim_ctx", "mel_jump", "bass_jump", "register_jump", "density_jump",
        "rhythm_change", "held_over", "first_onset", "held_x_onset"]
# pitch-only set: no timing or velocity, so cutting artefacts matter less
PITCH = ["key_sim", "key_sim_ctx", "mel_jump", "bass_jump", "register_jump"]
PERF = ["vel_edge_jump", "vel_mean_jump", "pulse_change", "poly_change"]
F = {n: i for i, n in enumerate(["note_density", "avg_duration", "std_duration", "avg_velocity", "std_velocity",
     "avg_melodic_interval", "most_common_interval", "chromatic_motion", "stepwise_motion", "melodic_thirds",
     "large_leaps", "repeated_notes", "rising_motion", "pitch_range", "mean_pitch", "std_pitch", "bass_register",
     "high_register", "pitch_class_variety", "pitch_variety", "strongest_pc_interval", "pc_entropy", "mean_ioi",
     "std_ioi", "strongest_pulse_bpm", "pulse_strength", "avg_polyphony"])}
S = {n: i for i, n in enumerate(["mel_first", "mel_last", "bass_first", "bass_last", "vel_first", "vel_last",
     "dens_first", "dens_last", "held_over", "first_onset"])}


def cos(a, b):
    return (a * b).sum(-1) / (np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1) + 1e-9)


def trans_features(X, pc, sc, a, b):
    """a, b: arrays of global bar indices. Returns [n, 14] features in COMP + PERF order."""
    ctx_a = pc[a] + pc[a - 1]; ctx_b = pc[b] + pc[np.minimum(b + 1, len(pc) - 1)]
    va = np.where(np.isnan(sc[a, S["vel_last"]]), X[a, F["avg_velocity"]], sc[a, S["vel_last"]])
    vb = np.where(np.isnan(sc[b, S["vel_first"]]), X[b, F["avg_velocity"]], sc[b, S["vel_first"]])
    lr = lambda u, v: np.abs(np.log((u + 1e-2) / (v + 1e-2)))
    cols = [
        cos(pc[a], pc[b]), cos(ctx_a, ctx_b),
        np.abs(sc[b, S["mel_first"]] - sc[a, S["mel_last"]]),
        np.abs(sc[b, S["bass_first"]] - sc[a, S["bass_last"]]),
        np.abs(X[b, F["mean_pitch"]] - X[a, F["mean_pitch"]]),
        np.abs(sc[b, S["dens_first"]] - sc[a, S["dens_last"]]),
        lr(X[b, F["mean_ioi"]], X[a, F["mean_ioi"]]),
        sc[a, S["held_over"]], sc[b, S["first_onset"]],
        sc[a, S["held_over"]] * sc[b, S["first_onset"]],
        np.abs(vb - va),
        np.abs(X[b, F["avg_velocity"]] - X[a, F["avg_velocity"]]),
        lr(X[b, F["strongest_pulse_bpm"]], X[a, F["strongest_pulse_bpm"]]),
        np.abs(X[b, F["avg_polyphony"]] - X[a, F["avg_polyphony"]]),
    ]
    return np.nan_to_num(np.stack(cols, 1).astype(np.float32))


def other_title_pick(rng, a, cand_pool, title):
    """Random splice: for each a, pick a random b from a different work."""
    B = cand_pool[rng.integers(0, len(cand_pool), len(a))]
    for _ in range(20):
        bad = title[B] == title[a]
        if not bad.any():
            break
        B[bad] = cand_pool[rng.integers(0, len(cand_pool), bad.sum())]
    return B


def smart_pick(rng, a, cand_pool, pc, sc, title, n_cand=50):
    """Selected splice: for each a, draw n_cand candidates and keep the one with the
    closest key and smallest melodic leap. Candidates from the same work are excluded."""
    C = cand_pool[rng.integers(0, len(cand_pool), (len(a), n_cand))]
    cost = (1 - cos(pc[a][:, None, :], pc[C])) + \
        np.abs(np.nan_to_num(sc[C, S["mel_first"]] - sc[a, S["mel_last"]][:, None])) / 12
    cost[title[C] == title[a][:, None]] = np.inf
    return C[np.arange(len(a)), cost.argmin(1)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("workdir"); ap.add_argument("outdir")
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--pairs", type=int, default=20000)
    a_ = ap.parse_args()
    os.makedirs(a_.outdir, exist_ok=True)

    d = np.load(os.path.join(a_.workdir, "features.npz")); e = np.load(os.path.join(a_.workdir, "edges.npz"))
    meta = pd.read_csv(os.path.join(a_.workdir, "meta.csv"))
    X, pid, pc, sc = d["X"], d["piece_id"], e["pc"], e["sc"]
    n = len(X)
    pos_in_piece = np.zeros(n, int)
    for p in np.unique(pid):
        idx = np.where(pid == p)[0]; pos_in_piece[idx] = np.arange(len(idx))
    piece_len = np.bincount(pid)[pid]
    isB = meta.canonical_composer.values[pid] == TARGET
    title = pd.factorize(meta.canonical_title)[0][pid]  # integer title codes
    sounding = X[:, F["note_density"]] > 0
    # valid a: Beethoven, not silent, with a-1 and a+1 in the same performance
    okA = isB & sounding & (pos_in_piece >= 1) & (pos_in_piece < piece_len - 2)
    okA &= np.roll(sounding, -1)  # a+1 not silent
    titlesB = np.unique(title[isB])

    rows, coefs = [], []
    for r in range(a_.runs):
        rng = np.random.default_rng(r)
        test_titles = rng.choice(titlesB, int(len(titlesB) * 0.3), replace=False)
        in_test = np.isin(title, test_titles)
        data = {}
        for part, mask in (("train", ~in_test), ("test", in_test)):
            A_all = np.where(okA & mask)[0]
            B_pool = np.where(okA & mask)[0]  # b comes from the same side of the split
            A = rng.choice(A_all, a_.pairs)
            real = trans_features(X, pc, sc, A, A + 1)
            Br = other_title_pick(rng, A, B_pool, title)
            rand = trans_features(X, pc, sc, A, Br)
            Bs = smart_pick(rng, A, B_pool, pc, sc, title)
            smart = trans_features(X, pc, sc, A, Bs)
            data[part] = dict(real=real, rand=rand, smart=smart, A=A, pool=B_pool)

        for featset, cols in (("all", COMP + PERF), ("composition_only", COMP), ("pitch_only", PITCH)):
            ci = [(COMP + PERF).index(c) for c in cols]
            for train_on in ("rand", "smart", "both"):
                if train_on == "both":  # half random, half selected splices
                    h = a_.pairs // 2
                    pos = np.concatenate([data["train"]["rand"][:h], data["train"]["smart"][h:]])
                else:
                    pos = data["train"][train_on]
                Xtr = np.concatenate([data["train"]["real"][:, ci], pos[:, ci]])
                ytr = np.r_[np.zeros(a_.pairs), np.ones(a_.pairs)]
                lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)).fit(Xtr, ytr)
                gb = HistGradientBoostingClassifier(max_iter=300, random_state=r).fit(Xtr, ytr)
                for test_on in ("rand", "smart"):
                    Xte = np.concatenate([data["test"]["real"][:, ci], data["test"][test_on][:, ci]])
                    yte = np.r_[np.zeros(a_.pairs), np.ones(a_.pairs)]
                    row = dict(run=r, features=featset, train_on=train_on, test_on=test_on,
                               auc_logreg=roc_auc_score(yte, lr.predict_proba(Xte)[:, 1]),
                               auc_boosting=roc_auc_score(yte, gb.predict_proba(Xte)[:, 1]))
                    rows.append(row); print(json.dumps(row), flush=True)
                if train_on == "both":
                    coefs.append(pd.Series(lr[-1].coef_[0], index=cols, name=f"{featset}_r{r}"))

                # segment level: real 32-bar segments vs. segments spliced every 4 bars
                for seg_kind in (("rand", "smart") if featset != "all" else ()):
                    seg_scores = {"real": [], "spliced": []}
                    T = data["test"]
                    starts = np.where(okA & in_test & (pos_in_piece < piece_len - 34))[0]
                    for kind in ("real", "spliced"):
                        for _ in range(300):
                            if kind == "real":
                                s0 = rng.choice(starts); bars = np.arange(s0, s0 + 32)
                            else:
                                # 4-bar chunks from held-out works; next start is random or selected
                                bars = []
                                st = rng.choice(T["pool"])
                                while len(bars) < 32:
                                    if st + 3 >= n or pid[st] != pid[st + 3]:
                                        st = rng.choice(T["pool"]); continue
                                    bars += list(range(st, st + 4))
                                    last = np.array([bars[-1]])
                                    st = (smart_pick(rng, last, T["pool"], pc, sc, title)[0] if seg_kind == "smart"
                                          else other_title_pick(rng, last, T["pool"], title)[0])
                                bars = np.array(bars[:32])
                            tf = trans_features(X, pc, sc, np.maximum(bars[:-1], 1), bars[1:])[:, ci]
                            sc_b = np.sort(gb.predict_proba(tf)[:, 1])[-7:].mean()
                            seg_scores[kind].append(sc_b)
                    y_seg = np.r_[np.zeros(300), np.ones(300)]
                    auc_seg = roc_auc_score(y_seg, np.r_[seg_scores["real"], seg_scores["spliced"]])
                    row = dict(run=r, features=featset, train_on=train_on, test_on=f"segment_{seg_kind}",
                               auc_boosting=auc_seg)
                    rows.append(row); print(json.dumps(row), flush=True)
        pd.DataFrame(rows).to_csv(os.path.join(a_.outdir, "splice_runs.csv"), index=False)

    df = pd.DataFrame(rows)
    summ = df.groupby(["features", "train_on", "test_on"])[["auc_logreg", "auc_boosting"]].agg(["mean", "std"]).round(3)
    summ.to_csv(os.path.join(a_.outdir, "splice_summary.csv")); print(summ.to_string())
    C = pd.concat(coefs, axis=1)
    C.to_csv(os.path.join(a_.outdir, "logreg_coefficients.csv"))
    print(C.T.groupby(lambda s: s.split("_r")[0]).mean().T.round(2).to_string())


if __name__ == "__main__":
    main()

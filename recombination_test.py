"""
Experiment A: does the classifier still say "Beethoven" when real Beethoven bars
are spliced together?

Each run:
  1. Train the GRU on a grouped split (see train_gru.py).
  2. Build test segments (32 bars) from held-out works only:
     real_beethoven    real Beethoven segments
     shuffle_within    real segments with bar order shuffled
     splice_random_k   random chunks of k bars from different Beethoven works (k = 1, 2, 4, 8, 16)
     splice_smart_4    4-bar chunks; each next chunk is the best of 50 candidates
                       (closest key + smallest melodic leap)
     mix_p             4-bar chunks, each from Beethoven with probability p, else another composer
     real_other        real segments by other composers
     splice_other_4    random 4-bar chunks from different non-Beethoven works
  3. Record the mean probability and the share predicted as Beethoven (p >= 0.5).

Usage: python recombination_test.py <work dir> <output dir> [--runs 3] [--epochs 12]
"""
import argparse, os, json
import numpy as np
import pandas as pd
import torch
from train_gru import make_segments, split, train_gru, predict, TARGET, SEG

N_PER = 400


def cos(a, b):
    return (a * b).sum(-1) / (np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1) + 1e-9)


class Pool_:
    """Bars of a set of performances (standardised features), for drawing chunks."""
    def __init__(self, pieces, Z_by_piece, title_of, pc_by_piece, mel_by_piece):
        self.pieces = [p for p in pieces if len(Z_by_piece[p]) >= 20]
        self.Z, self.title, self.pc, self.mel = Z_by_piece, title_of, pc_by_piece, mel_by_piece

    def chunk(self, rng, k, avoid_title=None):
        while True:
            p = self.pieces[rng.integers(len(self.pieces))]
            if self.title[p] != avoid_title:
                break
        s = rng.integers(0, len(self.Z[p]) - k + 1)
        return p, s


def build_splice(rng, k, pool, other_pool=None, p_beethoven=1.0):
    """Random splice of k-bar chunks; neighbouring chunks come from different works.
    With other_pool, each chunk is from pool with probability p_beethoven."""
    out, last_title = [], None
    while sum(len(c) for c in out) < SEG:
        kk = min(k, SEG - sum(len(c) for c in out))
        use = pool if (other_pool is None or rng.random() < p_beethoven) else other_pool
        p, s = use.chunk(rng, kk, last_title)
        out.append(use.Z[p][s:s + kk])
        last_title = use.title[p]
    return np.concatenate(out)[:SEG]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("workdir"); ap.add_argument("outdir")
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=12)
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    torch.set_num_threads(os.cpu_count())

    d = np.load(os.path.join(a.workdir, "features.npz"))
    e = np.load(os.path.join(a.workdir, "edges.npz"))
    meta = pd.read_csv(os.path.join(a.workdir, "meta.csv"))
    X, piece_id = d["X"], d["piece_id"]
    counts = meta.canonical_composer.value_counts()
    keep_comp = set(counts[counts > 1].index)

    S, y, pid, title = make_segments(X, piece_id, meta)
    keep = np.isin(meta.canonical_composer.values[pid], list(keep_comp))
    S, y, pid, title = S[keep], y[keep], pid[keep], title[keep]

    rows = []
    for r in range(a.runs):
        rng = np.random.default_rng(100 + r)
        tr, te = split(y, pid, title, "grouped", np.random.default_rng(r))
        mu = S[tr].reshape(-1, 27).mean(0); sd = S[tr].reshape(-1, 27).std(0) + 1e-6
        print(f"[run {r}] training GRU ...", flush=True)
        model = train_gru(((S[tr] - mu) / sd).astype(np.float32), y[tr], a.epochs, seed=r)
        torch.save(dict(state=model.state_dict(), mu=mu, sd=sd), os.path.join(a.outdir, f"gru_grouped_r{r}.pt"))

        # bar pools from held-out works
        test_pieces = np.unique(pid[te])
        Zp = {p: ((X[piece_id == p] - mu) / sd).astype(np.float32) for p in test_pieces}
        pcp = {p: e["pc"][e["piece_id"] == p] for p in test_pieces}
        melp = {p: e["sc"][e["piece_id"] == p][:, 1] for p in test_pieces}  # last melody note of each bar
        title_of = {p: meta.canonical_title[p] for p in test_pieces}
        isB = {p: meta.canonical_composer[p] == TARGET for p in test_pieces}
        poolB = Pool_([p for p in test_pieces if isB[p]], Zp, title_of, pcp, melp)
        poolO = Pool_([p for p in test_pieces if not isB[p]], Zp, title_of, pcp, melp)
        # first melody note of each bar, used to score the start of a candidate chunk
        melfirst = {p: e["sc"][e["piece_id"] == p][:, 0] for p in test_pieces}

        Zte = ((S[te] - mu) / sd).astype(np.float32)
        realB = Zte[y[te] == 1]; realO = Zte[y[te] == 0]
        conds = {"real_beethoven": realB,
                 "shuffle_within": np.stack([s[rng.permutation(SEG)] for s in realB]),
                 "real_other": realO[rng.choice(len(realO), min(N_PER, len(realO)), replace=False)]}
        for k in (1, 2, 4, 8, 16):
            conds[f"splice_random_{k}"] = np.stack([build_splice(rng, k, poolB) for _ in range(N_PER)])
        # selected splice: cost = key distance + melodic leap (in octaves) between chunks
        def smart_one():
            out, last = [], None
            while len(out) * 4 < SEG:
                cands = [poolB.chunk(rng, 4, title_of[last[0]] if last else None) for _ in range(50 if last else 1)]
                if last:
                    lp, li = last
                    cost = [(1 - cos(pcp[lp][li], pcp[p][s])) + abs(np.nan_to_num(melfirst[p][s] - melp[lp][li])) / 12
                            for p, s in cands]
                    p, s = cands[int(np.argmin(cost))]
                else:
                    p, s = cands[0]
                out.append(Zp[p][s:s + 4]); last = (p, s + 3)
            return np.concatenate(out)
        conds["splice_smart_4"] = np.stack([smart_one() for _ in range(N_PER)])
        for pb in (0.75, 0.5, 0.25, 0.0):
            conds[f"mix_{pb}"] = np.stack([build_splice(rng, 4, poolB, other_pool=poolO, p_beethoven=pb)
                                           for _ in range(N_PER)])
        conds["splice_other_4"] = np.stack([build_splice(rng, 4, poolO) for _ in range(N_PER)])

        for name, Z in conds.items():
            p = predict(model, Z)
            rows.append(dict(run=r, condition=name, n=len(Z), mean_prob=float(p.mean()),
                             called_beethoven=float((p >= 0.5).mean())))
            print(json.dumps(rows[-1]), flush=True)
        pd.DataFrame(rows).to_csv(os.path.join(a.outdir, "recombination_runs.csv"), index=False)

    df = pd.DataFrame(rows)
    order = list(dict.fromkeys(df.condition))
    summ = df.groupby("condition")[["mean_prob", "called_beethoven"]].agg(["mean", "std"]).round(3).loc[order]
    summ.to_csv(os.path.join(a.outdir, "recombination_summary.csv"))
    print(summ.to_string())


if __name__ == "__main__":
    main()

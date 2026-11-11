"""
Build bar-level datasets from ASAP.

  score      : every score (one version per movement), cut at annotated barlines
  perf_bars  : every Beethoven performance, cut at its annotated downbeats
               (notes up to 80 ms before a downbeat count as part of the new bar)
  perf_2s    : the same Beethoven performances, cut into 2-second windows
               (the original method, see extract_features.py / extract_edges.py)

Each dataset is saved as <name>.npz (X, pc, sc, piece_id, bar_len, timesig)
plus <name>_pieces.csv (one row per piece: folder, composer, group, file).

group = sonata (all movements of a sonata share a group); other works are
their own group. Splits in the experiments are made by group.

Usage: python build_bars.py <asap dir> <output dir> [score,perf_bars,perf_2s]
"""
import os
import re
import sys
import warnings
from multiprocessing import Pool

import numpy as np
import pandas as pd

# shared code (features, GRU, transition features) lives in the repository root
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import real_bars
from extract_features import piece_features
from extract_edges import piece_edges

warnings.filterwarnings("ignore")


def group_of(folder):
    """'Beethoven/Piano_Sonatas/8-1' -> 'Beethoven/Piano_Sonatas/8'."""
    return re.sub(r"/(\d+)-[^/]*$", r"/\1", folder)


def job_real(args):
    i, midi, ann = args
    try:
        return i, real_bars.piece_real_bars(midi, ann)
    except Exception as e:
        print("  skipped", midi, e, flush=True)
        return i, None


def job_real_perf(args):
    """Performances: allow 80 ms of chord spread around the annotated downbeat."""
    i, midi, ann = args
    try:
        return i, real_bars.piece_real_bars(midi, ann, tol_sec=0.08, chord=1 / 16)
    except Exception as e:
        print("  skipped", midi, e, flush=True)
        return i, None


def job_2s(args):
    i, midi, _ = args
    X = piece_features(midi)
    pc, sc = piece_edges(midi)
    n = min(len(X), len(pc))
    return i, dict(X=X[:n], pc=pc[:n], sc=sc[:n], bar_len=np.full(n, 2.0, np.float32),
                   timesig=np.array(["2s"] * n))


def build(name, pieces, fn, root, out):
    jobs = [(i, os.path.join(root, r.midi), os.path.join(root, r.ann)) for i, r in enumerate(pieces.itertuples())]
    res = [None] * len(jobs)
    with Pool() as pool:
        for k, (i, r) in enumerate(pool.imap_unordered(fn, jobs, chunksize=2), 1):
            res[i] = r
            if k % 25 == 0 or k == len(jobs):
                print(f"  {name}: {k}/{len(jobs)}", flush=True)
    keep = [i for i, r in enumerate(res) if r is not None and len(r["X"]) > 0]
    pieces = pieces.iloc[keep].reset_index(drop=True)
    res = [res[i] for i in keep]
    data = {k: np.concatenate([r[k] for r in res]) for k in ("X", "pc", "sc", "bar_len", "timesig")}
    data["piece_id"] = np.concatenate([np.full(len(r["X"]), i) for i, r in enumerate(res)])
    np.savez_compressed(os.path.join(out, f"{name}.npz"), **data)
    pieces.to_csv(os.path.join(out, f"{name}_pieces.csv"), index=False)
    b = pieces.composer.values[data["piece_id"]] == "Beethoven"
    print(f"{name}: {len(pieces)} pieces, {len(data['X'])} bars ({b.sum()} Beethoven)")


def main():
    root, out = sys.argv[1], sys.argv[2]
    os.makedirs(out, exist_ok=True)
    scores = pd.read_csv(os.path.join(root, "scores_used.csv"))
    sp = pd.DataFrame(dict(folder=scores.folder, composer=scores.composer,
                           group=scores.folder.map(group_of),
                           midi=scores.midi_score, ann=scores.midi_score_annotations))
    meta = pd.read_csv(os.path.join(root, "metadata.csv"))
    perf = meta[meta.composer == "Beethoven"]
    pp = pd.DataFrame(dict(folder=perf.folder, composer=perf.composer,
                           group=perf.folder.map(group_of),
                           midi=perf.midi_performance, ann=perf.performance_annotations))

    only = sys.argv[3].split(",") if len(sys.argv) > 3 else ["score", "perf_bars", "perf_2s"]
    if "score" in only:
        build("score", sp, job_real, root, out)
    if "perf_bars" in only:
        build("perf_bars", pp, job_real_perf, root, out)
    if "perf_2s" in only:
        build("perf_2s", pp, job_2s, root, out)


if __name__ == "__main__":
    main()

"""
Extract boundary information for each 2-second bar, used to score transitions.

Per bar:
  pc12        : pitch-class profile, weighted by sounding time, normalised
  mel_first   : highest note at the start of the bar
  mel_last    : highest note at the end of the bar
  bass_first  : lowest note at the start
  bass_last   : lowest note at the end
  vel_first   : mean velocity in the first 0.5 s
  vel_last    : mean velocity in the last 0.5 s
  dens_first  : number of onsets in the first 0.5 s
  dens_last   : number of onsets in the last 0.5 s
  held_over   : number of notes still sounding past the end of the bar
  first_onset : time from bar start to the first onset (s)

Bars line up with extract_features.py (same 2-second grid, same order).
Usage: python extract_edges.py <maestro dir> <output dir>
"""
import sys, os
import numpy as np
import pandas as pd
import pretty_midi
from multiprocessing import Pool

BAR = 2.0
EDGE = 0.5
SCALAR_NAMES = ["mel_first", "mel_last", "bass_first", "bass_last", "vel_first", "vel_last",
                "dens_first", "dens_last", "held_over", "first_onset"]


def piece_edges(path):
    m = pretty_midi.PrettyMIDI(path)
    notes = [nt for ins in m.instruments for nt in ins.notes]
    on = np.array([n.start for n in notes]); off = np.array([n.end for n in notes])
    pitch = np.array([n.pitch for n in notes]); vel = np.array([n.velocity for n in notes], float)
    nbar = int(np.ceil(m.get_end_time() / BAR))
    pc = np.zeros((nbar, 12), np.float32)
    sc = np.full((nbar, len(SCALAR_NAMES)), np.nan, np.float32)
    bar_idx = (on // BAR).astype(int)
    for b in range(nbar):
        t0, t1 = b * BAR, (b + 1) * BAR
        s = bar_idx == b
        # pitch-class profile of all notes sounding in the bar, weighted by time
        sound = (on < t1) & (off > t0)
        w = np.minimum(off[sound], t1) - np.maximum(on[sound], t0)
        np.add.at(pc[b], pitch[sound] % 12, w)
        if pc[b].sum() > 0:
            pc[b] /= pc[b].sum()
        sc[b, 8] = ((on < t1) & (off > t1 + 0.05)).sum()
        if s.sum() == 0:
            continue
        o, p, v = on[s], pitch[s], vel[s]
        first_t, last_t = o.min(), o.max()
        f = o <= first_t + 0.03; l = o >= last_t - 0.03
        sc[b, 0], sc[b, 1] = p[f].max(), p[l].max()
        sc[b, 2], sc[b, 3] = p[f].min(), p[l].min()
        e0 = o < t0 + EDGE; e1 = o >= t1 - EDGE
        sc[b, 4] = v[e0].mean() if e0.any() else np.nan
        sc[b, 5] = v[e1].mean() if e1.any() else np.nan
        sc[b, 6], sc[b, 7] = e0.sum(), e1.sum()
        sc[b, 9] = first_t - t0
    return pc, sc


def _job(a):
    i, p = a
    return i, piece_edges(p)


if __name__ == "__main__":
    root, outdir = sys.argv[1], sys.argv[2]
    meta = pd.read_csv(os.path.join(root, "maestro-v3.0.0.csv"))
    jobs = [(i, os.path.join(root, f)) for i, f in enumerate(meta.midi_filename)]
    res = [None] * len(jobs)
    with Pool() as pool:
        for k, (i, r) in enumerate(pool.imap_unordered(_job, jobs, chunksize=4)):
            res[i] = r
            if k % 100 == 0:
                print(f"{k}/{len(jobs)}", flush=True)
    pc = np.concatenate([r[0] for r in res]); sc = np.concatenate([r[1] for r in res])
    piece_id = np.concatenate([np.full(len(r[0]), i) for i, r in enumerate(res)])
    np.savez_compressed(os.path.join(outdir, "edges.npz"), pc=pc, sc=sc, piece_id=piece_id,
                        names=np.array(SCALAR_NAMES))
    print("bars:", pc.shape)

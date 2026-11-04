"""
Extract 27 features per bar from MAESTRO v3 MIDI files.

A "bar" is a 2-second window. MAESTRO files are performance MIDI with no real
barlines, and every file is marked 120 BPM in 4/4, so music21 also splits them
into 4-beat (2-second) measures.

The features follow jSymbolic definitions but are written in NumPy, which is
much faster than calling music21. Initial tempo and time signature are constant
in MAESTRO, so polyphony and pulse features are used instead.

Output: features.npz (X: [n_bars, 27], piece_id: performance index of each bar)
        and meta.csv
Usage:  python extract_features.py <maestro dir> <output dir>
"""
import sys, os
import numpy as np
import pandas as pd
import pretty_midi
from multiprocessing import Pool

BAR = 2.0  # seconds; one 4/4 bar at 120 BPM

FEATURE_NAMES = [
    "note_density",            # onsets in the bar
    "avg_duration",            # mean duration (s)
    "std_duration",            # duration spread
    "avg_velocity",            # mean velocity
    "std_velocity",            # velocity spread
    "avg_melodic_interval",    # mean melodic interval, top voice (semitones)
    "most_common_interval",    # most common melodic interval
    "chromatic_motion",        # share of 1-semitone steps
    "stepwise_motion",         # share of steps (1-2 semitones)
    "melodic_thirds",          # share of thirds (3-4 semitones)
    "large_leaps",             # share of leaps (> 7 semitones)
    "repeated_notes",          # share of repeated notes
    "rising_motion",           # share of rising moves
    "pitch_range",             # pitch range
    "mean_pitch",              # mean pitch
    "std_pitch",               # pitch spread
    "bass_register",           # share of notes <= F#3
    "high_register",           # share of notes >= C#5
    "pitch_class_variety",     # number of pitch classes used
    "pitch_variety",           # number of distinct pitches
    "strongest_pc_interval",   # interval between the two strongest pitch classes
    "pc_entropy",              # pitch-class entropy
    "mean_ioi",                # mean inter-onset interval
    "std_ioi",                 # inter-onset interval spread
    "strongest_pulse_bpm",     # strongest pulse (BPM)
    "pulse_strength",          # strength of the strongest pulse
    "avg_polyphony",           # mean number of sounding notes
]


def bar_features(on, off, pitch, vel, t0):
    """27 features for one bar. on/off/pitch/vel are the notes that start in this bar."""
    f = np.zeros(27, dtype=np.float32)
    n = len(on)
    f[0] = n
    if n == 0:
        return f
    dur = off - on
    f[1], f[2] = dur.mean(), dur.std()
    f[3], f[4] = vel.mean(), vel.std()

    # top-voice melody: highest note among onsets within 20 ms
    order = np.argsort(on)
    on_s, p_s = on[order], pitch[order]
    groups = np.floor((on_s - t0) / 0.02).astype(int)
    _, idx = np.unique(groups, return_index=True)
    mel = np.array([p_s[i:j].max() for i, j in zip(idx, list(idx[1:]) + [n])])
    if len(mel) > 1:
        iv = np.diff(mel)
        a = np.abs(iv)
        f[5] = a.mean()
        f[6] = np.bincount(a).argmax()
        f[7] = np.mean(a == 1)
        f[8] = np.mean((a >= 1) & (a <= 2))
        f[9] = np.mean((a >= 3) & (a <= 4))
        f[10] = np.mean(a > 7)
        f[11] = np.mean(a == 0)
        moving = iv[iv != 0]
        f[12] = np.mean(moving > 0) if len(moving) else 0.5

    f[13] = pitch.max() - pitch.min()
    f[14], f[15] = pitch.mean(), pitch.std()
    f[16] = np.mean(pitch <= 54)
    f[17] = np.mean(pitch >= 73)
    pc = np.bincount(pitch % 12, minlength=12).astype(float)
    f[18] = (pc > 0).sum()
    f[19] = len(np.unique(pitch))
    top2 = np.argsort(pc)[-2:]
    d = abs(int(top2[1]) - int(top2[0])) % 12
    f[20] = min(d, 12 - d) if pc[top2[0]] > 0 else 0
    p = pc / pc.sum()
    f[21] = -(p[p > 0] * np.log2(p[p > 0])).sum()

    onsets = np.unique(np.round(on_s, 2))
    if len(onsets) > 1:
        ioi = np.diff(onsets)
        f[22], f[23] = ioi.mean(), ioi.std()

    # pulse: autocorrelation of a 10 ms onset histogram, peak between 40 and 300 BPM
    hist = np.zeros(int(BAR / 0.01) + 1)
    np.add.at(hist, np.clip(((on - t0) / 0.01).astype(int), 0, len(hist) - 1), vel / 127.0)
    ac = np.correlate(hist, hist, mode="full")[len(hist) - 1:]
    if ac[0] > 0:
        lo, hi = 20, 150  # 0.2s-1.5s -> 300-40 BPM
        k = lo + np.argmax(ac[lo:hi])
        f[24] = 60.0 / (k * 0.01)
        f[25] = ac[k] / ac[0]

    # polyphony: notes sounding, sampled every 50 ms (notes from the previous bar are not counted)
    grid = t0 + np.arange(0, BAR, 0.05)
    f[26] = ((on[None, :] <= grid[:, None]) & (off[None, :] > grid[:, None])).sum(1).mean()
    return f


def piece_features(path):
    m = pretty_midi.PrettyMIDI(path)
    notes = [nt for ins in m.instruments for nt in ins.notes]
    on = np.array([nt.start for nt in notes])
    off = np.array([nt.end for nt in notes])
    pitch = np.array([nt.pitch for nt in notes])
    vel = np.array([nt.velocity for nt in notes], dtype=float)
    nbar = int(np.ceil(m.get_end_time() / BAR))
    bar_idx = (on // BAR).astype(int)
    out = np.zeros((nbar, 27), dtype=np.float32)
    for b in range(nbar):
        s = bar_idx == b
        out[b] = bar_features(on[s], off[s], pitch[s], vel[s], b * BAR)
    return out


def _job(args):
    i, path = args
    return i, piece_features(path)


if __name__ == "__main__":
    root, outdir = sys.argv[1], sys.argv[2]
    os.makedirs(outdir, exist_ok=True)
    meta = pd.read_csv(os.path.join(root, "maestro-v3.0.0.csv"))
    jobs = [(i, os.path.join(root, f)) for i, f in enumerate(meta.midi_filename)]
    feats = [None] * len(jobs)
    with Pool() as pool:
        for k, (i, F) in enumerate(pool.imap_unordered(_job, jobs, chunksize=4)):
            feats[i] = F
            if k % 100 == 0:
                print(f"{k}/{len(jobs)}", flush=True)
    piece_id = np.concatenate([np.full(len(F), i) for i, F in enumerate(feats)])
    X = np.concatenate(feats)
    np.savez_compressed(os.path.join(outdir, "features.npz"), X=X, piece_id=piece_id,
                        names=np.array(FEATURE_NAMES))
    meta.to_csv(os.path.join(outdir, "meta.csv"), index=False)
    print("bars:", X.shape)

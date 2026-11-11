"""
Cut a MIDI file into real bars using ASAP beat annotations, and compute
per-bar features in beat units.

Times are converted from seconds to quarter notes by linear interpolation
between annotated beats (each beat's length in quarters follows the time
signature), so a score and a performance of the same piece give the same bars.

Per bar this returns:
  X  : 27 features (same names and order as extract_features.py; time-based
       features are in quarter notes, and the pulse feature is a period in quarters)
  pc : 12-bin pitch-class profile weighted by sounding time
  sc : boundary values (same order as extract_edges.py; edge windows are 1 quarter)
  bar_len   : bar length in quarter notes
  timesig   : time signature in force at the bar
"""
import numpy as np
import pretty_midi

FEATURE_NAMES = [
    "note_density", "avg_duration", "std_duration", "avg_velocity", "std_velocity",
    "avg_melodic_interval", "most_common_interval", "chromatic_motion", "stepwise_motion",
    "melodic_thirds", "large_leaps", "repeated_notes", "rising_motion", "pitch_range",
    "mean_pitch", "std_pitch", "bass_register", "high_register", "pitch_class_variety",
    "pitch_variety", "strongest_pc_interval", "pc_entropy", "mean_ioi", "std_ioi",
    "strongest_pulse_bpm", "pulse_strength", "avg_polyphony",
]
EDGE_NAMES = ["mel_first", "mel_last", "bass_first", "bass_last", "vel_first", "vel_last",
              "dens_first", "dens_last", "held_over", "first_onset"]

CHORD = 1 / 32     # onsets closer than this (quarters) count as simultaneous
EDGE = 1.0         # edge window (quarters)
GRID = 1 / 24      # onset histogram resolution (quarters)


def read_annotations(path):
    """Return beat times, downbeat flags and time signatures (None where unchanged)."""
    times, is_db, ts = [], [], []
    with open(path) as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            fields = parts[2].split(",")
            times.append(float(parts[0]))
            is_db.append(fields[0] == "db")
            ts.append(fields[1] if len(fields) > 1 and fields[1] else None)
    return np.array(times), np.array(is_db), ts


def quarter_positions(is_db, ts):
    """Position of every annotated beat in quarter notes from the first beat."""
    n = len(is_db)
    q = np.zeros(n)
    db = list(np.where(is_db)[0]) + [n]
    beat_q = np.ones(n)
    cur = next((t for t in ts if t), "4/4")
    for k in range(len(db) - 1):
        a, b = db[k], db[k + 1]
        for i in range(a, b):
            if ts[i]:
                cur = ts[i]
        num, den = (int(x) for x in cur.split("/"))
        bar_q = num * 4 / den
        beat_q[a:b] = bar_q / max(b - a, 1)
    if db[0] > 0:
        beat_q[:db[0]] = beat_q[db[0]]
    q[1:] = np.cumsum(beat_q[:-1])
    return q


def to_quarters(t, beat_times, q):
    """Seconds -> quarter notes, linear between beats and extrapolated outside."""
    out = np.interp(t, beat_times, q)
    r0 = (q[1] - q[0]) / max(beat_times[1] - beat_times[0], 1e-3)
    r1 = (q[-1] - q[-2]) / max(beat_times[-1] - beat_times[-2], 1e-3)
    out = np.where(t < beat_times[0], q[0] + (t - beat_times[0]) * r0, out)
    out = np.where(t > beat_times[-1], q[-1] + (t - beat_times[-1]) * r1, out)
    return out


def load_notes(midi_path):
    m = pretty_midi.PrettyMIDI(midi_path)
    notes = [n for ins in m.instruments if not ins.is_drum for n in ins.notes]
    on = np.array([n.start for n in notes]); off = np.array([n.end for n in notes])
    pitch = np.array([n.pitch for n in notes]); vel = np.array([n.velocity for n in notes], float)
    o = np.argsort(on, kind="mergesort")
    return on[o], off[o], pitch[o], vel[o]


def bar_features(on, off, pitch, vel, t0, L, chord=CHORD):
    """27 features for one bar; on/off in beats, notes that start in [t0, t0+L)."""
    f = np.zeros(27, np.float32)
    n = len(on)
    f[0] = n
    if n == 0:
        return f
    dur = off - on
    f[1], f[2] = dur.mean(), dur.std()
    f[3], f[4] = vel.mean(), vel.std()
    groups = np.floor((on - t0) / chord).astype(int)
    _, idx = np.unique(groups, return_index=True)
    mel = np.array([pitch[i:j].max() for i, j in zip(idx, list(idx[1:]) + [n])])
    if len(mel) > 1:
        iv = np.diff(mel); a = np.abs(iv)
        f[5] = a.mean(); f[6] = np.bincount(a).argmax()
        f[7] = np.mean(a == 1); f[8] = np.mean((a >= 1) & (a <= 2)); f[9] = np.mean((a >= 3) & (a <= 4))
        f[10] = np.mean(a > 7); f[11] = np.mean(a == 0)
        moving = iv[iv != 0]
        f[12] = np.mean(moving > 0) if len(moving) else 0.5
    f[13] = pitch.max() - pitch.min()
    f[14], f[15] = pitch.mean(), pitch.std()
    f[16] = np.mean(pitch <= 54); f[17] = np.mean(pitch >= 73)
    pc = np.bincount(pitch % 12, minlength=12).astype(float)
    f[18] = (pc > 0).sum(); f[19] = len(np.unique(pitch))
    top2 = np.argsort(pc)[-2:]
    d = abs(int(top2[1]) - int(top2[0])) % 12
    f[20] = min(d, 12 - d) if pc[top2[0]] > 0 else 0
    p = pc / pc.sum(); f[21] = -(p[p > 0] * np.log2(p[p > 0])).sum()
    onsets = np.unique(np.round(on / chord) * chord)
    if len(onsets) > 1:
        ioi = np.diff(onsets); f[22], f[23] = ioi.mean(), ioi.std()
    # pulse: autocorrelation of the onset histogram, strongest period between 1/4 and 2 beats
    nb = int(np.ceil(L / GRID)) + 1
    hist = np.zeros(nb)
    np.add.at(hist, np.clip(((on - t0) / GRID).astype(int), 0, nb - 1), vel / 127.0)
    ac = np.correlate(hist, hist, mode="full")[nb - 1:]
    lo, hi = int(0.25 / GRID), min(int(2.0 / GRID) + 1, nb - 1)
    if ac[0] > 0 and hi > lo:
        k = lo + np.argmax(ac[lo:hi])
        f[24] = k * GRID          # period in beats (column name kept for compatibility)
        f[25] = ac[k] / ac[0]
    grid = t0 + np.arange(0, L, 1 / 8)
    f[26] = ((on[None, :] <= grid[:, None]) & (off[None, :] > grid[:, None])).sum(1).mean()
    return f


def piece_real_bars(midi_path, ann_path, tol_sec=0.0, chord=CHORD):
    """tol_sec: in performances the annotated downbeat often falls in the middle of the
    downbeat chord, so notes played up to tol_sec before a downbeat are moved into the
    new bar. chord: onsets closer than this (quarters) count as simultaneous."""
    times, is_db, ts = read_annotations(ann_path)
    on_s, off_s, pitch, vel = load_notes(midi_path)
    q = quarter_positions(is_db, ts)
    on, off = to_quarters(on_s + tol_sec, times, q), to_quarters(off_s + tol_sec, times, q)
    db_idx = np.where(is_db)[0]
    cur, ts_at = next((t for t in ts if t), "4/4"), []
    for i in range(len(times)):
        if ts[i]:
            cur = ts[i]
        if is_db[i]:
            ts_at.append(cur)
    starts = q[db_idx]
    last_num, last_den = (int(x) for x in ts_at[-1].split("/"))
    ends = np.r_[starts[1:], starts[-1] + last_num * 4 / last_den]
    nbar = len(starts)
    X = np.zeros((nbar, 27), np.float32); PC = np.zeros((nbar, 12), np.float32)
    SC = np.full((nbar, len(EDGE_NAMES)), np.nan, np.float32)
    for b in range(nbar):
        t0, t1 = starts[b], ends[b]
        L = t1 - t0
        s = (on >= t0) & (on < t1)
        X[b] = bar_features(on[s], off[s], pitch[s], vel[s], t0, L, chord)
        sound = (on < t1) & (off > t0)
        w = np.minimum(off[sound], t1) - np.maximum(on[sound], t0)
        np.add.at(PC[b], pitch[sound] % 12, w)
        if PC[b].sum() > 0:
            PC[b] /= PC[b].sum()
        SC[b, 8] = ((on < t1) & (off > t1 + 0.1)).sum()
        if s.sum() == 0:
            continue
        o, p, v = on[s], pitch[s], vel[s]
        fst, lst = o.min(), o.max()
        f = o <= fst + chord; l = o >= lst - chord
        SC[b, 0], SC[b, 1] = p[f].max(), p[l].max()
        SC[b, 2], SC[b, 3] = p[f].min(), p[l].min()
        e0 = o < t0 + min(EDGE, L / 2); e1 = o >= t1 - min(EDGE, L / 2)
        SC[b, 4] = v[e0].mean() if e0.any() else np.nan
        SC[b, 5] = v[e1].mean() if e1.any() else np.nan
        SC[b, 6], SC[b, 7] = e0.sum(), e1.sum()
        SC[b, 9] = fst - t0
    return dict(X=X, pc=PC, sc=SC, bar_len=(ends - starts).astype(np.float32),
                timesig=np.array(ts_at))

"""
Download the parts of the ASAP dataset used here, file by file, from GitHub.

For every score (one version per movement):
  midi_score.mid, midi_score_annotations.txt
For every Beethoven performance:
  <performance>.mid, <performance>_annotations.txt

Annotation files list beat times; lines marked "db" are downbeats (barlines),
with the time signature where it changes.

ASAP: Foscarin et al., "ASAP: a dataset of aligned scores and performances
for piano transcription", ISMIR 2020. https://github.com/fosfrancesco/asap-dataset

Usage: python download_asap.py <output dir>
Re-running skips files that are already there.
"""
import os
import re
import sys
import time
import urllib.request

import pandas as pd

BASE = "https://raw.githubusercontent.com/fosfrancesco/asap-dataset/master/"


def fetch(rel, out_root, tries=3):
    dst = os.path.join(out_root, rel)
    if os.path.exists(dst) and os.path.getsize(dst) > 0:
        return "skip"
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    url = BASE + urllib.request.quote(rel)
    for k in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                data = r.read()
            with open(dst, "wb") as f:
                f.write(data)
            return "ok"
        except Exception as e:
            if k == tries - 1:
                print("  failed:", rel, e)
                return "fail"
            time.sleep(2 * (k + 1))


def pick_versions(scores):
    """One folder per movement: prefer the version without repeats if there is one."""
    scores = scores.copy()
    scores["movement"] = scores.folder.str.replace(r"_no.*$", "", regex=True)
    scores["no_repeat"] = scores.folder.str.contains("_no_repeat")
    scores = scores.sort_values(["movement", "no_repeat"], ascending=[True, False])
    return scores.drop_duplicates("movement")


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else "asap"
    os.makedirs(out, exist_ok=True)
    print("metadata:", fetch("metadata.csv", out))
    meta = pd.read_csv(os.path.join(out, "metadata.csv"))
    scores = pick_versions(meta.drop_duplicates("folder"))
    perfs = meta[meta.composer == "Beethoven"]

    jobs = []
    for _, r in scores.iterrows():
        jobs += [r.midi_score, r.midi_score_annotations]
    for _, r in perfs.iterrows():
        jobs += [r.midi_performance, r.performance_annotations]
    print(f"{len(scores)} scores, {len(perfs)} Beethoven performances, {len(jobs)} files")

    counts = {"ok": 0, "skip": 0, "fail": 0}
    for i, rel in enumerate(jobs, 1):
        counts[fetch(rel, out)] += 1
        if i % 50 == 0 or i == len(jobs):
            print(f"  {i}/{len(jobs)}  {counts}", flush=True)
    scores.drop(columns=["no_repeat"]).to_csv(os.path.join(out, "scores_used.csv"), index=False)
    print("done:", counts)


if __name__ == "__main__":
    main()

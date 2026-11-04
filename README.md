# Thematic Composer Classification under Recombination

A reproduction of the segment-level Beethoven classifier from Galajda & Hua (IEEE MIPR 2024) on MAESTRO v3, followed by two experiments. The first tests whether the classifier can be fooled by imitations built from recombined real Beethoven bars. The second tests whether features of bar-to-bar transitions can detect such recombination.

## Key findings

1. **The reproduction works.** A two-layer GRU on 27 per-bar features reaches ROC AUC ≈ 0.96 for "Beethoven vs. other composers". The result holds when the data are split by piece, split by recording year, or stripped of dynamics and duration features.
2. **Recombination fools the classifier.** Segments spliced from bars of *different* Beethoven works are classified as Beethoven 83–92% of the time, matching or exceeding genuine Beethoven (83%). Splicing also raises the Beethoven rate for non-Beethoven material, from 8% to 25%.
3. **Transitions reveal recombination.** A detector that sees only features of bar-to-bar transitions separates spliced from real 32-bar segments with AUC 0.94–0.96, or 0.82–0.88 with pitch-based features only.
4. **Selected splicing leaves its own fingerprint.** When each splice is chosen to minimise melodic leap and key distance, transitions become *smoother* than real Beethoven (mean melodic leap 1.6 vs. 9.1 semitones). A detector trained on only one splicing strategy does not generalise to the other.

## Background

Galajda & Hua (2024) classify composer style per segment rather than per piece. Their binary "Beethoven vs. rest" classifier reports 85–87% precision and 100% recall on MAESTRO. Their earlier Neural-Base system (Galajda & Hua, 2023) generates Beethoven-style music by retrieving and recombining real bars. This repository asks whether a segment-level composer classifier can distinguish such recombinations from genuine works.

## Data

[MAESTRO v3.0.0](https://magenta.tensorflow.org/datasets/maestro) (MIDI only, ~58 MB): 1,276 performances by 60 composers. Following the original paper, composers with a single work are removed, leaving 35 composers. MAESTRO is licensed under CC BY-NC-SA 4.0 and is **not included** in this repository. Download `maestro-v3.0.0-midi.zip` and unzip it before running the scripts.

## Method

### Bars

MAESTRO files are performance MIDI. They have no notated barlines, and every file declares 120 BPM in 4/4. music21 therefore segments them into 4-beat (2-second) "measures", so the original paper's per-measure features are in effect 2-second windows. This repository uses the same unit: 358,815 bars in total.

### Features (`extract_features.py`)

27 features per bar, following jSymbolic definitions and reimplemented in NumPy:

- note density, duration (mean and spread), velocity (mean and spread)
- melodic intervals of the top voice: mean, mode, and the fractions of chromatic, stepwise, third, large-leap, repeated and rising motion
- pitch range, mean and spread; the share of bass and treble registers
- pitch and pitch-class variety, pitch-class entropy, the interval between the two strongest pitch classes
- inter-onset interval (mean and spread), strongest rhythmic pulse and its strength, average polyphony

Initial tempo and time signature, used in the original feature set, are constant in MAESTRO and are replaced by the polyphony and pulse features. The reimplementation processes the full dataset in about 2 minutes on 2 CPU cores; running music21's jSymbolic extractors directly is estimated at about 19 hours.

### Segments and model (`train_gru.py`)

- **Segments:** 32 consecutive bars per segment, non-overlapping. This gives 10,470 segments, 13.8% of them Beethoven.
- **Model:** two GRU layers (512 units each), then batch normalisation, fully connected layers (128 → 32 → 1) and dropout 0.4.
- **Training:** binary cross-entropy weighted for class imbalance; Adam, learning rate 1e-3, 12 epochs.
- **Baseline:** logistic regression on the mean feature vector of each segment, which ignores bar order.

### Data splits

| Split | Description |
|---|---|
| `random` | Segments shuffled and split 80/20, as in the original paper |
| `grouped` | Split by `canonical_title`: all performances of a work fall on the same side |
| `year` | Two recording years held out; works that also appear in training years are removed from training |
| `grouped --drop-perf` | Grouped split without velocity, pulse-strength and duration features, which depend heavily on the performer |

## Results

### Reproduction

GRU results; mean ± standard deviation over runs:

| Split | Runs | Precision | Recall | F1 | AUC | AUC (logistic regression) |
|---|---|---|---|---|---|---|
| random | 5 | 0.69 ± 0.05 | 0.86 ± 0.03 | 0.76 | 0.96 | 0.80 |
| grouped | 5 | 0.73 ± 0.11 | 0.80 ± 0.09 | 0.76 | 0.96 | 0.79 |
| year | 3 | 0.78 ± 0.07 | 0.80 ± 0.13 | 0.78 | 0.96 | 0.80 |
| grouped, no performance features | 3 | 0.60 ± 0.14 | 0.90 ± 0.03 | 0.71 | 0.96 | 0.75 |

The original paper reports 85–87% precision and 100% recall. AUC is unchanged by the stricter splits, so overlap of works between training and test sets does not explain the performance. The GRU clearly outperforms the order-blind baseline. Yet no single feature separates Beethoven from other composers by much (all |Cohen's d| < 0.2; see `feature_effects.txt`).

### Experiment A: Recombination test (`recombination_test.py`)

- **Training:** GRU on the grouped split, 3 runs.
- **Imitations:** built only from held-out works. Each condition has 400 segments of 32 bars, except the real and within-segment-shuffled conditions, which use all held-out Beethoven segments (about 230).
- **"Selected" splicing:** each next chunk is the best of 50 random candidates by key similarity plus melodic leap. This is a simple stand-in for retrieval-based generation.

| Condition | Classified as Beethoven |
|---|---|
| Real Beethoven segments | 83% ± 3 |
| Bars shuffled within each segment | 61% ± 9 |
| Random splice across Beethoven works, 1-bar chunks | 91% ± 7 |
| Random splice across Beethoven works, 2-bar chunks | 92% ± 4 |
| Random splice across Beethoven works, 4-bar chunks | 86% ± 3 |
| Random splice across Beethoven works, 8-bar chunks | 85% ± 2 |
| Random splice across Beethoven works, 16-bar chunks | 83% ± 5 |
| Selected splice, 4-bar chunks | 85% ± 6 |
| Mixed: 75% of chunks from Beethoven, rest from other composers | 73% ± 7 |
| Mixed: 50% of chunks from Beethoven | 61% ± 4 |
| Mixed: 25% of chunks from Beethoven | 46% ± 5 |
| Mixed: 0% of chunks from Beethoven | 29% ± 7 |
| Real segments by other composers | 8% ± 1 |
| Random splice across other composers' works, 4-bar chunks | 25% ± 2 |

- **The classifier does not penalise recombination.** Splices of real Beethoven bars score as high as genuine segments, or higher.
- **Splicing itself pushes predictions toward "Beethoven".** One possible explanation is that abrupt contrast between passages is learned as a Beethoven trait.
- **Open question:** shuffling bars *within* a segment lowers the score (83% → 61%), but splicing *across* works does not.

### Experiment B: Transition detector (`extract_edges.py`, `splice_detector.py`)

- **Task:** classify a bar pair a → b as an original transition (adjacent bars of the same performance) or a splice (b taken from a different Beethoven work).
- **Splices:** chosen either randomly or by the selection rule above.
- **Splits and size:** train and test are split by work title, 30% of titles held out; 5 runs, 20,000 pairs per class.

Transition features:

| Group | Features |
|---|---|
| Pitch | pitch-class cosine similarity (the two bars alone, and with one bar of context on each side); melodic leap; bass leap; register change |
| Timing | onset-density change at the boundary; change in inter-onset interval; notes held over the barline; delay of the first onset; their interaction |
| Performance (control) | velocity change at the boundary and between bar means; pulse change; polyphony change |

- **Models:** logistic regression and gradient boosting.
- **Segment score:** the mean of the 7 most suspicious of a segment's 31 transitions. With 4-bar chunks, a 32-bar segment contains 7 splice points.

**Detector trained on both splice types** (gradient boosting AUC, held-out works):

| Features | Pair: random | Pair: selected | 32-bar segment: random | 32-bar segment: selected |
|---|---|---|---|---|
| All (including performance) | 0.93 | 0.92 | – | – |
| Pitch and timing | 0.90 | 0.88 | 0.96 | 0.94 |
| Pitch only | 0.85 | 0.82 | 0.88 | 0.82 |

**Cross-strategy generalisation** (32-bar segments, pitch and timing features):

| Trained on → tested on | AUC |
|---|---|
| random → random | 0.99 |
| random → selected | 0.73 |
| selected → selected | 0.97 |
| selected → random | 0.56 |

**Mean transition features** (`transition_feature_means.csv`, Beethoven, 20,000 pairs):

| Feature | Real | Random splice | Selected splice |
|---|---|---|---|
| Melodic leap (semitones) | 9.1 | 15.1 | **1.6** |
| Key similarity (two bars) | 0.68 | 0.37 | 0.68 |
| Key similarity (with context) | 0.71 | 0.44 | 0.65 |
| Register change (semitones) | 5.0 | 9.7 | 7.5 |
| Onset-density change | 2.0 | 3.7 | 3.6 |

- **The selection rule overshoots.** Selected splices match real transitions on bar-level key similarity but have far smaller melodic leaps than real music. That is why a detector trained on random splices underestimates them, and why one trained on selected splices treats large leaps as evidence of authenticity.
- **The most informative cues** in the logistic regression, standardised coefficients from training on both types: contextual key similarity (−0.64), onset-density jump (+0.76), register jump (+0.63), and the held-note × first-onset interaction (−0.55). See `logreg_coefficients.csv`.

## Implementation notes

- **Segment count.** The original paper reports STRIDE = 32 and OFFSET = 31 (maximum overlap) and 11,047 training segments. The bar count here, divided by 32, gives 10,470 non-overlapping segments after the same composer filtering. The reported figure is therefore consistent with non-overlapping segmentation.
- **Splitting by title.** Works are grouped by `canonical_title`. Movements of one work listed under different titles may still fall on both sides. The year-based split partly addresses this. The same pianist may appear in more than one competition year.

## Limitations

- **Feature implementation.** Features follow jSymbolic definitions but are not identical to music21's implementation.
- **Splice points.** Splices fall on 2-second windows of performance MIDI rather than notated barlines, and join performances with different tempi and dynamics. The detection scores in Experiment B are therefore likely optimistic; pitch-only results are the more conservative estimate.
- **Selection rule.** It is a simple heuristic, not the hash-based retrieval used by Neural-Base.
- **Scope.** Piano repertoire only, one target composer, 3–5 runs per setting.

## Reproducing

Requires Python ≥ 3.10. Install the dependencies:

```bash
pip install -r requirements.txt
```

Then run:

```bash
unzip maestro-v3.0.0-midi.zip

# per-bar features and boundary information (~2–3 min each)
python extract_features.py maestro-v3.0.0 work
python extract_edges.py maestro-v3.0.0 work

# reproduction
python train_gru.py work out_main --runs 5 --modes random,grouped
python train_gru.py work out_year --runs 3 --modes year
python train_gru.py work out_noperf --runs 3 --modes grouped --drop-perf

# Experiment A
python recombination_test.py work out_recomb --runs 3

# Experiment B
python splice_detector.py work out_splice --runs 5
python transition_stats.py work out_splice/transition_feature_means.csv
```

On a 2-core CPU, one GRU run takes about 4.5 minutes. Experiment B takes about 3 minutes in total.

## Repository structure

| File | Contents |
|---|---|
| `extract_features.py` | 27 features per 2-second bar |
| `extract_edges.py` | Boundary information per bar: pitch-class profile, first and last melody and bass notes, edge velocity and density, held notes |
| `train_gru.py` | GRU classifier, data splits, logistic-regression baseline, within-segment shuffle test |
| `recombination_test.py` | Experiment A |
| `splice_detector.py` | Experiment B |
| `transition_stats.py` | Mean transition features by splice type |
| `all_runs.csv` | Per-run results of the reproduction (16 runs) |
| `feature_effects.txt` | Single-feature effect sizes, Beethoven vs. others |
| `recombination_runs.csv` | Per-run results of Experiment A |
| `splice_runs.csv` | Per-run results of Experiment B |
| `logreg_coefficients.csv` | Logistic-regression coefficients of the transition detector |
| `transition_feature_means.csv` | Mean transition features for real and spliced pairs |

## References

- J. E. Galajda and K. A. Hua, "Automated Thematic Composer Classification Using Segment Retrieval," *IEEE MIPR*, 2024, pp. 162–168. doi:10.1109/MIPR62202.2024.00032
- J. E. Galajda and K. A. Hua, "Neural-Base Music Generation for Intelligence Duplication," arXiv:2310.13691, 2023.
- C. Hawthorne et al., "Enabling Factorized Piano Music Modeling and Generation with the MAESTRO Dataset," *ICLR*, 2019.
- C. McKay and I. Fujinaga, "jSymbolic: A Feature Extractor for MIDI Files," *ICMC*, 2006.

# Thematic Composer Classification under Recombination

This project reproduces the segment-level Beethoven classifier from Galajda and Hua (IEEE MIPR 2024) on MAESTRO v3, and then uses it to ask two questions. Specifically, can the classifier be fooled by imitations built from real Beethoven bars? And if it can, can a detector that only looks at the transitions between bars catch these imitations instead? I first ran both experiments on MAESTRO, and then repeated them on ASAP scores with real barlines to check whether the results depend on how I cut the music.

## Key findings

1. The reproduction works. A two-layer GRU on 27 features per bar reaches an AUC of about 0.96 for "Beethoven or not", and the result stays the same when I split the data by piece, by recording year, or remove the dynamics and duration features.
2. Recombination fools the classifier. When I spliced together bars from different Beethoven works, 83 to 92 percent of the segments were classified as Beethoven, which is as often as or more often than real Beethoven (83 percent). On MAESTRO, splicing also raised the Beethoven rate of non-Beethoven music from 8 to 25 percent, but this did not hold when I repeated the test on scores.
3. Transitions reveal recombination. A detector that only sees features of the transitions between bars separates spliced 32-bar segments from real ones with an AUC of 0.94 to 0.96, or 0.82 to 0.88 with pitch features only.
4. Selected splicing leaves its own fingerprint. When each splice is chosen to keep the key close and the melodic leap small, the transitions become smoother than real Beethoven (a mean melodic leap of 1.6 semitones against 9.1). A detector trained on only one way of splicing does not work on the other.
5. The results hold with real barlines. When I repeated both experiments on ASAP scores, the transition detector worked as well as with 2-second windows, and the classifier still accepted spliced imitations.

## Background

Galajda and Hua (2024) classify composer style segment by segment instead of piece by piece. Their "Beethoven or not" classifier reports a precision of 85 to 87 percent and a recall of 100 percent on MAESTRO. In their earlier Neural-Base system (Galajda and Hua, 2023), they generate music in Beethoven's style by retrieving real bars and recombining them. This raises a natural question, which is whether a segment-level composer classifier can tell such recombinations apart from real works. Since Neural-Base outputs are not publicly available, I simulated this way of composing by splicing real Beethoven bars myself.

## Data

The main experiments use [MAESTRO v3.0.0](https://magenta.tensorflow.org/datasets/maestro), MIDI only (about 58 MB), which contains 1,276 performances by 60 composers. Following the original paper, I removed composers with only one work, which leaves 35 composers. MAESTRO is licensed under CC BY-NC-SA 4.0, so it is not included in this repository. To run the scripts, download `maestro-v3.0.0-midi.zip` and unzip it first.

The score-based replication uses ASAP (Foscarin et al., 2020), which I downloaded file by file from its GitHub repository. It is not included here either, and its terms are given in that repository.

## Method

### Bars

MAESTRO files are performance MIDI, so they have no notated barlines, and every file is marked 120 BPM in 4/4. Because of this, music21 cuts them into 4-beat measures, which are 2 seconds long. This means the "per-measure" features in the original paper are in fact 2-second windows. I used the same unit, which gives 358,815 bars in total.

### Features (`extract_features.py`)

I computed 27 features per bar, following the jSymbolic definitions but written in NumPy. They cover the following.

- Note density, and the mean and spread of durations and velocities
- Melodic intervals of the top voice, including the mean, the most common interval, and the share of chromatic, stepwise, third, large-leap, repeated and rising motion
- Pitch range, mean and spread, and the share of notes in the bass and treble registers
- Pitch variety, pitch-class variety, pitch-class entropy, and the interval between the two strongest pitch classes
- The mean and spread of inter-onset intervals, the strongest rhythmic pulse and its strength, and average polyphony

Since initial tempo and time signature are the same for every MAESTRO file, I replaced these two features from the original set with polyphony and pulse features. My version processes the whole dataset in about 2 minutes on 2 CPU cores, while running music21's jSymbolic extractors directly would take about 19 hours.

### Segments and model (`train_gru.py`)

Each segment is 32 consecutive bars, with no overlap, which gives 10,470 segments, 13.8 percent of them by Beethoven. The model has two GRU layers with 512 units each, followed by batch normalisation, fully connected layers (128, 32 and 1 units) and a dropout of 0.4. I trained it for 12 epochs with Adam at a learning rate of 1e-3, using binary cross-entropy weighted for the class imbalance. As a baseline, I also trained a logistic regression on the mean feature vector of each segment, which ignores the order of the bars.

### Data splits

I used four ways of splitting the data, to check whether the result depends on overlap between training and test sets.

| Split | Description |
|---|---|
| `random` | Segments shuffled and split 80/20, as in the original paper |
| `grouped` | Split by `canonical_title`, so all performances of a work fall on the same side |
| `year` | Two recording years held out, and works that also appear in the training years removed from training |
| `grouped --drop-perf` | Grouped split without the velocity, pulse-strength and duration features, which depend heavily on the performer |

## Results

### Reproduction

The table shows the GRU results as the mean and standard deviation over runs.

| Split | Runs | Precision | Recall | F1 | AUC | AUC (logistic regression) |
|---|---|---|---|---|---|---|
| random | 5 | 0.69 ± 0.05 | 0.86 ± 0.03 | 0.76 | 0.96 | 0.80 |
| grouped | 5 | 0.73 ± 0.11 | 0.80 ± 0.09 | 0.76 | 0.96 | 0.79 |
| year | 3 | 0.78 ± 0.07 | 0.80 ± 0.13 | 0.78 | 0.96 | 0.80 |
| grouped, no performance features | 3 | 0.60 ± 0.14 | 0.90 ± 0.03 | 0.71 | 0.96 | 0.75 |

The original paper reports a precision of 85 to 87 percent and a recall of 100 percent, so my precision is lower and my recall does not reach 100 percent. However, the AUC does not change under the stricter splits, which shows that overlap of works between training and test sets does not explain the result. The GRU also does much better than the baseline that ignores bar order. On the other hand, no single feature separates Beethoven from the other composers by much (every |Cohen's d| is below 0.2, see `feature_effects.txt`). This gap disappears on scores, see the score-based replication below.

### Experiment A: Recombination test (`recombination_test.py`)

In this experiment, I trained the GRU on the grouped split 3 times, and built imitations only from works that the model had not seen. Each condition has 400 segments of 32 bars, except the real and shuffled conditions, which use all held-out Beethoven segments (about 230). For "selected" splicing, each next chunk is the best of 50 random candidates, based on key similarity and melodic leap. This is a simple way to imitate a system that retrieves the next bar that fits.

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
| Mixed, 75% of chunks from Beethoven and the rest from other composers | 73% ± 7 |
| Mixed, 50% of chunks from Beethoven | 61% ± 4 |
| Mixed, 25% of chunks from Beethoven | 46% ± 5 |
| Mixed, 0% of chunks from Beethoven | 29% ± 7 |
| Real segments by other composers | 8% ± 1 |
| Random splice across other composers' works, 4-bar chunks | 25% ± 2 |

The results show that the classifier does not penalise recombination, since splices of real Beethoven bars score as high as real segments or even higher. This means it mainly judges whether each bar sounds like Beethoven, rather than whether the bars belong together. Splicing also pushed non-Beethoven music toward "Beethoven" (8 to 25 percent). One possible explanation is that the model learned sudden contrast between passages as a Beethoven trait, but this did not hold on scores. One question is still open. Shuffling bars within a segment lowers the score (83 to 61 percent), but splicing across works does not.

### Experiment B: Transition detector (`extract_edges.py`, `splice_detector.py`)

Based on Experiment A, I changed the question from "does this sound like Beethoven?" to "were these two bars next to each other in the original?" The detector classifies a pair of bars a → b as either an original transition, meaning two adjacent bars of the same performance, or a splice, where b comes from a different Beethoven work. The splices are chosen either randomly or with the selection rule above. I split training and test data by work title, holding out 30 percent of the titles, and used 20,000 pairs per class in each of 5 runs.

| Group | Transition features |
|---|---|
| Pitch | Pitch-class similarity between the two bars, and with one more bar of context on each side, melodic leap, bass leap, register change |
| Timing | Change in onset density at the boundary, change in inter-onset interval, notes held over the barline, delay of the first onset, and the interaction of the last two |
| Performance (control) | Velocity change at the boundary and between bar means, pulse change, polyphony change |

I trained both a logistic regression and a gradient boosting model. To score a whole 32-bar segment, I took the mean of its 7 most suspicious transitions out of 31, since a segment spliced every 4 bars contains 7 splice points.

When the detector is trained on both kinds of splices, it reaches the following AUC (gradient boosting, held-out works).

| Features | Pair: random | Pair: selected | 32-bar segment: random | 32-bar segment: selected |
|---|---|---|---|---|
| All, including performance | 0.93 | 0.92 | – | – |
| Pitch and timing | 0.90 | 0.88 | 0.96 | 0.94 |
| Pitch only | 0.85 | 0.82 | 0.88 | 0.82 |

However, when the detector is trained on only one way of splicing, it does not work on the other (32-bar segments, pitch and timing features).

| Trained on → tested on | AUC |
|---|---|
| random → random | 0.99 |
| random → selected | 0.73 |
| selected → selected | 0.97 |
| selected → random | 0.56 |

To understand why, I compared the mean transition features of real and spliced pairs (`transition_feature_means.csv`, Beethoven, 20,000 pairs).

| Feature | Real | Random splice | Selected splice |
|---|---|---|---|
| Melodic leap (semitones) | 9.1 | 15.1 | 1.6 |
| Key similarity (two bars) | 0.68 | 0.37 | 0.68 |
| Key similarity (with context) | 0.71 | 0.44 | 0.65 |
| Register change (semitones) | 5.0 | 9.7 | 7.5 |
| Onset-density change | 2.0 | 3.7 | 3.6 |

The table shows that the selection rule goes too far. Selected splices match real transitions in key similarity between two bars, but their melodic leaps are much smaller than in real music. This is why a detector trained on random splices underestimates them, and why a detector trained on selected splices treats large leaps as a sign of a real transition. In the logistic regression trained on both kinds of splices, the most useful cues are the key similarity with context (standardised coefficient −0.64), the jump in onset density (+0.76), the jump in register (+0.63), and the interaction between held notes and the delay of the first onset (−0.55), see `logreg_coefficients.csv`.

## Score-based replication (real barlines)

The experiments above have one clear weakness. Since MAESTRO only contains performance MIDI, I cut every piece into 2-second windows, so many splices fall in the middle of a phrase and join two performances with different tempi. This means the detector could be picking up the cut itself rather than the music. To test this, I repeated both experiments on ASAP (Foscarin et al., 2020), which contains score MIDI with annotated barlines and performances with annotated beats. The scripts and results are in `score_replication/`.

### Data and changes to the method

I used 223 scores, one version per movement. 57 of them are Beethoven sonata movements from 28 sonatas (12,182 bars), and the other 166 are works by other composers, mostly Bach (59), Chopin (34), Liszt (17), Schubert (13), Haydn (11), Schumann (10) and Mozart (6). I also used 271 Beethoven performances with beat annotations.

Compared with the MAESTRO experiments, I made four changes. Firstly, bars are cut at the annotated downbeats, and timing features are measured in quarter notes instead of seconds. Secondly, every split holds out whole sonatas, so all movements of a sonata stay on the same side, and each setting is repeated 10 times. Thirdly, in performances the annotated downbeat often falls in the middle of the downbeat chord, so notes played up to 80 ms before it are counted as part of the new bar. Without this step, the melodic leap at real transitions was much larger than in the scores. Finally, I added two harder ways of splicing, one that only joins bars with the same time signature and one that only splices at 4-bar boundaries. In the 4-bar case, the real transitions are also taken at 4-bar boundaries, so the comparison stays fair.

### Transition detector

The table below shows the gradient boosting AUC with pitch features only, averaged over 10 runs.

| Data | Pair: random | Pair: selected | Pair: selected, same meter, 4-bar aligned | Segment: random | Segment: selected |
|---|---|---|---|---|---|
| Scores, real barlines | 0.89 | 0.85 | 0.84 | 0.91 | 0.82 |
| Performances, real barlines | 0.89 | 0.84 | 0.83 | 0.91 | 0.82 |
| Same performances, 2-second windows | 0.86 | 0.83 | – | 0.91 | 0.83 |

The three rows are almost the same. This means the 2-second windows were not what the detector was picking up, since it works just as well on clean scores with real barlines. When timing features are added, the score results go up to 0.95, 0.93 and 0.92 for single transitions, and 0.98 and 0.95 for segments. The "too smooth" fingerprint of selected splices is weaker with real barlines, with a mean melodic leap of 6.0 semitones against 8.7 for real transitions (1.9 against 8.5 on the same performances with 2-second windows), but it still goes in the same direction.

### Composer classifier trained on scores

I also retrained the GRU classifier on the scores, holding out whole sonatas. It reached an AUC of 0.87 ± 0.03, with a precision of 0.73 and a recall of 0.70. A logistic regression on segment means reached 0.86.

| Condition | Classified as Beethoven |
|---|---|
| Real Beethoven segments | 70% ± 14 |
| Random splice across sonatas, 1–8-bar chunks | 80–86% |
| Random splice, same time signature | 82% ± 11 |
| Random splice, 4-bar aligned | 83% ± 12 |
| Selected splice, same time signature | 74% ± 11 |
| Bars shuffled within each segment | 65% ± 15 |
| Mixed, 50% of chunks from Beethoven | 45% ± 16 |
| Real segments by other composers | 16% ± 6 |
| Random splice across other composers' works | 11% ± 7 |
| Haydn and Mozart segments | 15% ± 20 (3–29 test segments per run) |

Overall, the classifier still does not penalise recombination. Random splices of Beethoven bars were labeled Beethoven more often than real Beethoven segments in all 10 runs, by 14 points on average, and selected splices were labeled Beethoven about as often as real ones.

However, two results from MAESTRO did not hold on scores. Firstly, on MAESTRO, splicing raised the Beethoven rate of non-Beethoven music from 8 to 25 percent, but on scores it went down from 16 to 11 percent. I therefore treat this effect as specific to performance MIDI cut into 2-second windows. Secondly, on MAESTRO the GRU was much better than the order-blind baseline (0.96 against 0.79), while on scores the two are almost the same (0.87 against 0.86). This suggests that part of the GRU's advantage on MAESTRO came from performance timing rather than from the composition, although the two datasets are different, so I cannot say this for sure.

## Implementation notes

There are two details about the setup that are worth mentioning. Firstly, the original paper reports a STRIDE of 32 and an OFFSET of 31, which means maximum overlap, and 11,047 training segments. However, after the same composer filtering, the number of bars here divided by 32 gives 10,470 segments without any overlap, so the reported number fits non-overlapping segments better. Secondly, I grouped works by `canonical_title`, so movements of one work that are listed under different titles may still end up on both sides. The year-based split partly addresses this, although the same pianist may appear in more than one competition year.

## Limitations

There are several limitations to this project. Firstly, my features follow the jSymbolic definitions but are not identical to music21's implementation. The second limitation is the splice points in the MAESTRO experiments, which fall on 2-second windows instead of notated barlines and join performances with different tempi and dynamics. I tested this directly in the score-based replication, and the detection results did not change. Thirdly, the selection rule is a simple heuristic, not the hash-based retrieval used by Neural-Base, so real Neural-Base outputs may behave differently. Finally, the project only covers piano music and one target composer. On scores, each test set holds only about 6 of the 28 Beethoven sonatas, and the Haydn and Mozart estimates come from very few segments.

## Reproducing

The code requires Python 3.10 or later. First install the dependencies.

```bash
pip install -r requirements.txt
```

Then run the MAESTRO experiments.

```bash
unzip maestro-v3.0.0-midi.zip

# per-bar features and boundary information (about 2–3 min each)
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

On a 2-core CPU, one GRU run takes about 4.5 minutes, and Experiment B takes about 3 minutes in total.

The score-based replication runs from the repository root as well.

```bash
# download ASAP scores, annotations and Beethoven performances (about 1,000 small files)
python score_replication/download_asap.py asap

# cut scores and performances into real bars, plus 2-second windows for comparison
python score_replication/build_bars.py asap bars

# transition detector on the three kinds of data
python score_replication/score_experiments.py detect bars out_score/detect_score --data score
python score_replication/score_experiments.py detect bars out_score/detect_perf_bars --data perf_bars
python score_replication/score_experiments.py detect bars out_score/detect_perf_2s --data perf_2s

# composer classifier trained on scores, with and without performance features
python score_replication/score_experiments.py classify bars out_score/classify
python score_replication/score_experiments.py classify bars out_score/classify_noperf --drop-perf
```

The detector takes a few minutes for each kind of data. The classifier uses a GPU when one is available.

## Repository structure

| File | Contents |
|---|---|
| `extract_features.py` | 27 features per 2-second bar |
| `extract_edges.py` | Boundary information per bar, including the pitch-class profile, first and last melody and bass notes, edge velocity and density, and held notes |
| `train_gru.py` | GRU classifier, data splits, logistic-regression baseline and the within-segment shuffle test |
| `recombination_test.py` | Experiment A |
| `splice_detector.py` | Experiment B |
| `transition_stats.py` | Mean transition features by splice type |
| `all_runs.csv` | Per-run results of the reproduction (16 runs) |
| `feature_effects.txt` | Single-feature effect sizes, Beethoven against other composers |
| `recombination_runs.csv` | Per-run results of Experiment A |
| `splice_runs.csv` | Per-run results of Experiment B |
| `logreg_coefficients.csv` | Logistic-regression coefficients of the transition detector |
| `transition_feature_means.csv` | Mean transition features for real and spliced pairs |
| `score_replication/download_asap.py` | Downloads the ASAP files used here |
| `score_replication/real_bars.py` | Cuts MIDI at annotated barlines and computes the same features in quarter notes |
| `score_replication/build_bars.py` | Builds the score, performance and 2-second-window bar datasets |
| `score_replication/score_experiments.py` | Transition detector and composer classifier on these datasets |
| `score_replication/results/` | Per-run results of the score-based replication |

## References

- F. Foscarin, A. McLeod, P. Rigaux, F. Jacquemard, and M. Sakai, "ASAP: a dataset of aligned scores and performances for piano transcription," *ISMIR*, 2020.
- J. E. Galajda and K. A. Hua, "Automated Thematic Composer Classification Using Segment Retrieval," *IEEE MIPR*, 2024, pp. 162–168. doi:10.1109/MIPR62202.2024.00032
- J. E. Galajda and K. A. Hua, "Neural-Base Music Generation for Intelligence Duplication," arXiv:2310.13691, 2023.
- C. Hawthorne et al., "Enabling Factorized Piano Music Modeling and Generation with the MAESTRO Dataset," *ICLR*, 2019.
- C. McKay and I. Fujinaga, "jSymbolic: A Feature Extractor for MIDI Files," *ICMC*, 2006.

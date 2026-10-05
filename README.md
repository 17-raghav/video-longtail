# video-longtail

Long-tail video classification with frozen CLIP features: what happens to rare classes when a
classifier is trained on an imbalanced set of action clips, and whether the standard fixes for
imbalance help.

An experimental second module, `experimental/anomaly_cascade/`, applies frozen image-text features
(SigLIP) to video anomaly detection. It has no score (see its README).

## Problem

UCF-101 has 13,320 short action clips in 101 classes and is close to balanced. To study long-tail
behaviour, `p3.py` resamples it with exponential decay (the CIFAR-LT recipe): 50 classes, 1,913 clips,
largest class 167 clips, smallest 8, imbalance factor 20.9 (the requested factor was 100; an 8-clip
floor clamps the tail). Split: stratified 70/30, seed 42. Metrics: overall accuracy, macro accuracy
(mean of per-class accuracy), head accuracy (top third of classes by training count), tail accuracy
(bottom third), and the number of classes with zero accuracy.

## Approach

1. Sample 16 uniform frames per clip; encode each with a frozen CLIP ViT-B/32 (open_clip, `openai`
   weights) into 512-d vectors; cache all features once in a Parquet file.
2. Zero-shot baseline: CLIP text embeddings of the class names, cosine similarity to the mean frame
   embedding. No training, so no class-frequency bias.
3. Temporal transformer head: CLS token + learned positional embeddings, 2 pre-norm encoder blocks,
   8 heads, d_model 512, feed-forward 1024; AdamW 1e-3, 30 epochs.
4. Two loss-based mitigations on the same transformer: class-balanced weights from the effective
   number of samples (Cui et al. 2019) and focal loss (gamma 2).
5. Mean-pooled baselines (frame order discarded): plain logistic regression, class-balanced logistic
   regression, and an MLP with one 512-unit hidden layer over 5 seeds.
6. Bootstrap confidence intervals for head/tail accuracy; a Polars vs pandas benchmark on the feature
   store.

## How to run

```bash
pip install -r requirements.txt
```

```python
# download UCF-101 (multi-GB) and unzip to ./ucf101, e.g. from the HF dataset quchenyuan/UCF101-ZIP
exec(open("p3.py").read())
smoke_test(video_root="ucf101")   # 4 classes, quick end-to-end check
res = run(video_root="ucf101")    # full eight-stage pipeline
```

No-data check of the model code (parameter counts and a smoke run on random features):

```bash
python scripts/check_models.py
```

## Results

**Evidence.** The only saved run of the experiment is the output stored in `Project_3.ipynb`
(Colab, a seven-stage version of the pipeline). Cell numbers below are 0-based cell indices. I could not re-run anything that needs the video
features: UCF-101 and the cached feature file are not on this machine. What is verified, and how:

| Result | Value | Source |
|---|---|---|
| Long-tailed set | 50 classes, 1,913 clips, 167 / 8 clips, factor 20.9 | notebook cell 7 output |
| Zero-shot CLIP: overall / macro / head / tail | 0.8025 / 0.7039 / 0.7905 / 0.5938 | notebook cell 9 output |
| Transformer, plain CE: overall / macro / head / tail | 0.9506 / 0.8109 / 0.9915 / 0.5938 | notebook cell 9 output |
| Transformer + class-balanced weights: macro / tail | 0.7855 / 0.5625 | notebook cell 9 output |
| Transformer + focal loss: macro / tail | 0.7764 / 0.5625 | notebook cell 9 output |
| Classes with zero accuracy under the transformer | 5 (YoYo, PommelHorse, RopeClimbing, JavelinThrow, StillRings) | notebook cell 9 output |
| Bootstrap (B=300), tail trained minus tail zero-shot | +0.0047, 95% CI [-0.2007, +0.1944] | notebook cell 11 output |
| Transformer parameters | 4,240,434 (nhead 8, d_model 512, ff 1024) | `results/check_models.log`, counted from the object |
| MLP(512) parameters on 512-d input, 50 classes | 288,306; transformer is 14.7x larger | `results/check_models.log` |
| Feature store group-by: Polars / pandas with `columns=` / naive pandas | 1.2 / 2.2 / 665.1 ms | notebook cell 7 output |

What the verified numbers say:
- Supervised training on frozen CLIP features raises head accuracy a lot (0.79 -> 0.99) but the
  transformer's tail accuracy equals zero-shot's (both 19/32 clips), so training did not help rare
  classes for this model.
- Both loss-based mitigations made the transformer slightly worse (macro 0.8109 -> 0.7855 / 0.7764).
- The honest Polars speedup is 1.8x against pandas reading the same single column; the 560x figure
  against naive pandas is mostly column pruning.

## Not yet verified

These appear in the earlier write-up and in CV drafts, but **no saved run supports them**. The
notebook output predates the `mean_pool_baselines` stage (7 stages saved, `p3.py` has 8).

- Mean-pooled MLP: overall 0.9703, macro 0.8951 +- 0.0046, tail 0.8375 +- 0.0125 over 5 seeds.
- Therefore the headline "a small MLP beats the temporal transformer by 24 points on rare classes"
  (0.8375 vs 0.5938) and "8 points on macro".
- Balanced logistic regression tail 0.7812 vs plain logistic 0.0312 (the "weighting moved rare classes
  from 0.03 to 0.78" claim).
- "2 zero-accuracy classes under the MLP" and which three classes it recovered.
- Bootstrap coverage of 91-93% vs nominal 95% (no simulation code is in the repository).
- The conclusion "capacity, not the loss function, drives rare-class failure" depends on the MLP rows,
  so it is a hypothesis until they are re-run.

What a re-run needs: the UCF-101 archive (a multi-GB download), the CLIP ViT-B/32 weights, then
feature extraction for 1,913 clips x 16 frames. On Colab this step took 449 s (notebook output);
on this laptop's CPU it would plausibly take tens of minutes. Everything after extraction trains in
seconds to minutes on CPU. No GPU strictly needed. I have not started it; it needs your go-ahead
because of the download and runtime.

## Corrections applied from the audit

From `Sony-Interview-Prep-Complete/01-audits` (00-CV-DISCREPANCY-REPORT.md, 50-github-readiness.md):
- Transformer size is 4,240,434 parameters, not ~6M; the ratio to the MLP is 14.7x, not ~20x
  (`dim_feedforward` is 1024, not PyTorch's default 2048). Now verified by counting.
- If the tail figures are ever quoted, use 0.838 vs 0.594 (difference 0.244), not "0.84 vs 0.59 = 24".
- The transformer's own variance rests on two runs of the same config: macro 0.8109 (cell 7) and
  0.8194 (cell 11, retrained), a spread of 0.0085.
- "2 classes remain at zero in all five seeds" is weakened to a mean, because the code averaged the
  count over seeds; `p3.py` now also records the per-seed counts.
- The "1.7x not 17x" fix in the audit belongs to the causal-inference project (Project_2), which is
  not part of this repository.

## Limitations

- **Possible feature bug.** The saved notebook output shows open_clip warning `QuickGELU mismatch
  between final model config (quick_gelu=False) and pretrained tag 'openai' (quick_gelu=True)`.
  `p3.py` loads `"ViT-B-32"` with `pretrained="openai"`; the OpenAI weights were trained with
  QuickGELU, so the features were computed with a slightly different activation than the weights
  expect. This probably lowers zero-shot accuracy and could affect every row. Fix for the re-run:
  use `"ViT-B-32-quickgelu"`.
- Tail accuracy is the mean over 16 classes with 2 test clips each, so it moves in steps of 1/32 =
  0.031; no tail figure should be quoted to four decimals.
- The head/tail boundary falls inside a 19-way tie at train count 6; three tied classes are excluded
  by sort order alone.
- Neither head was tuned; the transformer has 4.24M parameters for 1,346 training clips.
- The imbalance is constructed, not natural; 50 of 101 classes; one zero-shot prompt template.
- The Polars/pandas benchmark takes the minimum of 3 repeats without clearing the page cache.

## Next steps

1. Re-run `p3.py` end to end with the QuickGELU fix and commit the log; then decide which claims stand.
2. Add a mean-pooled transformer or a sequence MLP to separate "frame order" from "capacity".
3. Report tail accuracy for all classes below a frequency threshold instead of a fixed bottom third.

## Layout

```
p3.py                      pipeline (8 stages), bias audit, losses, feature store, benchmark
Project_3.ipynb            Colab driver with the only saved run output (7-stage version)
scripts/check_models.py    parameter counts + smoke run on random features (no data needed)
experimental/anomaly_cascade/   SigLIP frame scoring -> smoothing -> intervals; VLM step planned
results/                   logs of every run done for this repository
```

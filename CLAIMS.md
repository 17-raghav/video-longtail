# Claims

Status: **verified now** = run on 2026-10-05, log in `results/`; **existing output** = in the saved
output of `Project_3.ipynb` (0-based cell index); **unverified** = no saved run, do not use.

| Claim | Value | Source | Status |
|---|---|---|---|
| Long-tailed UCF-101 subset | 50 classes, 1,913 clips, 167/8, factor 20.9 | `Project_3.ipynb` cell 7 | existing output |
| Transformer plain CE overall / macro | 0.9506 / 0.8109 | cell 7 and cell 9 | existing output |
| Transformer head / tail | 0.9915 / 0.5938 | cell 9 | existing output |
| Zero-shot CLIP overall / macro / tail | 0.8025 / 0.7039 / 0.5938 | cells 7, 9 | existing output |
| Class-balanced / focal macro | 0.7855 / 0.7764 | cell 7 | existing output |
| Mitigations did not help the transformer | macro down 2.5 / 3.5 points | cell 7 | existing output |
| Tail gain from training (transformer) is zero | +0.0047, CI [-0.2007, +0.1944] | cell 11 | existing output |
| Transformer run-to-run spread | 0.8109 vs 0.8194 macro | cells 7, 11 | existing output (two runs only) |
| Transformer parameters | 4,240,434 | `results/check_models.log` | verified now |
| MLP parameters / ratio | 288,306 / 14.7x | `results/check_models.log` | verified now |
| Polars vs pandas (same column) | 1.8x (1.2 vs 2.2 ms) | cell 7 | existing output (min of 3, no cache clearing) |
| CLIP extraction time on Colab | 449 s for 1,913 clips | cell 7 | existing output |
| MLP beats transformer by 24 points on tail (0.8375 vs 0.5938) | | none | **unverified** |
| MLP macro 0.8951 +- 0.0046, overall 0.9703 | | none | **unverified** |
| Balanced logistic moves tail 0.03 -> 0.78 | | none | **unverified** |
| "Capacity, not loss function" | | depends on the MLP rows | **unverified (hypothesis)** |
| Bootstrap coverage 91-93% | | no simulation code committed | **unverified** |
| AHC anomaly detection score | - | saved submission is an empty template | **unverified - no score exists** |
| AHC scorer / post-processing self-tests pass | all pass | `results/anomaly_cascade/selftest.log` | verified now (tests the scorer, not a model) |
| AHC public test set size | 34 videos, 56.5 min | `results/anomaly_cascade/test_set_stats.log` | verified now |

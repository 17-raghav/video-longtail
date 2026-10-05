# anomaly_cascade (experimental, no score)

Video anomaly detection for a hackathon: given drone / CCTV / dashcam clips, report which anomaly
happened (11 classes such as traffic accident, fire, smoke, wrong-way driving, or normal) and, for
the harder levels, when (start and end time). Large hosted models were allowed only for development,
not at run time.

**Status: no score is claimed.** The saved `submission.json` in the original folder has 34
predictions with no events and zero wall time (`results/anomaly_cascade/original_submission_check.log`),
i.e. a template, not a run. The VLM step is planned, not built.

## Design

```
video -> frames at 2 fps, skip near-duplicate frames (mean abs grey-level diff <= 8 on 64x64)
      -> SigLIP ViT-B/16 image embeddings (frozen)
      -> per-frame class scores: zero-shot text prompts (cascade.py) or a small trained MLP head
         (train_head.py, embed_cache.py)
      -> temporal smoothing: open an interval when a class score passes OPEN, keep it while above
         HOLD, merge gaps, drop short runs (postprocess.build_intervals)
      -> intervals per class, or one clip label for level 1
      -> [planned] Qwen2.5-VL-3B confirms or rejects each candidate interval and writes the
         explanation (postprocess.parse_vlm exists and is unit-tested; the model call is not wired)
```

| File | Role |
|---|---|
| `embed_cache.py` | sample frames, embed with SigLIP, cache per video (`.npz`) |
| `train_head.py` | frame-level labels from ground-truth intervals, class-weighted MLP head |
| `cascade.py` | zero-shot variant: frame-to-prompt similarity, threshold, merge into intervals |
| `cascade_d1.py` | zero-shot variant with 8 frames per clip and a "normal scene" prompt |
| `postprocess.py` | hysteresis interval building, level-1 decision, VLM output parsing, explanations |
| `emit.py` | builds and validates the submission JSON (format rules) |
| `scorer.py` | my local re-implementation of the scoring rules (level 1 accuracy mix; temporal IoU >= 0.5 matching) |
| `selftest.py` | scorer and post-processing checks against the local ground truth |
| `build_manifest.py` | lists test videos |
| `colab_run.ipynb`, `colab_train.ipynb` | Colab drivers (no saved outputs) |

`run_tests.py` from the original folder was dropped: it calls `make_video` with an old signature and
fails (`results/anomaly_cascade/run_tests.log`); `selftest.py` covers the same checks and passes.

## What ran (2026-10-05)

`selftest.py` against the local 34-video public test ground truth: all checks pass
(`results/anomaly_cascade/selftest.log`): a perfect submission scores 1.0, an all-empty one 0.139,
format errors are caught, interval merging behaves as expected. This tests the scorer and
post-processing, not a model.

## What a real scoring run would need

- Data: the 34 public test videos are local (56.5 minutes of video, 75,242 frames; about 6,800
  frames at 2 fps before duplicate skipping). Not included here.
- Model: SigLIP ViT-B-16 `webli` weights from Hugging Face (not cached on this machine; a download).
- Zero-shot run (`python cascade.py test out.json`): CPU is enough; frame decoding plus at most
  ~6,800 SigLIP forward passes. My rough expectation is 15-30 minutes on this laptop's CPU; a T4 GPU
  would take a few minutes. Not run: needs your approval for the download and runtime.
- Trained-head run: needs the training videos (7 zip parts, not unpacked) for `embed_cache.py` +
  `train_head.py`; realistically a GPU session.
- VLM confirmation: Qwen2.5-VL-3B, needs a GPU (the Colab notebook targets a T4).

## Caveats found while reading the code

- `cascade.py` reads each video's level from `ground_truth.csv`. If the level is not given in the
  test manifest at run time, this is label leakage and must be removed before any score is reported.
- `scorer.py` encodes my reading of the rules; it is not the organisers' scorer.

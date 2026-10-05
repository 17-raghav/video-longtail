# Study guide: video-longtail

Read `CLAIMS.md` first: several numbers from the earlier write-up are **not** supported by a saved
run. Defend the project on the verified rows and on the honest provenance story.

---

## 1. Concepts from zero

### 1.1 Long-tailed classification
Real catalogues have a few classes with many examples (head) and many with few (tail). A classifier
trained on such data tends to be accurate on the head and poor on the tail, while overall accuracy
looks fine because the test set is also dominated by head classes. So report **macro accuracy** (mean
of per-class accuracies) and **head vs tail** accuracy, not just overall accuracy.

### 1.2 Building a long-tailed set (CIFAR-LT recipe)
Sort classes, give class i a target count n_max * (1/IF)^(i/(C-1)), where IF is the imbalance factor.
Here: requested IF 100, but each class keeps at least 8 clips (so that the 30% test split gives at
least 2 test clips), which clamps the tail and gives a realised IF of 167/8 = 20.9.

### 1.3 Frozen CLIP features
CLIP trains an image encoder and a text encoder so that matching image-text pairs have high cosine
similarity. Frozen = we never update its weights; we only use its 512-d image embeddings as features.
Zero-shot classification = embed "a photo of {class}" texts, pick the class whose text embedding is
closest to the image (here: mean of 16 frame) embedding.

### 1.4 Temporal transformer head
Input: 16 frame embeddings per clip. Prepend a learnable CLS vector, add learned positional
embeddings (17 positions), pass through 2 transformer encoder blocks (multi-head self-attention with 8
heads, feed-forward 512 -> 1024 -> 512, pre-norm), classify the CLS output. Parameter count:
per layer 1,050,624 (attention: Q/K/V projections 3x512x512 + biases, output 512x512 + bias) +
525,312 + 524,800 (feed-forward) + 2,048 (two LayerNorms) = 2,102,784; two layers 4,205,568; plus
CLS 512, positions 8,704, head 25,650 = 4,240,434 (counted in `results/check_models.log`).

### 1.5 Mitigations for imbalance
- **Class-balanced weights** (Cui et al. 2019): effective number of samples E_n = (1 - beta^n)/(1 - beta),
  weight each class by 1/E_n (beta 0.999). It accounts for near-duplicate examples adding less
  information than independent ones.
- **Focal loss** (Lin et al. 2017): multiply cross-entropy by (1 - p_t)^gamma, so confident (easy)
  examples contribute less and hard ones more.
Both change the loss. Neither changes how much the model can learn from 6 examples per class.

### 1.6 Bootstrap confidence intervals
Resample test clips with replacement B times, recompute the metric, take the 2.5th and 97.5th
percentiles. Here tail accuracy averages 16 classes of 2 clips each, so the metric is coarse (steps of
1/32) and i.i.d. clip resampling is not ideal for a class-averaged metric.

### 1.7 Feature store
Extract features once (the expensive part), store in Parquet (columnar), so later experiments read
only what they need. Polars `scan_parquet` reads lazily and prunes columns; pandas does the same only
if asked with `columns=`.

---

## 2. Code walkthrough (`p3.py`, execution order of `run()`)

1. `find_videos(root)` - list `.avi` files, class = parent folder name.
2. `make_long_tailed(items)` - pick 50 classes evenly spaced in the sorted list (`MAX_CLASSES`), assign
   exponentially decaying targets, enforce the 8-clip floor, report the realised imbalance factor.
3. `split_train_test(items)` - stratified 70/30 with seed 42.
4. `sample_frames(path)` - 16 uniformly spaced frames per clip with OpenCV.
5. `extract_features(items)` - open_clip `ViT-B-32` / `openai`; batch-encode frames; L2-normalise;
   array (n_clips, 16, 512). `save_feature_store` writes Parquet; `load_feature_store` reads it back.
6. `zero_shot(feats, labels, classes)` - text prompts from humanised class names, cosine similarity
   with the normalised mean frame embedding.
7. `TemporalTransformer(...).fit(...)` / `.predict(...)` - the head above, AdamW lr 1e-3, weight decay
   0.01, batch 64, 30 epochs; optional class weights or focal loss.
8. `effective_number_weights`, `_focal_loss` - the two mitigations.
9. `bias_audit(pred, y, classes, train_counts)` - per-class accuracy, macro, worst group, head/tail
   thirds, head minus tail, Spearman correlation between class frequency and accuracy.
10. `mean_pool_baselines(...)` - stage 7: logistic, balanced logistic, MLP over 5 seeds (this is the
    stage with no saved output).
11. `benchmark_store()` - Polars vs pandas, min of 3 repeats.
12. Results are written to `results.json`.

`scripts/check_models.py` - counts parameters from the real objects and runs the code on random
features to show every function executes.

Experimental module: see `experimental/anomaly_cascade/README.md` (frames at 2 fps with duplicate
skipping -> SigLIP embeddings -> per-frame class scores -> hysteresis intervals -> JSON).

---

## 3. Why this, not that

| Choice | Why | Alternative |
|---|---|---|
| Frozen CLIP | cheap, strong general features; isolates the head as the variable | fine-tuning a video model: GPU-heavy, confounds encoder and head |
| 16 uniform frames | fixed-size input, covers the whole clip | shot detection or dense sampling |
| Constructed imbalance | stock UCF-101 is balanced; a controlled factor makes the effect measurable | natural long-tail dataset (e.g. Kinetics subsets): more realistic, much bigger |
| 8-clip floor | guarantees 2 test clips per class | lower floor: single-clip classes make per-class accuracy a coin flip |
| Zero-shot as control | it never saw training frequencies, so it has no learned frequency bias | none comparable |
| Macro + head/tail metrics | overall accuracy hides tail failure | overall only |
| Parquet + Polars | columnar, lazy reads; extraction runs once | re-extract each experiment |

---

## 4. What failed or changed

- The first write-up claimed "supervised training amplifies imbalance" by comparing the transformer
  with zero-shot only. Adding simple baselines (MLP, logistic) reportedly reversed that, but that run
  was never saved; this is why those claims are now "not yet verified".
- Both loss-based mitigations lowered macro accuracy for the transformer (verified).
- The README's earlier "~6M parameters" and "20x" were wrong (4.24M, 14.7x).
- The saved run shows a QuickGELU mismatch warning: features were computed with the wrong activation
  for the OpenAI weights. Found while preparing this repo; not yet re-run.
- AHC: the cascade was designed and the scorer/post-processing were tested, but no scored run exists;
  the saved submission is an empty template.

---

## 5. Likely interview questions

1. **What was the question?** What happens to rare classes when you train a classifier on imbalanced
   video data with frozen CLIP features, and do standard imbalance fixes help.
2. **How did you make UCF-101 long-tailed?** Exponential decay over 50 classes with an 8-clip floor;
   realised imbalance 20.9.
3. **Why macro accuracy?** Overall accuracy is dominated by head classes; macro weights every class
   equally.
4. **What did the transformer get?** Overall 0.9506, macro 0.8109, head 0.9915, tail 0.5938.
5. **And zero-shot?** Overall 0.8025, macro 0.7039, tail 0.5938, the same tail as the trained
   transformer.
6. **So did training help rare classes?** Not for the transformer: tail difference +0.0047 with a
   95% bootstrap CI of [-0.20, +0.19].
7. **Did class-balanced loss or focal loss help?** No, macro fell to 0.7855 and 0.7764.
8. **Why might they fail?** They reweight the gradient but cannot create information: tail classes
   have 6 training clips against a 4.24M-parameter head. My hypothesis is capacity; the MLP baseline
   that would support it has no saved run, so I state it as a hypothesis.
9. **How many parameters, exactly, and why not 6M?** 4,240,434; the feed-forward width is 1024
   (dim*2), not PyTorch's default 2048.
10. **Single- or multi-head attention?** 8 heads here. (My separate SASRec project uses one head; don't
    mix them up.)
11. **Why is the tail metric coarse?** 16 tail classes x 2 test clips: steps of 1/32 = 0.031.
12. **What is the head/tail tie problem?** 19 classes have train count 6 and only 16 fit in the bottom
    third, so 3 are excluded by sort order alone.
13. **Is your bootstrap valid?** Clip-level i.i.d. resampling under-represents uncertainty for a
    class-averaged metric with 2 clips per class; intervals are probably too narrow.
14. **What did the feature store give you?** Extraction once (449 s on Colab), then experiments in
    seconds. Polars was 1.8x faster than pandas reading the same column; the 560x headline was mostly
    pandas reading all columns.
15. **What is effective number of samples?** (1 - beta^n)/(1 - beta): saturates as n grows because
    near-duplicates add little; class weight = 1/effective number.
16. **What does focal loss do?** Scales CE by (1 - p_t)^gamma, down-weighting easy examples.
17. **What is the QuickGELU warning about?** OpenAI CLIP weights expect the QuickGELU activation;
    loading them into a GELU model config gives slightly wrong features. Fix: `ViT-B-32-quickgelu`.
18. **Why not fine-tune CLIP?** Compute, and it would confound the comparison between heads.
19. **What would you do next?** Re-run end to end with the activation fix and commit the log; add a
    mean-pooled transformer to separate frame order from capacity; report tail as all classes below a
    count threshold.
20. **Hard follow-up: your CV said the MLP wins by 24 points. Where is the evidence?** The saved
    notebook run predates that stage; I found this in an audit, moved the claim to "not yet verified"
    and will re-run before quoting it.
21. **Hard follow-up: is 0.9506 overall good?** Not meaningful alone; macro 0.8109 and tail 0.5938
    show the imbalance cost.
22. **Hard follow-up: why is the head accuracy 0.99 but tail 0.59?** Head classes have up to about 117 (70% of 167)
    training clips; tail classes 6. The head memorises frequent classes.
23. **AHC: what is the design?** Frames at 2 fps with duplicate skipping, SigLIP embeddings, per-frame
    class scores, hysteresis smoothing into intervals, optional VLM confirmation.
24. **AHC: what was your score?** None. The saved submission is an empty template; I only verified the
    scorer and post-processing with self-tests.
25. **AHC: what would you watch for before reporting a score?** `cascade.py` reads each video's level
    from the ground-truth file, which would be leakage if levels are not given at test time.

"""Checks that need no video data.

1. Parameter counts of the two heads compared in the README, from the actual objects:
   p3.TemporalTransformer (50 classes) and the sklearn MLPClassifier(512) used in
   mean_pool_baselines (count = sizes of coefs_ and intercepts_ after one fit).
2. A smoke run of the training / audit code on RANDOM features (no signal), only to show
   the code paths execute. The accuracies printed here mean nothing.

usage: python scripts/check_models.py
"""
import os
import sys
from collections import Counter

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import p3  # noqa: E402

N_CLASSES = 50

tt = p3.TemporalTransformer(dim=512, n_classes=N_CLASSES)
n_tt = sum(p.numel() for p in tt.net.parameters())
by_part = {name: sum(p.numel() for p in mod.parameters()) for name, mod in tt.net.named_children()}
print(f"TemporalTransformer(50 classes): {n_tt:,} parameters")
print("  by part:", {k: f"{v:,}" for k, v in by_part.items()},
      f"cls+pos: {tt.net.cls.numel() + tt.net.pos.numel():,}")
layer = tt.net.enc.layers[0]
print(f"  nhead={layer.self_attn.num_heads}, d_model={layer.self_attn.embed_dim}, "
      f"dim_feedforward={layer.linear1.out_features}")

from sklearn.neural_network import MLPClassifier  # noqa: E402

rng = np.random.default_rng(0)
X = rng.normal(size=(N_CLASSES * 4, 512)).astype(np.float32)
y = np.repeat(np.arange(N_CLASSES), 4)
mlp = MLPClassifier(hidden_layer_sizes=(512,), max_iter=5, random_state=0).fit(X, y)
n_mlp = sum(w.size for w in mlp.coefs_) + sum(b.size for b in mlp.intercepts_)
print(f"MLPClassifier(512) on 512-d input, 50 classes: {n_mlp:,} parameters")
print(f"ratio transformer / MLP: {n_tt / n_mlp:.1f}x")

# ---- smoke run on random features (meaningless numbers)
print("\nsmoke run on random features (accuracies are meaningless):")
n_cls, per = 5, 12
feats = rng.normal(size=(n_cls * per, p3.N_FRAMES, 512)).astype(np.float32)
labels = [f"c{i}" for i in range(n_cls) for _ in range(per)]
classes = sorted(set(labels))
is_train = np.array([(k % per) < 8 for k in range(n_cls * per)])
ytr = np.array([classes.index(l) for l, t in zip(labels, is_train) if t])
yte = np.array([classes.index(l) for l, t in zip(labels, is_train) if not t])
model = p3.TemporalTransformer(dim=512, n_classes=n_cls).fit(feats[is_train], ytr, epochs=2)
audit = p3.bias_audit(model.predict(feats[~is_train]), yte, classes,
                      Counter(l for l, t in zip(labels, is_train) if t))
print("  transformer fit/predict/bias_audit ok; keys:", sorted(audit)[:6], "...")
tc = Counter(l for l, t in zip(labels, is_train) if t)
w = p3.effective_number_weights([tc[c] for c in classes])
print("  effective_number_weights ok, len", len(w))
p3.mean_pool_baselines(feats, labels, is_train, classes, seeds=(0,))
print("  mean_pool_baselines ok")

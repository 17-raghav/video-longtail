import os
import csv
import glob
import numpy as np
import torch
import torch.nn as nn

CLASSES = [
    "normal", "traffic_accident", "traffic_congestion", "stalled_or_broken_down_vehicle",
    "vehicle_blocking_traffic", "wrong_way_driving", "road_spill_or_debris",
    "waterlogging_or_flood", "fire", "smoke", "fighting_or_violence",
    "loitering_or_suspicious_presence",
]
IDX = {c: i for i, c in enumerate(CLASSES)}
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_frame_labels(cache_dir, gt_csv):
    gt = {}
    with open(gt_csv, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            vid = row["video_id"]
            cls = (row.get("class_name") or "normal").strip()
            s = row.get("start_time_sec", "")
            e = row.get("end_time_sec", "")
            s = float(s) if str(s).strip() != "" else None
            e = float(e) if str(e).strip() != "" else None
            gt.setdefault(vid, []).append((cls, s, e))
    X, y = [], []
    for npz in glob.glob(os.path.join(cache_dir, "*.npz")):
        vid = os.path.splitext(os.path.basename(npz))[0]
        if vid not in gt:
            continue
        d = np.load(npz)
        emb, times = d["emb"], d["times"]
        spans = gt[vid]
        for k in range(len(emb)):
            t = times[k]
            label = IDX["normal"]
            for cls, s, e in spans:
                if cls == "normal":
                    continue
                if s is None or e is None:
                    label = IDX.get(cls, IDX["normal"])
                    break
                if s <= t <= e:
                    label = IDX.get(cls, IDX["normal"])
                    break
            X.append(emb[k])
            y.append(label)
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int64)


class Head(nn.Module):
    def __init__(self, dim, n):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, 512), nn.GELU(), nn.Dropout(0.3), nn.Linear(512, n))

    def forward(self, x):
        return self.net(x)


def train(cache_dir, gt_csv, out_path="head.pt", epochs=30):
    X, y = load_frame_labels(cache_dir, gt_csv)
    counts = np.bincount(y, minlength=len(CLASSES))
    w = 1.0 / np.clip(counts, 1, None)
    w = w / w.sum() * len(CLASSES)
    Xt = torch.tensor(X).to(DEVICE)
    yt = torch.tensor(y).to(DEVICE)
    model = Head(X.shape[1], len(CLASSES)).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    lossf = nn.CrossEntropyLoss(weight=torch.tensor(w, dtype=torch.float32).to(DEVICE))
    n = len(Xt)
    for ep in range(epochs):
        perm = torch.randperm(n, device=DEVICE)
        model.train()
        for i in range(0, n, 4096):
            b = perm[i:i + 4096]
            opt.zero_grad()
            loss = lossf(model(Xt[b]), yt[b])
            loss.backward()
            opt.step()
    torch.save({"state": model.state_dict(), "dim": X.shape[1], "classes": CLASSES}, out_path)
    print("saved", out_path, "frames", n, "class counts", dict(zip(CLASSES, counts.tolist())))
    return out_path


if __name__ == "__main__":
    import sys
    train(sys.argv[1], sys.argv[2])

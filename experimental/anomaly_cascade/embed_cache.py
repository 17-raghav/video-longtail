import os
import csv
import glob
import numpy as np
import torch
import cv2
from PIL import Image
import open_clip

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32
FPS_SAMPLE = 2.0
DIFF_THRESH = 8.0

model, _, preprocess = open_clip.create_model_and_transforms(
    "ViT-B-16-SigLIP", pretrained="webli", device=DEVICE
)
model = model.to(DTYPE).eval()


def sample_frames(path, fps_sample=FPS_SAMPLE):
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    step = max(1, int(round(fps / fps_sample)))
    frames, times, idx, prev = [], [], 0, None
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % step == 0:
            small = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (64, 64))
            if prev is None or np.abs(small.astype(float) - prev).mean() > DIFF_THRESH:
                frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
                times.append(idx / fps)
            prev = small.astype(float)
        idx += 1
    cap.release()
    return frames, times, idx / fps if fps else 0.0


@torch.no_grad()
def embed_frames(frames, batch=64):
    out = []
    for i in range(0, len(frames), batch):
        px = torch.stack([preprocess(f) for f in frames[i:i + batch]]).to(DEVICE, DTYPE)
        feat = model.encode_image(px)
        feat = feat / feat.norm(dim=-1, keepdim=True)
        out.append(feat.float().cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0, 512), dtype=np.float32)


def build_cache(video_glob, out_dir, id_from_path=lambda p: os.path.splitext(os.path.basename(p))[0]):
    os.makedirs(out_dir, exist_ok=True)
    meta = []
    for path in sorted(glob.glob(video_glob)):
        vid = id_from_path(path)
        dst = os.path.join(out_dir, vid + ".npz")
        if os.path.exists(dst):
            continue
        frames, times, dur = sample_frames(path)
        emb = embed_frames(frames)
        np.savez_compressed(dst, emb=emb, times=np.array(times), duration=dur)
        meta.append((vid, len(frames), round(dur, 2)))
        print(vid, len(frames), round(dur, 2))
    with open(os.path.join(out_dir, "_index.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["video_id", "n_frames", "duration_sec"])
        w.writerows(meta)
    return meta


if __name__ == "__main__":
    import sys
    build_cache(sys.argv[1], sys.argv[2])

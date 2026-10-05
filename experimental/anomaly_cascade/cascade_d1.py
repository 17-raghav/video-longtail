import os
import sys
import csv
import json
import time
import numpy as np

from emit import make_event, make_video, build, validate
from scorer import score

SIM_THRESHOLD = 0.10
N_FRAMES = 8
PROMPT_TEMPLATE = "a photo of {}"
NORMAL_PROMPT = "a normal scene"
MODEL_NAME = "ViT-B-16-SigLIP"
PRETRAINED = "webli"
TEST_DIR = "test"
OUT_JSON = "submission_d1.json"

ANOM_CLASSES = [
    "traffic_accident", "traffic_congestion", "stalled_or_broken_down_vehicle",
    "vehicle_blocking_traffic", "wrong_way_driving", "road_spill_or_debris",
    "waterlogging_or_flood", "fire", "smoke", "fighting_or_violence",
    "loitering_or_suspicious_presence",
]
NORMAL_LABEL = "normal"
LABELS = ANOM_CLASSES + [NORMAL_LABEL]

_DEVICE = None
_DTYPE = None


def load_manifest(test_dir):
    out = {}
    with open(os.path.join(test_dir, "videos.csv"), newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            vid = (row.get("video_id") or "").strip()
            if not vid:
                continue
            fn = (row.get("filename") or f"videos/{vid}.mp4").strip()
            out[vid] = os.path.join(test_dir, fn)
    return out


def load_gt(gt_csv):
    level, cls = {}, {}
    with open(gt_csv, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            vid = (row.get("video_id") or "").strip()
            if not vid:
                continue
            level[vid] = int(row.get("level") or 1)
            c = (row.get("class_name") or "normal").strip()
            if c and c != "normal" and str(row.get("is_anomaly", "")).strip().lower() in ("1", "true", "yes"):
                cls[vid] = c
            else:
                cls.setdefault(vid, "normal")
    return level, cls


def get_model():
    global _DEVICE, _DTYPE
    import torch
    import open_clip
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(True)
    torch.backends.cuda.enable_math_sdp(True)
    _DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    _DTYPE = torch.float16 if _DEVICE == "cuda" else torch.float32
    model, _, preprocess = open_clip.create_model_and_transforms(
        MODEL_NAME, pretrained=PRETRAINED, device=_DEVICE)
    model = model.to(_DTYPE).eval()
    tokenizer = open_clip.get_tokenizer(MODEL_NAME)
    return model, preprocess, tokenizer


def text_features(model, tokenizer):
    import torch
    prompts = [PROMPT_TEMPLATE.format(c.replace("_", " ")) for c in ANOM_CLASSES] + [NORMAL_PROMPT]
    with torch.no_grad():
        tok = tokenizer(prompts).to(_DEVICE)
        f = model.encode_text(tok)
        f = f / f.norm(dim=-1, keepdim=True)
    return f.float().cpu().numpy()


def sample_frames_uniform(path, k=N_FRAMES):
    import cv2
    from PIL import Image
    cap = cv2.VideoCapture(path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    frames = []
    if total > 0:
        picks = sorted(set(int(round(x)) for x in np.linspace(0, total - 1, k)))
        for fi in picks:
            cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
            ok, frame = cap.read()
            if ok:
                frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
    else:
        idx = 0
        while len(frames) < k:
            ok, frame = cap.read()
            if not ok:
                break
            frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
            idx += 1
    cap.release()
    return frames


def embed_frames(model, preprocess, frames):
    import torch
    if not frames:
        return np.zeros((0, len(LABELS)), dtype=np.float32)
    with torch.no_grad():
        px = torch.stack([preprocess(f) for f in frames]).to(_DEVICE, _DTYPE)
        fe = model.encode_image(px)
        fe = fe / fe.norm(dim=-1, keepdim=True)
    return fe.float().cpu().numpy()


def choose_label(img_feats, txt_feats):
    if len(img_feats) == 0:
        return NORMAL_LABEL, 0.0
    mean_sims = (img_feats @ txt_feats.T).mean(axis=0)
    top = int(mean_sims.argmax())
    label = LABELS[top]
    if label == NORMAL_LABEL or mean_sims[top] < SIM_THRESHOLD:
        return NORMAL_LABEL, float(mean_sims[top])
    return label, float(mean_sims[top])


def run(test_dir=TEST_DIR, out_json=OUT_JSON):
    if os.path.basename(out_json) in ("submission.json", "submission_v2.json"):
        raise ValueError(f"refusing to write {out_json}; use a separate file")
    manifest = load_manifest(test_dir)
    level, gt_cls = load_gt(os.path.join(test_dir, "ground_truth.csv"))
    d1 = [v for v in manifest if level.get(v) == 1]
    model, preprocess, tokenizer = get_model()
    txt = text_features(model, tokenizer)
    preds = []
    chosen = {}
    for n, vid in enumerate(d1, 1):
        t0 = time.time()
        frames = sample_frames_uniform(manifest[vid])
        img = embed_frames(model, preprocess, frames)
        label, sim = choose_label(img, txt)
        chosen[vid] = (label, sim)
        events = [] if label == NORMAL_LABEL else [make_event(label, level=1)]
        ms = (time.time() - t0) * 1000.0
        preds.append(make_video(vid, events, frames_processed=len(frames),
                                chunks_processed=1, internal_ms=round(ms, 1)))
    sub = build(preds, submission_id="zeroshot-d1", model_name=f"{MODEL_NAME}-zeroshot-d1")
    errs = validate(sub, d1)
    if errs:
        print("VALIDATION ERRORS:", errs)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(sub, f, indent=2)
    print("wrote", out_json, "with", len(preds), "D1 videos")
    return d1, chosen, gt_cls


def main():
    test_dir = sys.argv[1] if len(sys.argv) > 1 else TEST_DIR
    out_json = sys.argv[2] if len(sys.argv) > 2 else OUT_JSON
    d1, chosen, gt_cls = run(test_dir, out_json)
    print("-" * 64)
    print(f"{'video_id':10} {'predicted':32} {'ground_truth':22} ok")
    correct = 0
    for vid in d1:
        pred = chosen[vid][0]
        truth = gt_cls.get(vid, "normal")
        ok = (pred == truth)
        correct += ok
        print(f"{vid:10} {pred:32} {truth:22} {'Y' if ok else '.'}")
    res = score(os.path.join(test_dir, "ground_truth.csv"), out_json)
    print("-" * 64)
    print(f"SIM_THRESHOLD = {SIM_THRESHOLD} | N_FRAMES = {N_FRAMES}")
    print(f"exact-label matches: {correct}/{len(d1)}")
    print(f"LEVEL-1 SCORE (24 D1 videos): {res['level1']:.4f}")


if __name__ == "__main__":
    main()

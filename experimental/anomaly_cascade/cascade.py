import os
import sys
import csv
import json
import time
from collections import Counter
import numpy as np

from emit import make_event, make_video, build, validate
from scorer import score

SIM_THRESHOLD = 0.10
MERGE_GAP_SEC = 2.0
MIN_EVENT_SEC = 1.0
FPS_SAMPLE = 2.0
DIFF_THRESH = 8.0
PROMPT_TEMPLATE = "a photo of {}"
MODEL_NAME = "ViT-B-16-SigLIP"
PRETRAINED = "webli"
TEST_DIR = "test"
OUT_JSON = "submission_v2.json"

ANOM_CLASSES = [
    "traffic_accident", "traffic_congestion", "stalled_or_broken_down_vehicle",
    "vehicle_blocking_traffic", "wrong_way_driving", "road_spill_or_debris",
    "waterlogging_or_flood", "fire", "smoke", "fighting_or_violence",
    "loitering_or_suspicious_presence",
]
FRAME_DT = 1.0 / FPS_SAMPLE

_DEVICE = None
_DTYPE = None


def load_manifest(test_dir):
    out = []
    with open(os.path.join(test_dir, "videos.csv"), newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            vid = (row.get("video_id") or "").strip()
            if not vid:
                continue
            fn = (row.get("filename") or f"videos/{vid}.mp4").strip()
            out.append((vid, os.path.join(test_dir, fn)))
    return out


def load_levels(gt_csv):
    lv = {}
    with open(gt_csv, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            vid = (row.get("video_id") or "").strip()
            if vid:
                lv[vid] = int(row.get("level") or 1)
    return lv


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
    prompts = [PROMPT_TEMPLATE.format(c.replace("_", " ")) for c in ANOM_CLASSES]
    with torch.no_grad():
        tok = tokenizer(prompts).to(_DEVICE)
        f = model.encode_text(tok)
        f = f / f.norm(dim=-1, keepdim=True)
    return f.float().cpu().numpy()


def sample_frames(path, fps_sample=FPS_SAMPLE, diff_thresh=DIFF_THRESH):
    import cv2
    from PIL import Image
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    step = max(1, int(round(fps / fps_sample)))
    frames, times, idx, prev = [], [], 0, None
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % step == 0:
            small = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (64, 64)).astype(np.float32)
            if prev is None or np.abs(small - prev).mean() > diff_thresh:
                frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
                times.append(idx / fps)
                prev = small
        idx += 1
    cap.release()
    return frames, times, (idx / fps if fps else 0.0)


def embed_frames(model, preprocess, frames, batch=64):
    import torch
    if not frames:
        return np.zeros((0, len(ANOM_CLASSES)), dtype=np.float32)
    out = []
    with torch.no_grad():
        for i in range(0, len(frames), batch):
            px = torch.stack([preprocess(f) for f in frames[i:i+batch]]).to(_DEVICE, _DTYPE)
            fe = model.encode_image(px)
            fe = fe / fe.norm(dim=-1, keepdim=True)
            out.append(fe.float().cpu().numpy())
    return np.concatenate(out)


def frame_scores(img_feats, txt_feats):
    sims = img_feats @ txt_feats.T
    return sims.argmax(1), sims.max(1)


def dominant_class(idx, best):
    flagged = [int(c) for c, s in zip(idx, best) if s > SIM_THRESHOLD]
    if not flagged:
        return None
    return Counter(flagged).most_common(1)[0][0]


def merge_intervals(times, idx, best):
    raw = []
    cur = None
    for t, c, s in zip(times, idx, best):
        if s > SIM_THRESHOLD:
            c = int(c)
            if cur is not None and c == cur[0] and (t - cur[2]) <= MERGE_GAP_SEC + 1e-9:
                cur[2] = t
            else:
                if cur is not None:
                    raw.append(cur)
                cur = [c, t, t]
    if cur is not None:
        raw.append(cur)
    out = []
    for c, s0, s1 in raw:
        end = s1 + FRAME_DT
        if end - s0 >= MIN_EVENT_SEC:
            out.append((c, s0, end))
    return out


def video_events(level, times, idx, best):
    if level == 1:
        c = dominant_class(idx, best)
        if c is None:
            return []
        return [make_event(ANOM_CLASSES[c], level=1)]
    return [make_event(ANOM_CLASSES[c], s, e, level=level) for c, s, e in merge_intervals(times, idx, best)]


def run(test_dir=TEST_DIR, out_json=OUT_JSON):
    if os.path.basename(out_json) == "submission.json":
        raise ValueError("refusing to write submission.json; use a different name")
    manifest = load_manifest(test_dir)
    levels = load_levels(os.path.join(test_dir, "ground_truth.csv"))
    model, preprocess, tokenizer = get_model()
    txt = text_features(model, tokenizer)
    preds = []
    for n, (vid, path) in enumerate(manifest, 1):
        t0 = time.time()
        frames, times, dur = sample_frames(path)
        img = embed_frames(model, preprocess, frames)
        if len(img):
            idx, best = frame_scores(img, txt)
        else:
            idx, best = np.zeros(0, dtype=int), np.zeros(0, dtype=float)
        events = video_events(levels.get(vid, 1), list(times), idx, best)
        ms = (time.time() - t0) * 1000.0
        preds.append(make_video(vid, events, frames_processed=len(frames),
                                chunks_processed=1, internal_ms=round(ms, 1)))
        print(f"[{n}/{len(manifest)}] {vid}: {len(frames)} frames -> {len(events)} events")
    sub = build(preds, submission_id="zeroshot-siglip", model_name=f"{MODEL_NAME}-zeroshot")
    errs = validate(sub, [v for v, _ in manifest])
    if errs:
        print("VALIDATION ERRORS:", errs)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(sub, f, indent=2)
    print("wrote", out_json)
    return sub


def main():
    test_dir = sys.argv[1] if len(sys.argv) > 1 else TEST_DIR
    out_json = sys.argv[2] if len(sys.argv) > 2 else OUT_JSON
    run(test_dir, out_json)
    res = score(os.path.join(test_dir, "ground_truth.csv"), out_json)
    print("=" * 40)
    print(f"SIM_THRESHOLD = {SIM_THRESHOLD}")
    for k in ("level1", "level2", "level3", "overall"):
        v = res.get(k)
        print(f"  {k:8}: {v:.4f}" if v is not None else f"  {k:8}: n/a")


if __name__ == "__main__":
    main()

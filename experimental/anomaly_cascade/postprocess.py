import numpy as np
from emit import make_event, make_video, build

ANOMALY = [
    "traffic_accident", "traffic_congestion", "stalled_or_broken_down_vehicle",
    "vehicle_blocking_traffic", "wrong_way_driving", "road_spill_or_debris",
    "waterlogging_or_flood", "fire", "smoke", "fighting_or_violence",
    "loitering_or_suspicious_presence",
]


def softmax_rows(x):
    x = np.asarray(x, dtype=np.float32)
    x = x - x.max(axis=1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=1, keepdims=True)


def head_scores_by_class(logits, classes):
    p = softmax_rows(logits)
    return {c: p[:, i] for i, c in enumerate(classes)}


def build_intervals(frame_classes, scores, times, open_score, hold_score,
                    merge_gap_sec, min_dur_sec, pad_sec, classes, max_events):
    n = len(times)
    runs = []
    cur = None

    def close(e):
        nonlocal cur
        if cur is not None:
            runs.append((cur[0], cur[1], cur[2]))
            cur = None

    for i in range(n):
        c = frame_classes[i]
        if c is None or c not in scores:
            close(i - 1)
            continue
        sc = float(scores[c][i])
        if cur is None:
            if sc >= open_score:
                cur = (c, i, i)
        else:
            if c == cur[0] and sc >= hold_score:
                cur = (c, cur[1], i)
            else:
                close(i - 1)
                if sc >= open_score:
                    cur = (c, i, i)
    close(n - 1)

    merged = []
    for c, s, e in runs:
        if merged and merged[-1][0] == c and (times[s] - times[merged[-1][2]]) <= merge_gap_sec:
            merged[-1] = (c, merged[-1][1], e)
        else:
            merged.append((c, s, e))

    out = []
    for c, s, e in merged:
        start = max(0.0, float(times[s]) - pad_sec)
        end = float(times[e]) + pad_sec
        if end - start >= min_dur_sec:
            out.append((c, round(start, 3), round(end, 3)))
    out.sort(key=lambda r: (r[1], r[2]))
    if len(out) > max_events:
        out = sorted(out, key=lambda r: r[2] - r[1], reverse=True)[:max_events]
        out.sort(key=lambda r: (r[1], r[2]))
    return out


def l1_class(frame_classes, scores, classes, confirm_score, min_fraction):
    n = len(frame_classes)
    if n == 0:
        return None
    votes = {c: 0 for c in classes}
    for i in range(n):
        c = frame_classes[i]
        if c in votes and c in scores and float(scores[c][i]) >= confirm_score:
            votes[c] += 1
    best = max(votes, key=votes.get)
    if votes[best] >= n * min_fraction:
        return best
    return None


def parse_vlm(text, classes):
    if not text:
        return None, None
    low = text.lower().strip()
    found = None
    for c in classes:
        if c in low:
            found = c
            break
    if found is None:
        return None, None
    expl = None
    for line in text.splitlines():
        ls = line.strip()
        if ls.startswith("explanation"):
            expl = ls.split(":", 1)[-1].strip()
            break
    if not expl:
        seg = text.replace(found, " ")
        for tok in ("class:", "explanation:", "class :", "explanation :"):
            seg = seg.replace(tok, " ")
        expl = " ".join(seg.split()).strip(" .,;:-")
    expl = expl or None
    return found, expl


def fallback_explanation(cls):
    return "Localized {} anomaly detected in the sampled surveillance footage.".format(
        cls.replace("_", " "))


def make_explanation(candidates, cls):
    for e in candidates:
        if e and 20 <= len(e) <= 500:
            return e
    fb = fallback_explanation(cls)
    if 20 <= len(fb) <= 500:
        return fb
    return None


def runtime_stats(name, ms):
    ms = [float(m) for m in ms]
    if not ms:
        return {
            "model_name": name, "call_count": 0, "total_time_ms": 0,
            "average_time_ms": 0.0, "p50_time_ms": 0.0,
            "p95_time_ms": 0.0, "max_time_ms": 0.0,
        }
    arr = np.array(ms, dtype=np.float64)
    cnt = arr.size
    total = float(arr.sum())
    avg = total / cnt
    return {
        "model_name": name, "call_count": int(cnt), "total_time_ms": round(total, 3),
        "average_time_ms": round(avg, 3), "p50_time_ms": round(float(np.percentile(arr, 50)), 3),
        "p95_time_ms": round(float(np.percentile(arr, 95)), 3),
        "max_time_ms": round(float(arr.max()), 3),
    }


def build_video_entry(video_id, level, events, frames_processed, chunks_processed,
                      internal_ms, model_runtimes):
    return make_video(video_id, events, frames_processed, chunks_processed,
                      int(internal_ms), model_runtimes)


def assemble(predictions, submission_id, model_name, wall_ms, hardware):
    return build(predictions, submission_id=submission_id, model_name=model_name,
                 wall_ms=int(wall_ms), hardware=hardware)


def events_for_level(level, intervals, explanations):
    evs = []
    for (c, s, e) in intervals:
        expl = make_explanation(explanations.get(c, []), c)
        if level == 1:
            evs.append(make_event(c, level=1, explanation=expl))
        else:
            evs.append(make_event(c, s, e, level=2 if level not in (2, 3) else level,
                                  explanation=expl))
    return evs
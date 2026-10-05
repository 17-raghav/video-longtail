import json
import csv
from collections import defaultdict


def load_ground_truth(csv_path):
    gt = defaultdict(lambda: {"level": 1, "is_anomaly": 0, "events": []})
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            vid = row["video_id"]
            lvl = int(row["level"]) if row.get("level") else 1
            gt[vid]["level"] = lvl
            is_anom = str(row.get("is_anomaly", "")).strip() in ("1", "True", "true")
            gt[vid]["is_anomaly"] = 1 if is_anom else gt[vid]["is_anomaly"]
            cls = (row.get("class_name") or "").strip()
            if cls and cls != "normal" and is_anom:
                s = row.get("start_time_sec", "")
                e = row.get("end_time_sec", "")
                s = float(s) if str(s).strip() != "" else None
                e = float(e) if str(e).strip() != "" else None
                gt[vid]["events"].append({"class_name": cls, "start": s, "end": e})
    return dict(gt)


def iou(a_s, a_e, b_s, b_e):
    inter = max(0.0, min(a_e, b_e) - max(a_s, b_s))
    union = max(a_e, b_e) - min(a_s, b_s)
    return inter / union if union > 0 else 0.0


def score_level1(gt, pred, vids):
    if not vids:
        return None
    anom_correct = 0
    cls_correct = 0
    cls_total = 0
    for v in vids:
        g = gt[v]
        p_events = pred.get(v, {}).get("events", [])
        p_is_anom = 1 if p_events else 0
        if p_is_anom == g["is_anomaly"]:
            anom_correct += 1
        if g["is_anomaly"]:
            cls_total += 1
            g_classes = {e["class_name"] for e in g["events"]}
            p_classes = {e["class_name"] for e in p_events}
            if g_classes & p_classes:
                cls_correct += 1
    anom_acc = anom_correct / len(vids)
    cls_acc = cls_correct / cls_total if cls_total else 1.0
    return 0.5 * anom_acc + 0.5 * cls_acc


def score_video_temporal(g_events, p_events, w_alert=0.3, w_match=0.4, w_timing=0.3):
    if not g_events:
        return 1.0 if not p_events else 0.0
    if not p_events:
        return 0.0
    used = [False] * len(p_events)
    matched = 0
    iou_sum = 0.0
    for ge in g_events:
        if ge["start"] is None or ge["end"] is None:
            continue
        best_j, best_iou = -1, 0.0
        for j, pe in enumerate(p_events):
            if used[j] or pe["class_name"] != ge["class_name"]:
                continue
            ps, peo = pe.get("start_time_sec"), pe.get("end_time_sec")
            if ps is None or peo is None:
                continue
            i = iou(ge["start"], ge["end"], ps, peo)
            if i >= 0.5 and i > best_iou:
                best_iou, best_j = i, j
        if best_j >= 0:
            used[best_j] = True
            matched += 1
            iou_sum += best_iou
    n_gt = len([e for e in g_events if e["start"] is not None])
    false_pos = sum(1 for u in used if not u)
    alert = 1.0
    precision = matched / (matched + false_pos) if (matched + false_pos) else 0.0
    recall = matched / n_gt if n_gt else 0.0
    match_score = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    timing = (iou_sum / matched) if matched else 0.0
    return w_alert * alert + w_match * match_score + w_timing * timing


def score(gt_csv, pred_json):
    gt = load_ground_truth(gt_csv)
    with open(pred_json, encoding="utf-8") as f:
        sub = json.load(f)
    pred = {p["video_id"]: p for p in sub.get("predictions", [])}

    l1 = [v for v in gt if gt[v]["level"] == 1]
    l2 = [v for v in gt if gt[v]["level"] == 2]
    l3 = [v for v in gt if gt[v]["level"] == 3]

    out = {}
    out["level1"] = score_level1(gt, pred, l1)
    for name, vids, w in [("level2", l2, (0.3, 0.4, 0.3)), ("level3", l3, (0.2, 0.3, 0.5))]:
        if not vids:
            out[name] = None
            continue
        s = 0.0
        for v in vids:
            g_ev = gt[v]["events"] if gt[v]["is_anomaly"] else []
            p_ev = pred.get(v, {}).get("events", [])
            s += score_video_temporal(g_ev, p_ev, *w)
        out[name] = s / len(vids)
    parts = [x for x in out.values() if x is not None]
    out["overall"] = sum(parts) / len(parts) if parts else 0.0
    return out


if __name__ == "__main__":
    import sys
    print(json.dumps(score(sys.argv[1], sys.argv[2]), indent=2))

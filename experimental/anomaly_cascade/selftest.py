import os
import sys
import json
import tempfile

import numpy as np

import postprocess
from emit import make_event, make_video, build, validate
from scorer import load_ground_truth, score, score_video_temporal

GT_CSV = os.path.join("test", "ground_truth.csv")
OPEN, HOLD, MIN_DUR, MERGE, PAD = 0.5, 0.3, 1.5, 3.0, 0.25


def write_tmp(sub):
    fd, path = tempfile.mkstemp(suffix=".json")
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        json.dump(sub, f)
    return path


def perfect_submission(gt):
    preds = []
    for vid, g in gt.items():
        events = []
        for e in g["events"]:
            events.append(make_event(e["class_name"], e["start"], e["end"],
                                     level=g["level"]))
        preds.append(make_video(vid, events, len(events), 1, 100))
    return build(preds)


def empty_submission(gt):
    return build([make_video(vid, [], 0, 0, 0) for vid in gt])


def test_scorer_mandatory():
    gt = load_ground_truth(GT_CSV)
    perfect = perfect_submission(gt)
    errs = validate(perfect, gt.keys())
    assert errs == [], errs
    p = write_tmp(perfect)
    s_perfect = score(GT_CSV, p)

    empty = empty_submission(gt)
    p2 = write_tmp(empty)
    s_empty = score(GT_CSV, p2)

    normal_l1 = next(v for v, g in gt.items() if g["level"] == 1 and not g["is_anomaly"])
    normal_l2 = next(v for v, g in gt.items() if g["level"] == 2 and not g["is_anomaly"])
    fired = json.loads(json.dumps(perfect))
    by_id = {p_["video_id"]: p_ for p_ in fired["predictions"]}
    by_id[normal_l1]["events"] = [make_event("fire", level=1)]
    by_id[normal_l2]["events"] = [make_event("fire", 5, 10, level=2)]
    p3 = write_tmp(fired)
    s_fired = score(GT_CSV, p3)

    print("TEST perfect:", json.dumps(s_perfect))
    print("TEST all-empty:", json.dumps(s_empty))
    print(f"TEST fired-on-normal {normal_l1}(L1) {normal_l2}(L2):", json.dumps(s_fired))
    assert all(abs(s_perfect[k] - 1.0) < 1e-9
               for k in ("level1", "level2", "level3", "overall")), s_perfect
    assert s_fired["level1"] < s_perfect["level1"]
    assert score_video_temporal(gt[normal_l2]["events"],
                                [{"class_name": "fire", "start": 5, "end": 10}]) == 0.0
    for q in (p, p2, p3):
        os.remove(q)


def test_emit_rejections():
    def vid(v, events, ms=1):
        return {"video_id": v, "events": events,
                "runtime_metadata": {"frames_processed": 1, "chunks_processed": 1,
                                     "end_to_end_internal_time_ms": ms,
                                     "model_runtimes": []}}
    bad = build([
        vid("X1", [{"class_name": "normal", "start_time_sec": None,
                    "end_time_sec": None}]),
        vid("X2", [{"class_name": "not_a_class", "start_time_sec": 0,
                    "end_time_sec": 5}]),
        vid("X3", [{"class_name": "fire", "start_time_sec": 10,
                    "end_time_sec": 0}]),
        vid("X4", [{"class_name": "fire", "start_time_sec": 0, "end_time_sec": 5,
                    "explanation": "short"}]),
        vid("X1", []),
        vid("X5", []),
    ])
    errs = validate(bad, ["X1", "X2", "X3", "X4"])
    print("TEST emit rejections caught:", len(errs), "issues:", errs)
    assert any("normal" in e for e in errs)
    assert any("bad class" in e for e in errs)
    assert any("bad interval" in e for e in errs)
    assert any("explanation length" in e for e in errs)
    assert any("duplicate" in e for e in errs)
    assert any("not in manifest" in e for e in errs)


def test_merge_logic():
    n = 90
    times = [i * 0.5 for i in range(n)]
    scores = {"fire": np.zeros(n, dtype=np.float32), "smoke": np.zeros(n, dtype=np.float32)}
    f, s = scores["fire"], scores["smoke"]
    runs = {"fire": [(4, 10), (17, 18), (25, 26), (28, 29), (40, 42)],
            "smoke": [(46, 48)]}
    frame_classes = [None] * n
    for c, spans in runs.items():
        for a, b in spans:
            for i in range(a, b + 1):
                frame_classes[i] = c
                scores[c][i] = 0.6

    intervals = postprocess.build_intervals(frame_classes, scores, times, OPEN, HOLD,
                                            MERGE, MIN_DUR, PAD, ["fire", "smoke"], 30)
    expected = [
        ("fire", 1.75, 5.25),
        ("fire", 12.25, 14.75),
        ("fire", 19.75, 21.25),
        ("smoke", 22.75, 24.25),
    ]
    print("TEST merge intervals:", intervals)
    assert intervals == expected, intervals


def test_l1_class():
    n = 20
    scores = {
        "fire": np.zeros(n, dtype=np.float32),
        "smoke": np.zeros(n, dtype=np.float32),
    }
    frame_classes = [None] * n
    for i in range(6):
        frame_classes[i] = "fire"
        scores["fire"][i] = 0.8
    for i in range(6, 10):
        frame_classes[i] = "smoke"
        scores["smoke"][i] = 0.7
    assert postprocess.l1_class(frame_classes, scores, ["fire", "smoke"], 0.3, 0.2) == "fire"
    scores["smoke"][6:10] = 0.95
    assert postprocess.l1_class(frame_classes, scores, ["fire", "smoke"], 0.3, 0.2) == "fire"
    assert postprocess.l1_class([None] * n, scores, ["fire", "smoke"], 0.3, 0.2) is None


def test_vlm_parse_and_expl():
    cls, expl = postprocess.parse_vlm(
        "class: fire\nexplanation: Thick smoke rises from a burning structure near the road.", ["fire"])
    assert cls == "fire" and expl and 20 <= len(expl) <= 60, (cls, expl)
    cls2, _ = postprocess.parse_vlm("class: normal", ["fire", "smoke"])
    assert cls2 is None
    cls3, expl3 = postprocess.parse_vlm("wrong_way_driving car on shoulder", ["wrong_way_driving"])
    assert cls3 == "wrong_way_driving"
    fb = postprocess.make_explanation([], "traffic_congestion")
    assert 20 <= len(fb) <= 500
    fb2 = postprocess.make_explanation(["x" * 700, "short", "ok enough".ljust(40, "x")], "smoke")
    assert 20 <= len(fb2) <= 500


def test_events_levels():
    ev1 = postprocess.events_for_level(1, [("fire", 0.0, 0.0)], {"fire": []})
    assert ev1[0]["start_time_sec"] is None and ev1[0]["end_time_sec"] is None
    assert ev1[0]["class_name"] == "fire"
    ev2 = postprocess.events_for_level(2, [("fire", 3.5, 9.25)], {"fire": ["A burning tanker blocks the right lane with flames and thick smoke."]})
    assert ev2[0]["start_time_sec"] == 3.5 and ev2[0]["end_time_sec"] == 9.25
    assert "explanation" in ev2[0]


def main():
    test_scorer_mandatory()
    test_emit_rejections()
    test_merge_logic()
    test_l1_class()
    test_vlm_parse_and_expl()
    test_events_levels()
    print("ALL LOCAL TESTS PASSED")


if __name__ == "__main__":
    main()
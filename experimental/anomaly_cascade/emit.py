import json

CLASSES = {
    "traffic_accident", "traffic_congestion", "stalled_or_broken_down_vehicle",
    "vehicle_blocking_traffic", "wrong_way_driving", "road_spill_or_debris",
    "waterlogging_or_flood", "fire", "smoke", "fighting_or_violence",
    "loitering_or_suspicious_presence",
}


def make_event(class_name, start=None, end=None, explanation=None, level=1):
    assert class_name in CLASSES, f"bad class {class_name}"
    ev = {"class_name": class_name}
    if level == 1:
        ev["start_time_sec"] = None
        ev["end_time_sec"] = None
    else:
        assert start is not None and end is not None and end > start >= 0
        ev["start_time_sec"] = start
        ev["end_time_sec"] = end
    if explanation and 20 <= len(explanation) <= 500:
        ev["explanation"] = explanation
    return ev


def make_video(video_id, events, frames_processed, chunks_processed, internal_ms, model_runtimes=None):
    return {
        "video_id": video_id,
        "events": events,
        "runtime_metadata": {
            "frames_processed": frames_processed,
            "chunks_processed": chunks_processed,
            "end_to_end_internal_time_ms": internal_ms,
            "model_runtimes": model_runtimes or [],
        },
    }


def build(predictions, submission_id="team-run", model_name="cascade-vad", wall_ms=0, hardware="1x T4"):
    return {
        "schema_version": "1.0",
        "submission_id": submission_id,
        "model_name": model_name,
        "run_metadata": {"total_wall_time_ms": wall_ms, "hardware": hardware},
        "predictions": predictions,
    }


def validate(sub, manifest_ids=None):
    errs = []
    assert "predictions" in sub, "missing predictions"
    seen = set()
    for p in sub["predictions"]:
        vid = p.get("video_id")
        if vid in seen:
            errs.append(f"{vid}: duplicate")
        seen.add(vid)
        if "events" not in p or "runtime_metadata" not in p:
            errs.append(f"{vid}: missing events or runtime_metadata")
            continue
        for ev in p["events"]:
            if ev.get("class_name") == "normal":
                errs.append(f"{vid}: class_name normal is illegal, use events:[]")
            if ev.get("class_name") not in CLASSES:
                errs.append(f"{vid}: bad class {ev.get('class_name')}")
            s, e = ev.get("start_time_sec"), ev.get("end_time_sec")
            if s is not None and e is not None and not (e > s >= 0):
                errs.append(f"{vid}: bad interval {s}-{e}")
            exp = ev.get("explanation")
            if exp is not None and not (20 <= len(exp) <= 500):
                errs.append(f"{vid}: explanation length {len(exp)} out of 20-500")
    if manifest_ids:
        missing = set(manifest_ids) - seen
        extra = seen - set(manifest_ids)
        if extra:
            errs.append(f"ids not in manifest: {extra}")
    return errs


if __name__ == "__main__":
    demo = build([
        make_video("E001", [], 90, 1, 4820),
        make_video("E009", [make_event("fire", level=1,
                    explanation="Thick smoke rises from a burning structure on the roadside.")], 30, 1, 2100),
        make_video("E021", [
            make_event("traffic_accident", 20, 40, level=2),
            make_event("traffic_accident", 60, 80, level=2),
        ], 600, 24, 91000),
    ])
    print("errors:", validate(demo))
    print(json.dumps(demo, indent=2))

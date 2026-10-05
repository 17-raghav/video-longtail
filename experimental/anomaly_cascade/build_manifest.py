import os
import csv
import json
import argparse


def load_levels(gt_csv):
    levels = {}
    with open(gt_csv, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            vid = row["video_id"].strip()
            lvl = int(row.get("level") or 1)
            if vid not in levels:
                levels[vid] = lvl
    return levels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--out", default="manifest.json")
    args = ap.parse_args()

    videos_csv = os.path.join(args.dir, "videos.csv")
    gt_csv = os.path.join(args.dir, "ground_truth.csv")
    levels = {}
    if os.path.exists(gt_csv):
        levels = load_levels(gt_csv)

    videos = []
    with open(videos_csv, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            vid = row["video_id"].strip()
            fn = row.get("filename", "").strip()
            videos.append({"video_id": vid, "level": levels.get(vid, 2),
                           "filename": fn})
    manifest = {"schema_version": "1.0", "videos": videos}
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print("wrote", args.out, "videos", len(videos))


if __name__ == "__main__":
    main()
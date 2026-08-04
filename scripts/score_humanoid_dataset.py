#!/usr/bin/env python3

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from score_lerobot_episodes.scores.humanoid import (  # noqa: E402
    build_time_stats,
    score_episode,
    signals_from_dataframe,
)


def find_episodes(root: str) -> dict[int, str]:
    """Map episode index -> parquet path."""
    out = {}
    for path in sorted(glob.glob(os.path.join(root, "data", "**", "episode_*.parquet"), recursive=True)):
        idx = int(os.path.basename(path).split("_")[1].split(".")[0])
        out[idx] = path
    return out


def find_video(root: str, episode: int, camera: str) -> str | None:
    hits = glob.glob(
        os.path.join(root, "videos", "**", camera, f"episode_{episode:06d}.mp4"), recursive=True
    )
    return hits[0] if hits else None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", help="dataset root (the directory holding meta/, data/, videos/)")
    ap.add_argument("--out", help="write per-episode scores to this CSV")
    ap.add_argument("--camera", default=None,
                    help="camera key to score (default: first observation.images.* in info.json)")
    ap.add_argument("--no-video", action="store_true", help="skip visual_clarity")
    ap.add_argument("--threshold", type=float, default=None,
                    help="if set, list episodes scoring below this")
    args = ap.parse_args()

    info = json.load(open(os.path.join(args.root, "meta", "info.json")))
    camera = args.camera
    if camera is None:
        cameras = [k for k in info["features"] if k.startswith("observation.images.")]
        camera = cameras[0] if cameras else None

    episodes = find_episodes(args.root)
    if not episodes:
        print(f"no episode parquet files under {args.root}/data", file=sys.stderr)
        return 1
    print(f"{len(episodes)} episodes, camera={camera}")

    # Duration statistics have to be dataset-wide, so build the signals first.
    signals = {ep: signals_from_dataframe(pd.read_parquet(p)) for ep, p in episodes.items()}
    time_stats = build_time_stats([s.duration for s in signals.values()])

    rows = []
    for ep in sorted(signals):
        video = None if args.no_video or camera is None else find_video(args.root, ep, camera)
        result = score_episode(signals[ep], video_path=video, time_stats=time_stats)
        row = {"episode": ep, "total": result["score"], **result["sub_scores"]}
        row["duration_s"] = signals[ep].duration
        row["degenerate"] = signals[ep].is_degenerate
        row["collision_events"] = result["detail"]["collision"].get("collision_events")
        row["grasp_transitions"] = result["detail"]["runtime"].get("grasp_transitions")
        row["idle_fraction"] = result["detail"]["runtime"].get("idle_fraction")
        rows.append(row)

    df = pd.DataFrame(rows).set_index("episode")
    pd.set_option("display.width", 200)

    print("\n=== score distribution ===")
    score_cols = [c for c in ("total", "smoothness", "collision", "runtime",
                              "acceleration", "visual_clarity") if c in df]
    print(df[score_cols].describe().T.to_string(float_format=lambda x: f"{x:8.4f}"))

    print("\n=== 10 lowest-scoring episodes ===")
    print(df.sort_values("total").head(10).round(3).to_string())

    if args.threshold is not None:
        flagged = df.index[df.total < args.threshold].tolist()
        print(f"\n{len(flagged)} episodes below {args.threshold}: {flagged}")

    degenerate = df.index[df.degenerate].tolist()
    if degenerate:
        print(f"\ndegenerate (no meaningful motion): {degenerate}")

    # Videos that exist on disk but have no corresponding episode.
    if camera:
        on_disk = glob.glob(os.path.join(args.root, "videos", "**", camera, "episode_*.mp4"),
                            recursive=True)
        orphans = sorted(
            int(os.path.basename(p).split("_")[1].split(".")[0]) for p in on_disk
            if int(os.path.basename(p).split("_")[1].split(".")[0]) not in episodes
        )
        if orphans:
            print(f"orphaned videos with no episode data: {orphans}")

    if args.out:
        df.to_csv(args.out)
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

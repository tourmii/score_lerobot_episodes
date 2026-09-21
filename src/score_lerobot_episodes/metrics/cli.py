#!/usr/bin/env python3
"""Measure, score, semantically filter and visualise a LeRobot humanoid dataset.

Runs the whole pipeline over a local dataset folder::

    parquet ─▶ raw measurements ─▶ calibration fit ─▶ scores ─▶ decision
                     │                                             ▲
                     └────────────▶ video ─▶ semantic verdict ─────┘
                                      │
                                      └─▶ synchronised HTML visualisation

Reads the episode parquet files directly instead of going through
``LeRobotDataset``, so it needs neither lerobot nor a GPU, and keeps the
humanoid channels (``eef_state``, ``projected_gravity``, the hand encoders)
that the metrics depend on.

For interactive review use the web app (``python -m app``) instead; this is the
batch path, for CI and for producing a report to hand on.

Examples::

    # measurements and scores only, no video decoding (fastest)
    python scripts/measure_dataset.py pickup_20260628_150622 --no-video

    # full run with the synchronised visualisation for the 15 worst episodes
    python scripts/measure_dataset.py pickup_20260628_150622 --html 15

    # add the semantic filter (needs a Cosmos-Reason vLLM server, see --semantic-*)
    python scripts/measure_dataset.py pickup_20260628_150622 --semantic
"""

from __future__ import annotations

import argparse
import concurrent.futures as futures
import glob
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from . import (
    Calibration,
    DEFAULT_WEIGHTS,
    Policy,
    QualityProfile,
    drift_report,
    format_drift,
    score_episode_absolute,
    groups_from_modality,
    measure_episode,
    score_episode,
    signals_from_dataframe,
)


# --------------------------------------------------------------------------
# Dataset layout
# --------------------------------------------------------------------------


def find_episodes(root: str | Path) -> dict[int, str]:
    """Map episode index -> parquet path."""
    out: dict[int, str] = {}
    pattern = os.path.join(str(root), "data", "**", "episode_*.parquet")
    for path in sorted(glob.glob(pattern, recursive=True)):
        out[_episode_index(path)] = path
    return out


def find_video(root: str | Path, episode: int, camera: str | None) -> str | None:
    if camera is None:
        return None
    hits = glob.glob(
        os.path.join(str(root), "videos", "**", camera, f"episode_{episode:06d}.mp4"),
        recursive=True,
    )
    return hits[0] if hits else None


def _episode_index(path: str) -> int:
    return int(os.path.basename(path).split("_")[1].split(".")[0])


def default_camera(root: str | Path) -> str | None:
    info_path = Path(root) / "meta" / "info.json"
    if not info_path.exists():
        return None
    info = json.loads(info_path.read_text(encoding="utf-8"))
    cameras = [k for k in info.get("features", {}) if k.startswith("observation.images.")]
    return cameras[0] if cameras else None


def orphaned_videos(root: str | Path, camera: str | None, episodes: dict[int, str]) -> list[int]:
    """Videos on disk with no episode data — a recording that never got saved."""
    if camera is None:
        return []
    pattern = os.path.join(str(root), "videos", "**", camera, "episode_*.mp4")
    indices = {_episode_index(p) for p in glob.glob(pattern, recursive=True)}
    return sorted(indices - set(episodes))


# --------------------------------------------------------------------------
# Pipeline
# --------------------------------------------------------------------------


def measure_all(
    root: str | Path,
    episodes: dict[int, str],
    camera: str | None,
    use_video: bool,
    keep_series: bool,
    workers: int,
    groups: dict[str, tuple[int, int]] | None,
    progress: bool = True,
):
    """Measure every episode, optionally in parallel (video decoding dominates)."""

    def one(episode: int):
        video = find_video(root, episode, camera) if use_video else None
        frame = pd.read_parquet(episodes[episode])
        sig = signals_from_dataframe(frame, groups=groups, episode=episode, video_path=video)
        return measure_episode(sig, video_path=video, keep_series=keep_series)

    out = {}
    if workers <= 1:
        for i, episode in enumerate(sorted(episodes), 1):
            out[episode] = one(episode)
            if progress:
                _tick(i, len(episodes))
    else:
        with futures.ThreadPoolExecutor(max_workers=workers) as pool:
            pending = {pool.submit(one, ep): ep for ep in sorted(episodes)}
            for i, future in enumerate(futures.as_completed(pending), 1):
                out[pending[future]] = future.result()
                if progress:
                    _tick(i, len(episodes))
    if progress:
        print(file=sys.stderr)
    return dict(sorted(out.items()))


def _tick(done: int, total: int) -> None:
    print(f"\r  measured {done}/{total}", end="", file=sys.stderr, flush=True)


def parse_weights(text: str | None) -> dict[str, float]:
    """``smoothness=0.3,acceleration=0.3`` -> dict, merged over the defaults."""
    weights = dict(DEFAULT_WEIGHTS)
    if not text:
        return weights
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        key, _, value = item.partition("=")
        key = key.strip()
        if key not in weights:
            raise SystemExit(f"unknown metric family {key!r}; expected one of {sorted(weights)}")
        weights[key] = float(value)
    return weights


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def parse_limits(items: list[str]) -> list:
    """``acc_arm_p99>18`` / ``ldlj_wrist<-20:review`` -> Rule objects."""
    from . import RULE_QUANTITIES, Rule

    rules = []
    for item in items:
        text, _, action = item.partition(":")
        for op in (">", "<"):
            quantity, sep, value = text.partition(op)
            if sep:
                break
        else:
            raise SystemExit(f"--limit needs '>' or '<': {item!r}")
        quantity = quantity.strip()
        if quantity not in RULE_QUANTITIES:
            raise SystemExit(
                f"unknown quantity {quantity!r}; expected one of {sorted(RULE_QUANTITIES)}")
        rules.append(Rule(quantity=quantity, op=op, limit=float(value),
                          action=action or "reject", enabled=True))
    return rules


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("root", help="dataset root (the folder holding meta/, data/, videos/)")
    p.add_argument("--out-dir", default="quality_report", help="where to write the outputs")
    p.add_argument("--camera", default=None,
                   help="camera key (default: first observation.images.* in info.json)")
    p.add_argument("--no-video", action="store_true",
                   help="skip video measurement — by far the slowest family")
    p.add_argument("--workers", type=int, default=4, help="parallel episodes")

    cal = p.add_argument_group("calibration")
    cal.add_argument("--calibration", help="load ranges from this JSON instead of fitting")
    cal.add_argument("--no-fit", action="store_true",
                     help="use the shipped reference ranges as-is (they are dataset-specific)")
    cal.add_argument("--lo-pct", type=float, default=5.0, help="lower percentile of each range")
    cal.add_argument("--hi-pct", type=float, default=95.0, help="upper percentile of each range")

    pro = p.add_argument_group("absolute profile (mode=absolute)")
    pro.add_argument("--profile", help="load anchors from this JSON and DO NOT refit — "
                                       "this is what makes one threshold transfer")
    pro.add_argument("--fit-profile", metavar="OUT.json",
                     help="fit the platform anchors on THIS dataset and write them out. "
                          "Run once, on a batch you have inspected; reuse with --profile")
    pro.add_argument("--good-sigma", type=float, default=1.0,
                     help="robust sigmas from the median to the 'stops mattering' anchor "
                          "(higher = more permissive, default 1.0)")
    pro.add_argument("--bad-sigma", type=float, default=4.0,
                     help="robust sigmas to the 'condemns on its own' anchor (default 4.0)")
    pro.add_argument("--task-duration", type=float, default=None,
                     help="seconds a clean run of this task takes; a task constant, not a "
                          "dataset statistic (default: median of the fitted batch)")
    pro.add_argument("--drift", action="store_true",
                     help="report how far this dataset sits outside the loaded profile")

    dec = p.add_argument_group("decision")
    dec.add_argument("--mode", choices=("absolute", "gate", "rules", "weighted"),
                     default="absolute",
                     help="absolute (default): one threshold on a noisy-OR of "
                          "physical-unit criteria, comparable across datasets; gate: "
                          "every weighted family must clear its own threshold; rules: a "
                          "limit per quantity, no aggregate; weighted: only the weighted "
                          "mean must pass")
    dec.add_argument("--limit", action="append", default=[], metavar="QTY>VALUE",
                     help="physical limit for rules mode, e.g. 'acc_arm_p99>18' or "
                          "'ldlj_wrist<-20'; repeatable. Suffix with ':review' to flag "
                          "instead of reject")
    dec.add_argument("--threshold", type=float, default=None,
                     help="accept above this. absolute (the default mode): P(no criterion "
                          "violated), 0.5 unless set, and the only knob you should need. "
                          "gate: per family, 0.35. weighted: ~0.6")
    dec.add_argument("--weights", help="family weights, e.g. 'acceleration=0.4,video=0.1'")
    dec.add_argument("--min", action="append", default=[], metavar="FAMILY=VALUE",
                     help="per-family floor sending an episode to review; repeatable")

    sem = p.add_argument_group("semantic filter (vision-language model)")
    sem.add_argument("--semantic", action="store_true",
                     help="judge task success from the video with Cosmos-Reason")
    sem.add_argument("--semantic-base-url", default=None, help="vLLM endpoint (COSMOS_BASE_URL)")
    sem.add_argument("--semantic-model", default=None, help="served model name (COSMOS_MODEL)")
    sem.add_argument("--api-key", default=None,
                     help="credential for a hosted endpoint (COSMOS_API_KEY). Leave it "
                          "off for a local vLLM server, which wants no credential")
    sem.add_argument("--semantic-workers", type=int, default=4, help="concurrent requests")
    sem.add_argument("--semantic-cache", default=None,
                     help="verdict JSONL, reused across runs (default: OUT_DIR/semantic.jsonl)")
    sem.add_argument("--semantic-max-frames", type=int, default=0,
                     help="subsample each clip to N frames with ffmpeg before sending")
    sem.add_argument("--semantic-inline-media", action="store_true",
                     help="send the video base64-encoded instead of by file:// path")
    sem.add_argument("--task", default=None,
                     help="task text (default: read from meta/episodes.jsonl or tasks.jsonl)")

    viz = p.add_argument_group("visualisation")
    viz.add_argument("--html", nargs="?", const="all", default=None, metavar="N",
                     help="write synchronised video+signal pages for the N worst episodes "
                          "(or 'all')")
    viz.add_argument("--embed-video", action="store_true",
                     help="inline the mp4 into each page so it is portable on its own")
    viz.add_argument("--overlay-mp4", default=None, metavar="EPISODES",
                     help="also burn the signal panel into an mp4, e.g. '3,17' or 'worst:5'")
    return p


def main() -> int:
    args = build_parser().parse_args()
    root = Path(args.root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    camera = args.camera or default_camera(root)
    episodes = find_episodes(root)
    if not episodes:
        print(f"no episode parquet files under {root}/data", file=sys.stderr)
        return 1

    modality = root / "meta" / "modality.json"
    groups = groups_from_modality(modality) if modality.exists() else None
    use_video = not args.no_video and camera is not None
    want_viz = args.html is not None or args.overlay_mp4 is not None

    print(f"{len(episodes)} episodes · camera={camera or 'none'} · "
          f"video={'on' if use_video else 'off'} · groups={list(groups) if groups else 'default'}")

    # ---- 1. measure -----------------------------------------------------
    measures = measure_all(root, episodes, camera, use_video, want_viz, args.workers, groups)

    raw = pd.DataFrame([m.to_row() for m in measures.values()], index=list(measures))
    raw.index.name = "episode"
    raw.to_csv(out_dir / "measurements.csv")

    invalid = raw.index[~raw["valid"]].tolist()
    print(f"\nvalid: {int(raw['valid'].sum())}/{len(raw)}"
          + (f" · excluded: {invalid}" if invalid else ""))

    # ---- 2. calibrate ---------------------------------------------------
    absolute = args.mode == "absolute" or args.profile or args.fit_profile
    profile = None
    if absolute:
        profile = build_profile(args, measures, root, out_dir)

    if args.calibration:
        calib = Calibration.load(args.calibration)
        print(f"calibration loaded from {args.calibration}")
    elif args.no_fit:
        calib = Calibration()
        print("calibration: shipped reference ranges (not refitted)")
    else:
        calib = Calibration.fit(measures.values(), lo_pct=args.lo_pct, hi_pct=args.hi_pct)
        print(f"calibration fitted on {calib.n_episodes_fitted} valid episodes "
              f"(p{args.lo_pct:g}/p{args.hi_pct:g}); grasp nominal={calib.grasp_nominal}, "
              f"duration median={calib.duration_median:.2f}s")
    calib.save(out_dir / "calibration.json")

    # ---- 3. semantic filter --------------------------------------------
    verdicts = {}
    if args.semantic:
        verdicts = run_semantic(args, root, episodes, camera, out_dir)

    # ---- 4. score and decide -------------------------------------------
    weights = parse_weights(args.weights)
    minimums = {}
    for item in args.min:
        key, _, value = item.partition("=")
        minimums[key.strip()] = float(value)
    policy = Policy(mode=args.mode if not absolute else "weighted",
                    accept_threshold=args.threshold if args.threshold is not None else 0.35,
                    minimums=minimums, rules=parse_limits(args.limit))

    scores = {}
    for episode, m in measures.items():
        verdict = verdicts.get(episode)
        semantic = None if verdict is None else verdict.score
        note = "" if verdict is None else verdict.summary
        if absolute:
            scores[episode] = score_episode_absolute(m, profile, semantic, note)
        else:
            scores[episode] = score_episode(m, calib, weights, policy, semantic, note)

    table = pd.DataFrame([s.to_row() for s in scores.values()]).set_index("episode")
    table.to_csv(out_dir / "scores.csv")

    if absolute:
        report_absolute(table, profile, root, camera, episodes)
        if args.drift:
            print("\ndrift against the loaded anchors "
                  "(high >bad on a lone criterion = the profile, not the batch):")
            print(format_drift(drift_report(measures.values(), profile)))
    else:
        report(raw, table, weights, root, camera, episodes)

    # ---- 5. visualise ---------------------------------------------------
    if want_viz:
        visualise(args, root, camera, measures, scores, table, out_dir)

    written = "measurements.csv, scores.csv, calibration.json"
    if absolute:
        written += ", profile.json"
    print(f"\nwrote {out_dir}/{written}")
    return 0


# --------------------------------------------------------------------------
# Absolute profile
# --------------------------------------------------------------------------


def build_profile(args, measures, root, out_dir) -> QualityProfile:
    """Load a frozen profile, or fit one and say plainly that it is not frozen.

    The distinction is the whole point of the mode, so it is printed rather than
    implied: a profile loaded from disk gives a verdict that does not depend on
    the batch, and a profile fitted here gives one that does.
    """
    if args.profile:
        profile = QualityProfile.load(args.profile)
        print(f"profile loaded from {args.profile}"
              + (f" (fitted on {profile.fitted_on}, "
                 f"{profile.n_episodes_fitted} episodes)" if profile.fitted_on else "")
              + " — anchors NOT refitted")
    else:
        profile = QualityProfile.fit(
            measures.values(), good_sigma=args.good_sigma, bad_sigma=args.bad_sigma,
            fitted_on=str(root),
        )
        print(f"profile fitted on {profile.n_episodes_fitted} valid episodes of THIS "
              f"dataset (good=+{args.good_sigma:g}s, bad=+{args.bad_sigma:g}s).")
        print("  the verdict therefore still depends on this batch. Save it with "
              "--fit-profile and pass it back with --profile on every later run.")

    if args.task_duration is not None:
        profile.task_duration_s = float(args.task_duration)
    if args.threshold is not None:
        profile.threshold = float(args.threshold)

    print(f"  threshold={profile.threshold:g} · task duration="
          f"{profile.task_duration_s:.2f}s · grasp nominal={profile.grasp_nominal} · "
          f"{len(profile.active())} active criteria")

    profile.save(out_dir / "profile.json")
    if args.fit_profile:
        profile.save(args.fit_profile)
        print(f"  profile written to {args.fit_profile}")
    return profile


def report_absolute(table, profile, root, camera, episodes) -> None:
    """Decisions, the score spread, and what the criteria actually cost."""
    counts = table["decision"].value_counts()
    n = len(table)
    print(f"\n{'decision':<10}{'n':>6}{'share':>9}")
    print("-" * 25)
    for name in ("accept", "review", "reject"):
        k = int(counts.get(name, 0))
        print(f"{name:<10}{k:>6}{k / n:>9.0%}")

    totals = table["total"].to_numpy(dtype=float)
    q = np.percentile(totals, [10, 25, 50, 75, 90])
    print(f"\nscore P(no criterion violated)  p10={q[0]:.3f}  p25={q[1]:.3f}  "
          f"median={q[2]:.3f}  p75={q[3]:.3f}  p90={q[4]:.3f}")
    print(f"threshold {profile.threshold:g} sits at the "
          f"{float((totals < profile.threshold).mean()):.0%} mark of this batch")

    rejected = table[table["decision"] == "reject"]
    if len(rejected):
        print(f"\nworst {min(10, len(rejected))} by severity:")
        for episode, row in rejected.nlargest(min(10, len(rejected)), "severity").iterrows():
            print(f"  ep {episode:<5} total={row['total']:.3f}  "
                  f"severity={row['severity']:>6.1f}  {row['reasons'][:96]}")


def run_semantic(args, root, episodes, camera, out_dir):
    """Judge task success for every episode that has a video."""
    from .semantic import SemanticFilter, default_task, server_is_reachable, tasks_from_meta

    if camera is None:
        print("semantic filter skipped: no camera in this dataset", file=sys.stderr)
        return {}
    api_key = (args.api_key or "").strip() or None
    if not server_is_reachable(args.semantic_base_url, api_key=api_key):
        base = args.semantic_base_url or os.environ.get(
            "COSMOS_BASE_URL", "http://localhost:8000/v1")
        print(f"semantic filter skipped: no vLLM server at {base}\n"
              "  start one with: vllm serve nvidia/Cosmos-Reason2-8B "
              f"--allowed-local-media-path {Path(root).resolve()}", file=sys.stderr)
        return {}

    per_episode = tasks_from_meta(root)
    fallback = args.task or default_task(root)
    jobs = []
    for episode in sorted(episodes):
        video = find_video(root, episode, camera)
        task = per_episode.get(episode, fallback)
        if video and task:
            jobs.append((episode, video, task))
    if not jobs:
        print("semantic filter skipped: no (video, task) pairs found", file=sys.stderr)
        return {}

    cache = args.semantic_cache or (out_dir / "semantic.jsonl")
    filt = SemanticFilter(
        base_url=args.semantic_base_url,
        model=args.semantic_model,
        api_key=api_key,
        cache=cache,
        inline_media=args.semantic_inline_media,
        max_frames=args.semantic_max_frames,
        structured=True,
    )
    print(f"semantic filter: {len(jobs)} episodes · cache {cache}")

    done = {"n": 0}

    def tick(verdict):
        done["n"] += 1
        mark = "cache" if verdict.cached else f"{verdict.score}"
        print(f"\r  judged {done['n']}/{len(jobs)} (last: ep {verdict.episode} -> {mark})",
              end="", file=sys.stderr, flush=True)

    verdicts = filt.evaluate_many(jobs, workers=args.semantic_workers, on_result=tick)
    print(file=sys.stderr)

    failed = [v for v in verdicts.values() if not v.ok]
    if failed:
        print(f"  {len(failed)} episodes could not be judged; first: {failed[0].error}",
              file=sys.stderr)
    scored = [v.score for v in verdicts.values() if v.ok]
    if scored:
        counts = {value: scored.count(value) for value in sorted(set(scored))}
        print("  verdicts: " + ", ".join(f"{k}: {v}" for k, v in counts.items()))
    return verdicts


def report(raw, table, weights, root, camera, episodes) -> None:
    """Print the distribution, the worst episodes and the integrity findings."""
    pd.set_option("display.width", 220)
    families = [c for c in weights if c in table.columns]

    print("\n=== score distribution ===")
    columns = ["total"] + families
    print(table[columns].describe().T.to_string(float_format=lambda x: f"{x:8.4f}"))

    print("\n=== decisions ===")
    for decision, count in table["decision"].value_counts().items():
        listed = table.index[table["decision"] == decision].tolist()
        shown = listed if len(listed) <= 20 else listed[:20] + ["..."]
        print(f"  {decision:<7} {count:>3}  {shown}")

    print("\n=== 10 lowest-scoring episodes ===")
    worst = table.sort_values("total").head(10)
    print(worst[columns + ["decision", "reasons"]].round(3).to_string())

    review = table.index[table["decision"] == "review"].tolist()
    if review:
        print("\n=== flagged for review ===")
        for episode in review:
            print(f"  ep {episode:>3}: {table.loc[episode, 'reasons']}")

    orphans = orphaned_videos(root, camera, episodes)
    if orphans:
        print(f"\nvideos on disk with no episode data: {orphans}")


def visualise(args, root, camera, measures, scores, table, out_dir) -> None:
    """Write the per-episode pages, the index, and any requested overlay mp4s."""
    from .visualize import render_episode_html, render_index_html, render_overlay_video

    pages_dir = out_dir / "episodes"
    order = table.sort_values("total").index.tolist()

    selected = order
    if args.html not in (None, "all"):
        selected = order[: int(args.html)]

    entries = []
    for episode in selected:
        m = measures[episode]
        page = pages_dir / f"episode_{episode:06d}.html"
        render_episode_html(
            m, page, score=scores[episode],
            video_path=find_video(root, episode, camera),
            embed_video=args.embed_video,
            index_href="../index.html",
        )
        entries.append({
            "episode": episode,
            "href": f"episodes/{page.name}",
            "total": float(scores[episode].total),
            "decision": scores[episode].decision,
            "families": {
                k: (None if not np.isfinite(v) else float(v))
                for k, v in scores[episode].families.items()
            },
            "semantic": scores[episode].semantic_score,
            "flags": m.flags.raised(),
        })

    if entries:
        index = render_index_html(entries, out_dir / "index.html",
                                  title=f"{Path(root).name} — episode quality")
        print(f"\nwrote {len(entries)} pages · open {index}")

    if args.overlay_mp4:
        for episode in _parse_selection(args.overlay_mp4, order):
            video = find_video(root, episode, camera)
            if video is None:
                continue
            target = out_dir / "overlay" / f"episode_{episode:06d}_overlay.mp4"
            render_overlay_video(measures[episode], target, video_path=video)
            print(f"wrote {target}")


def _parse_selection(spec: str, order: list[int]) -> list[int]:
    """``'3,17'`` -> those episodes; ``'worst:5'`` -> the five lowest-scoring."""
    spec = spec.strip()
    if spec.startswith("worst:"):
        return order[: int(spec.split(":", 1)[1])]
    if spec == "worst":
        return order[:1]
    return [int(x) for x in spec.split(",") if x.strip()]


if __name__ == "__main__":
    raise SystemExit(main())

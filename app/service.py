
from __future__ import annotations

import glob
import hashlib
import json
import os
import pickle
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np

from score_lerobot_episodes.metrics import (
    Calibration,
    Rule,
    rule_value,
    RULE_QUANTITIES,
    suggest_rules,
    violations,
    DEFAULT_WEIGHTS,
    EpisodeMeasures,
    EpisodeScore,
    Policy,
    groups_from_modality,
    measure_episode,
    score_episode,
    signals_from_dataframe,
)

#: Bump when the pickled measurement layout changes, so stale caches are ignored
#: rather than half-read.
CACHE_VERSION = 3


# --------------------------------------------------------------------------
# Jobs
# --------------------------------------------------------------------------


@dataclass
class Job:
    """A background task the UI polls: analysis, or a semantic pass."""

    id: str
    kind: str
    dataset: str
    state: str = "running"          # running | done | error
    done: int = 0
    total: int = 0
    message: str = ""
    error: str = ""
    started: float = field(default_factory=time.time)
    finished: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id, "kind": self.kind, "dataset": self.dataset,
            "state": self.state, "done": self.done, "total": self.total,
            "message": self.message, "error": self.error,
            "elapsed": round((self.finished or time.time()) - self.started, 1),
        }


class JobRegistry:
    def __init__(self) -> None:
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()

    def create(self, kind: str, dataset: str, total: int = 0) -> Job:
        job = Job(id=uuid.uuid4().hex[:12], kind=kind, dataset=dataset, total=total)
        with self._lock:
            self._jobs[job.id] = job
        return job

    def get(self, job_id: str) -> Job | None:
        return self._jobs.get(job_id)

    def run(self, job: Job, target: Callable[[Job], None]) -> Job:
        """Run ``target`` on a worker thread, recording success or failure."""
        def wrapper() -> None:
            try:
                target(job)
                job.state = "done"
            except Exception as exc:  # noqa: BLE001 - surfaced to the UI verbatim
                job.state = "error"
                job.error = f"{type(exc).__name__}: {exc}"
            finally:
                job.finished = time.time()

        threading.Thread(target=wrapper, daemon=True, name=f"job-{job.id}").start()
        return job


# --------------------------------------------------------------------------
# Dataset
# --------------------------------------------------------------------------


@dataclass
class DatasetHandle:
    """One LeRobot dataset folder plus whatever has been computed about it."""

    id: str
    root: Path
    name: str
    camera: str | None
    episode_paths: dict[int, str]
    groups: dict[str, tuple[int, int]] | None
    fps: float | None = None
    robot_type: str | None = None
    task: str = ""
    discarded: list[int] = field(default_factory=list)
    cameras: list[str] = field(default_factory=list)
    total_frames: int = 0

    measures: dict[int, EpisodeMeasures] = field(default_factory=dict)
    calibration: Calibration | None = None
    semantic: dict[int, dict[str, Any]] = field(default_factory=dict)
    analyzed_at: float | None = None
    analyzed_with_video: bool = False

    weights: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_WEIGHTS))
    threshold: float = 0.35
    minimums: dict[str, float] = field(default_factory=dict)
    mode: str = "gate"
    aggregate: str = "geometric"
    rules: list[Rule] = field(default_factory=list)

    @property
    def analyzed(self) -> bool:
        return bool(self.measures)

    def summary(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "path": str(self.root),
            "episodes": len(self.episode_paths),
            "frames": self.total_frames,
            "camera": self.camera,
            "cameras": self.cameras,
            "fps": self.fps,
            "robotType": self.robot_type,
            "task": self.task,
            "analyzed": self.analyzed,
            "analyzedWithVideo": self.analyzed_with_video,
            "analyzedAt": self.analyzed_at,
            "hasSemantic": bool(self.semantic),
            "discarded": self.discarded,
        }

    def policy(self) -> Policy:
        return Policy(mode=self.mode, accept_threshold=self.threshold,
                      minimums=dict(self.minimums), rules=list(self.rules),
                      aggregate=self.aggregate)

    def video_path(self, episode: int) -> str | None:
        return find_video(self.root, episode, self.camera)

    # ------------------------------------------------------------- scoring
    def scores(self) -> dict[int, EpisodeScore]:
        """Score every measured episode under the current weights and policy."""
        calib = self.calibration or Calibration()
        policy = self.policy()
        out: dict[int, EpisodeScore] = {}
        for episode, m in self.measures.items():
            verdict = self.semantic.get(episode)
            out[episode] = score_episode(
                m, calib, self.weights, policy,
                semantic_score=None if not verdict else verdict.get("score"),
                semantic_note="" if not verdict else (verdict.get("summary") or ""),
            )
        return out


def dataset_id(path: Path) -> str:
    return hashlib.sha1(str(path.resolve()).encode()).hexdigest()[:12]


def find_episodes(root: Path) -> dict[int, str]:
    out: dict[int, str] = {}
    for path in sorted(glob.glob(str(root / "data" / "**" / "episode_*.parquet"), recursive=True)):
        out[episode_index(path)] = path
    return out


def find_video(root: Path, episode: int, camera: str | None) -> str | None:
    if camera is None:
        return None
    hits = glob.glob(
        str(root / "videos" / "**" / camera / f"episode_{episode:06d}.mp4"), recursive=True
    )
    return hits[0] if hits else None


def episode_index(path: str) -> int:
    return int(os.path.basename(path).split("_")[1].split(".")[0])


def list_cameras(root: Path) -> list[str]:
    info = _info(root)
    return [k for k in info.get("features", {}) if k.startswith("observation.images.")]


def _info(root: Path) -> dict[str, Any]:
    path = root / "meta" / "info.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def open_dataset(path: str | Path, camera: str | None = None) -> DatasetHandle:
    """Validate a folder as a LeRobot dataset and build its handle."""
    root = Path(path).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"not a directory: {root}")

    episodes = find_episodes(root)
    if not episodes:
        raise ValueError(f"no episode parquet files under {root}/data")

    info = _info(root)
    cameras = list_cameras(root)
    modality = root / "meta" / "modality.json"

    task = ""
    tasks_file = root / "meta" / "tasks.jsonl"
    if tasks_file.exists():
        for line in tasks_file.read_text(encoding="utf-8").splitlines():
            if line.strip():
                task = str(json.loads(line).get("task", ""))
                break

    total_frames = int(info.get("total_frames") or 0)

    return DatasetHandle(
        id=dataset_id(root),
        root=root,
        name=root.name,
        camera=camera or (cameras[0] if cameras else None),
        episode_paths=episodes,
        groups=groups_from_modality(modality) if modality.exists() else None,
        fps=info.get("fps"),
        robot_type=info.get("robot_type"),
        task=task,
        discarded=list(info.get("discarded_episode_indices", []) or []),
        cameras=cameras,
        total_frames=total_frames,
    )


# --------------------------------------------------------------------------
# Service
# --------------------------------------------------------------------------


class QualityService:
    """Everything the HTTP layer needs, with no HTTP in it."""

    def __init__(self, state_dir: str | Path = ".quality_app") -> None:
        self.state_dir = Path(state_dir).expanduser().resolve()
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.datasets: dict[str, DatasetHandle] = {}
        self.jobs = JobRegistry()
        self._lock = threading.Lock()
        self._restore()

    # ---------------------------------------------------------- registry
    def register(self, path: str | Path, camera: str | None = None) -> DatasetHandle:
        handle = open_dataset(path, camera)
        with self._lock:
            existing = self.datasets.get(handle.id)
            if existing is not None:
                if camera and camera != existing.camera:
                    existing.camera = camera
                return existing
            self.datasets[handle.id] = handle
        self._load_cache(handle)
        self._persist_registry()
        return handle

    def forget(self, dataset_id: str) -> bool:
        with self._lock:
            removed = self.datasets.pop(dataset_id, None) is not None
        if removed:
            self._persist_registry()
        return removed

    def get(self, dataset_id: str) -> DatasetHandle:
        handle = self.datasets.get(dataset_id)
        if handle is None:
            raise KeyError(dataset_id)
        return handle

    # ----------------------------------------------------------- analysis
    def analyze(
        self,
        handle: DatasetHandle,
        use_video: bool = True,
        workers: int = 4,
        refit: bool = True,
        lo_pct: float = 5.0,
        hi_pct: float = 95.0,
    ) -> Job:
        """Measure every episode on a worker thread."""
        job = self.jobs.create("analyze", handle.id, total=len(handle.episode_paths))

        def work(job: Job) -> None:
            import concurrent.futures as futures
            import pandas as pd

            camera = handle.camera if use_video else None
            job.message = "measuring"

            def one(episode: int) -> tuple[int, EpisodeMeasures]:
                video = find_video(handle.root, episode, camera)
                frame = pd.read_parquet(handle.episode_paths[episode])
                sig = signals_from_dataframe(
                    frame, groups=handle.groups, episode=episode, video_path=video
                )
                return episode, measure_episode(sig, video_path=video, keep_series=True)

            measures: dict[int, EpisodeMeasures] = {}
            with futures.ThreadPoolExecutor(max_workers=max(workers, 1)) as pool:
                for episode, m in pool.map(one, sorted(handle.episode_paths)):
                    measures[episode] = m
                    job.done += 1

            handle.measures = dict(sorted(measures.items()))
            handle.analyzed_at = time.time()
            handle.analyzed_with_video = use_video
            if refit or handle.calibration is None:
                job.message = "fitting calibration"
                handle.calibration = Calibration.fit(
                    handle.measures.values(), lo_pct=lo_pct, hi_pct=hi_pct
                )
            if not handle.minimums:
                handle.minimums = {
                    f: (0.0 if f == "contact" else handle.threshold)
                    for f in handle.weights
                }
            if not handle.rules:
                # Seeded at each quantity's own percentile and left disabled:
                # a starting point to move, never a filter that fires by surprise.
                handle.rules = suggest_rules(handle.measures.values())
            job.message = "caching"
            self._save_cache(handle)
            job.message = f"{len(measures)} episodes measured"

        return self.jobs.run(job, work)

    def refit(self, handle: DatasetHandle, lo_pct: float = 5.0, hi_pct: float = 95.0) -> Calibration:
        handle.calibration = Calibration.fit(
            handle.measures.values(), lo_pct=lo_pct, hi_pct=hi_pct
        )
        self._save_cache(handle)
        return handle.calibration

    # ----------------------------------------------------------- semantic
    def run_semantic(
        self,
        handle: DatasetHandle,
        base_url: str | None = None,
        model: str | None = None,
        workers: int = 4,
        task: str | None = None,
        max_frames: int = 0,
        inline_media: bool = False,
        only: Iterable[int] | None = None,
    ) -> Job:
        """Judge task success for every episode that has a video."""
        from score_lerobot_episodes.metrics.semantic import (
            SemanticFilter, server_is_reachable, tasks_from_meta,
        )

        if handle.camera is None:
            raise ValueError("this dataset has no camera to judge")
        if not server_is_reachable(base_url):
            endpoint = base_url or os.environ.get("COSMOS_BASE_URL", "http://localhost:8000/v1")
            raise ConnectionError(
                f"no vision-language server at {endpoint} — start one with "
                f"`vllm serve nvidia/Cosmos-Reason2-8B --allowed-local-media-path {handle.root}`"
            )

        per_episode = tasks_from_meta(handle.root)
        fallback = task or handle.task
        wanted = set(only) if only is not None else set(handle.episode_paths)
        jobs: list[tuple[int, str, str]] = []
        for episode in sorted(wanted):
            video = find_video(handle.root, episode, handle.camera)
            text = per_episode.get(episode, fallback)
            if video and text:
                jobs.append((episode, video, text))
        if not jobs:
            raise ValueError("no (video, task) pairs to judge")

        job = self.jobs.create("semantic", handle.id, total=len(jobs))
        cache = self.state_dir / f"{handle.id}.semantic.jsonl"
        filt = SemanticFilter(
            base_url=base_url, model=model, cache=cache,
            max_frames=max_frames, inline_media=inline_media, structured=True,
        )

        def work(job: Job) -> None:
            def on_result(verdict) -> None:
                job.done += 1
                handle.semantic[verdict.episode] = {
                    "score": verdict.score,
                    "predicateHolds": verdict.predicate_holds,
                    "goalPredicate": verdict.goal_predicate,
                    "summary": verdict.summary,
                    "error": verdict.error,
                    "cached": verdict.cached,
                }
                job.message = f"episode {verdict.episode}: {verdict.score}"

            filt.evaluate_many(jobs, workers=max(workers, 1), on_result=on_result)
            judged = sum(1 for v in handle.semantic.values() if v.get("score") is not None)
            job.message = f"{judged} episodes judged"
            self._save_cache(handle)

        return self.jobs.run(job, work)

    # -------------------------------------------------------------- cache
    def _cache_path(self, handle: DatasetHandle) -> Path:
        return self.state_dir / f"{handle.id}.measures.pkl"

    def _save_cache(self, handle: DatasetHandle) -> None:
        payload = {
            "version": CACHE_VERSION,
            "root": str(handle.root),
            "camera": handle.camera,
            "measures": handle.measures,
            "calibration": handle.calibration.to_dict() if handle.calibration else None,
            "semantic": handle.semantic,
            "rules": [r.to_dict() for r in handle.rules],
            "analyzed_at": handle.analyzed_at,
            "analyzed_with_video": handle.analyzed_with_video,
        }
        tmp = self._cache_path(handle).with_suffix(".tmp")
        with open(tmp, "wb") as fh:
            pickle.dump(payload, fh, protocol=pickle.HIGHEST_PROTOCOL)
        tmp.replace(self._cache_path(handle))

    def _load_cache(self, handle: DatasetHandle) -> bool:
        path = self._cache_path(handle)
        if not path.exists():
            return False
        try:
            # Written by this app into its own state directory; a corrupt or
            # outdated file is discarded rather than trusted.
            with open(path, "rb") as fh:
                payload = pickle.load(fh)
            if payload.get("version") != CACHE_VERSION:
                return False
            handle.measures = payload.get("measures") or {}
            calibration = payload.get("calibration")
            handle.calibration = Calibration.from_dict(calibration) if calibration else None
            handle.semantic = payload.get("semantic") or {}
            handle.rules = [Rule.from_dict(r) for r in (payload.get("rules") or [])]
            handle.analyzed_at = payload.get("analyzed_at")
            handle.analyzed_with_video = bool(payload.get("analyzed_with_video"))
            return True
        except Exception:  # noqa: BLE001 - a bad cache must never block startup
            return False

    def _registry_path(self) -> Path:
        return self.state_dir / "datasets.json"

    def _persist_registry(self) -> None:
        entries = [
            {"path": str(h.root), "camera": h.camera} for h in self.datasets.values()
        ]
        self._registry_path().write_text(json.dumps(entries, indent=2), encoding="utf-8")

    def _restore(self) -> None:
        path = self._registry_path()
        if not path.exists():
            return
        try:
            entries = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            print(f"registry {path} is unreadable ({exc}); starting empty")
            return
        for entry in entries:
            try:
                self.register(entry["path"], entry.get("camera"))
            except Exception as exc:  # noqa: BLE001 - a moved dataset must not block startup
                # Report it: a silently empty dataset list looks like data loss.
                print(f"could not restore {entry.get('path')!r}: {type(exc).__name__}: {exc}")


# --------------------------------------------------------------------------
# Presentation helpers
# --------------------------------------------------------------------------


def clean(value):
    """JSON-safe: numpy scalars become Python, ``nan``/``inf`` become ``None``."""
    if isinstance(value, (np.floating, float)):
        value = float(value)
        return None if not np.isfinite(value) else value
    if isinstance(value, (np.integer, int)) and not isinstance(value, bool):
        return int(value)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, dict):
        return {k: clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, np.ndarray):
        return [clean(v) for v in value.tolist()]
    return value


def episode_rows(handle: DatasetHandle) -> list[dict[str, Any]]:
    """One row per episode for the overview table."""
    scores = handle.scores()
    rows = []
    for episode, m in handle.measures.items():
        score = scores[episode]
        rows.append(clean({
            "episode": episode,
            "total": score.total,
            "decision": score.decision,
            "reasons": score.reasons,
            "families": score.families,
            "flags": m.flags.raised(),
            "valid": m.flags.valid,
            "duration": m.duration_s,
            "idle": m.idle_fraction,
            "grasp": m.grasp_transitions,
            "contactEvents": m.contact_events,
            "semantic": score.semantic_score,
            "hasVideo": m.video_path is not None,
            "humanDiscarded": episode in handle.discarded,
        }))
    return rows


def overview(handle: DatasetHandle) -> dict[str, Any]:
    """Distributions and counts for the dataset page."""
    rows = episode_rows(handle)
    totals = [r["total"] for r in rows if r["total"] is not None]
    decisions: dict[str, int] = {}
    flag_counts: dict[str, int] = {}
    for row in rows:
        decisions[row["decision"]] = decisions.get(row["decision"], 0) + 1
        for flag in row["flags"]:
            flag_counts[flag] = flag_counts.get(flag, 0) + 1

    families = list(handle.weights)
    distributions = {}
    for name in families:
        values = [r["families"].get(name) for r in rows]
        values = [v for v in values if v is not None]
        if values:
            distributions[name] = clean({
                "values": values,
                "mean": float(np.mean(values)),
                "p5": float(np.percentile(values, 5)),
                "p50": float(np.percentile(values, 50)),
                "p95": float(np.percentile(values, 95)),
            })

    agreement = None
    if handle.discarded:
        discarded = set(handle.discarded) & set(handle.measures)
        kept = set(handle.measures) - discarded
        rejected = {r["episode"] for r in rows if r["decision"] == "reject"}
        flagged = {r["episode"] for r in rows if not r["valid"]}
        agreement = {
            "discarded": len(discarded),
            "kept": len(kept),
            "discardedRejected": len(discarded & rejected),
            "keptRejected": len(kept & rejected),
            "discardedFlagged": len(discarded & flagged),
        }

    return clean({
        "dataset": handle.summary(),
        "count": len(rows),
        "decisions": decisions,
        "flags": flag_counts,
        "mean": float(np.mean(totals)) if totals else None,
        "median": float(np.median(totals)) if totals else None,
        "histogram": _histogram(totals),
        "distributions": distributions,
        "weights": handle.weights,
        "threshold": handle.threshold,
        "minimums": handle.minimums,
        "mode": handle.mode,
        "aggregate": handle.aggregate,
        "rules": rule_report(handle),
        "calibration": handle.calibration.to_dict() if handle.calibration else None,
        "agreement": agreement,
    })


def _histogram(values: list[float], bins: int = 20) -> dict[str, list[float]]:
    if not values:
        return {"edges": [], "counts": []}
    counts, edges = np.histogram(values, bins=bins, range=(0.0, 1.0))
    return {"edges": [round(float(e), 4) for e in edges], "counts": [int(c) for c in counts]}


def episode_detail(handle: DatasetHandle, episode: int) -> dict[str, Any]:
    """Chart payload plus everything the episode page shows."""
    from score_lerobot_episodes.metrics.visualize import episode_payload

    m = handle.measures[episode]
    score = handle.scores()[episode]
    payload = episode_payload(m, score, video_src="video")

    # Every camera the dataset ships, not just the one the metrics scored: the
    # wrist views are usually what explains a number the base view cannot.
    videos = []
    for camera in (handle.cameras or ([handle.camera] if handle.camera else [])):
        if find_video(handle.root, episode, camera):
            videos.append({
                "key": camera,
                "label": camera.rsplit(".", 1)[-1].replace("_", " "),
                "url": f"/api/datasets/{handle.id}/episodes/{episode}/video?camera={camera}",
                "scored": camera == handle.camera,
            })
    payload["videos"] = videos
    payload["video"] = videos[0]["url"] if videos else None
    payload["task"] = handle.task
    payload["semanticDetail"] = handle.semantic.get(episode)
    payload["humanDiscarded"] = episode in handle.discarded
    payload["neighbours"] = _neighbours(sorted(handle.measures), episode)
    return clean(payload)


def _neighbours(order: list[int], episode: int) -> dict[str, int | None]:
    try:
        i = order.index(episode)
    except ValueError:
        return {"prev": None, "next": None}
    return {
        "prev": order[i - 1] if i > 0 else None,
        "next": order[i + 1] if i + 1 < len(order) else None,
    }


def rule_report(handle: DatasetHandle) -> list[dict[str, Any]]:
    """Every rule with how many episodes it currently catches.

    The count is what makes a limit settable: you move the number and see
    immediately how much of the dataset it takes out, in physical units, with no
    reference to how the rest of the batch scored.
    """
    out = []
    for rule in handle.rules:
        spec = RULE_QUANTITIES.get(rule.quantity, {})
        hits = 0
        for m in handle.measures.values():
            if m.flags.valid and rule.fires(rule_value(m, rule.quantity)):
                hits += 1
        values = np.asarray(
            [v for v in (rule_value(m, rule.quantity) for m in handle.measures.values()
                         if m.flags.valid) if np.isfinite(v)],
            dtype=float,
        )
        out.append(clean({
            "quantity": rule.quantity,
            "label": spec.get("label", rule.quantity),
            "unit": spec.get("unit", ""),
            "family": spec.get("family", ""),
            "op": rule.op,
            "limit": rule.limit,
            "action": rule.action,
            "enabled": rule.enabled,
            "hits": hits,
            "p5": float(np.percentile(values, 5)) if values.size else None,
            "p50": float(np.percentile(values, 50)) if values.size else None,
            "p95": float(np.percentile(values, 95)) if values.size else None,
            "min": float(values.min()) if values.size else None,
            "max": float(values.max()) if values.size else None,
        }))
    return out

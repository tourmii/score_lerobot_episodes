"""Semantic filter — did the episode actually accomplish the task?

The five measured families answer *how* the robot moved.  None of them can
answer *whether the task was done*: an episode can be smooth, gentle, promptly
finished and perfectly filmed while the teddy bear ends up beside the box
instead of in it.  Kinematics has no access to that; it needs a model that looks
at the recording and reads the final state.

This module wraps :class:`~score_lerobot_episodes.vlm.CosmosEvaluator` — a
Cosmos-Reason vision-language model served over a local vLLM OpenAI-compatible
endpoint — into a dataset-level gate:

* one verdict per episode: ``1.0`` goal reached, ``0.5`` attempted (object was
  grasped and moved toward the target), ``0.0`` not reached;
* verdicts are cached to JSONL keyed by (video, task), so a re-run costs
  nothing and an interrupted run resumes;
* the verdict enters the decision in :mod:`.normalize` as a hard reject at
  ``0.0`` and a review at ``0.5``, never as a weighted contribution — a
  half-done task is not compensated by a smooth trajectory.

The evaluator itself is imported lazily, so nothing here forces the ``openai``
dependency on a run that only wants the physical metrics.

Serving the model, for reference::

    vllm serve nvidia/Cosmos-Reason2-8B --allowed-local-media-path /path/to/dataset
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable

# --------------------------------------------------------------------------
# Verdict
# --------------------------------------------------------------------------


@dataclass
class SemanticVerdict:
    """One episode's task-success judgement."""

    episode: int | None
    video: str
    task: str
    score: float | None = None            # 1.0 reached / 0.5 attempted / 0.0 not
    predicate_holds: str | None = None    # "yes" / "no"
    goal_predicate: str = ""              # the task restated as a testable claim
    observed_final_state: str = ""        # what the model saw in the last frame
    note: str = ""
    error: str = ""
    cached: bool = False

    @property
    def ok(self) -> bool:
        return not self.error and self.score is not None

    @property
    def summary(self) -> str:
        if self.error:
            return f"error: {self.error}"
        return self.observed_final_state or self.goal_predicate or ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# --------------------------------------------------------------------------
# Filter
# --------------------------------------------------------------------------


class SemanticFilter:
    """Batch task-success evaluation with an on-disk cache.

    Parameters mirror the ones on :class:`CosmosEvaluator`; ``cache`` is a JSONL
    path holding one verdict per line.  A cached entry is reused whenever the
    (video, task) pair matches and the previous run did not error, so a failed
    server call is retried on the next run while a real verdict never is.
    """

    def __init__(
        self,
        base_url: str | None = None,
        model: str | None = None,
        api_key: str | None = None,
        cache: str | Path | None = None,
        inline_media: bool = False,
        max_frames: int = 0,
        structured: bool = True,
        temperature: float = 0.0,
        max_tokens: int = 10000,
        system_prompt: str | None = None,
        timeout: float = 900.0,
    ) -> None:
        self.cache_path = Path(cache) if cache else None
        self._cache: dict[tuple[str, str], dict[str, Any]] = {}
        if self.cache_path and self.cache_path.exists():
            self._cache = _read_cache(self.cache_path)

        self._evaluator = None
        self._evaluator_kwargs = {
            "base_url": base_url,
            "model": model,
            "api_key": api_key,
            "inline_media": inline_media,
            "max_frames": max_frames,
            "structured": structured,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "system_prompt": system_prompt,
            "timeout": timeout,
        }

    # ------------------------------------------------------------ backend
    @property
    def evaluator(self):
        """The underlying :class:`CosmosEvaluator`, built on first use."""
        if self._evaluator is None:
            try:
                from ..vlm import CosmosEvaluator, DEFAULT_API_KEY, DEFAULT_BASE_URL, DEFAULT_MODEL, SYSTEM_PROMPT
            except ImportError as exc:  # pragma: no cover - depends on the env
                raise RuntimeError(
                    "the semantic filter needs the OpenAI client: pip install openai"
                ) from exc
            kwargs = dict(self._evaluator_kwargs)
            kwargs["base_url"] = kwargs["base_url"] or DEFAULT_BASE_URL
            kwargs["model"] = kwargs["model"] or DEFAULT_MODEL
            kwargs["api_key"] = kwargs["api_key"] or DEFAULT_API_KEY
            kwargs["system_prompt"] = kwargs["system_prompt"] or SYSTEM_PROMPT
            self._evaluator = CosmosEvaluator(**kwargs)
        return self._evaluator

    # -------------------------------------------------------------- cache
    @staticmethod
    def _key(video: str | Path, task: str) -> tuple[str, str]:
        return (str(Path(video).resolve()), task.strip())

    def cached(self, video: str | Path, task: str) -> SemanticVerdict | None:
        entry = self._cache.get(self._key(video, task))
        if entry is None or entry.get("error"):
            return None
        known = set(SemanticVerdict.__dataclass_fields__)
        verdict = SemanticVerdict(**{k: v for k, v in entry.items() if k in known})
        verdict.cached = True
        return verdict

    def _store(self, verdict: SemanticVerdict) -> None:
        self._cache[self._key(verdict.video, verdict.task)] = verdict.to_dict()
        if self.cache_path is None:
            return
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.cache_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(verdict.to_dict(), ensure_ascii=False) + "\n")

    # ---------------------------------------------------------- evaluation
    def evaluate(self, video: str | Path, task: str, episode: int | None = None) -> SemanticVerdict:
        """Judge one episode, reusing a cached verdict when there is one."""
        hit = self.cached(video, task)
        if hit is not None:
            hit.episode = episode if episode is not None else hit.episode
            return hit

        result = self.evaluator.evaluate(video, task)
        verdict = SemanticVerdict(
            episode=episode,
            video=str(video),
            task=task,
            score=result.score,
            predicate_holds=result.predicate_holds,
            goal_predicate=result.goal_predicate or "",
            observed_final_state=result.observed_final_state or "",
            note=result.note,
            error=result.error,
        )
        self._store(verdict)
        return verdict

    def evaluate_many(
        self,
        jobs: Iterable[tuple[int | None, str | Path, str]],
        workers: int = 4,
        on_result: Callable[[SemanticVerdict], None] | None = None,
    ) -> dict[int | None, SemanticVerdict]:
        """Judge a batch of ``(episode, video, task)`` triples.

        Cached entries are resolved up front and never hit the server, so the
        worker pool only covers what is genuinely outstanding.
        """
        import concurrent.futures as futures

        jobs = list(jobs)
        out: dict[int | None, SemanticVerdict] = {}
        pending: list[tuple[int | None, str | Path, str]] = []

        for episode, video, task in jobs:
            hit = self.cached(video, task)
            if hit is not None:
                hit.episode = episode
                out[episode] = hit
                if on_result:
                    on_result(hit)
            else:
                pending.append((episode, video, task))

        if pending:
            with futures.ThreadPoolExecutor(max_workers=max(workers, 1)) as pool:
                futures_map = {
                    pool.submit(self.evaluate, video, task, episode): episode
                    for episode, video, task in pending
                }
                for future in futures.as_completed(futures_map):
                    verdict = future.result()
                    out[verdict.episode] = verdict
                    if on_result:
                        on_result(verdict)
        return out


def _read_cache(path: Path) -> dict[tuple[str, str], dict[str, Any]]:
    """Load a verdict JSONL; later lines win, so an append is an update."""
    cache: dict[tuple[str, str], dict[str, Any]] = {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            video, task = entry.get("video"), entry.get("task")
            if not video or task is None:
                continue
            cache[(str(Path(video).resolve()), str(task).strip())] = entry
    return cache


# --------------------------------------------------------------------------
# Dataset helpers
# --------------------------------------------------------------------------


def tasks_from_meta(root: str | Path) -> dict[int, str]:
    """Map episode index -> task description from a LeRobot ``meta/`` folder.

    Reads the per-episode task lists in ``episodes.jsonl``.  Datasets that only
    declare one global task leave this empty; use :func:`default_task` for those.
    """
    episodes_file = Path(root) / "meta" / "episodes.jsonl"
    if not episodes_file.exists():
        return {}

    tasks: dict[int, str] = {}
    with open(episodes_file, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            names = entry.get("tasks") or []
            if names:
                tasks[int(entry["episode_index"])] = str(names[0])
    return tasks


def default_task(root: str | Path) -> str:
    """The dataset's single task description, or an empty string."""
    tasks_file = Path(root) / "meta" / "tasks.jsonl"
    if not tasks_file.exists():
        return ""
    with open(tasks_file, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                return str(json.loads(line).get("task", ""))
    return ""


def server_is_reachable(base_url: str | None = None, timeout: float = 3.0) -> bool:
    """Cheap liveness probe, so a batch run can fail fast with a clear message."""
    import urllib.error
    import urllib.request

    base = base_url or os.environ.get("COSMOS_BASE_URL", "http://localhost:8000/v1")
    try:
        with urllib.request.urlopen(f"{base.rstrip('/')}/models", timeout=timeout) as response:
            return 200 <= response.status < 300
    except (urllib.error.URLError, OSError, ValueError):
        return False

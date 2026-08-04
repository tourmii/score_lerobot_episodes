"""
Success evaluation for robot manipulation episodes using a locally served
Cosmos-Reason2 model (vLLM OpenAI-compatible endpoint).

Usage as a batch CLI:

    # one task applied to every video in a folder
    python cosmos_eval.py --videos ./episodes \
        --task "put the teddy bear in the box" \
        --out results.jsonl

    # per-video tasks from a manifest
    python cosmos_eval.py --manifest episodes.jsonl --out results.jsonl

    # manifest format, one JSON object per line:
    #   {"video": "episodes/ep_0001.mp4", "task": "put the teddy bear in the box"}

Usage as a library:

    from cosmos_eval import CosmosEvaluator
    ev = CosmosEvaluator()
    result = ev.evaluate("episodes/ep_0001.mp4", "put the teddy bear in the box")
    print(result.score, result.predicate_holds)
"""

from __future__ import annotations

import argparse
import base64
import concurrent.futures as futures
import csv
import json
import mimetypes
import os
import re
import shutil
import statistics
import subprocess
import sys
import tempfile
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any, Iterator

from openai import OpenAI

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

DEFAULT_BASE_URL = os.environ.get("COSMOS_BASE_URL", "http://localhost:8000/v1")
DEFAULT_MODEL = os.environ.get("COSMOS_MODEL", "nvidia/Cosmos-Reason2-8B")
DEFAULT_API_KEY = os.environ.get("COSMOS_API_KEY", "EMPTY")

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}

SYSTEM_PROMPT = """\
You judge whether a robot manipulation episode reached its goal.

Judge the last frame in which the outcome is visible. Ignore how the episode got
there: a clumsy attempt that ends in the right place is a success, and a smooth
one that ends in the wrong place is a failure.

Reply with exactly one JSON object and nothing else. No preamble, no commentary,
no text after the closing brace. All four keys are required, in this order:

{"observed_final_state": "...", "goal_predicate": "...", "predicate_holds": "yes", "score": 1.0}

Field rules:
- observed_final_state: one sentence, at most 25 words, stating where the target
  object physically is in that frame. Position only. Do not narrate the sequence,
  the camera, the robot's motion, or unrelated objects.
- goal_predicate: one sentence restating the task as a testable claim, such as
  "the teddy bear is inside the box".
- predicate_holds: exactly "yes" or "no". Answer "yes" only if
  observed_final_state makes goal_predicate literally true. Answer "no" when the
  object is beside, on top of, or outside the target, when it is still held by
  the gripper, or when the outcome is not visible.
- score: exactly 0.0, 0.5, or 1.0, and never in contradiction with
  predicate_holds.
    predicate_holds "yes"                                        -> 1.0
    predicate_holds "no", object was grasped and moved toward it  -> 0.5
    predicate_holds "no", otherwise                               -> 0.0

A reply missing any of the four keys is invalid. Write all four before you stop."""


# --------------------------------------------------------------------------
# Result container
# --------------------------------------------------------------------------


@dataclass
class EvalResult:
    video: str
    task: str
    score: float | None = None
    predicate_holds: str | None = None
    goal_predicate: str | None = None
    observed_final_state: str | None = None
    reasoning: str = ""
    raw: str = ""
    note: str = ""
    error: str = ""
    prompt_tokens: int | None = None
    completion_tokens: int | None = None

    @property
    def ok(self) -> bool:
        return not self.error and self.score is not None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


VERDICT_SCHEMA = {
    "type": "object",
    "properties": {
        "observed_final_state": {"type": "string"},
        "goal_predicate": {"type": "string"},
        "predicate_holds": {"type": "string", "enum": ["yes", "no"]},
        "score": {"type": "number"},
    },
    "required": ["observed_final_state", "goal_predicate", "predicate_holds", "score"],
    "additionalProperties": False,
}

# "value"\n  "next_key"  ->  "value",\n  "next_key"
_MISSING_COMMA = re.compile(r'("|\d|\]|\}|true|false|null)(\s*\r?\n\s*)(")')
# a comma left dangling before the closing brace
_TRAILING_COMMA = re.compile(r",(\s*[}\]])")


def repair_json(blob: str) -> str:
    """
    Patch the malformations small models produce most often: commas dropped
    between fields, and a trailing comma before the close. Values are expected
    to be single-line, which is what the system prompt asks for.
    """
    blob = _MISSING_COMMA.sub(r"\1,\2\3", blob)
    blob = _TRAILING_COMMA.sub(r"\1", blob)
    return blob


def _try_parse(blob: str) -> tuple[dict[str, Any] | None, str]:
    try:
        parsed = json.loads(blob)
        return (parsed, "") if isinstance(parsed, dict) else (None, "")
    except json.JSONDecodeError:
        pass
    try:
        parsed = json.loads(repair_json(blob))
    except json.JSONDecodeError:
        return None, ""
    if isinstance(parsed, dict):
        return parsed, "model emitted malformed JSON; repaired client-side"
    return None, ""


def extract_last_json(text: str) -> tuple[dict[str, Any] | None, str]:
    """
    Return (payload, note) for the last JSON object in `text`.

    Falls back to a repair pass when strict parsing fails, and reports via the
    note whether that fallback was needed.
    """
    blocks: list[str] = []
    stack: list[int] = []
    for i, ch in enumerate(text):
        if ch == "{":
            stack.append(i)
        elif ch == "}" and stack:
            start = stack.pop()
            if not stack:
                blocks.append(text[start : i + 1])
    for blob in reversed(blocks):
        payload, note = _try_parse(blob)
        if payload is not None:
            return payload, note

    # Fallback: an unmatched "{" earlier in the prose left the stack dirty, so
    # the real object never closed at depth 0. Try every "{" before the final "}".
    end = text.rfind("}")
    if end == -1:
        return None, ""
    for start in reversed([i for i, ch in enumerate(text[:end]) if ch == "{"]):
        payload, note = _try_parse(text[start : end + 1])
        if payload is not None:
            return payload, note
    return None, ""


def reconcile_score(payload: dict[str, Any]) -> tuple[float | None, str]:
    """
    The system prompt makes the score a mechanical function of predicate_holds.
    Small models still contradict themselves, so enforce the rule client-side
    and record whenever a correction was applied.
    """
    holds = str(payload.get("predicate_holds", "")).strip().lower()
    raw_score = payload.get("score")

    try:
        score = float(raw_score)
    except (TypeError, ValueError):
        score = None

    if holds == "yes":
        if score != 1.0:
            return 1.0, f"score corrected {raw_score!r} -> 1.0 (predicate_holds=yes)"
        return 1.0, ""

    if holds == "no":
        if score is None:
            return 0.0, "score missing, defaulted to 0.0 (predicate_holds=no)"
        if score >= 1.0:
            return 0.5, f"score corrected {raw_score!r} -> 0.5 (predicate_holds=no)"
        if score not in (0.0, 0.5):
            snapped = 0.5 if score >= 0.25 else 0.0
            return snapped, f"score snapped {raw_score!r} -> {snapped}"
        return score, ""

    return score, f"predicate_holds unreadable: {payload.get('predicate_holds')!r}"


def message_text(message) -> tuple[str, str]:
    """
    Return (content, reasoning) for a chat message.

    The OpenAI SDK types only the fields it knows, and reasoning_content is a
    vLLM extension, so read the dumped dict and check the spellings different
    reasoning parsers emit.
    """
    if hasattr(message, "model_dump"):
        data = message.model_dump()
    elif isinstance(message, dict):
        data = message
    else:
        data = dict(getattr(message, "__dict__", {}))

    content = data.get("content") or ""
    for key in ("reasoning_content", "reasoning", "thinking"):
        reasoning = data.get(key)
        if reasoning:
            return content, str(reasoning)
    return content, ""


def subsample_video(src: Path, max_frames: int, workdir: Path) -> Path:
    """
    Uniformly reduce `src` to at most `max_frames` frames via ffmpeg.
    Used when the server was launched with num_frames=-1 and the clip would
    otherwise overflow the context window.
    """
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg not found on PATH; drop --max-frames or install ffmpeg")

    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-count_packets", "-show_entries", "stream=nb_read_packets",
         "-of", "csv=p=0", str(src)],
        capture_output=True, text=True, check=True,
    )
    total = int(probe.stdout.strip() or 0)
    if total == 0 or total <= max_frames:
        return src

    step = total / max_frames
    dst = workdir / f"{src.stem}__{max_frames}f.mp4"
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", str(src),
         "-vf", f"select='not(mod(n\\,{max(1, round(step))}))'",
         "-vsync", "vfr", "-an", str(dst)],
        check=True,
    )
    return dst


def video_content_part(path: Path, inline: bool) -> dict[str, Any]:
    """
    Build the `video_url` content block.

    inline=False -> file:// URL. Fast, zero-copy, but the path must live under
                    the server's --allowed-local-media-path.
    inline=True  -> base64 data URI. Works from anywhere (uploads, temp dirs)
                    at the cost of a larger request body.
    """
    if inline:
        mime = mimetypes.guess_type(path.name)[0] or "video/mp4"
        data = base64.b64encode(path.read_bytes()).decode("ascii")
        url = f"data:{mime};base64,{data}"
    else:
        url = path.resolve().as_uri()
    return {"type": "video_url", "video_url": {"url": url}}


def discover_videos(root: Path) -> list[Path]:
    if root.is_file():
        return [root]
    return sorted(p for p in root.rglob("*") if p.suffix.lower() in VIDEO_EXTS)


# --------------------------------------------------------------------------
# Evaluator
# --------------------------------------------------------------------------


class CosmosEvaluator:
    def __init__(
        self,
        base_url: str = DEFAULT_BASE_URL,
        model: str = DEFAULT_MODEL,
        api_key: str = DEFAULT_API_KEY,
        system_prompt: str = SYSTEM_PROMPT,
        temperature: float = 0.0,
        top_p: float = 1.0,
        max_tokens: int = 4096,
        inline_media: bool = False,
        max_frames: int = 0,
        structured: bool = False,
        timeout: float = 900.0,
    ) -> None:
        self.client = OpenAI(base_url=base_url, api_key=api_key, timeout=timeout)
        self.model = model
        self.system_prompt = system_prompt
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.inline_media = inline_media
        self.max_frames = max_frames
        self.structured = structured
        self._variant_index = 0
        self.structured_variant = ""

    # Ordered newest-first. vLLM renamed this API twice: guided_json was the
    # original, structured_outputs replaced it around 0.11, and response_format
    # is the OpenAI-standard spelling. An unrecognised extra_body key is dropped
    # silently rather than rejected, so an outdated spelling looks exactly like
    # structured output being off.
    STRUCTURED_VARIANTS = [
        ("response_format", lambda schema: {
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": "verdict", "schema": schema},
            }
        }),
        ("structured_outputs", lambda schema: {
            "extra_body": {"structured_outputs": {"json": schema}}
        }),
        ("guided_json", lambda schema: {
            "extra_body": {"guided_json": schema}
        }),
    ]

    def _structured_kwargs(self) -> dict[str, Any]:
        name, builder = self.STRUCTURED_VARIANTS[self._variant_index]
        self.structured_variant = name
        return builder(VERDICT_SCHEMA)

    def _create(self, messages: list[dict[str, Any]], kwargs: dict[str, Any]):
        """Send the request, stepping to an older structured API if rejected."""
        while True:
            try:
                return self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    max_tokens=self.max_tokens,
                    **kwargs,
                )
            except Exception as exc:  # noqa: BLE001
                retryable = self.structured and self._variant_index + 1 < len(self.STRUCTURED_VARIANTS)
                if not retryable or "400" not in str(exc):
                    raise
                self._variant_index += 1
                kwargs = self._structured_kwargs()

    def evaluate(self, video: str | Path, task: str) -> EvalResult:
        video = Path(video)
        result = EvalResult(video=str(video), task=task)

        if not video.exists():
            result.error = f"file not found: {video}"
            return result

        tmpdir: tempfile.TemporaryDirectory | None = None
        try:
            send_path = video
            if self.max_frames > 0:
                tmpdir = tempfile.TemporaryDirectory(prefix="cosmos_eval_")
                send_path = subsample_video(video, self.max_frames, Path(tmpdir.name))
                # a temp path is never under --allowed-local-media-path
                inline = True if send_path != video else self.inline_media
            else:
                inline = self.inline_media

            messages = [
                {"role": "system", "content": self.system_prompt},
                {
                    "role": "user",
                    "content": [
                        video_content_part(send_path, inline),
                        {"type": "text", "text": f"Task: {task}"},
                    ],
                },
            ]

            kwargs: dict[str, Any] = {}
            if self.structured:
                kwargs = self._structured_kwargs()

            response = self._create(messages, kwargs)
        except Exception as exc:  # noqa: BLE001 - surface the server message verbatim
            result.error = f"{type(exc).__name__}: {exc}"
            return result
        finally:
            if tmpdir is not None:
                tmpdir.cleanup()

        message = response.choices[0].message
        result.raw, result.reasoning = message_text(message)

        if response.usage:
            result.prompt_tokens = response.usage.prompt_tokens
            result.completion_tokens = response.usage.completion_tokens

        if response.choices[0].finish_reason == "length":
            result.note = "output truncated at max_tokens; raise --max-tokens"

        payload, parse_note = extract_last_json(result.raw)
        if payload is None:
            # reasoning parsers occasionally leave the JSON in the thinking trace
            payload, parse_note = extract_last_json(result.reasoning)
        if payload is None:
            hints = []
            if response.choices[0].finish_reason == "length":
                hints.append(
                    f"generation stopped at max_tokens={self.max_tokens} before the "
                    "JSON was written"
                )
            if not result.raw and result.reasoning:
                hints.append("the model never closed its reasoning block")
            if not result.raw and not result.reasoning:
                try:
                    keys = [k for k, v in message.model_dump().items() if v]
                except Exception:  # noqa: BLE001
                    keys = []
                hints.append(
                    "content and reasoning are both empty; non-empty message keys: "
                    + (", ".join(keys) if keys else "none")
                )
            if result.prompt_tokens:
                hints.append(
                    f"{result.prompt_tokens} prompt tokens consumed by the video "
                    f"(+{result.completion_tokens} generated)"
                )
            result.error = "no JSON object in model output"
            if hints:
                result.error += " — " + "; ".join(hints)
            return result

        missing = [k for k in ("goal_predicate", "predicate_holds", "score") if k not in payload]
        if missing:
            result.raw = result.raw or json.dumps(payload, ensure_ascii=False)
            result.error = (
                "model stopped early — JSON is missing " + ", ".join(missing)
                + ". Try --structured, which forces every key."
            )
            result.observed_final_state = payload.get("observed_final_state")
            return result

        result.observed_final_state = payload.get("observed_final_state")
        result.goal_predicate = payload.get("goal_predicate")
        result.predicate_holds = str(payload.get("predicate_holds", "")).strip().lower() or None

        score, note = reconcile_score(payload)
        result.score = score
        for extra in (parse_note, note):
            if extra:
                result.note = f"{result.note}; {extra}".strip("; ")
        return result

    def evaluate_many(
        self,
        jobs: list[tuple[Path, str]],
        workers: int = 4,
    ) -> Iterator[EvalResult]:
        """Yield results as they complete. Order is not preserved."""
        with futures.ThreadPoolExecutor(max_workers=workers) as pool:
            pending = {pool.submit(self.evaluate, v, t): v for v, t in jobs}
            for fut in futures.as_completed(pending):
                yield fut.result()


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def load_jobs(args: argparse.Namespace) -> list[tuple[Path, str]]:
    if args.manifest:
        jobs = []
        with open(args.manifest, encoding="utf-8") as fh:
            for lineno, line in enumerate(fh, 1):
                line = line.strip()
                if not line:
                    continue
                entry = json.loads(line)
                video = entry.get("video") or entry.get("path")
                task = entry.get("task") or args.task
                if not video or not task:
                    raise SystemExit(f"{args.manifest}:{lineno} needs both 'video' and 'task'")
                jobs.append((Path(video), task))
        return jobs

    if not args.task:
        raise SystemExit("--task is required when using --videos")
    return [(v, args.task) for v in discover_videos(Path(args.videos))]


def summarize(results: list[EvalResult]) -> str:
    scored = [r for r in results if r.ok]
    failed = [r for r in results if not r.ok]
    lines = [
        "",
        f"evaluated : {len(scored)}/{len(results)}",
    ]
    if scored:
        scores = [r.score for r in scored]
        buckets = {1.0: 0, 0.5: 0, 0.0: 0}
        for s in scores:
            buckets[s] = buckets.get(s, 0) + 1
        lines += [
            f"mean score: {statistics.mean(scores):.3f}",
            f"success   : {buckets.get(1.0, 0)}  partial: {buckets.get(0.5, 0)}  fail: {buckets.get(0.0, 0)}",
        ]
    corrected = [r for r in scored if "corrected" in r.note or "snapped" in r.note]
    if corrected:
        lines.append(f"score corrections applied: {len(corrected)}")
    if failed:
        lines.append(f"errors    : {len(failed)}")
        for r in failed[:5]:
            lines.append(f"  {Path(r.video).name}: {r.error}")
        if len(failed) > 5:
            lines.append(f"  ... and {len(failed) - 5} more")
    return "\n".join(lines)


def write_csv(results: list[EvalResult], path: Path) -> None:
    cols = ["video", "task", "score", "predicate_holds", "goal_predicate",
            "observed_final_state", "note", "error"]
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        writer.writeheader()
        for r in results:
            writer.writerow(r.to_dict())


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--videos", help="video file or directory to scan recursively")
    src.add_argument("--manifest", help="JSONL with one {'video','task'} object per line")

    p.add_argument("--task", help="task description applied to every video")
    p.add_argument("--out", default="results.jsonl", help="JSONL output path")
    p.add_argument("--csv", help="also write a flat CSV here")
    p.add_argument("--workers", type=int, default=10, help="concurrent requests")

    p.add_argument("--base-url", default=DEFAULT_BASE_URL)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--top-p", type=float, default=1.0)
    p.add_argument("--max-tokens", type=int, default=10000)
    p.add_argument("--inline-media", action="store_true",
                   help="send video as base64 instead of file:// (use when videos live "
                        "outside the server's --allowed-local-media-path)")
    p.add_argument("--structured", action="store_true",
                   help="constrain decoding to the verdict schema via vLLM guided_json")
    p.add_argument("--max-frames", type=int, default=0,
                   help="subsample to N frames with ffmpeg before sending (0 = off)")
    p.add_argument("--keep-reasoning", action="store_true",
                   help="store the full reasoning trace in the JSONL")

    args = p.parse_args()
    jobs = load_jobs(args)
    if not jobs:
        raise SystemExit("no videos found")

    evaluator = CosmosEvaluator(
        base_url=args.base_url,
        model=args.model,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        inline_media=args.inline_media,
        max_frames=args.max_frames,
        structured=args.structured,
    )

    print(f"evaluating {len(jobs)} videos against {args.base_url} ({args.workers} workers)")
    results: list[EvalResult] = []
    out_path = Path(args.out)

    with open(out_path, "w", encoding="utf-8") as fh:
        for i, result in enumerate(evaluator.evaluate_many(jobs, workers=args.workers), 1):
            results.append(result)
            record = result.to_dict()
            if not args.keep_reasoning:
                record.pop("reasoning", None)
                record.pop("raw", None)
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
            fh.flush()

            name = Path(result.video).name
            if result.ok:
                flag = " *" if result.note else ""
                print(f"[{i}/{len(jobs)}] {result.score:>4.1f}  {name}{flag}")
            else:
                print(f"[{i}/{len(jobs)}]  ERR  {name}: {result.error}", file=sys.stderr)

    if args.csv:
        write_csv(results, Path(args.csv))

    print(summarize(results))
    print(f"\nwrote {out_path}" + (f" and {args.csv}" if args.csv else ""))


if __name__ == "__main__":
    main()
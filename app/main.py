"""HTTP layer for the episode-quality app.
The API::
    GET    /api/health
    GET    /api/datasets                          registered datasets
    POST   /api/datasets                          {path, camera?} -> register
    DELETE /api/datasets/{id}                     forget (files untouched)
    GET    /api/datasets/{id}/overview            distributions, counts, calibration
    POST   /api/datasets/{id}/analyze             {useVideo, workers} -> job
    POST   /api/datasets/{id}/calibrate           {loPct, hiPct} -> refit ranges
    PUT    /api/datasets/{id}/policy              {mode, weights, threshold, minimums}
    POST   /api/datasets/{id}/profile/fit         {goodSigma, badSigma, taskDuration}
    POST   /api/datasets/{id}/profile             {profile} -> adopt frozen anchors
    GET    /api/datasets/{id}/profile.json        the anchors, to reuse elsewhere
    GET    /api/datasets/{id}/drift               this batch against those anchors
    POST   /api/datasets/{id}/semantic            {baseUrl, model, apiKey?, workers} -> job
    GET    /api/datasets/{id}/episodes            table rows
    GET    /api/datasets/{id}/episodes/{ep}       chart payload + detail
    GET    /api/datasets/{id}/episodes/{ep}/video range-streamed mp4
    GET    /api/datasets/{id}/export.csv          measurements + scores
    GET    /api/datasets/{id}/export.json         the accepted episode list
    GET    /api/jobs/{job_id}                     progress
"""

from __future__ import annotations

import io
import os
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from score_lerobot_episodes.metrics.visualize import ASSETS

from .service import DatasetHandle, QualityService, episode_detail, episode_rows, overview

STATIC = Path(__file__).resolve().parent / "static"
STATE_DIR = os.environ.get("QUALITY_STATE_DIR", ".quality_app")

app = FastAPI(title="LeRobot episode quality", version="2.0")
service = QualityService(STATE_DIR)

for _path in filter(None, os.environ.get("QUALITY_DATASETS", "").split(os.pathsep)):
    try:
        service.register(_path)
    except (FileNotFoundError, ValueError) as _exc:
        print(f"could not register {_path}: {_exc}")


# --------------------------------------------------------------------------
# Request bodies
# --------------------------------------------------------------------------


class RegisterBody(BaseModel):
    path: str
    camera: str | None = None


class AnalyzeBody(BaseModel):
    useVideo: bool = True
    workers: int = Field(default=4, ge=1, le=32)
    refit: bool = True
    loPct: float = Field(default=5.0, ge=0.0, le=49.0)
    hiPct: float = Field(default=95.0, ge=51.0, le=100.0)


class CalibrateBody(BaseModel):
    loPct: float = Field(default=5.0, ge=0.0, le=49.0)
    hiPct: float = Field(default=95.0, ge=51.0, le=100.0)


class PolicyBody(BaseModel):
    weights: dict[str, float] | None = None
    threshold: float | None = Field(default=None, ge=0.0, le=1.0)
    minimums: dict[str, float] | None = None
    mode: str | None = None
    aggregate: str | None = None
    #: Physical limits, one per quantity: {quantity, limit, op, action, enabled}
    rules: list[dict[str, Any]] | None = None


class ProfileFitBody(BaseModel):
    goodSigma: float = Field(default=1.0, gt=0.0, le=10.0)
    badSigma: float = Field(default=4.0, gt=0.0, le=20.0)
    taskDuration: float | None = Field(default=None, gt=0.0)


class ProfileBody(BaseModel):
    #: A whole ``QualityProfile.to_dict()``, as written by ``profile.json``.
    profile: dict[str, Any]
    source: str | None = None


class SemanticBody(BaseModel):
    baseUrl: str | None = None
    model: str | None = None
    #: For a hosted endpoint.  Absent or blank means "not set": the request is
    #: made without one, which is what a local vLLM server expects.  It is never
    #: stored on the dataset or returned by any route.
    apiKey: str | None = None
    workers: int = Field(default=4, ge=1, le=32)
    task: str | None = None
    maxFrames: int = Field(default=0, ge=0)
    inlineMedia: bool = False
    only: list[int] | None = None


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


def _dataset(dataset_id: str) -> DatasetHandle:
    try:
        return service.get(dataset_id)
    except KeyError:
        raise HTTPException(404, f"no such dataset: {dataset_id}") from None


def _analyzed(dataset_id: str) -> DatasetHandle:
    handle = _dataset(dataset_id)
    if not handle.analyzed:
        raise HTTPException(409, "dataset has not been analysed yet")
    return handle


# --------------------------------------------------------------------------
# API
# --------------------------------------------------------------------------


@app.get("/api/health")
def health() -> dict[str, Any]:
    return {"ok": True, "datasets": len(service.datasets), "stateDir": str(service.state_dir)}


@app.get("/api/datasets")
def list_datasets() -> list[dict[str, Any]]:
    return [h.summary() for h in service.datasets.values()]


@app.post("/api/datasets", status_code=201)
def register_dataset(body: RegisterBody) -> dict[str, Any]:
    try:
        handle = service.register(body.path, body.camera)
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc)) from None
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from None
    return handle.summary()


@app.delete("/api/datasets/{dataset_id}")
def forget_dataset(dataset_id: str) -> dict[str, bool]:
    if not service.forget(dataset_id):
        raise HTTPException(404, f"no such dataset: {dataset_id}")
    return {"ok": True}


@app.get("/api/datasets/{dataset_id}")
def get_dataset(dataset_id: str) -> dict[str, Any]:
    return _dataset(dataset_id).summary()


@app.get("/api/datasets/{dataset_id}/cameras")
def dataset_cameras(dataset_id: str) -> list[str]:
    from .service import list_cameras

    return list_cameras(_dataset(dataset_id).root)


@app.post("/api/datasets/{dataset_id}/analyze")
def analyze(dataset_id: str, body: AnalyzeBody) -> dict[str, Any]:
    handle = _dataset(dataset_id)
    job = service.analyze(
        handle, use_video=body.useVideo, workers=body.workers,
        refit=body.refit, lo_pct=body.loPct, hi_pct=body.hiPct,
    )
    return job.to_dict()


@app.post("/api/datasets/{dataset_id}/rules/seed")
def seed_rules(dataset_id: str) -> dict[str, Any]:
    """Re-seed every limit from this dataset's own distribution, all disabled."""
    from score_lerobot_episodes.metrics import suggest_rules

    handle = _analyzed(dataset_id)
    handle.rules = suggest_rules(handle.measures.values())
    return overview(handle)


@app.post("/api/datasets/{dataset_id}/calibrate")
def calibrate(dataset_id: str, body: CalibrateBody) -> dict[str, Any]:
    handle = _analyzed(dataset_id)
    return service.refit(handle, body.loPct, body.hiPct).to_dict()


@app.put("/api/datasets/{dataset_id}/policy")
def set_policy(dataset_id: str, body: PolicyBody) -> dict[str, Any]:
    handle = _analyzed(dataset_id)
    if body.weights is not None:
        unknown = set(body.weights) - set(handle.weights)
        if unknown:
            raise HTTPException(400, f"unknown metric families: {sorted(unknown)}")
        handle.weights.update({k: float(v) for k, v in body.weights.items()})
    if body.mode is not None:
        if body.mode not in ("absolute", "rules", "gate", "weighted"):
            raise HTTPException(
                400, "mode must be 'absolute', 'rules', 'gate' or 'weighted'")
        if body.mode == "absolute" and handle.profile is None:
            raise HTTPException(409, "no profile on this dataset; fit or upload one first")
        handle.mode = body.mode
    if body.threshold is not None:
        # The two thresholds are different quantities — a rank inside this batch
        # and a probability that no criterion is violated — so the slider writes
        # to whichever the active mode reads.  Sharing one field would silently
        # move the bar every time the mode changed.
        if handle.mode == "absolute" and handle.profile is not None:
            handle.profile.threshold = float(body.threshold)
        else:
            handle.threshold = float(body.threshold)
    if body.minimums is not None:
        handle.minimums = {k: float(v) for k, v in body.minimums.items()}
    if body.aggregate is not None:
        if body.aggregate not in ("geometric", "arithmetic"):
            raise HTTPException(400, "aggregate must be 'geometric' or 'arithmetic'")
        handle.aggregate = body.aggregate
    if body.rules is not None:
        from score_lerobot_episodes.metrics import RULE_QUANTITIES, Rule

        rules = []
        for entry in body.rules:
            if entry.get("quantity") not in RULE_QUANTITIES:
                raise HTTPException(400, f"unknown quantity: {entry.get('quantity')!r}")
            if entry.get("op") not in (">", "<"):
                raise HTTPException(400, "op must be '>' or '<'")
            if entry.get("action") not in ("reject", "review"):
                raise HTTPException(400, "action must be 'reject' or 'review'")
            rules.append(Rule.from_dict(entry))
        handle.rules = rules
    return overview(handle)


@app.post("/api/datasets/{dataset_id}/profile/fit")
def fit_profile(dataset_id: str, body: ProfileFitBody) -> dict[str, Any]:
    handle = _analyzed(dataset_id)
    service.fit_profile(handle, body.goodSigma, body.badSigma, body.taskDuration)
    return overview(handle)


@app.post("/api/datasets/{dataset_id}/profile")
def adopt_profile(dataset_id: str, body: ProfileBody) -> dict[str, Any]:
    handle = _analyzed(dataset_id)
    try:
        service.load_profile(handle, body.profile, body.source or "uploaded")
    except (TypeError, ValueError, KeyError) as exc:
        raise HTTPException(400, f"not a usable profile: {exc}") from None
    return overview(handle)


@app.get("/api/datasets/{dataset_id}/profile.json")
def download_profile(dataset_id: str) -> JSONResponse:
    handle = _dataset(dataset_id)
    if handle.profile is None:
        raise HTTPException(404, "this dataset has no profile yet")
    return JSONResponse(
        handle.profile.to_dict(),
        headers={"Content-Disposition":
                 f'attachment; filename="{handle.name}.profile.json"'},
    )


@app.get("/api/datasets/{dataset_id}/drift")
def dataset_drift(dataset_id: str) -> list[dict[str, Any]]:
    handle = _analyzed(dataset_id)
    if handle.profile is None:
        raise HTTPException(409, "this dataset has no profile yet")
    return service.drift(handle)


@app.post("/api/datasets/{dataset_id}/semantic")
def semantic(dataset_id: str, body: SemanticBody) -> dict[str, Any]:
    handle = _dataset(dataset_id)
    try:
        job = service.run_semantic(
            handle, base_url=body.baseUrl, model=body.model,
            api_key=(body.apiKey or "").strip() or None, workers=body.workers,
            task=body.task, max_frames=body.maxFrames,
            inline_media=body.inlineMedia, only=body.only,
        )
    except ConnectionError as exc:
        raise HTTPException(503, str(exc)) from None
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from None
    except RuntimeError as exc:  # openai missing
        raise HTTPException(501, str(exc)) from None
    return job.to_dict()


@app.get("/api/datasets/{dataset_id}/overview")
def dataset_overview(dataset_id: str) -> dict[str, Any]:
    return overview(_analyzed(dataset_id))


@app.get("/api/datasets/{dataset_id}/episodes")
def dataset_episodes(dataset_id: str) -> list[dict[str, Any]]:
    return episode_rows(_analyzed(dataset_id))


@app.get("/api/datasets/{dataset_id}/episodes/{episode}")
def dataset_episode(dataset_id: str, episode: int) -> dict[str, Any]:
    handle = _analyzed(dataset_id)
    if episode not in handle.measures:
        raise HTTPException(404, f"episode {episode} was not measured")
    return episode_detail(handle, episode)


@app.get("/api/datasets/{dataset_id}/episodes/{episode}/video")
def dataset_episode_video(dataset_id: str, episode: int, camera: str | None = Query(None)):
    from .service import find_video

    handle = _dataset(dataset_id)
    path = find_video(handle.root, episode, camera) if camera else handle.video_path(episode)
    if path is None or not Path(path).exists():
        raise HTTPException(404, f"no video for episode {episode}")
    # FileResponse honours Range, which is what makes the player seekable.
    return FileResponse(path, media_type="video/mp4")


@app.get("/api/datasets/{dataset_id}/export.csv")
def export_csv(dataset_id: str) -> StreamingResponse:
    """Raw measurements joined with the current scores, one row per episode."""
    import pandas as pd

    handle = _analyzed(dataset_id)
    scores = handle.scores()
    measures = pd.DataFrame(
        [m.to_row() for m in handle.measures.values()], index=list(handle.measures)
    ).drop(columns=["episode"], errors="ignore")   # already the index
    measures.index.name = "episode"
    scored = pd.DataFrame([s.to_row() for s in scores.values()]).set_index("episode")
    joined = measures.join(scored, rsuffix="_score")

    buffer = io.StringIO()
    joined.to_csv(buffer)
    buffer.seek(0)
    return StreamingResponse(
        iter([buffer.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{handle.name}_quality.csv"'},
    )


@app.get("/api/datasets/{dataset_id}/export.json")
def export_json(dataset_id: str, decision: str = Query("accept")) -> JSONResponse:
    """The episode list for one decision, ready to drive a dataset filter."""
    handle = _analyzed(dataset_id)
    wanted = {d.strip() for d in decision.split(",") if d.strip()}
    scores = handle.scores()
    episodes = sorted(ep for ep, s in scores.items() if s.decision in wanted)
    return JSONResponse({
        "dataset": handle.name,
        "path": str(handle.root),
        "decision": sorted(wanted),
        "count": len(episodes),
        "episodes": episodes,
        "mode": handle.mode,
        # The export has to carry the bar that produced it, or the accepted list
        # cannot be reproduced later.  In absolute mode that is the whole profile:
        # the threshold alone means nothing without the anchors it was applied to.
        **({"profile": handle.profile.to_dict(),
            "profileSource": handle.profile_source or "fitted on this dataset",
            "threshold": handle.profile.threshold}
           if handle.mode == "absolute" and handle.profile
           else {"weights": handle.weights, "threshold": handle.threshold}),
    })


@app.get("/api/jobs/{job_id}")
def job_status(job_id: str) -> dict[str, Any]:
    job = service.jobs.get(job_id)
    if job is None:
        raise HTTPException(404, f"no such job: {job_id}")
    return job.to_dict()


# --------------------------------------------------------------------------
# Frontend
# --------------------------------------------------------------------------

# The chart and the shared theme live with the metrics package so the app and
# the standalone pages `render_episode_html` writes cannot drift apart.
app.mount("/assets", StaticFiles(directory=str(ASSETS)), name="assets")
app.mount("/static", StaticFiles(directory=str(STATIC)), name="static")


#: Files whose URL gets a cache-busting token stamped into the shell.
_VERSIONED = ("/assets/theme.css", "/assets/chart.js", "/static/app.css", "/static/app.js")


def _asset_token() -> str:
    """Short token that changes whenever a served asset changes.

    Without it a browser keeps a cached ``chart.js`` after an upgrade and the
    page silently runs against the old one — which looks like a bug in the new
    code rather than a stale file.
    """
    stamps = []
    for url in _VERSIONED:
        root = ASSETS if url.startswith("/assets/") else STATIC
        path = root / url.rsplit("/", 1)[1]
        stamps.append(str(int(path.stat().st_mtime)) if path.exists() else "0")
    import hashlib

    return hashlib.sha1("|".join(stamps).encode()).hexdigest()[:8]


@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    shell = (STATIC / "index.html").read_text(encoding="utf-8")
    token = _asset_token()
    for url in _VERSIONED:
        shell = shell.replace(f'"{url}"', f'"{url}?v={token}"')
    return HTMLResponse(shell, headers={"Cache-Control": "no-cache"})

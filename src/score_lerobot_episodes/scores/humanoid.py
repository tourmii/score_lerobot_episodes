"""Backwards-compatible facade over :mod:`score_lerobot_episodes.metrics`.

The metric suite was split into a measurement layer (physical units and flags,
no thresholds) and a scoring layer (normalisation, weights, decision) — see
:mod:`score_lerobot_episodes.metrics`.  This module keeps the old single-tier
entry points alive for callers written against them: ``ui.py``,
``scripts/score_humanoid_dataset.py`` and anything using the ``DatasetScorer``
adapters.

New code should use the layered API directly::

    from score_lerobot_episodes.metrics import (
        signals_from_dataframe, measure_episode, Calibration, score_episode)

The one behavioural difference to be aware of: the layered API fits its
normalisation ranges to *your* dataset and excludes invalid episodes from that
fit, whereas the constants below are the ranges fitted once on
``pickup_20260628_150622``.
"""

from __future__ import annotations

import numpy as np

from ..metrics import measure as _measure
from ..metrics import normalize as _normalize
from ..metrics.measure import (  # noqa: F401 - re-exported for compatibility
    MIN_EEF_PATH_M,
    EpisodeMeasures,
    log_dimensionless_jerk,
    measure_episode,
)
from ..metrics.measure import count_events as _count_events  # noqa: F401
from ..metrics.measure import robust_spikes as _robust_spikes  # noqa: F401
from ..metrics.normalize import ramp as _ramp  # noqa: F401
from ..metrics.signals import (  # noqa: F401 - re-exported for compatibility
    EEF_SLICES,
    JOINT_GROUPS,
    EpisodeSignals,
    signals_from_dataframe,
    signals_from_states,
    states_from_dataframe,
)

__all__ = [
    "CALIB", "JOINT_GROUPS", "ARM_GROUPS", "BODY_GROUPS", "EEF_SLICES",
    "MIN_EEF_PATH_M", "COLLISION_RATE_REF", "HAND_CLOSED_COUNT",
    "NOMINAL_GRASP_TRANSITIONS", "EpisodeSignals", "signals_from_dataframe",
    "signals_from_states", "states_from_dataframe", "log_dimensionless_jerk",
    "smoothness", "acceleration", "collision", "runtime", "visual_clarity",
    "score_episode", "build_time_stats", "calibrate",
    "score_smoothness", "score_acceleration", "score_collision",
    "score_runtime", "score_visual_clarity",
]

# Coarse chains, kept as module constants because callers import them.
BODY_GROUPS = ("left_leg", "right_leg", "waist")
ARM_GROUPS = ("left_arm", "right_arm")

COLLISION_RATE_REF = _normalize.Calibration().contact_rate_ref
HAND_CLOSED_COUNT = _measure.HAND_CLOSED_COUNT
NOMINAL_GRASP_TRANSITIONS = _normalize.Calibration().grasp_nominal

#: ``{key: {"lo": ..., "hi": ...}}`` in the legacy spelling.  Mutating this
#: dict still changes how the functions below score, as it always did.
CALIB: dict[str, dict[str, float]] = {
    key: {"lo": lo, "hi": hi}
    for key, (lo, hi) in _normalize.DEFAULT_BRACKETS.items()
    if not key.startswith("video_")
}
CALIB["sharpness"] = {"lo": 58.0, "hi": 110.0}


def _calibration(time_stats: dict | None = None) -> _normalize.Calibration:
    """Build a :class:`Calibration` reflecting the current :data:`CALIB`."""
    calib = _normalize.Calibration(
        brackets={key: (bracket["lo"], bracket["hi"]) for key, bracket in CALIB.items()},
        grasp_nominal=NOMINAL_GRASP_TRANSITIONS,
        contact_rate_ref=COLLISION_RATE_REF,
        # The legacy idle term charged from the first idle frame rather than
        # granting the slow phases an allowance.
        idle_allowance=0.0,
        sharpness_fallback=(CALIB["sharpness"]["lo"], CALIB["sharpness"]["hi"]),
    )
    if time_stats:
        median = time_stats.get("median")
        mad = time_stats.get("mad")
        if median is None or not mad:
            median, mad = time_stats.get("mean"), time_stats.get("std")
        calib.duration_median = float(median) if median is not None else float("nan")
        calib.duration_mad = float(mad) if mad else float("nan")
    return calib


def _measured(sig: EpisodeSignals, video_path: str | None = None) -> EpisodeMeasures:
    return measure_episode(sig, video_path=video_path)


# --------------------------------------------------------------------------
# The five legacy metrics
# --------------------------------------------------------------------------


def smoothness(sig: EpisodeSignals) -> dict:
    """Trajectory smoothness from log dimensionless jerk (legacy dict shape)."""
    m = _measured(sig)
    detail = {
        "ldlj_arm": m.ldlj_arm,
        "ldlj_body": m.ldlj_body,
        "ldlj_wrist": m.ldlj_wrist,
        "working_side": m.working_side,
    }
    if m.flags.degenerate:
        return {"score": 0.0, "degenerate": True, **detail}

    terms = _normalize.score_smoothness(m, _calibration())
    score = terms["score"]
    return {
        "score": 0.0 if not np.isfinite(score) else float(score),
        "smoothness_arm": terms["arm"],
        "smoothness_body": terms["body"],
        "smoothness_wrist": terms["wrist"],
        **detail,
    }


def acceleration(sig: EpisodeSignals) -> dict:
    """Per-segment acceleration, split by kinematic group (legacy dict shape)."""
    m = _measured(sig)
    out: dict = {}
    for label in ("arm", "body", "wrist"):
        rms = getattr(m, f"acc_{label}_rms")
        p99 = getattr(m, f"acc_{label}_p99")
        if np.isfinite(rms):
            out[f"acc_{label}_rms"] = rms
        if np.isfinite(p99):
            out[f"acc_{label}_p99"] = p99
    for name, value in m.acc_group_rms.items():
        out[f"acc_{name}_rms"] = value

    terms = _normalize.score_acceleration(m, _calibration())
    for label in ("arm", "body", "wrist"):
        if label in terms:
            out[f"score_{label}"] = terms[label]

    if m.flags.degenerate:
        out["degenerate"] = True
        out["score"] = 0.0
        return out
    out["score"] = 0.0 if not np.isfinite(terms["score"]) else float(terms["score"])
    return out


def collision(sig: EpisodeSignals) -> dict:
    """Contact/impact proxy (legacy dict shape)."""
    m = _measured(sig)
    detail: dict = {
        "impact_events": m.impact_events,
        "base_disturbance_events": m.base_disturbance_events,
        "tracking_divergence_events": m.tracking_divergence_events,
        "collision_events": m.contact_events,
        "event_rate_hz": m.contact_rate_hz,
        "self_collision": m.flags.self_collision_suspect,
    }
    if np.isfinite(m.max_base_tilt_deg):
        detail["max_base_tilt_deg"] = m.max_base_tilt_deg
    if np.isfinite(m.min_wrist_separation_m):
        detail["min_wrist_separation_m"] = m.min_wrist_separation_m

    if m.flags.degenerate:
        detail["degenerate"] = True
        detail["score"] = 0.0
        return detail

    score = _normalize.score_contact(m, _calibration())["score"]
    if m.flags.self_collision_suspect:
        score = min(score, 0.2)
    detail["score"] = float(score)
    return detail


def runtime(sig: EpisodeSignals, stats: dict | None = None) -> dict:
    """Graded episode-duration score (legacy dict shape)."""
    m = _measured(sig)
    detail: dict = {"duration_s": m.duration_s, "idle_fraction": m.idle_fraction}
    if m.grasp_transitions >= 0:
        detail["grasp_transitions"] = m.grasp_transitions

    if m.flags.degenerate:
        detail["degenerate"] = True
        detail["score"] = 0.0
        return detail

    calib = _calibration(stats)
    terms = _normalize.score_timing(m, calib)
    duration_score = 1.0 if not np.isfinite(terms["duration"]) else float(terms["duration"])
    detail["duration_z"] = 0.0 if not np.isfinite(terms["duration_z"]) else terms["duration_z"]
    detail["duration_score"] = duration_score
    detail["idle_score"] = float(terms["idle"])

    if m.grasp_transitions < 0:
        retry = 1.0                       # no hand telemetry, do not guess
    elif m.flags.grasp_incomplete:
        retry = 0.0                       # the grasp cycle never completed
    else:
        retry = float(terms["grasp"])
    detail["retry_score"] = retry

    detail["score"] = float(
        0.5 * duration_score + 0.3 * detail["idle_score"] + 0.2 * retry
    )
    return detail


def visual_clarity(video_path: str, sig: EpisodeSignals | None = None,
                   n_samples: int = 16) -> dict:
    """Ego-camera quality, corrected for motion blur (legacy dict shape)."""
    video, _ = _measure.measure_video(video_path, n_samples=n_samples)
    detail: dict = {"video_path": str(video_path),
                    "frames_decoded": video.frames_decoded,
                    "frames_declared": video.frames_declared}
    if not video.decodable:
        return {"score": 0.0, "decodable": False, **detail}
    detail["decodable"] = True

    m = _measured(sig) if sig is not None else EpisodeMeasures()
    m.video = video
    m.video_path = str(video_path)
    m.flags = _measure.raise_flags(m, sig)

    if m.flags.video_truncated:
        detail["truncated"] = True
        return {"score": 0.0, **detail}

    detail.update({
        "sharpness": video.sharpness,
        "brightness": video.brightness,
        "contrast": video.contrast,
        "clipped_fraction": video.clipped_fraction,
        "interframe_diff": 0.0 if not np.isfinite(video.interframe_diff) else video.interframe_diff,
    })

    terms = _normalize.score_video(m, _calibration())
    detail["sharpness_score"] = terms["sharpness"]
    detail["exposure_score"] = terms["exposure"]
    detail["contrast_score"] = terms["contrast"]
    detail["clipping_score"] = terms["clipping"]
    detail["freeze_score"] = terms["freeze"]
    detail["score"] = float(terms["score"])
    return detail


# --------------------------------------------------------------------------
# Aggregation and calibration
# --------------------------------------------------------------------------


def score_episode(sig: EpisodeSignals, video_path: str | None = None,
                  time_stats: dict | None = None,
                  weights: dict[str, float] | None = None) -> dict:
    """Run every metric on one episode and return the weighted aggregate."""
    weights = weights or {"smoothness": 0.2, "collision": 0.2, "runtime": 0.2,
                          "acceleration": 0.2, "visual_clarity": 0.2}
    parts = {
        "smoothness": smoothness(sig),
        "collision": collision(sig),
        "runtime": runtime(sig, time_stats),
        "acceleration": acceleration(sig),
    }
    if video_path is not None:
        parts["visual_clarity"] = visual_clarity(video_path, sig)

    used = {k: w for k, w in weights.items() if k in parts and w > 0}
    denom = sum(used.values())
    total = sum(w * parts[k]["score"] for k, w in used.items()) / denom if denom else 0.0
    return {"score": float(total),
            "sub_scores": {k: parts[k]["score"] for k in parts},
            "detail": parts}


def build_time_stats(durations) -> dict:
    """Robust duration statistics for :func:`runtime`, computed dataset-wide."""
    d = np.asarray([x for x in durations if np.isfinite(x)], dtype=float)
    if d.size == 0:
        return {}
    med = float(np.median(d))
    return {
        "median": med,
        "mad": float(np.median(np.abs(d - med)) * 1.4826),
        "mean": float(d.mean()),
        "std": float(d.std()),
    }


def calibrate(values_by_key: dict, lo_pct=5, hi_pct=95,
              lower_is_better=()) -> dict[str, dict[str, float]]:
    """Refit :data:`CALIB` brackets from a new dataset."""
    out = {}
    for key, values in values_by_key.items():
        v = np.asarray([x for x in values if np.isfinite(x)], dtype=float)
        if v.size == 0:
            continue
        p_lo, p_hi = np.percentile(v, [lo_pct, hi_pct])
        if key in lower_is_better:
            out[key] = {"lo": float(p_hi), "hi": float(p_lo)}
        else:
            out[key] = {"lo": float(p_lo), "hi": float(p_hi)}
    return out


# --------------------------------------------------------------------------
# DatasetScorer adapters
# --------------------------------------------------------------------------


def score_smoothness(video_segment, sts, acts, vlm, task, nom) -> float:
    return smoothness(signals_from_states(sts, acts))["score"]


def score_acceleration(video_segment, sts, acts, vlm, task, nom) -> float:
    return acceleration(signals_from_states(sts, acts))["score"]


def score_collision(video_segment, sts, acts, vlm, task, nom) -> float:
    return collision(signals_from_states(sts, acts))["score"]


def score_runtime(video_segment, sts, acts, vlm, task, nom, time_stats=None) -> float:
    return runtime(signals_from_states(sts, acts), time_stats)["score"]


def score_visual_clarity(video_segment, sts, acts, vlm, task, nom) -> float:
    path = getattr(video_segment, "video_path", video_segment)
    return visual_clarity(path, signals_from_states(sts, acts))["score"]

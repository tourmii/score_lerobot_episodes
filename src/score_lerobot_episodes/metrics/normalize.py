"""Scoring layer — turn raw measurements into ``[0, 1]`` and then a decision.

This is the *second* and *third* tier.  :mod:`.measure` produced physical
quantities; here they are mapped onto a common scale and compared against
thresholds.  Everything platform-specific lives in this module, so retargeting
the suite to a different robot means refitting :class:`Calibration` and touching
nothing in the measurement code.

Two ways to turn a quantity into a criterion, both supported:

===========================  ==========================  =========================
                             raw value + absolute limit  percentile ramp (default)
===========================  ==========================  =========================
scale                        physical unit               ``[0, 1]``, 1 = good
thresholds to set            one per quantity, each in   one, shared
                             its own unit
controls the reject rate     not directly                approximately
keeps magnitude information  yes                         yes, except the clipped
                                                         tails
transfers to another robot   yes, if the limit is a      needs a refit
                             physical limit
depends on the dataset mix   no                          yes, through the range
===========================  ==========================  =========================

The percentile ramp is the default because no universal threshold for "too much
acceleration" exists — it depends on limb inertia, gear ratios and control rate.
Set :attr:`Policy.absolute_limits` to work in raw units instead.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from .measure import EpisodeMeasures

# --------------------------------------------------------------------------
# Normalisation primitives
# --------------------------------------------------------------------------


def ramp(value: float, lo: float, hi: float) -> float:
    """``clip((value - lo) / (hi - lo), 0, 1)`` — affine with clipping.

    The direction is encoded in the argument order rather than in a flag:
    ``lo > hi`` means "lower is better".  Non-finite input returns ``0.0``;
    callers that must distinguish "bad" from "not measurable" have to test the
    raw value themselves, which is what :func:`_combine` does.
    """
    if not np.isfinite(value) or lo == hi:
        return 0.0
    return float(np.clip((value - lo) / (hi - lo), 0.0, 1.0))


def gauss_asymmetric(z: float, tau_fast: float, tau_slow: float) -> float:
    """``exp(-(z/tau)**2)`` with a different width on each side of zero.

    Used for duration.  Overlong is nearly always a genuine defect, whereas
    short is a mixture of efficiency and truncation — and truncation already has
    its own flag — so ``tau_slow < tau_fast`` charges the slow side harder.  At
    ``|z| = 2`` with the shipped widths the slow side keeps 0.077 and the fast
    side 0.368, a factor of 4.8.
    """
    if not np.isfinite(z):
        return 0.0
    tau = tau_slow if z >= 0 else tau_fast
    if tau <= 0:
        return 0.0
    return float(math.exp(-((z / tau) ** 2)))


def exp_decay(value: float, reference: float) -> float:
    """``exp(-value / reference)`` — the normaliser for an event rate.

    It has a direct reading: for a rate ``lam`` and ``lam_ref = 0.5 Hz``
    (``tau = 2 s``) the score is the probability that an arbitrary 2-second
    window contains no event under a Poisson model.  At ``lam = lam_ref`` it is
    ``e**-1 = 0.368``.
    """
    if not np.isfinite(value) or reference <= 0:
        return 0.0
    return float(math.exp(-max(value, 0.0) / reference))


def robust_z(value: float, median: float, mad: float) -> float:
    """Median/MAD z-score, so one wild episode cannot move the centre."""
    if not np.isfinite(value):
        return float("nan")
    return float((value - median) / max(mad, 1e-6))


#: Floor applied to a term before the geometric mean takes its logarithm.  A
#: term at exactly 0 would otherwise annihilate the product and destroy all
#: ranking among the bad episodes; 0.02 leaves a catastrophic term costing about
#: a factor of four on its weight while keeping the order intact.
GEOMETRIC_FLOOR = 0.02


def combine(parts: Sequence[tuple[float, float, float]], mode: str = "geometric") -> float:
    """Combine ``(weight, score, raw)`` terms, skipping the unmeasurable ones.

    Filtering on the *raw* value and not on the normalised one is the point:
    :func:`ramp` maps ``nan`` to ``0.0``, so filtering afterwards would silently
    charge a dataset with no wrist channel a zero for the missing term instead
    of redistributing its weight.

    ``"geometric"`` (the default) is the weighted geometric mean.  It is used
    everywhere terms describing *different* failure modes meet, because the
    arithmetic mean is fully compensatory and these terms must not compensate:
    a trajectory that is smooth, prompt, well filmed and mechanically violent is
    not a three-quarters-good episode, it is a bad one.  Concretely, a family at
    0.0 beside three at 0.9 gives 0.582 arithmetically and 0.235 geometrically.

    Both agree closely on *ordering* (rank correlation 0.98 on the reference
    dataset); they disagree on how much a single collapse may be hidden.
    """
    usable = [(w, s) for w, s, raw in parts if np.isfinite(raw) and np.isfinite(s)]
    total_w = sum(w for w, _ in usable)
    if total_w <= 0:
        return float("nan")
    if mode == "arithmetic":
        return float(sum(w * s for w, s in usable) / total_w)
    log_mean = sum(w * math.log(max(s, GEOMETRIC_FLOOR)) for w, s in usable) / total_w
    return float(math.exp(log_mean))


# Kept for callers written against the private name.
_combine = combine


# --------------------------------------------------------------------------
# Calibration
# --------------------------------------------------------------------------

#: Quantities normalised by a percentile ramp, and their direction:
#: ``+1`` higher is better, ``-1`` lower is better.
RAMP_DIRECTION: dict[str, int] = {
    "ldlj_wrist": +1,
    "ldlj_arm": +1,
    "ldlj_body": +1,
    "acc_arm_rms": -1,
    "acc_arm_p99": -1,
    "acc_body_rms": -1,
    "acc_body_p99": -1,
    "acc_wrist_rms": -1,
    "acc_wrist_p99": -1,
    "video_sharpness_residual": +1,
    "video_contrast": +1,
}

#: Brackets fitted on ``pickup_20260628_150622`` (90 episodes, 50 Hz, Unitree
#: G1, one-armed pick-and-place).  They are the 5th/95th percentile of the
#: quantity, so a median episode lands near 0.5 rather than saturating.  They
#: are dataset-specific: applied unchanged to another robot, task or control
#: rate they produce plausible-looking scores that rank badly.  Refit with
#: :meth:`Calibration.fit`.
DEFAULT_BRACKETS: dict[str, tuple[float, float]] = {
    "ldlj_wrist": (-19.7, -17.0),
    "ldlj_arm": (-20.4, -17.5),
    "ldlj_body": (-21.4, -18.5),
    "acc_arm_rms": (5.4, 3.6),
    "acc_arm_p99": (21.1, 13.5),
    "acc_body_rms": (2.8, 1.3),
    "acc_body_p99": (9.0, 5.1),
    "acc_wrist_rms": (3.3, 2.0),
    "acc_wrist_p99": (12.1, 5.8),
    "video_contrast": (30.0, 60.0),
}

#: Relative weights across the five families.
#:
#: ``contact`` sits at zero on purpose.  Pick-and-place *requires* contact and
#: nothing in the data separates the intended kind from the accidental kind; on
#: the reference dataset 72 of 90 episodes register no event at all, so the
#: quantity has almost no ranking power.  It stays a review flag.  Raise the
#: weight only if your task genuinely forbids contact.
DEFAULT_WEIGHTS: dict[str, float] = {
    "smoothness": 0.30,
    "acceleration": 0.30,
    "timing": 0.25,
    "video": 0.15,
    "contact": 0.0,
}


@dataclass
class Calibration:
    """Everything the scoring layer needs that depends on the dataset."""

    brackets: dict[str, tuple[float, float]] = field(
        default_factory=lambda: dict(DEFAULT_BRACKETS)
    )
    # Duration is meaningless in isolation and is judged against the dataset.
    duration_median: float = float("nan")
    duration_mad: float = float("nan")
    tau_fast: float = 2.00
    tau_slow: float = 1.25
    # Grasp transitions are not monotonic — wrong in either direction is bad —
    # so the quantity is re-expressed as "excess over the nominal cycle" first.
    # The nominal is the mode of the dataset: 4 when the hand rests closed
    # (rest-closed -> open -> close -> open -> rest-closed), 2 when it rests open.
    grasp_nominal: int = 4
    grasp_decay: float = 2.0
    # Contact rate costing ~63% of the contact score, in events per second.
    contact_rate_ref: float = 0.5
    # Fraction of the episode that may legitimately be slow — approach, settling
    # and release all move slowly on purpose.
    idle_allowance: float = 0.25
    # sharpness ~= a + b * arm_speed_rms, fitted across the dataset.  Laplacian
    # variance drops both from defocus (a real fault) and from motion blur (a
    # sign the robot was working); scoring the residual removes the second.
    sharpness_intercept: float = float("nan")
    sharpness_slope: float = float("nan")
    # Fallback when no fit is available: credit back the blur the episode's own
    # speed explains, capped, because without the cap fast motion would buy the
    # whole sharpness score and invert the metric the other way.
    sharpness_fallback: tuple[float, float] = (58.0, 110.0)
    sharpness_allowance_scale: float = 1.5
    sharpness_allowance_cap: float = 0.5
    n_episodes_fitted: int = 0

    # ---------------------------------------------------------------- fit
    @classmethod
    def fit(
        cls,
        measures: Iterable[EpisodeMeasures],
        lo_pct: float = 5.0,
        hi_pct: float = 95.0,
        base: "Calibration | None" = None,
    ) -> "Calibration":
        """Fit the ranges and references from a dataset.

        **Only episodes that pass their preconditions take part.**  Every motion
        metric is a decreasing function of some derivative norm, so a motionless
        recording is simultaneously at the global optimum of all of them; on the
        reference dataset a reset recording once scored the best smoothness in
        the set.  Leaving such an episode in the percentile sample skews both
        ends of every range and corrupts the thresholds of all the others.

        Quantities that are absent or ``nan`` across the dataset keep their
        value from ``base`` (the shipped defaults if omitted).
        """
        base = base or cls()
        items = [m for m in measures if m.flags.valid]
        if not items:
            return base

        brackets = dict(base.brackets)
        for key, direction in RAMP_DIRECTION.items():
            values = np.asarray(
                [v for v in (_quantity(m, key, base) for m in items) if np.isfinite(v)],
                dtype=float,
            )
            if values.size < 3:
                continue
            p_lo, p_hi = np.percentile(values, [lo_pct, hi_pct])
            if p_lo == p_hi:
                continue
            brackets[key] = (float(p_lo), float(p_hi)) if direction > 0 else (float(p_hi), float(p_lo))

        durations = np.asarray(
            [m.duration_s for m in items if np.isfinite(m.duration_s)], dtype=float
        )
        duration_median = float(np.median(durations)) if durations.size else base.duration_median
        duration_mad = (
            float(np.median(np.abs(durations - duration_median)) * 1.4826)
            if durations.size else base.duration_mad
        )

        transitions = [m.grasp_transitions for m in items if m.grasp_transitions >= 2]
        grasp_nominal = base.grasp_nominal
        if transitions:
            values, counts = np.unique(np.asarray(transitions), return_counts=True)
            grasp_nominal = int(values[int(np.argmax(counts))])

        intercept, slope = _fit_sharpness(items)

        fitted = cls(
            brackets=brackets,
            duration_median=duration_median,
            duration_mad=duration_mad,
            tau_fast=base.tau_fast,
            tau_slow=base.tau_slow,
            grasp_nominal=grasp_nominal,
            grasp_decay=base.grasp_decay,
            contact_rate_ref=base.contact_rate_ref,
            idle_allowance=base.idle_allowance,
            sharpness_intercept=intercept,
            sharpness_slope=slope,
            sharpness_fallback=base.sharpness_fallback,
            sharpness_allowance_scale=base.sharpness_allowance_scale,
            sharpness_allowance_cap=base.sharpness_allowance_cap,
            n_episodes_fitted=len(items),
        )
        # The residual range can only be measured once the fit exists.
        residuals = np.asarray(
            [r for r in (_quantity(m, "video_sharpness_residual", fitted) for m in items)
             if np.isfinite(r)],
            dtype=float,
        )
        if residuals.size >= 3:
            p_lo, p_hi = np.percentile(residuals, [lo_pct, hi_pct])
            if p_lo != p_hi:
                fitted.brackets["video_sharpness_residual"] = (float(p_lo), float(p_hi))
        return fitted

    # ------------------------------------------------------------- storage
    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["brackets"] = {k: list(v) for k, v in self.brackets.items()}
        data["sharpness_fallback"] = list(self.sharpness_fallback)
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Calibration":
        data = dict(data)
        data["brackets"] = {k: tuple(v) for k, v in data.get("brackets", {}).items()}
        if "sharpness_fallback" in data:
            data["sharpness_fallback"] = tuple(data["sharpness_fallback"])
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in data.items() if k in known})

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "Calibration":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    def _ramp(self, key: str, value: float) -> float:
        bracket = self.brackets.get(key)
        return ramp(value, *bracket) if bracket else 0.0


def _fit_sharpness(items: Sequence[EpisodeMeasures]) -> tuple[float, float]:
    """Least-squares ``sharpness ~ intercept + slope * arm_speed_rms``."""
    xy = [
        (m.arm_speed_rms, m.video.sharpness)
        for m in items
        if np.isfinite(m.arm_speed_rms) and np.isfinite(m.video.sharpness)
    ]
    if len(xy) < 5:
        return float("nan"), float("nan")
    x = np.asarray([a for a, _ in xy], dtype=float)
    y = np.asarray([b for _, b in xy], dtype=float)
    if float(x.std()) < 1e-9:
        return float("nan"), float("nan")
    slope, intercept = np.polyfit(x, y, 1)
    return float(intercept), float(slope)


def _quantity(m: EpisodeMeasures, key: str, calib: Calibration) -> float:
    """Read a ramp quantity off a measurement, including the derived ones."""
    if key == "video_sharpness_residual":
        if not np.isfinite(calib.sharpness_slope) or not np.isfinite(m.video.sharpness):
            return float("nan")
        if not np.isfinite(m.arm_speed_rms):
            return float("nan")
        predicted = calib.sharpness_intercept + calib.sharpness_slope * m.arm_speed_rms
        return float(m.video.sharpness - predicted)
    if key.startswith("video_"):
        return float(getattr(m.video, key[len("video_"):], float("nan")))
    return float(getattr(m, key, float("nan")))


# --------------------------------------------------------------------------
# Family scores
# --------------------------------------------------------------------------


def score_smoothness(m: EpisodeMeasures, calib: Calibration,
                     aggregate: str = "geometric") -> dict[str, float]:
    """Weighted mean of the three LDLJ chains.

    The Cartesian term leads because it is what a downstream policy has to
    reproduce and it does not move when the redundant joints re-pose.
    """
    parts = [
        (0.5, calib._ramp("ldlj_wrist", m.ldlj_wrist), m.ldlj_wrist),
        (0.3, calib._ramp("ldlj_arm", m.ldlj_arm), m.ldlj_arm),
        (0.2, calib._ramp("ldlj_body", m.ldlj_body), m.ldlj_body),
    ]
    return {
        "score": combine(parts, aggregate),
        "wrist": parts[0][1],
        "arm": parts[1][1],
        "body": parts[2][1],
    }


def score_acceleration(m: EpisodeMeasures, calib: Calibration,
                       aggregate: str = "geometric") -> dict[str, float]:
    """Per chain, the *worse* of the sustained load and the peak jolt.

    Taking the minimum of the RMS and the p99 term means an episode that is calm
    on average but contains one violent transient cannot pass on its average.
    """
    out: dict[str, float] = {}
    usable = []
    for label in ("arm", "body", "wrist"):
        rms = getattr(m, f"acc_{label}_rms")
        p99 = getattr(m, f"acc_{label}_p99")
        if not np.isfinite(rms) and not np.isfinite(p99):
            continue
        value = min(
            calib._ramp(f"acc_{label}_rms", rms),
            calib._ramp(f"acc_{label}_p99", p99),
        )
        out[label] = value
        usable.append(value)
    # Same rule one level down: a violent leg is a real fault, and averaging it
    # against a calm arm and a calm wrist hides it.  Five episodes on the
    # reference dataset have a limb at ~0 yet scored above 0.4 under a mean.
    out["score"] = combine([(1.0, v, v) for v in usable], aggregate) if usable else float("nan")
    return out


def score_contact(m: EpisodeMeasures, calib: Calibration,
                  aggregate: str = "geometric") -> dict[str, float]:
    """Exponential in the event *rate*, so it does not depend on episode length."""
    return {"score": exp_decay(m.contact_rate_hz, calib.contact_rate_ref)}


def score_timing(m: EpisodeMeasures, calib: Calibration,
                 aggregate: str = "geometric") -> dict[str, float]:
    """Duration, dead time and regrasps, each normalised on its own terms."""
    z = robust_z(m.duration_s, calib.duration_median, calib.duration_mad)
    duration = (
        gauss_asymmetric(z, calib.tau_fast, calib.tau_slow)
        if np.isfinite(calib.duration_median) and np.isfinite(z) else float("nan")
    )
    # Idle below the allowance is free; beyond it the score falls linearly to 0
    # at a wholly motionless episode.
    idle = ramp(m.idle_fraction, 1.0, calib.idle_allowance)

    if m.grasp_transitions < 0:
        grasp = float("nan")                       # no hand telemetry, do not guess
    elif m.flags.grasp_incomplete:
        # Not a low score but a flag: the cycle never happened, so there is no
        # grasp quality to speak of.  The episode is rejected by its flag.
        grasp = float("nan")
    else:
        excess = max(0, m.grasp_transitions - calib.grasp_nominal)
        grasp = float(math.exp(-excess / calib.grasp_decay))

    parts = [(0.5, duration, duration), (0.3, idle, m.idle_fraction), (0.2, grasp, grasp)]
    return {
        "score": combine(parts, aggregate),
        "duration": duration,
        "duration_z": z,
        "idle": idle,
        "grasp": grasp,
    }


def score_video(m: EpisodeMeasures, calib: Calibration,
                aggregate: str = "geometric") -> dict[str, float]:
    """Appearance, gated by the integrity checks.

    A corrupt or frozen recording looks pristine to any appearance statistic, so
    the integrity terms multiply rather than average: they can zero the result.
    """
    if m.video_path is None:
        return {"score": float("nan")}
    if not m.video.decodable or m.flags.video_truncated:
        return {"score": 0.0, "sharpness": 0.0, "exposure": 0.0,
                "contrast": 0.0, "clipping": 0.0, "freeze": 0.0}

    residual = _quantity(m, "video_sharpness_residual", calib)
    if np.isfinite(residual) and "video_sharpness_residual" in calib.brackets:
        sharpness = calib._ramp("video_sharpness_residual", residual)
    else:
        # No dataset fit: fall back to a raw ramp plus a capped motion credit.
        sharpness = ramp(m.video.sharpness, *calib.sharpness_fallback)
        if np.isfinite(m.arm_speed_rms):
            allowance = float(np.clip(
                m.arm_speed_rms / calib.sharpness_allowance_scale,
                0.0, calib.sharpness_allowance_cap,
            ))
            sharpness = float(np.clip(sharpness + allowance, 0.0, 1.0))

    exposure = float(np.clip(1.0 - abs(m.video.brightness - 127.5) / 127.5, 0.0, 1.0))
    contrast = calib._ramp("video_contrast", m.video.contrast)
    clipping = float(np.clip(1.0 - m.video.clipped_fraction / 0.02, 0.0, 1.0))
    freeze = 1.0
    if np.isfinite(m.video.interframe_diff) and m.video.interframe_diff < 1.0:
        freeze = float(np.clip(m.video.interframe_diff, 0.0, 1.0))

    appearance = 0.45 * sharpness + 0.30 * exposure + 0.25 * contrast
    return {
        "score": float(min(freeze, clipping) * appearance),
        "sharpness": sharpness,
        "sharpness_residual": residual,
        "exposure": exposure,
        "contrast": contrast,
        "clipping": clipping,
        "freeze": freeze,
    }


FAMILY_SCORERS = {
    "smoothness": score_smoothness,
    "acceleration": score_acceleration,
    "contact": score_contact,
    "timing": score_timing,
    "video": score_video,
}


# --------------------------------------------------------------------------
# Aggregation and decision
# --------------------------------------------------------------------------


# --------------------------------------------------------------------------
# Per-quantity rules — the decision in physical units
# --------------------------------------------------------------------------

#: Quantities a rule can be written against, with the direction that is bad.
#: ``worse="higher"`` -> the rule fires when the measurement exceeds the limit.
#: ``seed`` is the percentile used when seeding a limit from a dataset.
RULE_QUANTITIES: dict[str, dict[str, Any]] = {
    "ldlj_wrist":            {"label": "wrist smoothness (LDLJ)", "unit": "",       "worse": "lower",  "seed": 2,  "family": "smoothness"},
    "ldlj_arm":              {"label": "arm smoothness (LDLJ)",   "unit": "",       "worse": "lower",  "seed": 2,  "family": "smoothness"},
    "ldlj_body":             {"label": "body smoothness (LDLJ)",  "unit": "",       "worse": "lower",  "seed": 2,  "family": "smoothness"},
    "acc_arm_rms":           {"label": "arm acceleration RMS",    "unit": "rad/s²", "worse": "higher", "seed": 98, "family": "acceleration"},
    "acc_arm_p99":           {"label": "arm acceleration p99",    "unit": "rad/s²", "worse": "higher", "seed": 98, "family": "acceleration"},
    "acc_body_rms":          {"label": "body acceleration RMS",   "unit": "rad/s²", "worse": "higher", "seed": 98, "family": "acceleration"},
    "acc_body_p99":          {"label": "body acceleration p99",   "unit": "rad/s²", "worse": "higher", "seed": 98, "family": "acceleration"},
    "acc_wrist_rms":         {"label": "wrist acceleration RMS",  "unit": "m/s²",   "worse": "higher", "seed": 98, "family": "acceleration"},
    "acc_wrist_p99":         {"label": "wrist acceleration p99",  "unit": "m/s²",   "worse": "higher", "seed": 98, "family": "acceleration"},
    "contact_rate_hz":       {"label": "contact event rate",      "unit": "Hz",     "worse": "higher", "seed": 98, "family": "contact"},
    "max_base_tilt_deg":     {"label": "max base tilt",           "unit": "deg",    "worse": "higher", "seed": 98, "family": "contact"},
    "min_wrist_separation_m": {"label": "min wrist separation",   "unit": "m",      "worse": "lower",  "seed": 2,  "family": "contact"},
    "duration_s":            {"label": "duration (too long)",     "unit": "s",      "worse": "higher", "seed": 95, "family": "timing"},
    "duration_short_s":      {"label": "duration (too short)",    "unit": "s",      "worse": "lower",  "seed": 2,  "family": "timing"},
    "idle_fraction":         {"label": "idle fraction",           "unit": "",       "worse": "higher", "seed": 98, "family": "timing"},
    "grasp_transitions":     {"label": "hand transitions",        "unit": "",       "worse": "higher", "seed": 98, "family": "timing"},
    "eef_path_m":            {"label": "wrist path length",       "unit": "m",      "worse": "lower",  "seed": 2,  "family": "timing"},
    "video_sharpness":       {"label": "sharpness",               "unit": "lap.var", "worse": "lower", "seed": 2,  "family": "video"},
    "video_clipped_fraction": {"label": "clipped pixels",         "unit": "",       "worse": "higher", "seed": 98, "family": "video"},
    "video_interframe_diff": {"label": "inter-frame difference",  "unit": "grey",   "worse": "lower",  "seed": 2,  "family": "video"},
}


@dataclass
class Rule:
    """One limit, in the quantity's own physical unit.

    This is the decision that does *not* depend on the rest of the dataset: a
    limit of 6 rad/s² is 6 rad/s² whether the batch that came with it was good
    or bad.  A percentile-fitted score cannot say that — it is a rank, so it
    rejects the same fraction of any dataset you hand it.
    """

    quantity: str
    limit: float
    op: str = ">"                 # ">" fires above the limit, "<" fires below it
    action: str = "reject"        # "reject" | "review"
    enabled: bool = True

    def fires(self, value: float) -> bool:
        if value is None or not np.isfinite(value):
            return False          # not measurable is not a violation
        return value > self.limit if self.op == ">" else value < self.limit

    @property
    def label(self) -> str:
        spec = RULE_QUANTITIES.get(self.quantity, {})
        unit = spec.get("unit", "")
        return (f"{spec.get('label', self.quantity)} {self.op} {self.limit:g}"
                + (f" {unit}" if unit else ""))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Rule":
        known = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in data.items() if k in known})


def rule_value(m: EpisodeMeasures, quantity: str) -> float:
    """Read a rule quantity off a measurement, in its physical unit."""
    if quantity == "duration_short_s":
        return float(m.duration_s)
    if quantity.startswith("video_"):
        return float(getattr(m.video, quantity[len("video_"):], float("nan")))
    value = getattr(m, quantity, float("nan"))
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def violations(m: EpisodeMeasures, rules: Sequence[Rule]) -> list[tuple[Rule, float]]:
    """Every enabled rule this episode breaks, with the offending value."""
    out = []
    for rule in rules:
        if not rule.enabled:
            continue
        value = rule_value(m, rule.quantity)
        if rule.fires(value):
            out.append((rule, value))
    return out


def suggest_rules(
    measures: Iterable[EpisodeMeasures],
    quantities: Sequence[str] | None = None,
    action: str = "review",
) -> list[Rule]:
    """Seed a limit per quantity from the valid episodes of a dataset.

    A starting point, not an answer: it puts each limit at that quantity's
    ``seed`` percentile so nothing but the existing outliers fires, and leaves
    you to move it to whatever your robot or task actually tolerates.  Once you
    have moved it, the limit is absolute and transfers to the next dataset.
    """
    items = [m for m in measures if m.flags.valid]
    rules: list[Rule] = []
    for quantity in (quantities or RULE_QUANTITIES):
        spec = RULE_QUANTITIES[quantity]
        values = np.asarray(
            [v for v in (rule_value(m, quantity) for m in items) if np.isfinite(v)],
            dtype=float,
        )
        if values.size < 3:
            continue
        limit = float(np.percentile(values, spec["seed"]))
        rules.append(Rule(
            quantity=quantity,
            limit=round(limit, 4),
            op=">" if spec["worse"] == "higher" else "<",
            action=action,
            enabled=False,          # opt in deliberately; nothing rejects by surprise
        ))
    return rules


@dataclass
class Policy:
    """Where the accept/review/reject line sits.

    Three decision rules:

    ``"gate"`` (default)
        Every family must clear **its own** threshold, taken from
        :attr:`minimums` and falling back to :attr:`accept_threshold`.  One
        slider per family, and no averaging: each family targets a different
        failure mode, so a good one must not be allowed to hide a bad one.

        Relative — the ramp brackets are the dataset's own p5/p95, so a family
        score is close to a percentile rank.  Set the sliders by looking at what
        each one costs rather than by picking a number that sounds strict.
    ``"rules"``
        A limit per measured quantity in its own physical unit, no aggregate.
        The only mode whose verdict does not depend on the rest of the batch, at
        the price of twenty numbers to maintain instead of five.
    ``"weighted"``
        The weighted mean must clear the threshold.  Compensatory, and the
        loosest of the three; useful when a single triage number is what you
        want.

    The weighted total is computed in every mode — it is the ranking column —
    but only ``"weighted"`` decides anything with it.
    """

    mode: str = "gate"
    #: How the family scores are combined into the ranking total, and how the
    #: terms inside each family are combined.  "geometric" is non-compensatory.
    aggregate: str = "geometric"
    #: Limits in physical units.  Used by ``mode="rules"``.
    rules: list[Rule] = field(default_factory=list)
    #: The same number means very different things in the two modes.  The ranges
    #: are fitted to the dataset's own p5/p95, so a median episode lands near 0.5
    #: on *every* family; requiring 0.6 from all five at once keeps only a
    #: handful, while 0.6 from their mean keeps most.  This default is chosen for
    #: ``"gate"``: roughly "above the dataset's own lower third on every axis".
    #: Raise it to about 0.6 when switching to ``"weighted"``.
    accept_threshold: float = 0.35
    #: Per-family floors, overriding ``accept_threshold`` for that family.  In
    #: ``"weighted"`` mode a family below its floor sends the episode to review;
    #: in ``"gate"`` mode it rejects.
    minimums: dict[str, float] = field(default_factory=dict)
    #: Contact events are a review trigger, not a rejection: this task needs
    #: contact and nothing separates the intended kind from the rest.
    review_on_contact: bool = True
    contact_review_rate_hz: float = 0.5
    #: Semantic verdict below this sends the episode to review; a hard failure
    #: (0.0) is a rejection.  See :mod:`.semantic`.
    semantic_accept: float = 1.0
    semantic_reject: float = 0.0
    #: Raw-unit limits, applied on top of the normalised score.  Maps a
    #: measurement attribute to ``(">", value)`` or ``("<", value)``; violating
    #: one rejects the episode.  Empty by default — use it when you have a real
    #: physical limit (a joint acceleration rating, say) rather than a
    #: percentile.
    absolute_limits: dict[str, tuple[str, float]] = field(default_factory=dict)


@dataclass
class EpisodeScore:
    episode: int | None
    total: float
    families: dict[str, float]
    terms: dict[str, dict[str, float]]
    weights: dict[str, float]
    flags: dict[str, bool]
    decision: str = "accept"
    reasons: list[str] = field(default_factory=list)
    semantic_score: float | None = None
    semantic_note: str = ""
    #: Evidence against the episode, in nats: ``-log(total)``.  ``total`` is a
    #: probability, and a product over a dozen criteria spans so many decades
    #: that every hopeless episode prints as ``0.000`` and stops being sortable
    #: — exactly the episodes a reviewer most wants ordered.  ``severity`` is
    #: the same quantity on a scale that stays legible: 0 is clean, ~0.7 is the
    #: default accept line, 10+ is condemned several times over.  It ranks;
    #: ``total`` decides.
    severity: float = 0.0

    def to_row(self) -> dict[str, Any]:
        row: dict[str, Any] = {"episode": self.episode, "total": self.total,
                               "severity": self.severity, "decision": self.decision}
        row.update(self.families)
        row.update(self.flags)
        if self.semantic_score is not None:
            row["semantic"] = self.semantic_score
        row["reasons"] = "; ".join(self.reasons)
        return row


def score_episode(
    m: EpisodeMeasures,
    calib: Calibration | None = None,
    weights: dict[str, float] | None = None,
    policy: Policy | None = None,
    semantic_score: float | None = None,
    semantic_note: str = "",
) -> EpisodeScore:
    """Normalise one episode's measurements, aggregate, and decide.

    Weights are relative: they are divided by their sum over the families that
    were actually computable, so a dataset without video does not lose 15% of
    its score to a term that could not be measured.
    """
    calib = calib or Calibration()
    weights = weights or DEFAULT_WEIGHTS
    policy = policy or Policy()

    terms = {name: fn(m, calib, policy.aggregate) for name, fn in FAMILY_SCORERS.items()}
    families = {name: terms[name]["score"] for name in terms}

    used = {
        name: w for name, w in weights.items()
        if w > 0 and name in families and np.isfinite(families[name])
    }
    total = combine(
        [(w, families[name], families[name]) for name, w in used.items()],
        policy.aggregate,
    )
    if not np.isfinite(total):
        total = 0.0

    score = EpisodeScore(
        episode=m.episode,
        total=total,
        families=families,
        terms=terms,
        weights=used,
        flags=m.flags.to_dict(),
        semantic_score=semantic_score,
        semantic_note=semantic_note,
        severity=-math.log(max(total, 1e-12)),
    )
    decide(score, m, policy)
    return score


def decide(score: EpisodeScore, m: EpisodeMeasures, policy: Policy) -> EpisodeScore:
    """Apply the preconditions, the limits and the threshold, in that order."""
    reasons: list[str] = []

    for name in m.flags.raised():
        if name in m.flags.HARD:
            reasons.append(f"flag:{name}")
    if reasons:
        score.decision, score.reasons = "reject", reasons
        return score

    for attribute, (operator, limit) in policy.absolute_limits.items():
        value = getattr(m, attribute, float("nan"))
        if not np.isfinite(value):
            continue
        if (operator == ">" and value > limit) or (operator == "<" and value < limit):
            reasons.append(f"limit:{attribute}{operator}{limit:g} (got {value:.3g})")
    if reasons:
        score.decision, score.reasons = "reject", reasons
        return score

    if score.semantic_score is not None and score.semantic_score <= policy.semantic_reject:
        detail = f": {score.semantic_note}" if score.semantic_note else ""
        score.decision = "reject"
        score.reasons = [f"semantic:goal not reached{detail}"]
        return score

    review: list[str] = []

    if policy.mode == "rules":
        # Physical limits only.  Nothing here consults the aggregate, and
        # nothing here depends on how the rest of the dataset scored.
        for rule, value in violations(m, policy.rules):
            text = f"{rule.label} (got {value:.4g})"
            (reasons if rule.action == "reject" else review).append(text)
        if reasons:
            score.decision, score.reasons = "reject", reasons
            return score
    elif policy.mode == "gate":
        # Every measurable family stands on its own; no averaging away a fault.
        #
        # Gating is decided by the family's *threshold*, never by its weight.
        # Weight is for the ranking total; tying the two meant a family weighted
        # 0 — contact, by default — silently ignored its own slider.  Set a
        # family's threshold to 0 to opt it out of the gate.
        for family, value in score.families.items():
            floor = policy.minimums.get(family, policy.accept_threshold)
            if floor <= 0 or not np.isfinite(value):
                continue
            if value < floor:
                reasons.append(f"{family} {value:.3f} < {floor:g}")
        if reasons:
            score.decision, score.reasons = "reject", reasons
            return score
    else:
        if score.total < policy.accept_threshold:
            score.decision = "reject"
            score.reasons = [f"score {score.total:.3f} < {policy.accept_threshold:g}"]
            return score

    for name in m.flags.raised():
        if name in m.flags.SOFT:
            review.append(f"flag:{name}")
    if policy.mode == "weighted":
        for family, floor in policy.minimums.items():
            value = score.families.get(family, float("nan"))
            if np.isfinite(value) and value < floor:
                review.append(f"{family} {value:.3f} < {floor:g}")
    if policy.review_on_contact and m.contact_rate_hz > policy.contact_review_rate_hz:
        review.append(f"contact {m.contact_events} events ({m.contact_rate_hz:.2f} Hz)")
    if score.semantic_score is not None and score.semantic_score < policy.semantic_accept:
        review.append(f"semantic {score.semantic_score:g}")

    score.decision = "review" if review else "accept"
    score.reasons = review
    return score

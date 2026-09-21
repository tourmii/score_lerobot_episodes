"""Absolute scoring — one threshold that means the same thing on every dataset.

:mod:`.normalize` fits its ramps to the batch it is scoring, which makes a
family score a *percentile rank* inside that batch.  That is useful for triage
and useless for a standing quality bar: ``--threshold 0.35`` rejects roughly the
same fraction of any dataset handed to it, so a uniformly excellent batch still
loses its bottom third and a uniformly bad one still keeps two thirds.  Scores
from two datasets are not comparable, and changing a weight moves every
episode's total, so the threshold has to be re-tuned after every edit.

This module takes the other route.  Each quantity gets two anchors **in its own
physical unit**:

``good``
    the level below which the quantity stops mattering.  Not "typical" — the
    point where you would no longer look twice.
``bad``
    the level at which this quantity *alone* condemns the episode, whatever the
    rest of the recording looks like.

Both are fitted **once**, on a reference dataset you trust, and then frozen into
a :class:`QualityProfile` on disk.  Applying the saved profile unchanged to the
next dataset is what makes ``6 rad/s²`` mean ``6 rad/s²`` in January and in
June, on batch 3 and on batch 40.

Aggregation is a **noisy-OR** rather than a weighted mean::

    u_i      = (v_i - good_i) / (bad_i - good_i)   # 0 at good, 1 at bad
    p_i      = P_MAX * smoothstep(u_i)             # this criterion is violated
    severity = sum_i w_i * -log(1 - p_i)           # evidence against, in nats
    total    = exp(-severity)                      # no criterion is violated

``total`` is the number the threshold is compared against; ``severity`` is the
same quantity on a log scale and is what the tables sort on, because a product
over a dozen criteria underflows to a shared ``0.000`` exactly for the episodes
a reviewer most wants ranked.

Three properties follow, and each one removes a knob:

*The total keeps its meaning when the weights change.*  ``w_i`` is a
multiplicity, not a share of a budget that must sum to one.  Under
:func:`~.normalize.combine` the exponents are ``w_i / sum(w)``, so raising one
weight rescales every episode's total and the threshold has to move with it.
Here, raising ``w_i`` can only lower the episodes that trip criterion ``i``.

*It is non-compensatory without a floor.*  A single ``p_i`` near one drags the
product to zero however clean the other terms are, which is the property the
geometric mean was chosen for — but here it falls out of the algebra instead of
needing ``GEOMETRIC_FLOOR`` to stop a zero term annihilating the ranking.
``P_MAX < 1`` leaves the condemned episodes rankable among themselves.

*An unmeasurable quantity leaves no trace.*  A ``nan`` term drops out of the
product entirely.  ``total`` is then "no criterion **of those measurable** is
violated", which is the honest reading; the weight-redistribution in
:func:`~.normalize.score_episode` exists only because a weighted mean has no
such option.

What is left to tune is one number: :attr:`QualityProfile.threshold`.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from .measure import EpisodeMeasures
from .normalize import EpisodeScore

#: Ceiling on a single criterion's violation probability.  At exactly ``1.0`` a
#: condemned episode scores ``0.0`` regardless of everything else, and every
#: condemned episode ties at zero — which destroys the ranking among the worst
#: episodes, the ones a human most wants ordered.  ``0.98`` costs a factor of 50
#: per condemned criterion, so the order survives.
P_MAX = 0.98

#: MAD-multiples used when fitting anchors from a reference dataset.
#: ``good`` sits at 1 robust sigma into the bad direction, so a median episode
#: lands at ``u < 0`` and pays nothing at all; ``bad`` sits at 4, which on a
#: normal-ish sample is a genuine outlier rather than a lower quartile.
GOOD_SIGMA = 1.0
BAD_SIGMA = 4.0


def smoothstep(u: float) -> float:
    """``3u^2 - 2u^3`` clipped to ``[0, 1]`` — zero slope at both anchors.

    A plain ``clip(u, 0, 1)`` would put a corner exactly at ``good``, so an
    episode a hair over the anchor starts paying at the full rate.  Smoothstep
    is flat there: the first measurements past ``good`` cost almost nothing and
    the penalty only bites in the middle of the band.  That is what makes a
    single threshold behave — the accept/reject line lands on a slope, not on a
    cliff where a rounding difference flips the verdict.
    """
    if not np.isfinite(u):
        return float("nan")
    u = float(np.clip(u, 0.0, 1.0))
    return u * u * (3.0 - 2.0 * u)


# --------------------------------------------------------------------------
# Criteria
# --------------------------------------------------------------------------


@dataclass
class Criterion:
    """One quantity, two anchors in its unit, and a multiplicity.

    Direction is carried by the anchors, not by a flag: ``bad < good`` means
    lower is worse.  That mirrors :func:`~.normalize.ramp` and keeps the two
    layers readable side by side.
    """

    quantity: str
    good: float
    bad: float
    weight: float = 1.0
    #: ``"reject"`` counts toward the total; ``"review"`` never lowers the score
    #: and only raises a note.  Use it for quantities that mark an episode worth
    #: a human look without being evidence of a defect.
    action: str = "reject"
    unit: str = ""
    label: str = ""
    enabled: bool = True

    def u(self, value: float) -> float:
        """Position in the band: ``0`` at ``good``, ``1`` at ``bad``."""
        if value is None or not np.isfinite(value) or self.good == self.bad:
            return float("nan")
        return float((value - self.good) / (self.bad - self.good))

    def penalty(self, value: float) -> float:
        """``P(this criterion is violated)``, or ``nan`` if not measurable."""
        u = self.u(value)
        if not np.isfinite(u):
            return float("nan")
        if u <= 0.0:
            return 0.0
        if u <= 1.0:
            return P_MAX * smoothstep(u)
        # Past ``bad`` the probability is already 0.98 and has nowhere useful to
        # go — every catastrophic episode would read the same.  Continuing as
        # ``1 - (1 - P_MAX)**u`` keeps it below one while :meth:`cost` stays
        # linear in ``u``, which is what preserves the ranking among them.
        return float(1.0 - (1.0 - P_MAX) ** min(u, 50.0))

    def cost(self, value: float) -> float:
        """``-log(1 - p)`` in nats — the term's actual contribution to the total.

        Computed in closed form rather than from :meth:`penalty`, because past
        ``bad`` the probability is within a float's rounding of one and the
        logarithm of it saturates.  In nats there is no ceiling: an episode four
        times over the limit costs four times what one at the limit costs, so
        the worst recordings stay ordered instead of all tying at the floor.
        """
        u = self.u(value)
        if not np.isfinite(u):
            return float("nan")
        if u <= 0.0:
            return 0.0
        unit_cost = -math.log(1.0 - P_MAX)
        if u <= 1.0:
            return -math.log(max(1.0 - P_MAX * smoothstep(u), 1e-300))
        return float(min(u, 50.0) * unit_cost)

    @property
    def text(self) -> str:
        name = self.label or self.quantity
        unit = f" {self.unit}" if self.unit else ""
        return f"{name}: good {self.good:g}{unit} -> bad {self.bad:g}{unit}"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Criterion":
        known = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in data.items() if k in known})


#: Quantities whose anchors are genuine constants — dimensionless ratios, event
#: rates, grey levels.  They do not depend on the robot, the control rate or the
#: camera, so they ship as-is and :meth:`QualityProfile.fit` leaves them alone.
#: This is the part of the profile that never needs a refit.
ABSOLUTE_CRITERIA: tuple[Criterion, ...] = (
    Criterion("idle_fraction", good=0.30, bad=0.75, weight=1.0,
              label="idle fraction"),
    Criterion("grasp_excess", good=0.0, bad=4.0, weight=0.7,
              label="extra hand transitions"),
    Criterion("contact_rate_hz", good=0.20, bad=1.50, weight=0.5,
              action="review", unit="Hz", label="contact event rate"),
    Criterion("max_base_tilt_deg", good=8.0, bad=25.0, weight=0.8,
              unit="deg", label="max base tilt"),
    Criterion("duration_ratio", good=1.35, bad=2.50, weight=1.0,
              label="duration vs nominal (long)"),
    Criterion("duration_ratio_short", good=0.70, bad=0.35, weight=1.0,
              label="duration vs nominal (short)"),
    Criterion("video_clipped_fraction", good=0.005, bad=0.050, weight=0.8,
              label="clipped pixels"),
    Criterion("video_brightness_dev", good=30.0, bad=90.0, weight=0.8,
              unit="grey", label="exposure deviation"),
    Criterion("video_interframe_diff", good=3.0, bad=0.5, weight=0.6,
              unit="grey", label="inter-frame difference"),
)

#: Quantities set by the machine and the lens: joint inertia and gear ratios fix
#: what acceleration is violent, the control rate and the filter fix where LDLJ
#: sits, the sensor fixes what Laplacian variance counts as sharp.  These are the
#: ones :meth:`QualityProfile.fit` estimates — **once per robot and camera**, not
#: once per batch.  The shipped values come from ``pickup_20260628_150622``
#: (Unitree G1, 50 Hz, one-armed pick-and-place).
PLATFORM_CRITERIA: tuple[Criterion, ...] = (
    Criterion("ldlj_wrist", good=-19.2, bad=-21.7, weight=1.0,
              label="wrist smoothness (LDLJ)"),
    Criterion("ldlj_arm", good=-19.9, bad=-22.4, weight=0.7,
              label="arm smoothness (LDLJ)"),
    Criterion("ldlj_body", good=-20.9, bad=-23.4, weight=0.5,
              label="body smoothness (LDLJ)"),
    Criterion("acc_arm_rms", good=5.4, bad=9.5, weight=1.0,
              unit="rad/s2", label="arm acceleration RMS"),
    Criterion("acc_arm_p99", good=21.0, bad=38.0, weight=1.0,
              unit="rad/s2", label="arm acceleration p99"),
    Criterion("acc_body_rms", good=2.8, bad=6.0, weight=0.7,
              unit="rad/s2", label="body acceleration RMS"),
    Criterion("acc_body_p99", good=9.0, bad=20.0, weight=0.7,
              unit="rad/s2", label="body acceleration p99"),
    Criterion("acc_wrist_rms", good=3.3, bad=7.0, weight=0.7,
              unit="m/s2", label="wrist acceleration RMS"),
    Criterion("acc_wrist_p99", good=12.0, bad=26.0, weight=0.7,
              unit="m/s2", label="wrist acceleration p99"),
    Criterion("video_sharpness_residual", good=-18.0, bad=-55.0, weight=0.8,
              unit="lap.var", label="sharpness (motion-corrected)"),
    Criterion("video_contrast", good=45.0, bad=18.0, weight=0.6,
              unit="grey", label="contrast"),
)

DEFAULT_CRITERIA: tuple[Criterion, ...] = ABSOLUTE_CRITERIA + PLATFORM_CRITERIA

#: The subset :meth:`QualityProfile.fit` re-estimates.  Everything else in the
#: profile is a constant and stays put, which is the line between "this is a
#: different robot" and "this is a different batch of the same robot".
FITTED_QUANTITIES: tuple[str, ...] = tuple(c.quantity for c in PLATFORM_CRITERIA)

#: Which side of the median is the bad one, for the quantities that get fitted.
_WORSE_LOWER = {
    "ldlj_wrist", "ldlj_arm", "ldlj_body",
    "video_sharpness_residual", "video_contrast",
}


# --------------------------------------------------------------------------
# Reading a quantity
# --------------------------------------------------------------------------


def quantity_value(m: EpisodeMeasures, key: str, profile: "QualityProfile") -> float:
    """Read a criterion's quantity off a measurement, in its physical unit.

    The derived ones are computed here rather than stored on
    :class:`~.measure.EpisodeMeasures`, so the measurement layer stays free of
    anything that depends on a profile.
    """
    if key == "grasp_excess":
        # Wrong in either direction, but too few transitions is already the
        # ``grasp_incomplete`` flag's business, so only the excess is scored.
        if m.grasp_transitions < 0 or m.flags.grasp_incomplete:
            return float("nan")
        return float(max(0, m.grasp_transitions - profile.grasp_nominal))

    if key in ("duration_ratio", "duration_ratio_short"):
        nominal = profile.task_duration_s
        if not np.isfinite(nominal) or nominal <= 0 or not np.isfinite(m.duration_s):
            return float("nan")
        return float(m.duration_s / nominal)

    if key == "video_brightness_dev":
        if not np.isfinite(m.video.brightness):
            return float("nan")
        return float(abs(m.video.brightness - 127.5))

    if key == "video_sharpness_residual":
        # Laplacian variance falls both from defocus, a real fault, and from
        # motion blur, a sign the robot was working.  Scoring the residual
        # against the fitted speed/sharpness line removes the second.
        if not np.isfinite(profile.sharpness_slope) or not np.isfinite(m.video.sharpness):
            return float("nan")
        if not np.isfinite(m.arm_speed_rms):
            return float("nan")
        predicted = profile.sharpness_intercept + profile.sharpness_slope * m.arm_speed_rms
        return float(m.video.sharpness - predicted)

    if key.startswith("video_"):
        if m.video_path is None or not m.video.decodable:
            return float("nan")
        return float(getattr(m.video, key[len("video_"):], float("nan")))

    value = getattr(m, key, float("nan"))
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


# --------------------------------------------------------------------------
# The profile
# --------------------------------------------------------------------------


@dataclass
class QualityProfile:
    """An absolute quality bar: anchors in physical units, plus one threshold.

    Fit it once on a dataset you have inspected and trust, save it, and pass the
    same file to every run afterwards.  Refitting per batch throws away exactly
    the property this class exists for.
    """

    criteria: list[Criterion] = field(
        default_factory=lambda: [Criterion.from_dict(c.to_dict()) for c in DEFAULT_CRITERIA]
    )
    #: The only number meant for routine tuning.  ``total`` is a probability, so
    #: this reads directly: ``0.5`` keeps episodes more likely clean than not,
    #: ``0.8`` is strict, ``0.3`` is permissive.
    threshold: float = 0.50
    #: How long this task takes when it goes well, in seconds.  A task property,
    #: not a dataset property — a slower batch of the same task should score
    #: worse, which is precisely what a per-batch median would hide.
    task_duration_s: float = float("nan")
    #: Hand transitions in one clean cycle: 4 when the hand rests closed, 2 when
    #: it rests open.
    grasp_nominal: int = 4
    #: ``sharpness ~ intercept + slope * arm_speed_rms``, fitted once per camera.
    sharpness_intercept: float = float("nan")
    sharpness_slope: float = float("nan")
    #: Provenance, so a profile in a report can be traced to what produced it.
    fitted_on: str = ""
    n_episodes_fitted: int = 0
    notes: str = ""

    # ---------------------------------------------------------------- access
    def active(self) -> list[Criterion]:
        return [c for c in self.criteria if c.enabled and c.weight > 0]

    def get(self, quantity: str) -> Criterion | None:
        for c in self.criteria:
            if c.quantity == quantity:
                return c
        return None

    # ---------------------------------------------------------------- fit
    @classmethod
    def fit(
        cls,
        measures: Iterable[EpisodeMeasures],
        base: "QualityProfile | None" = None,
        good_sigma: float = GOOD_SIGMA,
        bad_sigma: float = BAD_SIGMA,
        fitted_on: str = "",
    ) -> "QualityProfile":
        """Estimate the platform anchors from a reference dataset.

        Anchors are placed at ``median ± k * 1.4826 * MAD`` rather than at a
        percentile pair.  The difference matters: a percentile is a rank and
        reproduces itself on any batch, whereas a median and a robust spread are
        *locations in the unit*, and once written down they stop moving.  The
        median/MAD pair also survives the handful of violent episodes that are
        the reason for scoring at all — a p5/p95 range is partly defined by them.

        Only episodes passing their preconditions take part.  Every motion
        metric is a decreasing function of some derivative norm, so a motionless
        recording sits at the global optimum of all of them at once; leaving one
        in the sample drags ``good`` toward a value no working episode can meet.

        Quantities in :data:`ABSOLUTE_CRITERIA` are deliberately untouched.
        """
        base = base or cls()
        items = [m for m in measures if m.flags.valid]
        profile = cls.from_dict(base.to_dict())
        profile.fitted_on = fitted_on
        if not items:
            return profile

        # Duration and grasp nominal come first: they are the task anchors the
        # derived quantities are read against.
        durations = np.asarray(
            [m.duration_s for m in items if np.isfinite(m.duration_s)], dtype=float)
        if durations.size and not np.isfinite(profile.task_duration_s):
            profile.task_duration_s = float(np.median(durations))

        transitions = [m.grasp_transitions for m in items if m.grasp_transitions >= 2]
        if transitions:
            values, counts = np.unique(np.asarray(transitions), return_counts=True)
            profile.grasp_nominal = int(values[int(np.argmax(counts))])

        intercept, slope = _fit_sharpness(items)
        if np.isfinite(slope):
            profile.sharpness_intercept, profile.sharpness_slope = intercept, slope

        for criterion in profile.criteria:
            if criterion.quantity not in FITTED_QUANTITIES:
                continue
            values = np.asarray(
                [v for v in (quantity_value(m, criterion.quantity, profile) for m in items)
                 if np.isfinite(v)],
                dtype=float,
            )
            if values.size < 5:
                continue
            median = float(np.median(values))
            sigma = float(np.median(np.abs(values - median)) * 1.4826)
            if sigma <= 0:
                # A degenerate spread means the quantity carries no information
                # on this dataset; keeping the shipped anchors beats inventing a
                # zero-width band that condemns everything a hair off the median.
                continue
            sign = -1.0 if criterion.quantity in _WORSE_LOWER else 1.0
            criterion.good = round(median + sign * good_sigma * sigma, 4)
            criterion.bad = round(median + sign * bad_sigma * sigma, 4)

        profile.n_episodes_fitted = len(items)
        return profile

    # ------------------------------------------------------------- storage
    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["criteria"] = [c.to_dict() for c in self.criteria]
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "QualityProfile":
        data = dict(data)
        criteria = data.get("criteria")
        if criteria is not None:
            data["criteria"] = [Criterion.from_dict(c) for c in criteria]
        known = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in data.items() if k in known})

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "QualityProfile":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


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


# --------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------


def penalties(m: EpisodeMeasures, profile: QualityProfile) -> dict[str, dict[str, float]]:
    """Per-criterion ``value``, ``u`` and ``p`` — the audit trail for a verdict.

    Reported for every enabled criterion including the unmeasurable ones, so a
    silent ``nan`` is visible in the export rather than inferred from a missing
    column.
    """
    out: dict[str, dict[str, float]] = {}
    for criterion in profile.active():
        value = quantity_value(m, criterion.quantity, profile)
        out[criterion.quantity] = {
            "value": value,
            "u": criterion.u(value),
            "p": criterion.penalty(value),
            "cost": criterion.cost(value),
            "weight": criterion.weight,
        }
    return out


def combine_cost(terms: Sequence[tuple[float, float]]) -> float:
    """Total evidence against, in nats: ``sum(w * cost)``.

    The aggregation runs in nats and is exponentiated once at the end.  Doing it
    the other way — multiplying a dozen probabilities — costs both the ranking
    among the condemned episodes, whose product underflows to a shared zero, and
    the meaning of the weight, which in a product has to be applied as a power.
    """
    usable = [(w, c) for w, c in terms if np.isfinite(c) and w > 0]
    if not usable:
        return float("nan")
    return float(sum(w * c for w, c in usable))


def combine_noisy_or(terms: Sequence[tuple[float, float]]) -> float:
    """``prod (1 - p) ** w`` over the measurable terms; ``nan`` if none are.

    Takes ``(weight, cost)`` pairs, not ``(weight, p)`` — see
    :meth:`Criterion.cost` for why the costs are carried rather than the
    probabilities.
    """
    total_cost = combine_cost(terms)
    if not np.isfinite(total_cost):
        return float("nan")
    return float(math.exp(-total_cost))


def score_episode_absolute(
    m: EpisodeMeasures,
    profile: QualityProfile | None = None,
    semantic_score: float | None = None,
    semantic_note: str = "",
) -> EpisodeScore:
    """Score and decide against an absolute profile.

    Returns the same :class:`~.normalize.EpisodeScore` the percentile path
    returns, so the CSV export, the app and the visualiser need no changes.
    ``families`` carries ``1 - p`` per family for continuity with the existing
    columns, but nothing decides on it any more: the verdict is ``total``
    against the single threshold.
    """
    profile = profile or QualityProfile()
    detail = penalties(m, profile)

    scored = [c for c in profile.active() if c.action != "review"]
    severity = combine_cost([(c.weight, detail[c.quantity]["cost"]) for c in scored])
    # No measurable criterion is not the same as every criterion violated.  A
    # dataset missing the channels a profile is written against would otherwise
    # be rejected wholesale on evidence nobody ever collected; ``nan`` sends it
    # to review, where a human finds out why in one look.
    total = float(math.exp(-severity)) if np.isfinite(severity) and severity < 700 else (
        float("nan") if not np.isfinite(severity) else 0.0)

    score = EpisodeScore(
        episode=m.episode,
        total=total,
        families=_family_view(detail, profile),
        terms={k: dict(v) for k, v in detail.items()},
        weights={c.quantity: c.weight for c in profile.active()},
        flags=m.flags.to_dict(),
        semantic_score=semantic_score,
        semantic_note=semantic_note,
        severity=severity,
    )
    return decide_absolute(score, m, profile, detail)


#: Only for the ``families`` columns the existing report and app already print.
_FAMILY_OF = {
    "ldlj_wrist": "smoothness", "ldlj_arm": "smoothness", "ldlj_body": "smoothness",
    "acc_arm_rms": "acceleration", "acc_arm_p99": "acceleration",
    "acc_body_rms": "acceleration", "acc_body_p99": "acceleration",
    "acc_wrist_rms": "acceleration", "acc_wrist_p99": "acceleration",
    "contact_rate_hz": "contact", "max_base_tilt_deg": "contact",
    "idle_fraction": "timing", "grasp_excess": "timing",
    "duration_ratio": "timing", "duration_ratio_short": "timing",
    "video_clipped_fraction": "video", "video_brightness_dev": "video",
    "video_interframe_diff": "video", "video_sharpness_residual": "video",
    "video_contrast": "video",
}


def _family_view(detail: dict[str, dict[str, float]],
                 profile: QualityProfile) -> dict[str, float]:
    """The same noisy-OR restricted to each family — a breakdown, not a gate."""
    grouped: dict[str, list[tuple[float, float]]] = {}
    for criterion in profile.active():
        family = _FAMILY_OF.get(criterion.quantity, "other")
        grouped.setdefault(family, []).append(
            (criterion.weight, detail[criterion.quantity]["cost"]))
    return {name: combine_noisy_or(terms) for name, terms in grouped.items()}


def decide_absolute(
    score: EpisodeScore,
    m: EpisodeMeasures,
    profile: QualityProfile,
    detail: dict[str, dict[str, float]] | None = None,
) -> EpisodeScore:
    """Preconditions, then the semantic gate, then the one threshold."""
    detail = detail if detail is not None else penalties(m, profile)

    hard = [f"flag:{name}" for name in m.flags.raised() if name in m.flags.HARD]
    if hard:
        score.decision, score.reasons = "reject", hard
        return score

    if score.semantic_score is not None and score.semantic_score <= 0.0:
        note = f": {score.semantic_note}" if score.semantic_note else ""
        score.decision, score.reasons = "reject", [f"semantic:goal not reached{note}"]
        return score

    if not np.isfinite(score.total):
        score.decision = "review"
        score.reasons = ["no criterion measurable on this episode"]
        return score

    if score.total < profile.threshold:
        score.decision = "reject"
        score.reasons = [f"score {score.total:.3f} < {profile.threshold:g}"]
        score.reasons += _top_offenders(detail, profile)
        return score

    review: list[str] = [f"flag:{n}" for n in m.flags.raised() if n in m.flags.SOFT]
    for criterion in profile.active():
        if criterion.action != "review":
            continue
        p = detail[criterion.quantity]["p"]
        if np.isfinite(p) and p > 0.5:
            value = detail[criterion.quantity]["value"]
            review.append(f"{criterion.label or criterion.quantity} {value:.4g}"
                          + (f" {criterion.unit}" if criterion.unit else ""))
    if score.semantic_score is not None and score.semantic_score < 1.0:
        review.append(f"semantic {score.semantic_score:g}")

    score.decision = "review" if review else "accept"
    score.reasons = review
    return score


def _top_offenders(detail: dict[str, dict[str, float]], profile: QualityProfile,
                   limit: int = 3) -> list[str]:
    """The criteria that actually cost the episode its score.

    Ranked by ``w * -log(1 - p)``, the term's real contribution to the product,
    not by ``p`` alone — a weight-0.5 criterion at ``p = 0.9`` costs less than a
    weight-1.0 one at ``p = 0.7``, and a reason list that says otherwise sends
    the reader to the wrong knob.
    """
    ranked = []
    for criterion in profile.active():
        if criterion.action == "review":
            continue
        entry = detail.get(criterion.quantity)
        if entry is None or not np.isfinite(entry["cost"]) or entry["cost"] <= 0.0:
            continue
        ranked.append((criterion.weight * entry["cost"], criterion, entry))
    ranked.sort(key=lambda item: item[0], reverse=True)

    out = []
    for _, criterion, entry in ranked[:limit]:
        unit = f" {criterion.unit}" if criterion.unit else ""
        out.append(
            f"{criterion.label or criterion.quantity} {entry['value']:.4g}{unit} "
            f"(good {criterion.good:g}, bad {criterion.bad:g}, p={entry['p']:.2f})"
        )
    return out


# --------------------------------------------------------------------------
# Drift
# --------------------------------------------------------------------------


def drift_report(
    measures: Iterable[EpisodeMeasures],
    profile: QualityProfile,
) -> list[dict[str, Any]]:
    """How far a new dataset sits outside the anchors of a frozen profile.

    A frozen profile cannot tell a genuinely worse batch from a different robot,
    a different control rate or a re-aimed camera — both look like "everything
    got worse".  Neither can it tell you by adapting, which is what the
    percentile path does silently.  So it reports instead: per criterion, the
    share of episodes past ``good`` and past ``bad``, and the median ``u``.

    Read it as a domain-shift alarm.  A handful of criteria mildly over is a
    batch with problems; ``over_bad`` above ~0.3 on a criterion whose neighbours
    are clean usually means the anchor no longer describes this hardware, and
    the profile wants refitting rather than the batch wanting rejecting.
    """
    items = [m for m in measures if m.flags.valid]
    rows: list[dict[str, Any]] = []
    for criterion in profile.active():
        values = np.asarray(
            [quantity_value(m, criterion.quantity, profile) for m in items], dtype=float)
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            rows.append({"quantity": criterion.quantity, "label": criterion.label,
                         "n": 0, "median_u": float("nan"),
                         "over_good": float("nan"), "over_bad": float("nan")})
            continue
        us = np.asarray([criterion.u(v) for v in finite], dtype=float)
        rows.append({
            "quantity": criterion.quantity,
            "label": criterion.label or criterion.quantity,
            "n": int(finite.size),
            "median_value": float(np.median(finite)),
            "median_u": float(np.median(us)),
            "over_good": float(np.mean(us > 0.0)),
            "over_bad": float(np.mean(us >= 1.0)),
        })
    rows.sort(key=lambda r: (r["over_bad"] if np.isfinite(r["over_bad"]) else -1),
              reverse=True)
    return rows


def format_drift(rows: Sequence[dict[str, Any]], limit: int = 8) -> str:
    """The drift report as a short table for a terminal."""
    head = f"{'criterion':<34}{'n':>5}{'median u':>11}{'>good':>8}{'>bad':>8}"
    lines = [head, "-" * len(head)]
    for row in rows[:limit]:
        lines.append(
            f"{(row['label'] or row['quantity'])[:33]:<34}"
            f"{row['n']:>5}"
            f"{row['median_u']:>11.2f}"
            f"{row['over_good']:>8.0%}"
            f"{row['over_bad']:>8.0%}"
        )
    return "\n".join(lines)

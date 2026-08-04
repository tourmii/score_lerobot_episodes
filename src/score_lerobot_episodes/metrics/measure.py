"""Raw measurement layer — physical quantities and boolean flags, nothing else.

Nothing in this module maps anything onto ``[0, 1]`` and nothing here reads a
threshold.  That is deliberate and it is the main structural decision of the
metric suite: **measurement is separated from scoring**.

An arm RMS acceleration of 3.63 rad/s² is 3.63 rad/s² whatever robot produced
it; only the normalisation range and the accept threshold depend on the
platform and the task.  Keeping them apart means moving to another robot is a
change to the calibration table in :mod:`.normalize`, not a change to any
measurement here.

Five families, each aimed at a specific failure mode rather than at an abstract
notion of "quality":

============  =========================================================
family        what it measures
============  =========================================================
smoothness    trajectory *shape* — is the motion jerky (LDLJ)
acceleration  trajectory *magnitude* — mechanical stress (RMS and p99)
contact       collision events, inferred; there is no force/torque sensor
timing        duration, dead time, regrasp count
video         recording quality and integrity
============  =========================================================

Smoothness and acceleration are deliberately near-orthogonal: LDLJ is scale
invariant and so cannot say whether a trajectory is violent, while the
acceleration family is scale dependent and so cannot say whether it is jerky.
Their negative correlation on real data is the intended consequence of covering
two axes, not a redundancy.

Three time scales are covered by three different mechanisms, which is the
structural reason all three families are kept:

===================  ===========  ==========================
anomaly              duty cycle   caught by
===================  ===========  ==========================
single frame         < 0.5%       contact event measures
burst, 0.1-0.5 s     1-3%         acceleration p99
whole episode        ~100%        acceleration RMS, LDLJ
===================  ===========  ==========================
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .signals import EpisodeSignals

# np.trapz was renamed in numpy 2.0 and the project does not pin a major version.
_trapezoid = getattr(np, "trapezoid", None) or np.trapz

# --------------------------------------------------------------------------
# Measurement constants
# --------------------------------------------------------------------------
# These are properties of the *measurement*, not scoring thresholds: they decide
# whether a quantity is defined at all, not whether its value is good.

#: Below this wrist path length an episode has no trajectory to describe, so
#: every shape metric is undefined rather than excellent.
MIN_EEF_PATH_M = 0.30
#: Wrists closer than this are flagged for manual review as a self-collision.
SELF_COLLISION_M = 0.10
#: Idle threshold = max(floor, fraction x p95 speed).  The relative part adapts
#: to fast and slow episodes; the absolute floor breaks a pathological loop —
#: on a near-motionless episode p95 collapses, the relative threshold follows it
#: down, no frame counts as idle, and the worst episode scores perfectly.
IDLE_SPEED_FLOOR_RAD_S = 0.10
IDLE_SPEED_FRACTION = 0.05
#: Finger encoder count below which the hand reads closed, plus the hysteresis
#: band that stops a signal resting near the threshold from chattering.
HAND_CLOSED_COUNT = 300.0
HAND_HYSTERESIS_COUNT = 60.0
#: Only the four finger channels take part in the open/close test.  The
#: thumb-proximal-yaw channel sweeps its whole range on a parked hand and would
#: manufacture transitions that never happened.
HAND_FINGER_CHANNELS = (0, 1, 2, 3)
#: A grasp cycle needs at least one close and one release; fewer transitions
#: than this means it never happened, so "grasp quality" does not exist.
MIN_GRASP_TRANSITIONS = 2
#: Spike detection width, in robust standard deviations above the median.
SPIKE_N_MAD = 6.0
#: How many of the three contact signatures must agree for an event to count.
CONTACT_QUORUM = 2
#: Decoded/declared frame ratio below which the recording is truncated.
VIDEO_DECODE_RATIO = 0.95
#: Mean absolute inter-frame difference below which the video is frozen.
VIDEO_FREEZE_DIFF = 1.0


# --------------------------------------------------------------------------
# Primitives
# --------------------------------------------------------------------------


def log_dimensionless_jerk(x: np.ndarray, t: np.ndarray) -> float:
    """Log dimensionless jerk of a position-like trajectory.

    ``D = (T**3 / v_peak**2) * integral ||d3x/dt3||**2 dt``, ``LDLJ = -ln(D)``.

    The two normalisers make ``D`` dimensionless and invariant to both
    amplitude (``x -> a*x``) and time scale (``t -> t/b``), so LDLJ depends on
    the *shape* of the trajectory and nothing else — which is what "smooth"
    ought to mean.  Higher (closer to zero) is smoother.

    This is what RMS acceleration cannot do: it gives a slow careful episode and
    a fast jerky one the same value, because it mixes shape with magnitude.

    Returns ``nan`` — meaning *not measurable*, never zero — when the episode is
    too short, has no positive duration, or never reaches a non-zero peak speed.
    """
    x = np.atleast_2d(np.asarray(x, dtype=float))
    if x.shape[0] != len(t):
        x = x.T
    if x.shape[0] < 4 or x.size == 0:
        return float("nan")

    vel = np.gradient(x, t, axis=0)
    jerk = np.gradient(np.gradient(vel, t, axis=0), t, axis=0)

    duration = float(t[-1] - t[0])
    peak_speed = float(np.linalg.norm(vel, axis=1).max())
    if duration <= 0 or peak_speed <= 1e-9:
        return float("nan")

    integral = float(_trapezoid(np.sum(jerk**2, axis=1), t))
    if integral <= 0:
        return float("nan")
    return -float(np.log(duration**3 / peak_speed**2 * integral))


def robust_spikes(signal: np.ndarray, n_mad: float = SPIKE_N_MAD) -> np.ndarray:
    """Samples more than ``n_mad`` robust deviations above the median.

    ``threshold = median + n_mad * 1.4826 * MAD``.

    Median/MAD rather than mean/std because of masking: the very spikes this is
    meant to find would inflate a mean and a standard deviation, dragging the
    threshold up with them until the spikes look normal.  The median has a 50%
    breakdown point, so a minority of extreme samples cannot move it.
    """
    signal = np.asarray(signal, dtype=float)
    if signal.size == 0:
        return np.zeros(0, dtype=bool)
    med = float(np.median(signal))
    mad = float(np.median(np.abs(signal - med)))
    if mad <= 1e-12:
        return np.zeros_like(signal, dtype=bool)
    return signal > med + n_mad * 1.4826 * mad


def event_starts(mask: np.ndarray) -> np.ndarray:
    """Start index of every contiguous ``True`` run in ``mask``."""
    mask = np.asarray(mask, dtype=bool)
    if mask.size == 0 or not mask.any():
        return np.zeros(0, dtype=int)
    edges = np.flatnonzero(np.diff(mask.astype(np.int8)) == 1) + 1
    return np.concatenate(([0], edges)) if mask[0] else edges


def count_events(mask: np.ndarray) -> int:
    """Number of contiguous ``True`` runs, so an 8-frame impact counts once."""
    return int(len(event_starts(mask)))


def binary_state(x: np.ndarray, threshold: float, hysteresis: float = 0.0) -> np.ndarray:
    """Threshold ``x`` into ``True`` below / ``False`` above, with hysteresis.

    The band keeps a signal loitering at the threshold from producing a run of
    phantom transitions; the state only flips once ``x`` clears the far side.
    """
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return np.zeros(0, dtype=bool)
    lo, hi = threshold - hysteresis, threshold + hysteresis
    state = np.empty(x.size, dtype=bool)
    current = bool(x[0] < threshold)
    for i, value in enumerate(x):
        if current and value > hi:
            current = False
        elif not current and value < lo:
            current = True
        state[i] = current
    return state


def _rms(a: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.asarray(a, dtype=float) ** 2))) if np.size(a) else float("nan")


def _p99(a: np.ndarray) -> float:
    return float(np.percentile(np.abs(np.asarray(a, dtype=float)), 99)) if np.size(a) else float("nan")


# --------------------------------------------------------------------------
# Results
# --------------------------------------------------------------------------


@dataclass
class Flags:
    """Preconditions, evaluated before any distribution is looked at.

    An episode failing a hard flag must be excluded from the percentile fit as
    well as from the output set.  Every motion metric is a decreasing function
    of some derivative norm, so a motionless recording maxes out all of them at
    once — leaving one in the calibration sample drags both ends of every
    normalisation range and corrupts the thresholds for every other episode.
    """

    degenerate: bool = False             # wrist path too short: no real motion
    grasp_incomplete: bool = False       # fewer than 2 hand transitions
    video_unreadable: bool = False       # container will not decode
    video_truncated: bool = False        # fewer frames than declared
    self_collision_suspect: bool = False # wrists came too close; review by hand
    video_frozen: bool = False           # duplicated frames, review by hand

    #: Hard flags disqualify; soft flags request a human look.
    HARD = ("degenerate", "grasp_incomplete", "video_unreadable", "video_truncated")
    SOFT = ("self_collision_suspect", "video_frozen")

    @property
    def valid(self) -> bool:
        return not any(getattr(self, name) for name in self.HARD)

    @property
    def needs_review(self) -> bool:
        return any(getattr(self, name) for name in self.SOFT)

    def raised(self) -> list[str]:
        return [n for n in self.HARD + self.SOFT if getattr(self, n)]

    def to_dict(self) -> dict[str, bool]:
        return {n: bool(getattr(self, n)) for n in self.HARD + self.SOFT}


@dataclass
class VideoMeasures:
    """Raw camera quantities.  ``sharpness`` is *not* a quality figure on its own."""

    decodable: bool = False
    frames_declared: int = 0
    frames_decoded: int = 0
    decode_ratio: float = 0.0
    sharpness: float = float("nan")          # variance of the Laplacian
    brightness: float = float("nan")         # mean grey level, 0-255
    contrast: float = float("nan")           # std of grey level
    clipped_fraction: float = float("nan")   # px > 250 or < 5
    interframe_diff: float = float("nan")    # mean |frame - previous frame|

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class EpisodeMeasures:
    """Every raw quantity for one episode, in physical units, plus the flags.

    ``nan`` means *not measurable* (channel absent, episode too short) and must
    be re-weighted around by the scoring layer, never read as zero.
    """

    episode: int | None = None
    video_path: str | None = None

    # --- geometry / bookkeeping ---
    duration_s: float = float("nan")
    n_frames: int = 0
    fps: float = float("nan")
    working_side: str = "right"
    eef_path_m: float = float("nan")

    # --- smoothness: trajectory shape (higher = smoother) ---
    ldlj_wrist: float = float("nan")   # Cartesian, what a policy must reproduce
    ldlj_arm: float = float("nan")     # joint space, sees null-space shake
    ldlj_body: float = float("nan")    # legs + waist, sees induced postural sway

    # --- acceleration: magnitude (rad/s^2, wrist in m/s^2) ---
    acc_arm_rms: float = float("nan")
    acc_arm_p99: float = float("nan")
    acc_body_rms: float = float("nan")
    acc_body_p99: float = float("nan")
    acc_wrist_rms: float = float("nan")
    acc_wrist_p99: float = float("nan")
    acc_group_rms: dict[str, float] = field(default_factory=dict)

    # --- contact: inferred events ---
    contact_events: int = 0
    contact_rate_hz: float = 0.0
    impact_events: int = 0
    base_disturbance_events: int = 0
    tracking_divergence_events: int = 0
    contact_phases: list[float] = field(default_factory=list)
    min_wrist_separation_m: float = float("nan")
    max_base_tilt_deg: float = float("nan")

    # --- timing ---
    idle_fraction: float = float("nan")
    idle_threshold_rad_s: float = float("nan")
    grasp_transitions: int = -1        # -1 = no hand telemetry
    arm_speed_rms: float = float("nan")

    # --- video ---
    video: VideoMeasures = field(default_factory=VideoMeasures)

    flags: Flags = field(default_factory=Flags)
    series: dict[str, np.ndarray] = field(default_factory=dict)

    def to_row(self) -> dict[str, Any]:
        """Flat dict for a DataFrame; series are dropped."""
        row: dict[str, Any] = {
            k: v for k, v in asdict(self).items()
            if k not in ("acc_group_rms", "contact_phases", "video", "flags", "series")
        }
        row.update({f"acc_{name}_rms": v for name, v in self.acc_group_rms.items()})
        row.update({f"video_{k}": v for k, v in self.video.to_dict().items()})
        row.update(self.flags.to_dict())
        row["valid"] = self.flags.valid
        row["needs_review"] = self.flags.needs_review
        return row


# --------------------------------------------------------------------------
# Family 1 — smoothness
# --------------------------------------------------------------------------


def measure_smoothness(sig: EpisodeSignals) -> dict[str, Any]:
    """LDLJ on the three signal chains.

    Each chain catches a fault the other two are blind to:

    * **wrist (Cartesian)** — the quantity a downstream policy has to
      reproduce, and invariant to how the redundant joints are configured.
    * **arm joints** — null-space shake: the joints oscillate while the wrist
      stands still, which is completely invisible in the Cartesian chain.
    * **body joints** — postural sway induced by the arm motion.  Nothing
      commands this chain during a standing task, so anything measured here is
      pure reaction.
    """
    arm_idx, body_idx = sig.arm_indices, sig.body_indices
    side = sig.working_side

    out = {
        "ldlj_arm": log_dimensionless_jerk(sig.q[:, arm_idx], sig.t) if arm_idx else float("nan"),
        "ldlj_body": log_dimensionless_jerk(sig.q[:, body_idx], sig.t) if body_idx else float("nan"),
        "ldlj_wrist": (
            log_dimensionless_jerk(sig.wrist_pos[side], sig.t)
            if side in sig.wrist_pos else float("nan")
        ),
        "working_side": side,
    }
    return out


# --------------------------------------------------------------------------
# Family 2 — acceleration
# --------------------------------------------------------------------------


def measure_acceleration(sig: EpisodeSignals, keep_series: bool = False) -> dict[str, Any]:
    """RMS and 99th percentile acceleration, per chain and per named group.

    Both statistics are reported because they answer different questions on
    different time scales.  RMS measures sustained load; p99 answers "was there
    a single jolt".  p99 reacts as soon as a burst covers more than ~1% of the
    frames, whereas RMS only moves once the burst energy is comparable to the
    whole episode's baseline energy — so an 8-frame spike that RMS barely
    notices is plainly visible to p99, and a uniformly aggressive episode is the
    other way round.

    The per-group breakdown exists to attribute a bad number to a specific limb.
    """
    out: dict[str, Any] = {}
    series: dict[str, np.ndarray] = {}

    for label, groups in (("arm", sig.arm_groups), ("body", sig.body_groups)):
        idx = sig.joint_indices(groups)
        if not idx:
            continue
        a = sig.acc[:, idx]
        out[f"acc_{label}_rms"] = _rms(a)
        out[f"acc_{label}_p99"] = _p99(a)
        if keep_series:
            series[f"acc_{label}"] = np.sqrt(np.mean(a**2, axis=1))

    # Per named group, for attribution.
    group_rms = {}
    for name in sig.groups:
        idx = sig.joint_indices([name])
        if idx:
            group_rms[name] = _rms(sig.acc[:, idx])
    out["acc_group_rms"] = group_rms

    side = sig.working_side
    if side in sig.wrist_acc:
        mag = np.linalg.norm(sig.wrist_acc[side], axis=1)
        out["acc_wrist_rms"] = _rms(mag)
        out["acc_wrist_p99"] = float(np.percentile(mag, 99))
        if keep_series:
            series["acc_wrist"] = mag
            series["speed_wrist"] = np.linalg.norm(sig.wrist_vel[side], axis=1)

    arm_idx = sig.arm_indices
    out["arm_speed_rms"] = _rms(sig.vel[:, arm_idx]) if arm_idx else float("nan")

    if keep_series:
        out["series"] = series
    return out


# --------------------------------------------------------------------------
# Family 3 — contact
# --------------------------------------------------------------------------


def measure_contact(sig: EpisodeSignals, keep_series: bool = False) -> dict[str, Any]:
    """Contact events, inferred from their dynamic consequences.

    There is no force/torque sensing anywhere in this data, so contact has to be
    read off three independent signatures — independent in the sense that each
    has its own false-alarm mode, uncorrelated with the other two:

    ==================  ==============================================  ====================
    signature           physical mechanism                              its own false alarm
    ==================  ==============================================  ====================
    impact              momentum change: a wrist acceleration spike     deliberate fast
                        while decelerating and still moving fast        motion
    base disturbance    reaction force travelling up the chain into     ordinary balance
                        the floating base                               correction
    tracking residual   the controller pushing into an obstacle: a      step command
                        spike in the setpoint-minus-state residual
    ==================  ==============================================  ====================

    An event is confirmed only when at least two signatures agree
    (:data:`CONTACT_QUORUM`), because one channel spiking alone is usually just
    a fast intentional motion.

    Events are counted as contiguous runs, so an 8-frame impact counts once, and
    the quantity carried forward is the **rate** (events per second) rather than
    the count — a rate is invariant to episode length.

    The tracking residual is detrended per joint first.  ``action.wbc`` is not a
    position command the robot must reproduce: it is the setpoint of a compliant
    controller, which needs a standing steady-state error to generate its
    gravity-compensating torque.  That leaves a large constant per-joint offset
    (one locked waist joint sits -0.49 rad off a setpoint it never tracks), and
    sweeping the lag 0-7 frames moves the residual RMS by under 2%, confirming
    it is structural rather than a latency artefact.  Without the median
    subtraction this measures how stiff the controller is instead of where its
    compliance changed.

    .. note::
       Pick-and-place *requires* contact — grasping the object and setting it
       down are contacts — and nothing in the data distinguishes intended from
       unintended.  Treat the output as "worth reviewing", not as a reason to
       drop or rank an episode.
    """
    n = sig.n_frames
    votes = np.zeros(n, dtype=int)
    out: dict[str, Any] = {}
    series: dict[str, np.ndarray] = {}

    side = sig.working_side
    if side in sig.wrist_vel and n > 1:
        speed = np.linalg.norm(sig.wrist_vel[side], axis=1)
        acc_mag = np.linalg.norm(sig.wrist_acc[side], axis=1)
        decelerating = np.gradient(speed, sig.t) < 0
        moving = speed > max(float(np.percentile(speed, 25)), 1e-3)
        impact = robust_spikes(acc_mag) & decelerating & moving
        votes += impact.astype(int)
        out["impact_events"] = count_events(impact)
        if keep_series:
            series["contact_impact"] = impact

    if sig.gravity is not None and n > 2:
        g = np.asarray(sig.gravity, dtype=float)
        g_acc = np.gradient(np.gradient(g, sig.t, axis=0), sig.t, axis=0)
        base = robust_spikes(np.linalg.norm(g_acc, axis=1))
        votes += base.astype(int)
        out["base_disturbance_events"] = count_events(base)
        norm = np.linalg.norm(g, axis=1)
        tilt = np.degrees(np.arccos(np.clip(-g[:, 2] / np.maximum(norm, 1e-9), -1.0, 1.0)))
        out["max_base_tilt_deg"] = float(tilt.max())
        if keep_series:
            series["contact_base"] = base
            series["base_tilt_deg"] = tilt

    if sig.action is not None and np.shape(sig.action) == np.shape(sig.q):
        arm_idx = sig.arm_indices
        if arm_idx:
            residual = sig.action[:, arm_idx] - sig.q[:, arm_idx]
            residual = residual - np.median(residual, axis=0, keepdims=True)
            r = np.linalg.norm(residual, axis=1)
            track = robust_spikes(r)
            votes += track.astype(int)
            out["tracking_divergence_events"] = count_events(track)
            if keep_series:
                series["contact_tracking"] = track
                series["tracking_residual"] = r

    confirmed = votes >= CONTACT_QUORUM
    starts = event_starts(confirmed)
    out["contact_events"] = int(len(starts))
    out["contact_rate_hz"] = float(len(starts) / sig.duration) if sig.duration > 0 else 0.0
    out["contact_phases"] = [float(i / max(n - 1, 1)) for i in starts]

    if "left" in sig.wrist_pos and "right" in sig.wrist_pos:
        d = np.linalg.norm(sig.wrist_pos["right"] - sig.wrist_pos["left"], axis=1)
        out["min_wrist_separation_m"] = float(d.min())
        if keep_series:
            series["wrist_separation"] = d

    if keep_series:
        series["contact_confirmed"] = confirmed
        out["series"] = series
    return out


# --------------------------------------------------------------------------
# Family 4 — timing
# --------------------------------------------------------------------------


def measure_timing(sig: EpisodeSignals, keep_series: bool = False) -> dict[str, Any]:
    """Duration, dead time and regrasp count.

    Raw duration is a badly confounded variable — long can mean hesitation or
    regrasping (a real defect) or simply slow and clean; short can mean an
    efficient run or a truncated recording — so it is decomposed by cause into
    three separate quantities and each is judged on its own.

    * **duration** is reported in seconds and left alone.  It only becomes
      meaningful against the rest of the dataset ("11.8 seconds is good" is not
      a statement), so the comparison happens in the scoring layer.
    * **idle fraction** uses an adaptive velocity threshold: a small fraction of
      the episode's own peak speed, floored by an absolute value.  The relative
      part follows fast and slow episodes; the floor stops a near-motionless
      episode from declaring none of its frames idle.
    * **grasp transitions** come from the finger encoders, whose distribution is
      strongly bimodal so the threshold is not sensitive.  Only the four finger
      channels take part — the thumb-proximal-yaw channel sweeps its entire
      range while the hand is parked and would fabricate transitions.
    """
    out: dict[str, Any] = {
        "duration_s": sig.duration,
        "n_frames": sig.n_frames,
        "fps": sig.fps,
    }
    series: dict[str, np.ndarray] = {}

    speed = sig.joint_speed
    threshold = max(
        IDLE_SPEED_FLOOR_RAD_S,
        IDLE_SPEED_FRACTION * float(np.percentile(speed, 95)) if speed.size else 0.0,
    )
    idle = speed < threshold
    out["idle_threshold_rad_s"] = float(threshold)
    out["idle_fraction"] = float(idle.mean()) if speed.size else float("nan")

    hand = sig.working_hand
    if hand is None:
        out["grasp_transitions"] = -1
    else:
        hand = np.asarray(hand, dtype=float)
        channels = [c for c in HAND_FINGER_CHANNELS if c < hand.shape[1]]
        finger = hand[:, channels].mean(axis=1) if channels else hand.mean(axis=1)
        closed = binary_state(finger, HAND_CLOSED_COUNT, HAND_HYSTERESIS_COUNT)
        out["grasp_transitions"] = int(np.count_nonzero(np.diff(closed.astype(np.int8))))
        if keep_series:
            series["hand_closed"] = closed
            series["hand_finger_mean"] = finger

    if keep_series:
        series["joint_speed"] = speed
        series["idle"] = idle
        out["series"] = series
    return out


# --------------------------------------------------------------------------
# Family 5 — video
# --------------------------------------------------------------------------


def measure_video(
    video_path: str | Path,
    n_samples: int = 16,
    diff_width: int = 160,
    keep_series: bool = False,
) -> tuple[VideoMeasures, dict[str, np.ndarray]]:
    """Recording quality and integrity.

    Appearance statistics (sharpness, brightness, contrast, clipping) are
    sampled at ``n_samples`` frames; the inter-frame difference is computed on
    every consecutive pair, downscaled to ``diff_width``, because a frozen or
    duplicated frame only shows up between neighbours.

    Sharpness is measured but **not interpreted here**.  Laplacian variance
    falls for two unrelated reasons — genuine defocus, which is the fault worth
    catching, and motion blur, which means the robot was working.  On real data
    it correlates about -0.46 with joint velocity, so read directly it ranks the
    least active episodes as the sharpest.  Removing that dependence needs the
    whole dataset and therefore belongs to the scoring layer.

    Measuring video is by orders of magnitude the slowest of the five families;
    skip it while iterating.
    """
    import cv2  # imported lazily: everything else here is numpy-only

    out = VideoMeasures()
    series: dict[str, np.ndarray] = {}

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        cap.release()
        return out, series

    declared = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(declared // n_samples, 1) if declared > 0 else 25

    sharp, bright, contrast, clipped, diffs = [], [], [], [], []
    prev_small = None
    index, decoded = -1, 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        index += 1
        decoded += 1
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        scale = diff_width / max(gray.shape[1], 1)
        small = (
            cv2.resize(gray, (diff_width, max(int(gray.shape[0] * scale), 1)))
            if scale < 1.0 else gray
        ).astype(np.int16)
        if prev_small is not None:
            diffs.append(float(np.abs(small - prev_small).mean()))
        prev_small = small

        if index % step:
            continue
        sharp.append(float(cv2.Laplacian(gray, cv2.CV_64F).var()))
        bright.append(float(gray.mean()))
        contrast.append(float(gray.std()))
        clipped.append(float(((gray > 250) | (gray < 5)).mean()))
    cap.release()

    out.frames_declared = declared
    out.frames_decoded = decoded
    out.decode_ratio = float(decoded / declared) if declared > 0 else (1.0 if decoded else 0.0)
    if not sharp:
        return out, series

    out.decodable = True
    out.sharpness = float(np.mean(sharp))
    out.brightness = float(np.mean(bright))
    out.contrast = float(np.mean(contrast))
    out.clipped_fraction = float(np.mean(clipped))
    out.interframe_diff = float(np.mean(diffs)) if diffs else float("nan")

    if keep_series and diffs:
        series["interframe_diff"] = np.asarray([diffs[0]] + diffs, dtype=float)
    return out, series


# --------------------------------------------------------------------------
# Assembly
# --------------------------------------------------------------------------


def measure_episode(
    sig: EpisodeSignals,
    video_path: str | Path | None = None,
    keep_series: bool = False,
    video_samples: int = 16,
) -> EpisodeMeasures:
    """Run all five families on one episode and collect the raw quantities.

    Every value returned carries a physical unit or is a count; nothing is
    normalised and nothing is compared to a threshold that would make it "good".
    Pass the result to :mod:`.normalize` for that.
    """
    m = EpisodeMeasures(episode=sig.episode, video_path=str(video_path) if video_path else None)
    series: dict[str, np.ndarray] = {}

    if keep_series:
        series["t"] = np.asarray(sig.t, dtype=float)

    smooth = measure_smoothness(sig)
    m.ldlj_arm = smooth["ldlj_arm"]
    m.ldlj_body = smooth["ldlj_body"]
    m.ldlj_wrist = smooth["ldlj_wrist"]
    m.working_side = smooth["working_side"]
    m.eef_path_m = sig.eef_path_length(m.working_side) if sig.wrist_pos else float("nan")

    acc = measure_acceleration(sig, keep_series)
    series.update(acc.pop("series", {}))
    m.acc_group_rms = acc.pop("acc_group_rms", {})
    for key, value in acc.items():
        setattr(m, key, value)

    contact = measure_contact(sig, keep_series)
    series.update(contact.pop("series", {}))
    for key, value in contact.items():
        setattr(m, key, value)

    timing = measure_timing(sig, keep_series)
    series.update(timing.pop("series", {}))
    for key, value in timing.items():
        setattr(m, key, value)

    if video_path is not None:
        m.video, video_series = measure_video(
            video_path, n_samples=video_samples, keep_series=keep_series
        )
        series.update(video_series)

    m.flags = raise_flags(m, sig)
    if keep_series:
        m.series = series
    return m


def raise_flags(m: EpisodeMeasures, sig: EpisodeSignals | None = None) -> Flags:
    """Evaluate the preconditions from already-measured quantities."""
    flags = Flags()

    if np.isfinite(m.eef_path_m):
        flags.degenerate = m.eef_path_m < MIN_EEF_PATH_M
    elif sig is not None:
        # No Cartesian channel: fall back to "did any joint ever move".
        flags.degenerate = bool(sig.joint_speed.max() < 1e-3) if sig.n_frames else True
    if not np.isfinite(m.duration_s) or m.duration_s <= 0:
        flags.degenerate = True

    flags.grasp_incomplete = 0 <= m.grasp_transitions < MIN_GRASP_TRANSITIONS

    if np.isfinite(m.min_wrist_separation_m):
        flags.self_collision_suspect = m.min_wrist_separation_m < SELF_COLLISION_M

    if m.video_path is not None:
        flags.video_unreadable = not m.video.decodable
        flags.video_truncated = m.video.decodable and m.video.decode_ratio < VIDEO_DECODE_RATIO
        flags.video_frozen = (
            m.video.decodable
            and np.isfinite(m.video.interframe_diff)
            and m.video.interframe_diff < VIDEO_FREEZE_DIFF
        )

    return flags

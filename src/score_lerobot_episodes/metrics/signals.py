"""Signal substrate for one episode.

This module owns nothing but the *data*: it loads an episode, works out which
joint channels are alive, and computes every derivative exactly once so the
measurement functions in :mod:`.measure` never recompute them.

Two layout facts drive the design and both are read from the dataset rather
than hard-coded, so the metrics survive a change of robot:

* **Dead channels.** The Unitree G1 publishes 43 state dimensions but 16 are
  structurally zero (the unwired hand slots plus two locked waist joints; the
  hands report on ``observation.*_hand_q`` instead).  An RMS pooled over all 43
  is deflated by ``sqrt(27/43) ~ 0.79``, i.e. -21%, and that factor is
  robot-specific — thresholds fitted with it do not transfer.  ``active`` masks
  the dead dimensions off, and it is derived per episode.
* **Two chains with different roles.** The arms receive the task commands; the
  legs and waist receive none during a standing task, so everything measured
  there is *reaction*.  That makes the body chain an indirect sensor for upper
  body motion quality, which no other group provides.  ``split_chains`` labels
  the groups by name so both roles stay separable on another layout.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# --------------------------------------------------------------------------
# Layout
# --------------------------------------------------------------------------

#: Index ranges into ``observation.state`` for the Unitree G1.  Used only when
#: the dataset ships no ``meta/modality.json``; prefer :func:`groups_from_modality`.
JOINT_GROUPS: dict[str, tuple[int, int]] = {
    "left_leg": (0, 6),
    "right_leg": (6, 12),
    "waist": (12, 15),
    "left_arm": (15, 22),
    "right_arm": (29, 36),
}

#: Substrings that put a named joint group into the commanded (arm) chain or
#: the reactive (body) chain.
ARM_TOKENS = ("arm", "shoulder", "elbow", "wrist")
BODY_TOKENS = ("leg", "waist", "torso", "hip", "knee", "ankle", "spine")

#: Slices into ``observation.eef_state`` (14 dims: pos3 + quat4 per wrist).
EEF_SLICES = {
    "left_wrist_pos": slice(0, 3),
    "left_wrist_quat": slice(3, 7),
    "right_wrist_pos": slice(7, 10),
    "right_wrist_quat": slice(10, 14),
}

#: A channel is considered alive when its standard deviation over the episode
#: clears this.  Well below encoder noise, well above exact structural zero.
ACTIVE_STD_RAD = 1e-6


def split_chains(groups: dict[str, tuple[int, int]]) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Split named joint groups into ``(arm_groups, body_groups)`` by name.

    Anything that matches neither token list is left out of both chains rather
    than guessed into one, so an unrecognised group never silently pollutes a
    chain-level RMS.
    """
    arm = tuple(n for n in groups if any(tok in n.lower() for tok in ARM_TOKENS))
    body = tuple(n for n in groups if any(tok in n.lower() for tok in BODY_TOKENS))
    return arm, body


def groups_from_modality(path: str | Path) -> dict[str, tuple[int, int]]:
    """Read joint groups from a LeRobot ``meta/modality.json``.

    Only entries that actually index into ``observation.state`` are returned;
    the ones carrying ``original_key`` describe a different column (hands,
    wrist poses, gravity) and are handled by their own fields.
    """
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    state = data.get("state", {})
    return {
        name: (int(spec["start"]), int(spec["end"]))
        for name, spec in state.items()
        if "original_key" not in spec and "start" in spec and "end" in spec
    }


# --------------------------------------------------------------------------
# Container
# --------------------------------------------------------------------------


@dataclass
class EpisodeSignals:
    """Time series for one episode, with derivatives computed once.

    Only ``t`` and ``q`` are required.  Every other channel is optional and the
    measurement functions degrade to whatever is present rather than failing.
    """

    t: np.ndarray                          # (N,)    seconds
    q: np.ndarray                          # (N, D)  joint positions, rad
    action: np.ndarray | None = None       # (N, D)  whole-body-control setpoint
    eef: np.ndarray | None = None          # (N, 14) wrist poses
    gravity: np.ndarray | None = None      # (N, 3)  gravity in base frame
    left_hand: np.ndarray | None = None    # (N, 6)  finger encoder counts
    right_hand: np.ndarray | None = None   # (N, 6)  finger encoder counts
    groups: dict[str, tuple[int, int]] = field(default_factory=lambda: dict(JOINT_GROUPS))
    episode: int | None = None
    video_path: str | None = None

    def __post_init__(self) -> None:
        self.t = np.asarray(self.t, dtype=float).reshape(-1)
        self.q = np.atleast_2d(np.asarray(self.q, dtype=float))
        if self.q.shape[0] != len(self.t) and self.q.shape[1] == len(self.t):
            self.q = self.q.T

        self.n_frames = int(len(self.t))
        self.dt = float(np.median(np.diff(self.t))) if self.n_frames > 1 else 0.0
        self.duration = float(self.t[-1] - self.t[0]) if self.n_frames > 1 else 0.0
        self.fps = 1.0 / self.dt if self.dt > 0 else 0.0

        self.active = self.q.std(axis=0) > ACTIVE_STD_RAD

        if self.n_frames > 1:
            self.vel = np.gradient(self.q, self.t, axis=0)
            self.acc = np.gradient(self.vel, self.t, axis=0)
        else:
            self.vel = np.zeros_like(self.q)
            self.acc = np.zeros_like(self.q)

        self.arm_groups, self.body_groups = split_chains(self.groups)

        self.wrist_pos: dict[str, np.ndarray] = {}
        self.wrist_vel: dict[str, np.ndarray] = {}
        self.wrist_acc: dict[str, np.ndarray] = {}
        if self.eef is not None and self.n_frames > 1:
            eef = np.asarray(self.eef, dtype=float)
            for side in ("left", "right"):
                sl = EEF_SLICES[f"{side}_wrist_pos"]
                if eef.shape[1] < sl.stop:
                    continue
                pos = eef[:, sl]
                vel = np.gradient(pos, self.t, axis=0)
                self.wrist_pos[side] = pos
                self.wrist_vel[side] = vel
                self.wrist_acc[side] = np.gradient(vel, self.t, axis=0)

    # --- joint selection -------------------------------------------------

    def joint_indices(self, groups) -> list[int]:
        """Indices of the *active* joints belonging to ``groups``."""
        idx: list[int] = []
        width = self.q.shape[1]
        for name in groups:
            span = self.groups.get(name)
            if span is None:
                continue
            start, end = span
            idx.extend(i for i in range(start, min(end, width)) if self.active[i])
        return idx

    @property
    def arm_indices(self) -> list[int]:
        return self.joint_indices(self.arm_groups)

    @property
    def body_indices(self) -> list[int]:
        return self.joint_indices(self.body_groups)

    # --- end effector ----------------------------------------------------

    def eef_path_length(self, side: str) -> float:
        """Cartesian path length of one wrist, in metres.

        The degeneracy test uses this rather than joint velocity or joint range
        because both of those are confounded: null-space motion makes the joints
        move while the wrist stays put, and an in-place oscillation has a large
        joint range but zero net displacement.
        """
        pos = self.wrist_pos.get(side)
        if pos is None or len(pos) < 2:
            return 0.0
        return float(np.linalg.norm(np.diff(pos, axis=0), axis=1).sum())

    @property
    def working_side(self) -> str:
        """Whichever wrist actually performed the task, by path length."""
        if not self.wrist_pos:
            return "right"
        return max(self.wrist_pos, key=self.eef_path_length)

    @property
    def working_hand(self) -> np.ndarray | None:
        """Finger encoders for the working arm, or any hand channel available."""
        hands = {"left": self.left_hand, "right": self.right_hand}
        preferred = hands.get(self.working_side)
        if preferred is not None:
            return preferred
        return next((h for h in hands.values() if h is not None), None)

    @property
    def joint_speed(self) -> np.ndarray:
        """Per-frame Euclidean joint speed over the active channels, rad/s."""
        if not self.active.any():
            return np.zeros(self.n_frames)
        return np.linalg.norm(self.vel[:, self.active], axis=1)

    @property
    def is_degenerate(self) -> bool:
        """Whether nothing meaningful happened — an idle or reset recording.

        Kept for callers written against the single-tier API.  The current home
        of this test is the ``degenerate`` flag on
        :class:`~score_lerobot_episodes.metrics.measure.Flags`, which is
        evaluated alongside the other preconditions.
        """
        from .measure import MIN_EEF_PATH_M  # deferred: measure imports this module

        if self.duration <= 0:
            return True
        if self.wrist_pos:
            return self.eef_path_length(self.working_side) < MIN_EEF_PATH_M
        return bool(self.joint_speed.max() < 1e-3)


# --------------------------------------------------------------------------
# Constructors
# --------------------------------------------------------------------------


def signals_from_dataframe(
    df,
    groups: dict[str, tuple[int, int]] | None = None,
    episode: int | None = None,
    video_path: str | None = None,
) -> EpisodeSignals:
    """Build :class:`EpisodeSignals` from an episode parquet frame."""

    def stack(col):
        if col not in df.columns:
            return None
        return np.vstack(df[col].to_numpy())

    return EpisodeSignals(
        t=np.asarray(df["timestamp"], dtype=float).reshape(-1),
        q=stack("observation.state"),
        action=stack("action.wbc"),
        eef=stack("observation.eef_state"),
        gravity=stack("observation.projected_gravity"),
        left_hand=stack("observation.left_hand_q"),
        right_hand=stack("observation.right_hand_q"),
        groups=dict(groups or JOINT_GROUPS),
        episode=episode,
        video_path=video_path,
    )


def signals_from_states(sts, acts=None, groups=None) -> EpisodeSignals:
    """Build :class:`EpisodeSignals` from the pipeline's list-of-dict states."""

    def stack(key):
        if not sts or key not in sts[0] or sts[0][key] is None:
            return None
        return np.asarray([np.asarray(st[key], dtype=float) for st in sts])

    return EpisodeSignals(
        t=np.asarray([st["t"] for st in sts], dtype=float),
        q=np.asarray([st["q"] for st in sts], dtype=float),
        action=np.asarray(acts, dtype=float) if acts is not None else None,
        eef=stack("eef"),
        gravity=stack("gravity"),
        left_hand=stack("left_hand"),
        right_hand=stack("right_hand"),
        groups=dict(groups or JOINT_GROUPS),
    )


def states_from_dataframe(df) -> tuple[list[dict], np.ndarray | None]:
    """Convert an episode frame into ``(states, actions)`` for the scorer pipeline.

    Carries the humanoid channels through; ``organize_by_episode`` emits only
    ``q``/``t``, which silently drops the wrist, gravity and hand signals.
    """

    def col(name):
        return np.vstack(df[name].to_numpy()) if name in df.columns else None

    q = col("observation.state")
    t = np.asarray(df["timestamp"], dtype=float).reshape(-1)
    eef, grav = col("observation.eef_state"), col("observation.projected_gravity")
    lh, rh = col("observation.left_hand_q"), col("observation.right_hand_q")

    states = [
        {
            "q": q[i],
            "t": t[i],
            "eef": None if eef is None else eef[i],
            "gravity": None if grav is None else grav[i],
            "left_hand": None if lh is None else lh[i],
            "right_hand": None if rh is None else rh[i],
        }
        for i in range(len(t))
    ]
    return states, col("action.wbc")

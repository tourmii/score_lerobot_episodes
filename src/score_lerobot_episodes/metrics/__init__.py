"""Episode-quality metrics for teleoperated humanoid datasets.

Four layers, each usable on its own:

``signals``   load an episode, mask the dead channels, differentiate once.
``measure``   raw physical quantities and boolean flags.  No thresholds.
``normalize`` map onto ``[0, 1]`` against a fitted calibration, then decide.
``semantic``  did the episode actually achieve its task?  (vision-language model)
``visualize`` play the measured signals alongside the episode video.

The split between the first two and the third is the point of the design: an
arm RMS acceleration of 3.63 rad/s² is 3.63 rad/s² on any robot, and only the
normalisation range and the accept threshold are platform-specific.  Moving to
another robot is a refit of :class:`Calibration`, not an edit to any measurement.

Typical use::

    import glob, pandas as pd
    from score_lerobot_episodes.metrics import (
        signals_from_dataframe, measure_episode, Calibration, score_episode)

    measures = []
    for path in sorted(glob.glob("data/chunk-000/*.parquet")):
        sig = signals_from_dataframe(pd.read_parquet(path))
        measures.append(measure_episode(sig, video_path=None))

    calib = Calibration.fit(measures)             # excludes invalid episodes
    scores = [score_episode(m, calib) for m in measures]
"""

from .measure import (
    EpisodeMeasures,
    Flags,
    VideoMeasures,
    binary_state,
    count_events,
    event_starts,
    log_dimensionless_jerk,
    measure_acceleration,
    measure_contact,
    measure_episode,
    measure_smoothness,
    measure_timing,
    measure_video,
    raise_flags,
    robust_spikes,
)
from .normalize import (
    Calibration,
    RULE_QUANTITIES,
    Rule,
    rule_value,
    suggest_rules,
    violations,
    DEFAULT_BRACKETS,
    DEFAULT_WEIGHTS,
    EpisodeScore,
    Policy,
    decide,
    exp_decay,
    gauss_asymmetric,
    ramp,
    robust_z,
    score_acceleration,
    score_contact,
    score_episode,
    score_smoothness,
    score_timing,
    score_video,
)
from .signals import (
    ARM_TOKENS,
    BODY_TOKENS,
    EEF_SLICES,
    EpisodeSignals,
    JOINT_GROUPS,
    groups_from_modality,
    signals_from_dataframe,
    signals_from_states,
    split_chains,
    states_from_dataframe,
)

__all__ = [
    # signals
    "EpisodeSignals", "signals_from_dataframe", "signals_from_states",
    "states_from_dataframe", "groups_from_modality", "split_chains",
    "JOINT_GROUPS", "EEF_SLICES", "ARM_TOKENS", "BODY_TOKENS",
    # measure
    "EpisodeMeasures", "Flags", "VideoMeasures", "measure_episode",
    "measure_smoothness", "measure_acceleration", "measure_contact",
    "measure_timing", "measure_video", "raise_flags",
    "log_dimensionless_jerk", "robust_spikes", "count_events", "event_starts",
    "binary_state",
    # normalize
    "Calibration", "Policy", "EpisodeScore", "score_episode", "decide",
    "Rule", "RULE_QUANTITIES", "suggest_rules", "violations", "rule_value",
    "score_smoothness", "score_acceleration", "score_contact", "score_timing",
    "score_video", "ramp", "gauss_asymmetric", "exp_decay", "robust_z",
    "DEFAULT_BRACKETS", "DEFAULT_WEIGHTS",
    # optional layers, imported lazily by __getattr__
    "SemanticFilter", "SemanticVerdict", "tasks_from_meta", "default_task",
    "render_episode_html", "render_index_html", "render_overlay_video",
    "build_tracks", "Track",
]

_LAZY = {
    "SemanticFilter": ".semantic",
    "SemanticVerdict": ".semantic",
    "tasks_from_meta": ".semantic",
    "default_task": ".semantic",
    "server_is_reachable": ".semantic",
    "render_episode_html": ".visualize",
    "render_index_html": ".visualize",
    "render_overlay_video": ".visualize",
    "episode_payload": ".visualize",
    "build_tracks": ".visualize",
    "Track": ".visualize",
}


def __getattr__(name: str):
    """Defer ``semantic`` (needs openai) and ``visualize`` (needs cv2) imports."""
    module = _LAZY.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from importlib import import_module

    return getattr(import_module(module, __name__), name)

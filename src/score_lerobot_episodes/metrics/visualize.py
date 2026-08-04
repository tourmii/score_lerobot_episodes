"""Play the measurements next to the video they were measured from.

A number like "arm RMS 5.65 rad/s²" says an episode was rough but not *when* or
*why*.  Watching the signals advance alongside the recording answers both, and
it is the fastest way to tell a genuine fault from a measurement artefact — a
contact event lines up with the box rim coming into frame, or it does not.

Two renderers, same data:

* :func:`render_episode_html` — a self-contained page: the mp4 next to a grid
  of small charts, one per signal, sharing a playhead, click-to-seek, contact
  markers and idle shading.  Modelled on huggingface/lerobot-dataset-visualizer,
  minus the server: one file, opens offline, embeds its data as JSON.
* :func:`render_overlay_video` — the same signals burned in beside the frames as
  a new mp4, for sharing somewhere a browser is not available.  This one does
  stack them into lanes: a panel that has to sit next to a video frame has no
  room for a grid.

Both draw the *same arrays the metrics consumed*, taken from
``EpisodeMeasures.series`` (measure with ``keep_series=True``), so what is on
screen is what was scored rather than a re-derivation of it.
"""

from __future__ import annotations

import base64
import html
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .measure import EpisodeMeasures
from .normalize import EpisodeScore

#: Points kept per trace in the HTML payload.  Beyond this the traces are
#: min/max decimated, which preserves spikes that plain subsampling drops.
MAX_POINTS = 3000


@dataclass
class Track:
    """One trace in the synchronised panel."""

    key: str
    label: str
    unit: str = ""
    color: str = "#4f9cf9"
    kind: str = "line"                       # "line" | "step"
    values: np.ndarray = field(default_factory=lambda: np.zeros(0))
    reference: float | None = None           # dashed horizontal line
    reference_label: str = ""


# --------------------------------------------------------------------------
# Track assembly
# --------------------------------------------------------------------------

_TRACK_SPECS: tuple[tuple[str, str, str, str, str], ...] = (
    # key,               label,                          unit,     colour,    kind
    ("speed_wrist",      "wrist speed",                  "m/s",    "#4f9cf9", "line"),
    ("acc_wrist",        "wrist acceleration",           "m/s²",   "#f2994a", "line"),
    ("acc_arm",          "arm joint acceleration (RMS)", "rad/s²", "#eb5757", "line"),
    ("acc_body",         "body joint acceleration (RMS)", "rad/s²", "#bb6bd9", "line"),
    ("joint_speed",      "joint speed",                  "rad/s",  "#27ae60", "line"),
    ("tracking_residual", "tracking residual",           "rad",    "#e0b341", "line"),
    ("base_tilt_deg",    "base tilt",                    "deg",    "#56ccf2", "line"),
    ("hand_closed",      "hand closed",                  "",       "#f2c94c", "step"),
    ("interframe_diff",  "inter-frame difference",       "grey",   "#9aa0a6", "line"),
)


def build_tracks(m: EpisodeMeasures, keys: Sequence[str] | None = None) -> list[Track]:
    """Pick the traces available for this episode, in a fixed reading order."""
    wanted = set(keys) if keys else None
    tracks: list[Track] = []
    for key, label, unit, color, kind in _TRACK_SPECS:
        if wanted is not None and key not in wanted:
            continue
        values = m.series.get(key)
        if values is None or len(values) == 0:
            continue
        track = Track(key=key, label=label, unit=unit, color=color, kind=kind,
                      values=np.asarray(values, dtype=float))
        if key == "joint_speed" and np.isfinite(m.idle_threshold_rad_s):
            track.reference = float(m.idle_threshold_rad_s)
            track.reference_label = "idle threshold"
        tracks.append(track)
    return tracks


def _timebase(m: EpisodeMeasures, length: int) -> np.ndarray:
    """Seconds from episode start, for a series of ``length`` samples."""
    t = m.series.get("t")
    if t is not None and len(t) >= length:
        t = np.asarray(t, dtype=float)[:length]
        return t - t[0]
    fps = m.fps if np.isfinite(m.fps) and m.fps > 0 else 50.0
    return np.arange(length, dtype=float) / fps


def _decimate(t: np.ndarray, y: np.ndarray, limit: int = MAX_POINTS) -> tuple[list, list]:
    """Min/max decimation: keeps the extremes of each bucket, so spikes survive."""
    n = len(y)
    if n <= limit:
        return [round(float(v), 4) for v in t], [None if not np.isfinite(v) else round(float(v), 5) for v in y]

    buckets = max(limit // 2, 1)
    edges = np.linspace(0, n, buckets + 1, dtype=int)
    out_t: list[float] = []
    out_y: list[float | None] = []
    for start, end in zip(edges[:-1], edges[1:]):
        if end <= start:
            continue
        chunk = y[start:end]
        finite = np.isfinite(chunk)
        if not finite.any():
            out_t.append(round(float(t[start]), 4))
            out_y.append(None)
            continue
        lo = start + int(np.argmin(np.where(finite, chunk, np.inf)))
        hi = start + int(np.argmax(np.where(finite, chunk, -np.inf)))
        for i in sorted((lo, hi)):
            out_t.append(round(float(t[i]), 4))
            out_y.append(round(float(y[i]), 5))
    return out_t, out_y


def _spans(mask: np.ndarray, t: np.ndarray) -> list[list[float]]:
    """Contiguous ``True`` runs of ``mask`` as ``[start_s, end_s]`` pairs.

    The end is pushed out by one sample period so a single-frame event — which
    is exactly what a contact looks like — still has a visible width.
    """
    mask = np.asarray(mask, dtype=bool)
    if mask.size == 0 or not mask.any() or len(t) == 0:
        return []
    dt = float(np.median(np.diff(t))) if len(t) > 1 else 0.02
    padded = np.concatenate(([False], mask, [False]))
    edges = np.diff(padded.astype(np.int8))
    starts = np.flatnonzero(edges == 1)
    ends = np.flatnonzero(edges == -1) - 1
    out = []
    for start, end in zip(starts, ends):
        start = min(int(start), len(t) - 1)
        end = min(int(end), len(t) - 1)
        out.append([round(float(t[start]), 4), round(float(t[end]) + dt, 4)])
    return out


# --------------------------------------------------------------------------
# Payload
# --------------------------------------------------------------------------


def episode_payload(
    m: EpisodeMeasures,
    score: EpisodeScore | None = None,
    video_src: str | None = None,
    tracks: Sequence[Track] | None = None,
) -> dict[str, Any]:
    """The JSON blob the page renders from."""
    tracks = list(tracks) if tracks is not None else build_tracks(m)

    payload_tracks = []
    for track in tracks:
        t = _timebase(m, len(track.values))
        xs, ys = _decimate(t, track.values)
        finite = track.values[np.isfinite(track.values)]
        payload_tracks.append({
            "label": track.label,
            "unit": track.unit,
            "color": track.color,
            "kind": track.kind,
            "t": xs,
            "y": ys,
            "min": round(float(finite.min()), 5) if finite.size else 0.0,
            "max": round(float(finite.max()), 5) if finite.size else 1.0,
            "reference": track.reference,
            "referenceLabel": track.reference_label,
        })

    duration = float(m.duration_s) if np.isfinite(m.duration_s) else 0.0
    base_t = _timebase(m, m.n_frames or 1)
    contact = m.series.get("contact_confirmed")
    idle = m.series.get("idle")

    measures = _measure_table(m)
    families = score.families if score else {}

    return {
        "episode": m.episode,
        "duration": round(duration, 3),
        "frames": int(m.n_frames),
        "fps": round(float(m.fps), 3) if np.isfinite(m.fps) else 0.0,
        "video": video_src,
        "tracks": payload_tracks,
        "contact": _spans(contact, base_t) if contact is not None else [],
        "idle": _spans(idle, base_t) if idle is not None else [],
        "flags": m.flags.to_dict(),
        "raisedFlags": m.flags.raised(),
        "measures": measures,
        "score": {
            "total": None if score is None else round(float(score.total), 4),
            "decision": None if score is None else score.decision,
            "reasons": [] if score is None else score.reasons,
            "families": {
                k: (None if not np.isfinite(v) else round(float(v), 4))
                for k, v in families.items()
            },
            "semantic": None if score is None else score.semantic_score,
            "semanticNote": "" if score is None else score.semantic_note,
        },
    }


def _measure_table(m: EpisodeMeasures) -> list[dict[str, Any]]:
    """Raw quantities, grouped by family, with their units."""
    def fmt(value, digits=3):
        if value is None:
            return "-"
        if isinstance(value, (int, np.integer)):
            return str(int(value))
        value = float(value)
        return "-" if not np.isfinite(value) else f"{value:.{digits}f}"

    rows = [
        ("smoothness", "LDLJ wrist", fmt(m.ldlj_wrist, 2), ""),
        ("smoothness", "LDLJ arm", fmt(m.ldlj_arm, 2), ""),
        ("smoothness", "LDLJ body", fmt(m.ldlj_body, 2), ""),
        ("acceleration", "arm RMS", fmt(m.acc_arm_rms), "rad/s²"),
        ("acceleration", "arm p99", fmt(m.acc_arm_p99), "rad/s²"),
        ("acceleration", "body RMS", fmt(m.acc_body_rms), "rad/s²"),
        ("acceleration", "body p99", fmt(m.acc_body_p99), "rad/s²"),
        ("acceleration", "wrist RMS", fmt(m.acc_wrist_rms), "m/s²"),
        ("acceleration", "wrist p99", fmt(m.acc_wrist_p99), "m/s²"),
        ("contact", "events", fmt(m.contact_events), ""),
        ("contact", "rate", fmt(m.contact_rate_hz), "Hz"),
        ("contact", "impact / base / tracking",
         f"{m.impact_events} / {m.base_disturbance_events} / {m.tracking_divergence_events}", ""),
        ("contact", "min wrist separation", fmt(m.min_wrist_separation_m), "m"),
        ("contact", "max base tilt", fmt(m.max_base_tilt_deg, 1), "deg"),
        ("timing", "duration", fmt(m.duration_s, 2), "s"),
        ("timing", "idle fraction", fmt(m.idle_fraction), ""),
        ("timing", "grasp transitions",
         "-" if m.grasp_transitions < 0 else str(m.grasp_transitions), ""),
        ("timing", "wrist path", fmt(m.eef_path_m), "m"),
        ("timing", "working side", m.working_side, ""),
    ]
    if m.video_path is not None:
        v = m.video
        rows += [
            ("video", "sharpness", fmt(v.sharpness, 1), "Laplacian var"),
            ("video", "brightness", fmt(v.brightness, 1), "0-255"),
            ("video", "contrast", fmt(v.contrast, 1), "std"),
            ("video", "clipped", fmt(v.clipped_fraction, 4), "fraction"),
            ("video", "inter-frame diff", fmt(v.interframe_diff, 2), "grey"),
            ("video", "frames decoded", f"{v.frames_decoded} / {v.frames_declared}", ""),
        ]
    return [{"family": f, "name": n, "value": val, "unit": u} for f, n, val, u in rows]


# --------------------------------------------------------------------------
# HTML
# --------------------------------------------------------------------------

ASSETS = Path(__file__).resolve().parent / "assets"


def _asset(name: str) -> str:
    """Read a shared asset.  The app serves these; the standalone page inlines them."""
    return (ASSETS / name).read_text(encoding="utf-8")


#: Layout that only the standalone page needs; the shared look lives in theme.css.
_PAGE_CSS = """
.wrap { max-width: 1400px; margin: 0 auto; padding: 20px 18px 48px; }
header { display: flex; flex-wrap: wrap; align-items: baseline; gap: 12px; margin-bottom: 12px; }
.layout { display: grid; grid-template-columns: minmax(320px, 460px) 1fr;
          gap: 16px; align-items: start; }
@media (max-width: 900px) { .layout { grid-template-columns: 1fr; } }
video { width: 100%; border-radius: var(--radius); background: #000; display: block; }
.card + .card { margin-top: 12px; }
.col > .card:first-child { margin-top: 12px; }
"""

#: Glue between the shared charts and this page's own elements.
_PAGE_JS = r"""
const P = JSON.parse(document.getElementById('payload').textContent);
const video = document.getElementById('video');
const readout = document.getElementById('readout');
const charts = new SyncChart(document.getElementById('charts'), P, {
  onSeek: t => { if (video) video.currentTime = t; },
});
charts.mountLegend(document.getElementById('legend'));

// Each card shows its own value, so the read-out only carries the shared state.
charts.onCursor = t => {
  readout.innerHTML = `<b>t = ${t.toFixed(2)} s</b> / ${P.duration.toFixed(2)} s` +
    (charts.inContact(t) ? ' &nbsp;\u00b7&nbsp; <b style="color:#eb5757">contact</b>' : '');
};

if (video) {
  const follow = () => charts.setCursor(video.currentTime);
  video.addEventListener('timeupdate', follow);
  video.addEventListener('seeked', follow);
  const loop = () => {
    if (!video.paused && !video.ended) follow();
    requestAnimationFrame(loop);
  };
  requestAnimationFrame(loop);
}
charts.setCursor(0);
"""



def _video_src(video_path: str | Path | None, out_path: Path, embed: bool) -> str | None:
    """Relative link by default; a data URI when the page must stand alone."""
    if video_path is None:
        return None
    video_path = Path(video_path)
    if not video_path.exists():
        return None
    if embed:
        data = base64.b64encode(video_path.read_bytes()).decode("ascii")
        return f"data:video/mp4;base64,{data}"
    try:
        import os
        return os.path.relpath(video_path.resolve(), out_path.resolve().parent).replace("\\", "/")
    except ValueError:  # different drive on Windows
        return video_path.resolve().as_uri()


def render_episode_html(
    m: EpisodeMeasures,
    out_path: str | Path,
    score: EpisodeScore | None = None,
    video_path: str | Path | None = None,
    embed_video: bool = False,
    tracks: Sequence[Track] | None = None,
    index_href: str | None = None,
) -> Path:
    """Write a self-contained page playing ``video_path`` beside the traces.

    ``m`` must have been produced with ``keep_series=True``; without the series
    there is nothing to plot.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    src = _video_src(video_path or m.video_path, out_path, embed_video)
    payload = episode_payload(m, score, src, tracks)
    title = f"Episode {m.episode:06d}" if m.episode is not None else "Episode"

    decision = payload["score"]["decision"]
    total = payload["score"]["total"]
    badge = (
        f'<span class="badge {decision}">{decision}</span>' if decision else ""
    )
    head_bits = [f'<span class="sub">{payload["duration"]:.2f} s · {payload["fps"]:.0f} fps</span>']
    if total is not None:
        head_bits.insert(0, f'<span class="sub">score <b>{total:.3f}</b></span>')
    if index_href:
        head_bits.append(f'<a class="sub" href="{html.escape(index_href)}">← all episodes</a>')

    families_html = "".join(
        f'<div class="fam"><span>{html.escape(name)}</span>'
        f'<span class="bar"><i style="width:{0 if value is None else value * 100:.1f}%"></i></span>'
        f'<span class="num">{"–" if value is None else f"{value:.3f}"}</span></div>'
        for name, value in payload["score"]["families"].items()
    )
    flags_html = "".join(
        f'<span class="flag">{html.escape(flag)}</span>' for flag in payload["raisedFlags"]
    ) or '<span class="sub">none</span>'
    reasons = payload["score"]["reasons"]
    reasons_html = (
        f'<div class="hint">{html.escape("; ".join(reasons))}</div>' if reasons else ""
    )
    semantic = payload["score"]["semantic"]
    semantic_html = ""
    if semantic is not None:
        semantic_html = (
            f'<div class="card"><h2>semantic verdict</h2>'
            f'<div><b>{semantic:.1f}</b> — {html.escape(payload["score"]["semanticNote"] or "")}</div></div>'
        )

    rows = "".join(
        f'<tr><td>{html.escape(r["family"])}</td><td>{html.escape(r["name"])}</td>'
        f'<td class="v">{html.escape(str(r["value"]))}</td>'
        f'<td class="u">{html.escape(r["unit"])}</td></tr>'
        for r in payload["measures"]
    )
    video_html = (
        f'<video id="video" src="{html.escape(src)}" controls playsinline preload="metadata"></video>'
        if src else '<div class="card sub">no video for this episode</div>'
    )

    document = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title} — episode metrics</title>
<style>{_asset("theme.css")}{_PAGE_CSS}</style></head>
<body><div class="wrap">
<header><h1>{title}</h1>{badge}{"".join(head_bits)}</header>
{reasons_html}
<div class="layout">
  <div class="col">
    {video_html}
    <div class="card"><h2>score</h2><div class="families">{families_html}</div></div>
    {semantic_html}
    <div class="card"><h2>flags</h2><div class="flags">{flags_html}</div></div>
  </div>
  <div class="col">
    <div class="card">
      <h2>signals, synchronised with playback</h2>
      <div class="legend" id="legend"></div>
      <div id="charts"></div>
      <div class="readout" id="readout"></div>
      <div class="hint">click any chart to seek the video there · click a legend chip to hide a signal</div>
    </div>
    <div class="card scrollx"><h2>raw measurements</h2>
      <table><thead><tr><th>family</th><th>quantity</th><th class="v">value</th><th>unit</th></tr></thead>
      <tbody>{rows}</tbody></table>
    </div>
  </div>
</div></div>
<script id="payload" type="application/json">{json.dumps(payload)}</script>
<script>{_asset("chart.js")}</script>
<script>{_PAGE_JS}</script>
</body></html>
"""
    out_path.write_text(document, encoding="utf-8")
    return out_path


def render_index_html(
    entries: Sequence[dict[str, Any]],
    out_path: str | Path,
    title: str = "Episode quality",
) -> Path:
    """Write the dataset overview linking to every episode page.

    Each entry needs ``episode`` and ``href``; ``total``, ``decision``,
    ``families``, ``flags`` and ``semantic`` are shown when present.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    families: list[str] = []
    for entry in entries:
        for name in entry.get("families", {}):
            if name not in families:
                families.append(name)

    def cell(value):
        return "–" if value is None or (isinstance(value, float) and not math.isfinite(value)) \
            else f"{value:.3f}" if isinstance(value, float) else str(value)

    header = "".join(f"<th class='v'>{html.escape(n)}</th>" for n in families)
    rows = []
    for entry in sorted(entries, key=lambda e: e.get("total") if e.get("total") is not None else -1):
        decision = entry.get("decision", "")
        fam_cells = "".join(
            f"<td class='v'>{cell(entry.get('families', {}).get(n))}</td>" for n in families
        )
        flags = ", ".join(entry.get("flags", []))
        rows.append(
            f"<tr><td><a href='{html.escape(entry['href'])}'>"
            f"{entry['episode'] if entry.get('episode') is not None else '?'}</a></td>"
            f"<td><span class='badge {html.escape(decision)}'>{html.escape(decision)}</span></td>"
            f"<td class='v'>{cell(entry.get('total'))}</td>{fam_cells}"
            f"<td class='v'>{cell(entry.get('semantic'))}</td>"
            f"<td class='u'>{html.escape(flags)}</td></tr>"
        )

    counts: dict[str, int] = {}
    for entry in entries:
        counts[entry.get("decision", "?")] = counts.get(entry.get("decision", "?"), 0) + 1
    summary = " · ".join(f"{k}: <b>{v}</b>" for k, v in sorted(counts.items()))

    document = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(title)}</title>
<style>{_asset("theme.css")}{_PAGE_CSS}</style></head>
<body><div class="wrap">
<header><h1>{html.escape(title)}</h1>
<span class="sub">{len(entries)} episodes · {summary}</span></header>
<div class="card scrollx"><table>
<thead><tr><th>episode</th><th>decision</th><th class="v">total</th>{header}
<th class="v">semantic</th><th>flags</th></tr></thead>
<tbody>{"".join(rows)}</tbody></table></div>
<p class="hint">sorted worst first · click an episode to watch it with its signals</p>
</div></body></html>
"""
    out_path.write_text(document, encoding="utf-8")
    return out_path


# --------------------------------------------------------------------------
# Burned-in video
# --------------------------------------------------------------------------


def render_overlay_video(
    m: EpisodeMeasures,
    out_path: str | Path,
    video_path: str | Path | None = None,
    tracks: Sequence[Track] | None = None,
    panel_width: int = 560,
    fps: float | None = None,
) -> Path:
    """Write an mp4 with the traces drawn beside the frames and a moving playhead.

    The static panel is rasterised once and only the playhead and the current
    values are redrawn per frame, so the cost is dominated by the codec rather
    than the plotting.
    """
    import cv2

    video_path = Path(video_path or m.video_path or "")
    if not video_path.exists():
        raise FileNotFoundError(f"no video to overlay: {video_path}")

    tracks = list(tracks) if tracks is not None else build_tracks(m)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        cap.release()
        raise RuntimeError(f"cannot open {video_path}")
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or m.n_frames
    out_fps = fps or cap.get(cv2.CAP_PROP_FPS) or (m.fps if np.isfinite(m.fps) else 30.0)

    panel, lanes = _render_panel(m, tracks, panel_width, frame_h)
    writer = cv2.VideoWriter(
        str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), out_fps, (frame_w + panel_width, frame_h)
    )

    duration = float(m.duration_s) if np.isfinite(m.duration_s) and m.duration_s > 0 else \
        (n_frames / out_fps if out_fps else 1.0)
    left, right = 8, panel_width - 8

    index = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t = index / out_fps if out_fps else 0.0
        canvas = panel.copy()
        x = int(left + (min(t, duration) / duration) * (right - left)) if duration > 0 else left
        cv2.line(canvas, (x, 18), (x, frame_h - 8), (240, 240, 240), 1)

        for lane in lanes:
            value = _sample(lane["values"], lane["t"], t)
            label = "-" if value is None else f"{value:.2f}"
            cv2.putText(canvas, label, (right - 62, lane["top"] + 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, lane["color"], 1, cv2.LINE_AA)
        cv2.putText(canvas, f"t={t:5.2f}s", (left, frame_h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1, cv2.LINE_AA)

        writer.write(np.hstack([frame, canvas]))
        index += 1

    cap.release()
    writer.release()
    return out_path


def _hex_to_bgr(color: str) -> tuple[int, int, int]:
    color = color.lstrip("#")
    r, g, b = (int(color[i:i + 2], 16) for i in (0, 2, 4))
    return (b, g, r)


def _sample(values: np.ndarray, t: np.ndarray, when: float) -> float | None:
    if len(values) == 0:
        return None
    i = int(np.searchsorted(t, when))
    i = max(0, min(i, len(values) - 1))
    value = float(values[i])
    return value if np.isfinite(value) else None


def _render_panel(m: EpisodeMeasures, tracks: Sequence[Track], width: int, height: int):
    """Rasterise the static part of the panel once."""
    import cv2

    panel = np.full((height, width, 3), 26, dtype=np.uint8)
    if not tracks:
        cv2.putText(panel, "no series", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (180, 180, 180), 1, cv2.LINE_AA)
        return panel, []

    left, right = 8, width - 8
    top_margin, bottom_margin = 18, 26
    lane_h = max((height - top_margin - bottom_margin) // len(tracks), 24)
    duration = float(m.duration_s) if np.isfinite(m.duration_s) and m.duration_s > 0 else 1.0

    contact = m.series.get("contact_confirmed")
    idle = m.series.get("idle")
    base_t = _timebase(m, m.n_frames or 1)

    lanes = []
    for i, track in enumerate(tracks):
        top = top_margin + i * lane_h
        bottom = top + lane_h - 12
        t = _timebase(m, len(track.values))
        finite = track.values[np.isfinite(track.values)]
        lo = min(0.0, float(finite.min()) if finite.size else 0.0)
        hi = float(finite.max()) if finite.size else 1.0
        if track.reference is not None:
            hi = max(hi, track.reference)
        if hi - lo < 1e-9:
            hi = lo + 1.0
        color = _hex_to_bgr(track.color)

        def to_xy(k):
            x = int(left + (t[k] / duration) * (right - left))
            y = int(bottom - ((track.values[k] - lo) / (hi - lo)) * (bottom - top))
            return x, max(top, min(y, bottom))

        for mask, shade in ((idle, (52, 52, 52)), (contact, (40, 40, 120))):
            if mask is None:
                continue
            for a, b in _spans(mask, base_t):
                xa = int(left + (a / duration) * (right - left))
                xb = max(int(left + (b / duration) * (right - left)), xa + 1)
                panel[top:bottom, xa:xb] = shade

        points = np.array(
            [to_xy(k) for k in range(len(track.values)) if np.isfinite(track.values[k])],
            dtype=np.int32,
        )
        if len(points) > 1:
            cv2.polylines(panel, [points], False, color, 1, cv2.LINE_AA)
        if track.reference is not None:
            y = int(bottom - ((track.reference - lo) / (hi - lo)) * (bottom - top))
            cv2.line(panel, (left, y), (right, y), (110, 110, 110), 1)
        cv2.line(panel, (left, bottom), (right, bottom), (70, 70, 70), 1)
        label = f"{track.label}" + (f" [{track.unit}]" if track.unit else "")
        cv2.putText(panel, label, (left + 2, top + 12), cv2.FONT_HERSHEY_SIMPLEX,
                    0.36, color, 1, cv2.LINE_AA)
        lanes.append({"top": top, "color": color, "values": track.values, "t": t})

    title = f"episode {m.episode}" if m.episode is not None else "episode"
    cv2.putText(panel, title, (left, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                (225, 225, 225), 1, cv2.LINE_AA)
    return panel, lanes

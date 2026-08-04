# Architecture

How the pieces fit together, and why they are split where they are.

```
                     ┌───────────────────────────────────────────┐
  parquet + mp4 ────▶│ signals.py    load, mask dead channels,   │
                     │               differentiate once          │
                     └──────────────────┬────────────────────────┘
                                        ▼
                     ┌───────────────────────────────────────────┐
                     │ measure.py    RAW QUANTITIES + FLAGS       │  rad/s², m, Hz, s
                     │               no thresholds, no [0,1]      │  + boolean flags
                     └──────────────────┬────────────────────────┘
                                        ▼
                     ┌───────────────────────────────────────────┐
                     │ normalize.py  ranges fitted to YOUR data   │  [0,1], 1 = good
                     │               per-family gate → decision    │  accept/review/reject
                     └──────────────────┬────────────────────────┘
                                        │
        ┌───────────────────────────────┼───────────────────────────────┐
        ▼                               ▼                               ▼
┌───────────────┐          ┌─────────────────────────┐      ┌─────────────────────┐
│ semantic.py   │          │ visualize.py            │      │ app/ (FastAPI + SPA)│
│ did the task  │          │ standalone HTML page,   │      │ measure once,       │
│ actually      │          │ burned-in overlay mp4   │      │ re-score live       │
│ succeed?      │          └─────────────────────────┘      └─────────────────────┘
└───────────────┘
```

## Why measurement is separate from scoring

An arm RMS acceleration of 3.63 rad/s² is 3.63 rad/s² whatever robot produced
it. What *changes* between platforms is the range that counts as normal and the
line above which an episode is accepted. Keeping those in a different module
means:

- retargeting to another robot is a refit of `Calibration`, not an edit to any
  measurement code;
- the raw numbers are auditable — `measurements.csv` has physical units, so a
  domain expert can disagree with a threshold without re-deriving the metric;
- scoring is cheap. Measuring 90 episodes with video takes minutes; re-scoring
  them takes milliseconds, which is what lets the app move a weight slider and
  re-rank the whole dataset instantly.

## The five families

Each targets a specific failure mode, and together they cover three time scales
that no single statistic spans:

| anomaly | duty cycle | caught by |
|---|---|---|
| single frame (20–60 ms) | < 0.5 % | contact event measures |
| burst, 0.1–0.5 s | 1–3 % | acceleration p99 |
| whole episode | ~100 % | acceleration RMS, LDLJ |

| family | measures | key quantity |
|---|---|---|
| smoothness | trajectory *shape* | LDLJ on wrist, arm, body chains |
| acceleration | trajectory *magnitude* | RMS and p99 per chain |
| contact | inferred collisions | events/second, 2-of-3 quorum |
| timing | duration, dead time, regrasps | seconds, fraction, transition count |
| video | recording quality and integrity | Laplacian variance, decode ratio |

Smoothness and acceleration are near-orthogonal by construction: LDLJ is scale
invariant so it cannot say whether a motion is violent; the acceleration family
is scale dependent so it cannot say whether it is jerky. Their negative
correlation is the point, not a redundancy.

## Flags come before distributions

Four preconditions are checked before any percentile is computed:

| flag | condition | why it is not just a low score |
|---|---|---|
| `degenerate` | wrist path < 0.30 m | every motion metric is a decreasing function of a derivative norm, so a motionless episode is at the global optimum of all of them at once |
| `grasp_incomplete` | < 2 hand transitions | the grasp cycle never happened, so "grasp quality" does not exist |
| `video_unreadable` / `video_truncated` | decode fails or < 95 % of declared frames | a corrupt file looks pristine to an appearance metric |
| `self_collision_suspect`, `video_frozen` | wrists < 0.10 m apart; near-zero inter-frame difference | review by hand, not an automatic reject |

`Calibration.fit` excludes flagged episodes from the percentile sample. On the
reference dataset a reset recording once scored the *best* smoothness in the
set; leaving it in skews both ends of every range and corrupts the thresholds of
all the others.

## What the semantic filter adds

The five families answer *how* the robot moved. None can answer *whether the
task was done* — an episode can be smooth, gentle, prompt and well filmed while
the object ends up beside the box instead of in it.

On `pickup_20260628_150622` this is measurable: 23 episodes were discarded by
hand, and the hard flags catch 4 of them outright, but the aggregate score of
the discarded set (0.656) is indistinguishable from the kept set (0.644). The
remaining discards are task failures, which is exactly the gap
[`semantic.py`](../src/score_lerobot_episodes/metrics/semantic.py) fills using a
vision-language model. Its verdict enters as a gate — reject at 0.0, review at
0.5 — never as a weighted term, because a half-done task is not compensated by a
smooth trajectory.

## The decision is a limit, not a score

The default rule (`Policy.mode = "rules"`) compares each measurement to a limit
**in its own physical unit** and never consults the aggregate.

The reason is that a normalised family score is a percentile rank in disguise —
it correlates 0.97–0.99 with rank, because the ramp brackets are fitted to the
dataset's own p5/p95. A threshold on it removes about the same share of *any*
dataset, however good: it sets a reject *rate*, not a quality bar.

On the reference dataset the ranking is also pointing the wrong way. Against its
own hand-discard list, smoothness scores AUC **0.39** and video **0.36** — below
the 0.5 coin flip, meaning the discarded episodes are on average *smoother* than
the kept ones. Compared in raw units, the episodes a percentile gate rejected
were physically indistinguishable from the ones it kept (arm RMS 4.28 vs
4.45 rad/s², identical grasp counts). Motion metrics reward doing less, and a
successful pick-and-place involves vigorous contact.

So: the flags reject (absolute, checkable), the limits reject (absolute, yours
to set), the semantic filter rejects (task success), and the weighted score
ranks. `"gate"` and `"weighted"` remain available when a relative cut is what
you want.

## Entry points

| what | command |
|---|---|
| web app (FE + BE) | `python -m app` → http://127.0.0.1:8000 |
| batch CLI | `python scripts/measure_dataset.py <dataset> --html 15` |
| library | `from score_lerobot_episodes.metrics import ...` |
| legacy single-tier API | `score_lerobot_episodes.scores.humanoid` (facade) |

See [`app.md`](app.md) for the application and
[`metrics_reference.md`](metrics_reference.md) for every formula and contract.

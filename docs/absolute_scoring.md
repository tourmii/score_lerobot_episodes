# Absolute scoring — one threshold across every dataset

`metrics/normalize.py` fits its ramps to the batch it is scoring. That makes a
family score a **percentile rank inside that batch**, which is useful for triage
and useless as a standing quality bar:

```
percentile path, calibration refitted per batch, threshold 0.5

  golden     median total 0.469   accept 46%   reject 54%
  reference  median total 0.464   accept 42%   reject 58%
  bad        median total 0.266   accept  1%   reject 99%
```

The golden batch and the reference batch are *not* the same quality — the first
was generated a full 1.6 sigma cleaner on every quantity — and they reject within
three points of each other. The threshold is not asking "is this episode good",
it is asking "how does this episode compare to its neighbours", so it removes
roughly the same fraction of anything handed to it.

Two further consequences: a score of 0.6 in January and a score of 0.6 in June
are not the same episode quality, and because `combine()` normalises the
exponents by `sum(w)`, changing any weight rescales every episode's total and the
threshold has to be re-tuned after the edit.

`metrics/profile.py` is the other route.

## The model

Each quantity gets two anchors **in its own physical unit**:

| anchor | meaning |
|---|---|
| `good` | the level below which the quantity stops mattering. Not "typical" — the point where you would no longer look twice. |
| `bad`  | the level at which this quantity *alone* condemns the episode, whatever the rest of the recording looks like. |

```
u_i      = (v_i - good_i) / (bad_i - good_i)      # 0 at good, 1 at bad
p_i      = P_MAX * smoothstep(u_i)                # P(criterion i is violated)
severity = sum_i w_i * -log(1 - p_i)              # evidence against, in nats
total    = exp(-severity)                         # P(no criterion is violated)
```

`total` is what the threshold compares against. `severity` is the same quantity
on a log scale and is what the tables sort on — a product over a dozen criteria
underflows to a shared `0.000` for exactly the episodes a reviewer most wants
ranked.

### Why noisy-OR and not a weighted mean

**The scale does not move when the weights do.** `w_i` is a multiplicity, not a
share of a budget that has to sum to one. Doubling every acceleration weight
leaves a clean batch's reject rate at 2% — under a weighted mean the whole
distribution shifts and the threshold follows it.

**Non-compensatory without a floor.** A single `p_i` near one drags the product
toward zero however clean the other terms are. That is the property the geometric
mean was chosen for, but here it falls out of the algebra instead of needing
`GEOMETRIC_FLOOR` to stop a zero term annihilating the ranking.

**An unmeasurable quantity leaves no trace.** A `nan` term drops out of the sum
entirely, so `total` reads "no criterion *of those measurable* is violated". The
weight-redistribution in `score_episode()` exists only because a weighted mean
has no such option. A dataset with no video simply has fewer terms, and its
scores stay comparable with everything else.

## What has to be set, and how often

| layer | quantities | changes when |
|---|---|---|
| `ABSOLUTE_CRITERIA` | idle fraction, extra hand transitions, contact rate, base tilt, duration ratio, clipped pixels, exposure deviation, inter-frame difference | never — these are dimensionless ratios, event rates and grey levels |
| `PLATFORM_CRITERIA` | the three LDLJ chains, six acceleration terms, sharpness residual, contrast | the robot, the control rate or the camera changes |
| task constants | `task_duration_s`, `grasp_nominal` | the task changes |
| `threshold` | — | whenever you want a different bar. **This is the only routine knob.** |

`QualityProfile.fit()` estimates only the platform layer, at
`median ± k * 1.4826 * MAD` rather than at a percentile pair. The difference is
the whole point: a percentile is a rank and reproduces itself on any batch,
whereas a median and a robust spread are *locations in the unit*, and once
written down they stop moving.

## Using it

```bash
# once, on a batch you have actually inspected
python scripts/measure_dataset.py golden_20260628 \
    --mode absolute --fit-profile g1_50hz.json --task-duration 8.0

# every run after that — anchors frozen, one number to turn
python scripts/measure_dataset.py pickup_20260901 \
    --mode absolute --profile g1_50hz.json --threshold 0.5 --drift
```

`absolute` is the default mode, so `--mode` only needs naming to leave it.
Running it *without* `--profile` still fits on the batch being scored and prints
that it has done so. That is a starting point, not the mode
working as designed: the verdict still depends on the batch.

| flag | purpose |
|---|---|
| `--profile FILE` | load anchors and do not refit — what makes the threshold transfer |
| `--fit-profile OUT` | fit on this dataset and write the anchors out |
| `--threshold T` | the bar on `total`; default 0.5 |
| `--good-sigma` / `--bad-sigma` | where the anchors sit, in robust sigmas (1.0 / 4.0) |
| `--task-duration S` | seconds a clean run takes; a task constant, not a dataset statistic |
| `--drift` | report how far this batch sits outside the loaded anchors |

### Reading the threshold

`total` is a probability, so the number reads directly.

| threshold | on the reference batch |
|---|---|
| 0.35 | 7% rejected — permissive |
| 0.50 | 13% rejected — the default |
| 0.80 | 27% rejected |
| 0.90 | 41% rejected — strict |

Note that fitting a profile on a batch and then scoring that same batch at 0.5
rejects around 10–15% of it. That is `good` sitting one sigma out: roughly a
sixth of episodes exceed it on any one criterion and the excesses accumulate.
Raise `--good-sigma` to 1.5 for a more permissive profile.

## Drift, and the one thing a frozen profile cannot do

A frozen profile cannot distinguish a genuinely worse batch from a different
robot, a changed control rate or a re-aimed camera. Both look like "everything
got worse". Neither can it tell you by quietly adapting, which is what the
percentile path does. So it reports instead — per criterion, the share of
episodes past `good` and past `bad`:

```
criterion                             n   median u   >good    >bad
arm acceleration p99                 90       1.55    100%     92%
body acceleration RMS                90       1.64    100%     90%
```

Read it as a domain-shift alarm. A handful of criteria mildly over is a batch
with problems. `>bad` above ~30% on one criterion whose neighbours are clean
usually means the anchor no longer describes this hardware, and the profile
wants refitting rather than the batch wanting rejecting.

## Checking the implementation

```bash
python scripts/check_absolute_transfer.py
```

Six checks on synthetic batches of known relative quality: that one threshold
separates them, that the weights do not move the scale, that `severity` still
ranks the episodes whose `total` has floored, and that a strictly worse episode
never scores higher (2000 trials).

Three more scripts cover the app:

```bash
python scripts/check_app_absolute.py   # the service layer, on datasets on disk
python scripts/check_app_routes.py     # the HTTP routes
python scripts/check_semantic_key.py   # the default mode, and blank API keys
```

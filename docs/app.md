# The quality app

A FastAPI backend and a single-page frontend for measuring a dataset once and
reviewing it interactively. It replaces the Streamlit dashboard, which could not
show a video and its signals side by side, could not stream video at all, and
re-ran the whole pipeline on every widget change.

```bash
pip install -e ".[app]"
python -m app                              # http://127.0.0.1:8000
python -m app --port 9000 --reload
python -m app --dataset ./pickup_20260628_150622   # register at startup
```

There is no build step and no npm: the frontend is three files under
`app/static/`, and the chart it draws with is the same
`metrics/assets/chart.js` that the standalone HTML export inlines — so a page
saved to disk and the live app cannot drift apart.

The layout follows [huggingface/lerobot-dataset-visualizer](https://github.com/huggingface/lerobot-dataset-visualizer):
dataset stats and a scored episode list in a fixed rail, the cameras in a row
above the signals, and one floating transport bar driving all of them. Dark by
default — this is read against dark video frames — with a toggle in the rail
corner that persists.

## The workflow

1. **Add a dataset** — paste the path to the folder holding `meta/`, `data/`
   and `videos/`. It is validated immediately (episode files must exist).
2. **Run measurement** — a background job with live progress. Uncheck *include
   video* for a first pass: it is the slowest family by orders of magnitude.
   Results are cached under `.quality_app/`, so a restart does not re-measure.
3. **Review the overview** — decision counts, the score histogram against the
   accept threshold, per-family spread, flags raised, and the sortable episode
   table.
4. **Open an episode** — the recording plays beside the signals it was judged
   on. The playhead follows the video; clicking a trace seeks the video there.
5. **Set the decision rule** — *every family must pass* (the default) or
   *weighted mean must pass*. Re-scoring is in-process and takes milliseconds,
   so the whole dataset re-decides as you drag. See
   [`metrics_reference.md` §5.6](metrics_reference.md) — the same threshold is
   far stricter per-family than on a mean, and the rail says how many episodes
   the current setting accepts.
6. **Run the semantic filter** — if a Cosmos-Reason vLLM server is reachable,
   each episode gets a task-success verdict. Verdicts are cached to JSONL and
   never re-requested.
7. **Export** — measurements + scores as CSV, or the accepted episode list as
   JSON.

## Why measure and score are separate endpoints

Measurement decodes video and differentiates trajectories; scoring is a few
hundred floating-point operations. Splitting them is what makes the weight
sliders usable — `PUT /policy` re-scores 90 cached episodes without touching a
parquet file. It is the same separation the metrics themselves are built on
(see [`architecture.md`](architecture.md)), carried up into the application.

## API

| method | path | purpose |
|---|---|---|
| `GET` | `/api/health` | liveness, cache location |
| `GET` | `/api/datasets` | registered datasets |
| `POST` | `/api/datasets` | `{path, camera?}` → register |
| `DELETE` | `/api/datasets/{id}` | forget (files are untouched) |
| `GET` | `/api/datasets/{id}/cameras` | available camera keys |
| `POST` | `/api/datasets/{id}/analyze` | `{useVideo, workers, loPct, hiPct}` → job |
| `GET` | `/api/datasets/{id}/overview` | distributions, counts, calibration |
| `PUT` | `/api/datasets/{id}/policy` | `{mode, aggregate, weights, threshold, minimums, rules}` → overview |
| `POST` | `/api/datasets/{id}/calibrate` | `{loPct, hiPct}` → refit the ranges |
| `POST` | `/api/datasets/{id}/semantic` | `{baseUrl, model, workers}` → job |
| `GET` | `/api/datasets/{id}/episodes` | table rows |
| `GET` | `/api/datasets/{id}/episodes/{ep}` | chart payload + full detail |
| `GET` | `/api/datasets/{id}/episodes/{ep}/video` | range-streamed mp4 |
| `GET` | `/api/datasets/{id}/export.csv` | measurements joined with scores |
| `GET` | `/api/datasets/{id}/export.json?decision=accept` | episode list |
| `GET` | `/api/jobs/{job_id}` | progress: `state`, `done`, `total`, `message` |

Jobs are polled rather than streamed — measurement emits one event per episode,
which is far too coarse to justify a socket.

Interactive API docs are at `/docs` (FastAPI's own Swagger UI).

## The episode view

| element | what it shows |
|---|---|
| rail | frames / episodes / fps, then every episode with a decision dot and its score; filter by decision, click to jump |
| camera row | every camera the dataset ships, not just the scored one — the wrist views usually explain what the base view cannot. All are driven by one transport |
| transport | play/pause, frame step, restart, scrub, frame counter. <kbd>space</kbd> plays, <kbd>↑</kbd><kbd>↓</kbd> change episode, <kbd>←</kbd><kbd>→</kbd> step a frame |
| combine all | every trace on one axis, each scaled to its own range — for spotting *coincidence* between signals, never for reading a value |
| video | streamed with HTTP range support, so the player seeks |
| charts | one card per signal: wrist speed and acceleration, arm and body joint acceleration, joint speed, tracking residual, base tilt, hand open/close, inter-frame difference |
| card header | the signal, its unit, and its live value at the playhead |
| red bands | confirmed contact events (2-of-3 signature quorum) |
| grey bands | frames below the adaptive idle velocity threshold |
| dashed line | the idle threshold on the joint-speed chart |
| raw table | the measured quantities, in their physical units |

Each signal gets **its own chart** rather than a lane in one tall plot. Nine
traces sharing a single y-axis give each of them the same sliver of it and none
of them enough; separate cards let every trace keep its own scale, label and
value while a common time axis, a shared playhead and the same contact/idle
shading keep them one instrument.

The traces are the arrays the metrics consumed, not a re-derivation of them —
what is on screen is what was scored.

Click a legend chip to hide a trace; click anywhere on the chart to seek.

## State and caching

Everything the app computes lives in `.quality_app/` next to where you started
it (override with `--state-dir` or `QUALITY_STATE_DIR`):

| file | contents |
|---|---|
| `datasets.json` | registered dataset paths |
| `<id>.measures.pkl` | cached measurements, calibration and semantic verdicts |
| `<id>.semantic.jsonl` | one verdict per line, append-only |

The pickle is written and read only by this app; a corrupt or version-stale file
is discarded rather than trusted. Deleting the directory costs one re-measure.

## Layout

```
app/
  __main__.py     python -m app
  main.py         FastAPI: routes, video streaming, static mounts
  service.py      registry, jobs, caching, scoring — no HTTP in it
  static/
    index.html    shell
    app.css       app-specific layout
    app.js        routing and the two views
```

The shared look and the chart come from
`src/score_lerobot_episodes/metrics/assets/`, served at `/assets`.

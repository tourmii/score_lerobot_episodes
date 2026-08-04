import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import os
import sys
import glob

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

# Only the humanoid scorers are imported at module scope; they depend on nothing
# beyond numpy/pandas/opencv.  The legacy HuggingFace path pulls in lerobot and
# a VLM SDK, so it is imported lazily inside `run_scoring_analysis` — that way
# the app still starts when those heavier optional deps are absent.
from score_lerobot_episodes.scores.humanoid import (
    CALIB,
    _ramp,
    build_time_stats as build_humanoid_time_stats,
    score_episode,
    signals_from_dataframe,
)

st.set_page_config(
    page_title="LeRobot Episode Scoring Toolkit",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

HUMANOID_METRICS = ["smoothness", "collision", "runtime", "acceleration", "visual_clarity"]

# Weights are normalised by their sum, so these are relative, not shares.
# `acceleration` is weighted up because it is the metric that most directly
# reflects the mechanical stress a trajectory puts on the robot, and unlike
# `smoothness` it is not invariant to how fast the motion was.
DEFAULT_WEIGHTS = {
    "smoothness": 0.20,
    "collision": 0.20,
    "runtime": 0.20,
    "acceleration": 0.40,
    "visual_clarity": 0.20,
}

# Raw diagnostics pulled out of the scorers' detail dicts for the results table.
DETAIL_FIELDS = {
    "runtime": ["duration_s", "idle_fraction", "grasp_transitions", "duration_z"],
    "collision": ["collision_events", "event_rate_hz", "max_base_tilt_deg",
                  "min_wrist_separation_m", "self_collision"],
    "acceleration": ["acc_arm_rms", "acc_body_rms", "acc_wrist_rms",
                     "acc_arm_p99", "acc_body_p99", "acc_wrist_p99"],
    "smoothness": ["ldlj_arm", "ldlj_body", "ldlj_wrist", "working_side"],
    "visual_clarity": ["sharpness", "brightness", "contrast", "interframe_diff", "decodable"],
}


# ---------------------------------------------------------------------------
# Humanoid scoring (local LeRobot folder)
# ---------------------------------------------------------------------------


def find_episode_files(root):
    """Map episode index -> parquet path for a local LeRobot dataset."""
    out = {}
    for path in sorted(glob.glob(os.path.join(root, "data", "**", "episode_*.parquet"),
                                 recursive=True)):
        out[int(os.path.basename(path).split("_")[1].split(".")[0])] = path
    return out


def find_episode_video(root, episode, camera):
    hits = glob.glob(os.path.join(root, "videos", "**", camera, f"episode_{episode:06d}.mp4"),
                     recursive=True)
    return hits[0] if hits else None


def list_cameras(root):
    meta = os.path.join(root, "meta", "info.json")
    if os.path.exists(meta):
        import json
        info = json.load(open(meta))
        cams = [k for k in info.get("features", {}) if k.startswith("observation.images.")]
        if cams:
            return cams
    return sorted({os.path.basename(os.path.dirname(p))
                   for p in glob.glob(os.path.join(root, "videos", "**", "*.mp4"), recursive=True)})


@st.cache_data(show_spinner=False)
def score_humanoid_dataset(root, camera, use_video, weights_key):
    """Score every episode. Cached on the argument tuple, so re-running with
    different weights or thresholds does not re-decode the videos."""
    episodes = find_episode_files(root)
    if not episodes:
        raise FileNotFoundError(f"No episode parquet files under {root}/data")

    progress = st.progress(0.0, text="Reading episodes…")
    signals = {}
    for i, (ep, path) in enumerate(sorted(episodes.items())):
        signals[ep] = signals_from_dataframe(pd.read_parquet(path))
        progress.progress((i + 1) / len(episodes) / 2, text=f"Reading episode {ep}…")

    # Duration scoring is relative to the dataset, so stats must be global.
    time_stats = build_humanoid_time_stats([s.duration for s in signals.values()])

    rows = []
    for i, ep in enumerate(sorted(signals)):
        video = find_episode_video(root, ep, camera) if (use_video and camera) else None
        result = score_episode(signals[ep], video_path=video, time_stats=time_stats)

        row = {"Episode": ep, **result["sub_scores"]}
        row["degenerate"] = bool(signals[ep].is_degenerate)
        row["video_path"] = video or ""
        for part, fields in DETAIL_FIELDS.items():
            detail = result["detail"].get(part, {})
            for f in fields:
                if f in detail:
                    row[f] = detail[f]
        rows.append(row)
        progress.progress(0.5 + (i + 1) / len(signals) / 2, text=f"Scoring episode {ep}…")
    progress.empty()

    df = pd.DataFrame(rows).set_index("Episode").sort_index()
    return df, time_stats


def apply_weights(df, weights, threshold):
    """Recompute the aggregate from the per-metric sub-scores.

    Kept separate from scoring so the weight sliders respond instantly instead
    of re-running the whole pipeline.
    """
    used = {m: w for m, w in weights.items() if m in df.columns and w > 0}
    out = df.copy()
    if not used:
        out["Aggregate Score"] = 0.0
    else:
        total = sum(used.values())
        out["Aggregate Score"] = sum(df[m] * w for m, w in used.items()) / total
    out["Status"] = np.where(out["Aggregate Score"] >= threshold, "GOOD", "BAD")
    return out


def scan_dataset_integrity(root, camera, scored_episodes):
    """Video files with no episode data, and videos that will not decode."""
    import cv2

    issues = {"orphaned": [], "unreadable": []}
    if not camera:
        return issues
    for path in sorted(glob.glob(os.path.join(root, "videos", "**", camera, "episode_*.mp4"),
                                 recursive=True)):
        ep = int(os.path.basename(path).split("_")[1].split(".")[0])
        if ep not in scored_episodes:
            issues["orphaned"].append(ep)
        cap = cv2.VideoCapture(path)
        ok = cap.isOpened() and cap.read()[0]
        cap.release()
        if not ok:
            issues["unreadable"].append(ep)
    return issues


# ---------------------------------------------------------------------------
# Views
# ---------------------------------------------------------------------------


def create_scoring_dashboard(results_df, distributions, agg_mean, criteria_names):
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Episode Scores Overview")

        plot_df = results_df.reset_index() if results_df.index.name == "Episode" else results_df
        hover = [c for c in criteria_names if c in plot_df.columns]
        fig = px.scatter(
            plot_df,
            x="Episode",
            y="Aggregate Score",
            color="Status",
            color_discrete_map={"GOOD": "#2ecc71", "BAD": "#e74c3c"},
            hover_data=hover,
            title="Episode Performance by Aggregate Score"
        )
        threshold = st.session_state.get("threshold", 0.5)
        fig.add_hline(y=threshold, line_dash="dash", line_color="gray",
                      annotation_text=f"Threshold ({threshold:.2f})")
        fig.update_layout(height=400)
        st.plotly_chart(fig, width="stretch")

        good_episodes = len(results_df[results_df["Status"] == "GOOD"])
        total_episodes = len(results_df)

        col_metric1, col_metric2, col_metric3 = st.columns(3)
        with col_metric1:
            st.metric("Total Episodes", total_episodes)
        with col_metric2:
            st.metric("Good Episodes", good_episodes,
                      f"{good_episodes/total_episodes*100:.1f}%" if total_episodes else "0%")
        with col_metric3:
            st.metric("Average Score", f"{agg_mean:.3f}")

    with col2:
        st.subheader("Score Distribution")

        fig_hist = go.Figure()
        fig_hist.add_trace(go.Histogram(
            x=results_df["Aggregate Score"],
            nbinsx=20,
            marker_color="#3498db",
            opacity=0.7
        ))
        fig_hist.add_vline(x=st.session_state.get("threshold", 0.5),
                           line_dash="dash", line_color="red", annotation_text="Threshold")
        fig_hist.update_layout(
            title="Aggregate Score Distribution",
            xaxis_title="Score",
            yaxis_title="Frequency",
            height=400
        )
        st.plotly_chart(fig_hist, width="stretch")


def create_criteria_analysis(distributions, criteria_names):
    st.subheader("Criteria Analysis")
    st.caption(
        "A metric whose box collapses to a line is not discriminating between "
        "episodes — check its calibration before trusting the aggregate."
    )

    cols = st.columns(min(3, max(len(criteria_names), 1)))

    for i, criterion in enumerate(criteria_names):
        with cols[i % 3]:
            scores = np.asarray(distributions[criterion], dtype=float)

            fig = go.Figure()
            fig.add_trace(go.Box(y=scores, name=criterion, marker_color="#9b59b6",
                                 boxpoints="outliers"))
            fig.update_layout(
                title=f"{criterion.replace('_', ' ').title()}",
                yaxis_title="Score",
                yaxis_range=[-0.05, 1.05],
                height=300,
                showlegend=False
            )
            st.plotly_chart(fig, width="stretch")

            spread = float(np.nanmax(scores) - np.nanmin(scores)) if scores.size else 0.0
            st.metric(f"Avg {criterion}", f"{np.nanmean(scores):.3f}", f"spread {spread:.3f}",
                      delta_color="off")


def create_metric_correlation(results_df, criteria_names):
    """Two metrics that correlate strongly are double-counting in the aggregate."""
    present = [m for m in criteria_names if m in results_df.columns]
    if len(present) < 2:
        return
    st.subheader("Metric Independence")
    corr = results_df[present].corr()
    fig = px.imshow(corr, text_auto=".2f", zmin=-1, zmax=1,
                    color_continuous_scale="RdBu_r", aspect="auto",
                    title="Sub-score correlation")
    fig.update_layout(height=380)
    st.plotly_chart(fig, width="stretch")
    st.caption(
        "High |correlation| means the two metrics measure overlapping things and "
        "the aggregate weights them twice."
    )


def create_data_quality_panel(results_df, integrity):
    st.subheader("Data Quality Flags")

    degenerate = results_df.index[results_df["degenerate"]].tolist() \
        if "degenerate" in results_df.columns else []
    orphaned = integrity.get("orphaned", [])
    unreadable = integrity.get("unreadable", [])

    c1, c2, c3 = st.columns(3)
    c1.metric("Degenerate episodes", len(degenerate))
    c2.metric("Orphaned videos", len(orphaned))
    c3.metric("Unreadable videos", len(unreadable))

    if degenerate:
        st.warning(
            f"**Degenerate (no meaningful end-effector motion): {degenerate}** — "
            "idle or reset recordings. These score 0 on the motion metrics by "
            "design; a plain RMS smoothness metric would rank them best."
        )
    if orphaned:
        st.warning(
            f"**Orphaned videos with no episode data: {orphaned}** — leftover "
            "files from an interrupted recording. Safe to delete."
        )
    if unreadable:
        st.error(f"**Unreadable / corrupt videos: {unreadable}** — these score 0 on visual_clarity.")
    if not (degenerate or orphaned or unreadable):
        st.success("No structural problems found.")


def create_score_breakdown(row, weights, episode):
    """Show the arithmetic behind one episode's aggregate score."""
    st.markdown("**How this score is calculated**")

    used = {m: w for m, w in weights.items() if m in row.index and w > 0}
    total_w = sum(used.values())
    if not total_w:
        st.info("All weights are zero — nothing contributes to the aggregate.")
        return

    breakdown = pd.DataFrame([
        {
            "metric": m,
            "sub-score": row[m],
            "weight": w,
            "share": w / total_w,
            "contribution": row[m] * w / total_w,
        }
        for m, w in used.items()
    ]).sort_values("contribution", ascending=False)

    aggregate = breakdown["contribution"].sum()

    st.dataframe(
        breakdown.style.format({
            "sub-score": "{:.3f}", "weight": "{:.2f}",
            "share": "{:.1%}", "contribution": "{:.4f}",
        }).bar(subset=["contribution"], color="#3498db"),
        width="stretch", hide_index=True,
    )

    terms = "  +  ".join(f"{r['share']:.3f}×{r['sub-score']:.3f}"
                         for _, r in breakdown.iterrows())
    st.markdown(f"**Episode {episode} aggregate** = {terms} = **{aggregate:.4f}**")
    st.caption(
        "Each metric's share is its weight divided by the sum of all weights, so "
        "the shares always add to 100% however you set the sliders."
    )

    # The acceleration term has its own internal aggregation worth exposing,
    # especially now that it carries the largest weight.
    groups = [g for g in ("arm", "body", "wrist") if f"acc_{g}_rms" in row.index]
    if "acceleration" in used and groups:
        with st.expander("…and how the acceleration sub-score is built"):
            rows = []
            for g in groups:
                rms, p99 = row[f"acc_{g}_rms"], row[f"acc_{g}_p99"]
                s_rms = _ramp(rms, **CALIB[f"acc_{g}_rms"])
                s_p99 = _ramp(p99, **CALIB[f"acc_{g}_p99"])
                rows.append({
                    "group": g,
                    "RMS": rms,
                    "RMS score": s_rms,
                    "p99": p99,
                    "p99 score": s_p99,
                    "group score": min(s_rms, s_p99),
                })
            acc_df = pd.DataFrame(rows)
            st.dataframe(
                acc_df.style.format({c: "{:.3f}" for c in acc_df.columns if c != "group"}),
                width="stretch", hide_index=True,
            )
            st.markdown(
                f"Each group takes **min(RMS score, p99 score)** — so one violent "
                f"transient cannot hide behind a calm average. The sub-score is the "
                f"mean over groups: **{acc_df['group score'].mean():.3f}**"
            )
            st.caption(
                "Raw values map to [0,1] against the dataset's 5th–95th percentile "
                "brackets in `CALIB`; lower acceleration scores higher. Units: "
                "rad/s² for arm and body, m/s² for the Cartesian wrist."
            )


def create_episode_detail(results_df, weights):
    st.subheader("Episode Detail")

    episode = st.selectbox("Episode", results_df.index.tolist(), key="detail_episode")
    row = results_df.loc[episode]

    left, right = st.columns([1, 1])

    with left:
        present = [m for m in HUMANOID_METRICS if m in results_df.columns]
        fig = go.Figure(go.Bar(
            x=[row[m] for m in present],
            y=[m.replace("_", " ") for m in present],
            orientation="h",
            marker_color=["#e74c3c" if row[m] < 0.4 else "#f39c12" if row[m] < 0.7 else "#2ecc71"
                          for m in present],
        ))
        fig.update_layout(title=f"Episode {episode} — sub-scores",
                          xaxis_range=[0, 1], height=300, margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig, width="stretch")

        create_score_breakdown(row, weights, episode)

        st.markdown("**Raw diagnostics**")
        # Formatted to strings: the column mixes floats with `working_side`,
        # and Arrow cannot serialise a mixed-dtype column.
        diag = {k: (f"{row[k]:.4g}" if isinstance(row[k], (int, float, np.number)) else str(row[k]))
                for k in ["duration_s", "idle_fraction", "grasp_transitions", "collision_events",
                          "max_base_tilt_deg", "min_wrist_separation_m", "working_side",
                          "acc_arm_rms", "acc_body_rms", "acc_wrist_rms", "sharpness"]
                if k in results_df.columns and pd.notna(row.get(k))}
        st.dataframe(pd.Series(diag, name="value").to_frame(), width="stretch")

    with right:
        video_path = row.get("video_path", "")
        if isinstance(video_path, str) and video_path and os.path.exists(video_path):
            st.video(video_path)
        else:
            st.info("No video available for this episode.")

        if "acc_arm_rms" in results_df.columns:
            st.markdown("**Acceleration by kinematic group** (this episode vs dataset)")
            groups = [("arm", "acc_arm_rms"), ("body", "acc_body_rms"), ("wrist", "acc_wrist_rms")]
            groups = [(g, c) for g, c in groups if c in results_df.columns]
            fig = go.Figure()
            fig.add_trace(go.Bar(name="this episode", x=[g for g, _ in groups],
                                 y=[row[c] for _, c in groups], marker_color="#3498db"))
            fig.add_trace(go.Bar(name="dataset median", x=[g for g, _ in groups],
                                 y=[results_df[c].median() for _, c in groups],
                                 marker_color="#95a5a6"))
            fig.update_layout(height=280, barmode="group",
                              yaxis_title="RMS accel (rad/s² · m/s² for wrist)",
                              margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(fig, width="stretch")


def create_detailed_table(results_df):
    st.subheader("Detailed Results")

    status_filter = st.selectbox("Filter by Status", ["All", "GOOD", "BAD"])

    filtered_df = results_df
    if status_filter != "All":
        filtered_df = results_df[results_df["Status"] == status_filter]

    filtered_df = filtered_df.drop(columns=["video_path"], errors="ignore")
    numeric = filtered_df.select_dtypes(include=[np.number]).columns

    st.dataframe(
        filtered_df.style.format({col: "{:.3f}" for col in numeric}),
        width="stretch"
    )

    st.download_button(
        "⬇️ Download scores as CSV",
        filtered_df.to_csv().encode("utf-8"),
        file_name="episode_scores.csv",
        mime="text/csv",
    )


# ---------------------------------------------------------------------------
# Legacy HuggingFace path
# ---------------------------------------------------------------------------


def run_scoring_analysis(repo_id, root_path, nominal_time):
    with st.spinner("Loading dataset and analyzing episodes..."):
        try:
            # Imported here rather than at module scope: these require lerobot
            # and a VLM SDK, which the local humanoid path does not need.
            from score_lerobot_episodes.data import (
                organize_by_episode, load_dataset_hf, get_scorable_video_segment)
            from score_lerobot_episodes.scores import build_time_stats, DatasetScorer

            dataset = load_dataset_hf(repo_id, root=root_path)
            episode_map = organize_by_episode(dataset)

            all_states = [episode_map[i]['states'] for i in episode_map]
            time_stats = build_time_stats(all_states)

            scorer = DatasetScorer(None, time_stats=time_stats)

            rows = []
            agg_mean = 0.0

            progress_bar = st.progress(0)
            total_episodes = len(episode_map)

            for idx, episode_index in enumerate(episode_map):
                episode = episode_map[episode_index]
                episode_total = 0

                for camera_type in episode['vid_paths']:
                    vid_path = episode['vid_paths'][camera_type]
                    video_info = episode.get('video_info', {}).get(camera_type, None)
                    states = episode['states']
                    actions = episode['actions']
                    task = episode['task']

                    scorable_video_segment = get_scorable_video_segment(vid_path, video_info)

                    total, subs = scorer.score(scorable_video_segment, states, actions,
                                               task, nominal_time)
                    rows.append((episode_index, camera_type, vid_path, total, subs))
                    episode_total += total

                agg_mean += episode_total / len(episode['vid_paths'])
                progress_bar.progress((idx + 1) / total_episodes)

            agg_mean /= len(rows)

            criteria_names = list(scorer.criteria.keys())
            distributions = {k: [] for k in criteria_names}

            results_data = []
            for ep_idx, cam, vid_path, total, subs in rows:
                row_data = {
                    "Episode": ep_idx,
                    "Camera": cam,
                    "Video Path": vid_path,
                    "Aggregate Score": total,
                    "Status": "GOOD" if total >= 0.5 else "BAD"
                }

                for k in criteria_names:
                    distributions[k].append(subs[k])
                    row_data[k] = subs[k]

                results_data.append(row_data)

            results_df = pd.DataFrame(results_data)

            return results_df, distributions, agg_mean, criteria_names, episode_map

        except ImportError as e:
            st.error(
                f"The HuggingFace path needs optional dependencies that are not installed: {e}. "
                "Install them with `pip install -e .`, or use **Humanoid (local folder)** mode, "
                "which needs only numpy/pandas/opencv."
            )
            return None, None, None, None, None
        except Exception as e:
            st.error(f"Error during analysis: {str(e)}")
            return None, None, None, None, None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def humanoid_sidebar():
    root_path = st.text_input(
        "Dataset folder",
        value=st.session_state.get("humanoid_root", ""),
        placeholder="e.g., pickup_20260628_150622",
        help="Local LeRobot dataset root — the folder containing meta/, data/, videos/",
    )

    cameras = list_cameras(root_path) if root_path and os.path.isdir(root_path) else []
    camera = st.selectbox("Camera", cameras) if cameras else None
    if root_path and not cameras:
        st.caption("No cameras found yet — check the folder path.")

    use_video = st.checkbox(
        "Score visual clarity", value=True,
        help="Decodes frames from every video. Uncheck for a much faster run.",
    )

    threshold = st.slider("GOOD threshold", 0.0, 1.0, 0.65, 0.01)

    with st.expander("Metric weights", expanded=True):
        st.caption(
            "Relative weights — they are normalised by their sum, so only the "
            "ratios matter. Set one to 0 to drop that metric entirely."
        )
        weights = {m: st.slider(m.replace("_", " "), 0.0, 1.0,
                                0.0 if (m == "visual_clarity" and not use_video)
                                else DEFAULT_WEIGHTS[m],
                                0.05, key=f"w_{m}")
                   for m in HUMANOID_METRICS}
        total = sum(weights.values())
        if total > 0:
            st.caption("Effective share: " + " · ".join(
                f"{m.split('_')[0]} {w/total:.0%}" for m, w in weights.items() if w > 0))

    run = st.button("🔍 Score Episodes", type="primary")
    return root_path, camera, use_video, threshold, weights, run


def main():
    st.title("🤖 LeRobot Episode Scoring Toolkit")
    st.markdown("Analyze and visualize robot episode performance with interactive dashboards")

    with st.sidebar:
        st.header("Configuration")

        mode = st.radio(
            "Scoring mode",
            ["Humanoid (local folder)", "Legacy (HuggingFace)"],
            help="Humanoid mode reads a local LeRobot folder and applies the "
                 "humanoid-calibrated metrics. Legacy mode uses the original "
                 "HuggingFace pipeline.",
        )

        export_filtered = False
        output_path = "./filtered_output"

        if mode == "Humanoid (local folder)":
            root_path, camera, use_video, threshold, weights, run = humanoid_sidebar()
        else:
            repo_id = st.text_input(
                "Repository ID",
                placeholder="e.g., lerobot/svla_so101_pickplace",
                help="HuggingFace repository ID for the dataset"
            )
            root_path = st.text_input(
                "Root Path (Optional)",
                placeholder="Leave empty for default cache",
                help="Local path to dataset root directory"
            )
            nominal_time = st.number_input(
                "Nominal Time", min_value=0.0, value=10.0, step=0.1,
                help="Reference time for runtime scoring"
            )
            run = st.button("🔍 Analyze Episodes", type="primary")

        st.markdown("---")
        with st.expander("Export Options"):
            export_filtered = st.checkbox("Save filtered dataset")
            output_path = st.text_input("Output Path", value="./filtered_output")

    # --- run ---
    if mode == "Humanoid (local folder)" and run:
        if not root_path or not os.path.isdir(root_path):
            st.error("Enter a valid dataset folder.")
        else:
            try:
                scores_df, time_stats = score_humanoid_dataset(
                    root_path, camera, use_video, tuple(sorted(weights.items())))
                integrity = scan_dataset_integrity(root_path, camera, set(scores_df.index))

                st.session_state.humanoid_scores = scores_df
                st.session_state.humanoid_integrity = integrity
                st.session_state.humanoid_root = root_path
                st.session_state.time_stats = time_stats
                st.session_state.mode = "humanoid"
                st.session_state.pop("results_df", None)
                st.success(f"✅ Scored {len(scores_df)} episodes.")
            except Exception as e:
                st.error(f"Error during scoring: {e}")

    if mode == "Legacy (HuggingFace)" and run and repo_id:
        root = root_path if root_path.strip() else None
        results_df, distributions, agg_mean, criteria_names, episode_map = run_scoring_analysis(
            repo_id, root, nominal_time
        )
        if results_df is not None:
            st.session_state.results_df = results_df
            st.session_state.distributions = distributions
            st.session_state.agg_mean = agg_mean
            st.session_state.criteria_names = criteria_names
            st.session_state.episode_map = episode_map
            st.session_state.repo_id = repo_id
            st.session_state.root_path = root
            st.session_state.output_path = output_path
            st.session_state.mode = "legacy"
            st.session_state.pop("humanoid_scores", None)
            st.success(
                f"✅ Analysis complete! Processed {len(results_df)} video segments "
                f"from {len(results_df['Episode'].unique())} episodes."
            )

    # --- render humanoid results ---
    if st.session_state.get("mode") == "humanoid" and "humanoid_scores" in st.session_state:
        scores_df = st.session_state.humanoid_scores
        st.session_state.threshold = threshold
        results_df = apply_weights(scores_df, weights, threshold)

        present = [m for m in HUMANOID_METRICS if m in results_df.columns]
        distributions = {m: results_df[m].tolist() for m in present}

        create_scoring_dashboard(results_df, distributions,
                                 float(results_df["Aggregate Score"].mean()), present)
        st.markdown("---")
        create_data_quality_panel(results_df, st.session_state.humanoid_integrity)
        st.markdown("---")
        create_criteria_analysis(distributions, present)
        st.markdown("---")
        create_metric_correlation(results_df, present)
        st.markdown("---")
        create_episode_detail(results_df, weights)
        st.markdown("---")
        create_detailed_table(results_df)

        with st.sidebar:
            if export_filtered and st.button("💾 Export Filtered Dataset"):
                from score_lerobot_episodes.data import save_filtered_dataset
                with st.spinner("Exporting filtered dataset..."):
                    try:
                        good = results_df.index[results_df["Status"] == "GOOD"].tolist()
                        save_filtered_dataset(st.session_state.humanoid_root, output_path, good)
                        st.success(f"✅ Filtered dataset saved to {output_path}")
                    except Exception as e:
                        st.error(f"Error exporting dataset: {str(e)}")

    # --- render legacy results ---
    elif st.session_state.get("mode") == "legacy" and "results_df" in st.session_state:
        st.session_state.threshold = 0.5
        create_scoring_dashboard(
            st.session_state.results_df,
            st.session_state.distributions,
            st.session_state.agg_mean,
            st.session_state.criteria_names
        )
        st.markdown("---")
        create_criteria_analysis(
            st.session_state.distributions,
            st.session_state.criteria_names
        )
        st.markdown("---")
        create_detailed_table(st.session_state.results_df)

        with st.sidebar:
            if export_filtered and st.button("💾 Export Filtered Dataset"):
                from score_lerobot_episodes.data import save_filtered_dataset
                with st.spinner("Exporting filtered dataset..."):
                    try:
                        good_episodes = st.session_state.results_df[
                            st.session_state.results_df["Status"] == "GOOD"
                        ]["Episode"].unique().tolist()

                        dataset_path = st.session_state.root_path
                        if not dataset_path:
                            cache_dir = os.path.expanduser("~/.cache/huggingface/lerobot/")
                            dataset_path = os.path.join(cache_dir, st.session_state.repo_id)

                        save_filtered_dataset(dataset_path, st.session_state.output_path,
                                              good_episodes)
                        st.success(f"✅ Filtered dataset saved to {st.session_state.output_path}")
                    except Exception as e:
                        st.error(f"Error exporting dataset: {str(e)}")

    else:
        st.info("👆 Configure a dataset in the sidebar and start scoring.")

        with st.expander("ℹ️ How to use this tool", expanded=True):
            st.markdown("""
            **Humanoid (local folder)** — point at a local LeRobot dataset folder
            (the one containing `meta/`, `data/`, `videos/`) and score it with metrics
            calibrated for teleoperated humanoid data. Needs only numpy/pandas/opencv.

            | metric | what it measures |
            |---|---|
            | **smoothness** | Log dimensionless jerk on the arm chain and the working wrist — normalised by duration and peak speed, so slow-careful and fast-jerky episodes are distinguishable |
            | **collision** | Contact proxy combining wrist impact deceleration, base disturbance and whole-body-control tracking divergence; an event needs two channels to agree |
            | **runtime** | Duration vs the dataset (robust z-score, overlong penalised harder), idle fraction, and regrasp count |
            | **acceleration** | Split by kinematic group — body (legs+waist), arms, wrist (Cartesian) — scoring the worse of RMS and 99th percentile |
            | **visual_clarity** | Sharpness/exposure/contrast, with a motion-blur allowance, gated on decode integrity and frozen-frame detection |

            Episodes with no meaningful end-effector motion are flagged **degenerate**
            and score 0 on the motion metrics — an idle recording would otherwise look
            maximally smooth.

            **Legacy (HuggingFace)** — the original pipeline, driven by a HF repo ID.
            Requires `lerobot` and a VLM SDK to be installed.
            """)


if __name__ == "__main__":
    main()

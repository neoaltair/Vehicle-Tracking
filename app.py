"""Interactive Web Application for Incident-Triggered Multi-Camera Vehicle Search."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

# Ensure project src is in python path
_REPO_ROOT = Path(__file__).resolve().parent
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

import streamlit as st  # noqa: E402
from PIL import Image  # noqa: E402

from incident_search.pipeline import IncidentSearchPipeline  # noqa: E402

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("app")

st.set_page_config(
    page_title="Incident-Triggered Vehicle Search",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom header styling
st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        color: #4B5563;
        font-size: 1.05rem;
        margin-bottom: 1.5rem;
    }
    .metric-box {
        background-color: #F3F4F6;
        border-radius: 8px;
        padding: 12px;
        text-align: center;
        border: 1px solid #E5E7EB;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="main-title">🚗 Incident-Triggered Multi-Camera Vehicle Search</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">Automated video ingestion, tracking, FastReID representation, and cross-camera spatiotemporal retrieval.</div>',
    unsafe_allow_html=True,
)


@st.cache_resource
def get_pipeline():
    weights = Path("outputs/weights/veri_sbs_R50-ibn.pth")
    return IncidentSearchPipeline(weights_path=weights)


pipeline = get_pipeline()

# Session state initialization
if "processed_cameras" not in st.session_state:
    st.session_state["processed_cameras"] = {}
if "selected_query_id" not in st.session_state:
    st.session_state["selected_query_id"] = None
if "search_results" not in st.session_state:
    st.session_state["search_results"] = None

# Sidebar: Configuration & Controls
with st.sidebar:
    st.header("⚙️ Configuration")
    time_window_s = st.slider(
        "Temporal Window (seconds)",
        min_value=10,
        max_value=600,
        value=60,
        step=10,
        help="Maximum time delta for cross-camera retrieval.",
    )
    alpha = st.slider(
        "Spatio-Temporal Prior Weight (α)",
        min_value=0.0,
        max_value=0.5,
        value=0.05,
        step=0.01,
        help="Balance between appearance similarity and spatiotemporal prior.",
    )
    top_k = st.number_input("Top-K Candidates", min_value=1, max_value=20, value=5)
    max_frames_process = st.number_input(
        "Max frames per video (0 = all)",
        min_value=0,
        max_value=2000,
        value=150,
        step=50,
        help="Use a small limit for quick interactive demonstration.",
    )

    st.markdown("---")
    st.subheader("📁 Input Feeds")
    feed_mode = st.radio("Source Mode", ["Use Sample Feeds", "Upload Custom Videos"])

# Video Processing Tabs
tab_ingest, tab_search = st.tabs(["🎥 1. Ingest & Track Feeds", "🔍 2. Cross-Camera Incident Search"])

with tab_ingest:
    st.markdown("### Step 1: Ingest Videos and Extract Vehicle Tracklets")
    st.write("Process camera video streams to detect vehicles, assign persistent tracklets, and extract deep ReID embeddings.")

    sample_video_path = Path("data/raw/samples/demo_traffic.mp4")

    col_c1, col_c2 = st.columns(2)

    with col_c1:
        st.info("📹 Camera 1 (North Intersection)")
        c1_time = st.number_input("Cam 1 Time Offset (s)", value=0.0, step=1.0)
        c1_video = None
        if feed_mode == "Use Sample Feeds":
            c1_video = sample_video_path if sample_video_path.exists() else None
            st.caption(f"Using sample feed: `{c1_video}`")
        else:
            c1_uploaded = st.file_uploader("Upload Cam 1 Video", type=["mp4", "avi", "mov"], key="c1_up")
            if c1_uploaded:
                temp_c1 = Path("outputs/temp_c1.mp4")
                temp_c1.parent.mkdir(parents=True, exist_ok=True)
                temp_c1.write_bytes(c1_uploaded.read())
                c1_video = temp_c1

        if st.button("Process Camera 1", key="btn_c1"):
            if c1_video and Path(c1_video).exists():
                with st.spinner("Tracking and extracting Cam 1..."):
                    limit = None if max_frames_process == 0 else int(max_frames_process)
                    trks = pipeline.process_camera_video(
                        camera_id="Cam_North",
                        video_path=c1_video,
                        time_offset_s=float(c1_time),
                        max_frames=limit,
                    )
                    st.session_state["processed_cameras"]["Cam_North"] = trks
                    st.success(f"Tracked {len(trks)} vehicles in Cam_North!")
            else:
                st.error("Cam 1 video not found.")

    with col_c2:
        st.info("📹 Camera 2 (South Junction)")
        c2_time = st.number_input("Cam 2 Time Offset (s)", value=15.0, step=1.0)
        c2_video = None
        if feed_mode == "Use Sample Feeds":
            c2_video = sample_video_path if sample_video_path.exists() else None
            st.caption(f"Using sample feed: `{c2_video}` (Simulating secondary view)")
        else:
            c2_uploaded = st.file_uploader("Upload Cam 2 Video", type=["mp4", "avi", "mov"], key="c2_up")
            if c2_uploaded:
                temp_c2 = Path("outputs/temp_c2.mp4")
                temp_c2.parent.mkdir(parents=True, exist_ok=True)
                temp_c2.write_bytes(c2_uploaded.read())
                c2_video = temp_c2

        if st.button("Process Camera 2", key="btn_c2"):
            if c2_video and Path(c2_video).exists():
                with st.spinner("Tracking and extracting Cam 2..."):
                    limit = None if max_frames_process == 0 else int(max_frames_process)
                    trks = pipeline.process_camera_video(
                        camera_id="Cam_South",
                        video_path=c2_video,
                        time_offset_s=float(c2_time),
                        max_frames=limit,
                    )
                    st.session_state["processed_cameras"]["Cam_South"] = trks
                    st.success(f"Tracked {len(trks)} vehicles in Cam_South!")
            else:
                st.error("Cam 2 video not found.")

    # Show summary of current tracklet pool
    st.markdown("---")
    st.markdown("#### Current Surveillance Tracklet Pool")
    if pipeline.tracklets:
        st.write(f"Total active tracklets in system: **{len(pipeline.tracklets)}** across **{len(set(t.camera_id for t in pipeline.tracklets))}** cameras.")

        # Display sample crops
        cols = st.columns(6)
        displayed = 0
        for trk in pipeline.tracklets:
            crops = [p for p in trk.crop_paths if Path(p).exists()]
            if crops and displayed < 6:
                img = Image.open(crops[len(crops) // 2])
                with cols[displayed]:
                    st.image(img, caption=f"{trk.camera_id} - {trk.tracklet_id}", use_container_width=True)
                displayed += 1
    else:
        st.info("No tracklets processed yet. Click 'Process Camera 1' & 'Process Camera 2' above.")

with tab_search:
    st.markdown("### Step 2: Trigger Incident Search")
    st.write("Select a suspect / query vehicle from an incident camera, and retrieve corresponding sightings across downstream cameras.")

    if not pipeline.tracklets:
        st.warning("Please process at least one video in Tab 1 first.")
    else:
        # Choose query vehicle
        cams = sorted(list(set(t.camera_id for t in pipeline.tracklets)))
        q_cam = st.selectbox("Incident Camera Location", cams)

        eligible_queries = [t for t in pipeline.tracklets if t.camera_id == q_cam and t.crop_paths]

        if not eligible_queries:
            st.warning(f"No valid vehicle tracklets with crops found in {q_cam}.")
        else:
            st.markdown("#### Pick Target Vehicle Crop:")
            q_cols = st.columns(min(len(eligible_queries), 6))
            for i, q_trk in enumerate(eligible_queries[:6]):
                crops = [p for p in q_trk.crop_paths if Path(p).exists()]
                if crops:
                    img = Image.open(crops[len(crops) // 2])
                    with q_cols[i]:
                        st.image(img, use_container_width=True)
                        if st.button(f"Select #{q_trk.tracklet_id}", key=f"sel_{q_trk.tracklet_id}"):
                            st.session_state["selected_query_id"] = q_trk.tracklet_id

            if st.session_state["selected_query_id"]:
                chosen_trk = next(
                    (t for t in pipeline.tracklets if t.tracklet_id == st.session_state["selected_query_id"]),
                    None,
                )
                if chosen_trk:
                    st.success(f"Selected Query: `{chosen_trk.tracklet_id}` (Observed at t={chosen_trk.start_time_s:.1f}s)")

                    if st.button("🚀 Search Vehicle Across Surveillance Network", type="primary"):
                        with st.spinner("Searching cross-camera network with Spatio-Temporal fusion..."):
                            res = pipeline.search_vehicle(
                                query_tracklet=chosen_trk,
                                time_window_s=float(time_window_s),
                                alpha=float(alpha),
                                top_k=int(top_k),
                            )
                            st.session_state["search_results"] = res

            # Render Results
            res = st.session_state["search_results"]
            if res is not None:
                st.markdown("---")
                st.markdown("### 🏆 Cross-Camera Retrieval Results")
                m1, m2, m3 = st.columns(3)
                m1.metric("Candidates Evaluated", res.candidates_before_pruning)
                m2.metric("Temporal Pruning Reduction", f"{res.pruning_reduction * 100:.1f}%")
                m3.metric("Top Matches Retrieved", len(res.results))

                if not res.results:
                    st.info("No cross-camera candidate vehicles met the temporal window criteria.")
                else:
                    res_cols = st.columns(len(res.results))
                    for idx, item in enumerate(res.results):
                        with res_cols[idx]:
                            crops = [p for p in item.crop_paths if Path(p).exists()]
                            if crops:
                                img = Image.open(crops[len(crops) // 2])
                                st.image(img, use_container_width=True)
                            st.markdown(f"**Rank {idx + 1}**")
                            st.caption(f"Camera: `{item.camera_id}`")
                            st.caption(f"Score: `{item.score:.3f}`")
                            st.caption(f"Appearance: `{item.appearance_score:.3f}`")
                            st.caption(f"Time: `{item.start_time_s:.1f}s`")

"""Interactive Live Web Application for Incident-Triggered Multi-Camera Vehicle Search."""

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
    page_title="Incident Vehicle Search - Live Feed Surveillance",
    page_icon="🚔",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.1rem;
        font-weight: 800;
        color: #1E3A8A;
        margin-bottom: 0.1rem;
    }
    .sub-title {
        color: #4B5563;
        font-size: 1.0rem;
        margin-bottom: 1.2rem;
    }
    .live-badge {
        background-color: #DC2626;
        color: white;
        padding: 3px 8px;
        border-radius: 4px;
        font-weight: bold;
        font-size: 0.75rem;
        margin-left: 6px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="main-title">🚔 Incident-Triggered Multi-Camera Surveillance</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">Real-time video tracking, vehicle identification badges, and cross-camera spatiotemporal re-identification.</div>',
    unsafe_allow_html=True,
)


@st.cache_resource
def get_pipeline():
    weights = Path("outputs/weights/veri_sbs_R50-ibn.pth")
    return IncidentSearchPipeline(weights_path=weights)


pipeline = get_pipeline()

# Session states
if "selected_query_id" not in st.session_state:
    st.session_state["selected_query_id"] = None
if "search_results" not in st.session_state:
    st.session_state["search_results"] = None

# Sidebar Controls
with st.sidebar:
    st.header("⚙️ Surveillance Controls")
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
    top_k = st.number_input("Top Matches to Retrieve", min_value=1, max_value=20, value=5)
    max_frames = st.number_input(
        "Max Stream Frames to Process",
        min_value=30,
        max_value=1000,
        value=150,
        step=30,
        help="Number of video frames to stream and track in real-time.",
    )

    st.markdown("---")
    st.subheader("📁 Video Feeds")
    feed_mode = st.radio("Source Mode", ["Built-in Traffic Feeds", "Upload Camera Feeds"])

tab_live, tab_search = st.tabs(["🎥 Live Camera Stream & Tracking", "🔍 Incident Re-Identification"])

sample_video_path = Path("data/raw/samples/demo_traffic.mp4")

with tab_live:
    st.markdown("### 🎥 Live Video Processing & Target Identification")
    st.write("Watch the live feed identify, assign persistent IDs, and extract vehicle crops frame-by-frame.")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### 📹 Camera 1: North Intersection <span class='live-badge'>FEED A</span>", unsafe_allow_html=True)
        c1_time = st.number_input("Cam 1 Timestamp Offset (s)", value=0.0, step=1.0)
        c1_video = None
        if feed_mode == "Built-in Traffic Feeds":
            c1_video = sample_video_path if sample_video_path.exists() else None
            st.caption(f"Active feed: `{c1_video}`")
        else:
            c1_uploaded = st.file_uploader("Upload Cam 1 Video", type=["mp4", "avi", "mov"], key="live_c1")
            if c1_uploaded:
                temp_c1 = Path("outputs/live_c1.mp4")
                temp_c1.parent.mkdir(parents=True, exist_ok=True)
                temp_c1.write_bytes(c1_uploaded.read())
                c1_video = temp_c1

        start_c1 = st.button("▶️ Start Live Stream (Cam 1)", type="primary", key="btn_stream_c1")
        view_c1 = st.empty()
        status_c1 = st.empty()

        if start_c1 and c1_video and Path(c1_video).exists():
            status_c1.info("Connecting to feed and initializing YOLO tracking...")
            for frame_rgb, idx, count in pipeline.stream_camera_video(
                camera_id="Cam_North",
                video_path=c1_video,
                time_offset_s=float(c1_time),
                max_frames=int(max_frames),
            ):
                view_c1.image(
                    frame_rgb,
                    caption=f"Cam_North | Frame #{idx} | Active Vehicles: {count}",
                    use_container_width=True,
                )
            status_c1.success("Camera 1 processing complete! Tracklets registered.")

    with col2:
        st.markdown("#### 📹 Camera 2: South Junction <span class='live-badge'>FEED B</span>", unsafe_allow_html=True)
        c2_time = st.number_input("Cam 2 Timestamp Offset (s)", value=15.0, step=1.0)
        c2_video = None
        if feed_mode == "Built-in Traffic Feeds":
            c2_video = sample_video_path if sample_video_path.exists() else None
            st.caption(f"Active feed: `{c2_video}` (Simulating downstream junction)")
        else:
            c2_uploaded = st.file_uploader("Upload Cam 2 Video", type=["mp4", "avi", "mov"], key="live_c2")
            if c2_uploaded:
                temp_c2 = Path("outputs/live_c2.mp4")
                temp_c2.parent.mkdir(parents=True, exist_ok=True)
                temp_c2.write_bytes(c2_uploaded.read())
                c2_video = temp_c2

        start_c2 = st.button("▶️ Start Live Stream (Cam 2)", type="primary", key="btn_stream_c2")
        view_c2 = st.empty()
        status_c2 = st.empty()

        if start_c2 and c2_video and Path(c2_video).exists():
            status_c2.info("Connecting to feed and initializing YOLO tracking...")
            for frame_rgb, idx, count in pipeline.stream_camera_video(
                camera_id="Cam_South",
                video_path=c2_video,
                time_offset_s=float(c2_time),
                max_frames=int(max_frames),
            ):
                view_c2.image(
                    frame_rgb,
                    caption=f"Cam_South | Frame #{idx} | Active Vehicles: {count}",
                    use_container_width=True,
                )
            status_c2.success("Camera 2 processing complete! Tracklets registered.")

    st.markdown("---")
    st.markdown("#### 🚗 Extracted Vehicle Gallery (Ready for Search)")
    if pipeline.tracklets:
        st.write(f"Total Registered Vehicles: **{len(pipeline.tracklets)}** across **{len(set(t.camera_id for t in pipeline.tracklets))}** cameras.")
        cols = st.columns(6)
        displayed = 0
        for trk in pipeline.tracklets:
            crops = [p for p in trk.crop_paths if Path(p).exists()]
            if crops and displayed < 6:
                img = Image.open(crops[len(crops) // 2])
                with cols[displayed]:
                    st.image(img, caption=f"{trk.camera_id}\n{trk.tracklet_id}", use_container_width=True)
                displayed += 1
    else:
        st.info("No vehicles tracked yet. Click ▶️ 'Start Live Stream' above to run tracking on camera feeds.")

with tab_search:
    st.markdown("### 🔍 Incident-Triggered Cross-Camera Search")
    st.write("Click on any suspect vehicle from an incident scene to retrieve its path and appearances across other cameras.")

    if not pipeline.tracklets:
        st.warning("Please run tracking on at least one camera feed in Tab 1 first.")
    else:
        cams = sorted(list(set(t.camera_id for t in pipeline.tracklets)))
        q_cam = st.selectbox("Select Incident Camera", cams)

        eligible_queries = [t for t in pipeline.tracklets if t.camera_id == q_cam and t.crop_paths]

        if not eligible_queries:
            st.warning(f"No tracked vehicles available in {q_cam}.")
        else:
            st.markdown("#### Select Suspect Vehicle:")
            q_cols = st.columns(min(len(eligible_queries), 6))
            for i, q_trk in enumerate(eligible_queries[:6]):
                crops = [p for p in q_trk.crop_paths if Path(p).exists()]
                if crops:
                    img = Image.open(crops[len(crops) // 2])
                    with q_cols[i]:
                        st.image(img, use_container_width=True)
                        if st.button(f"Pick #{q_trk.tracklet_id}", key=f"btn_pick_{q_trk.tracklet_id}"):
                            st.session_state["selected_query_id"] = q_trk.tracklet_id

            if st.session_state["selected_query_id"]:
                chosen_trk = next(
                    (t for t in pipeline.tracklets if t.tracklet_id == st.session_state["selected_query_id"]),
                    None,
                )
                if chosen_trk:
                    st.success(f"Selected Suspect: **{chosen_trk.tracklet_id}** at **{chosen_trk.camera_id}** (Time: {chosen_trk.start_time_s:.1f}s)")

                    if st.button("🚨 Run Cross-Camera Search", type="primary"):
                        with st.spinner("Searching surveillance network using FastReID & Spatio-Temporal fusion..."):
                            res = pipeline.search_vehicle(
                                query_tracklet=chosen_trk,
                                time_window_s=float(time_window_s),
                                alpha=float(alpha),
                                top_k=int(top_k),
                            )
                            st.session_state["search_results"] = res

            res = st.session_state["search_results"]
            if res is not None:
                st.markdown("---")
                st.markdown("### 🎯 Retrieval Matches Across Cameras")
                m1, m2, m3 = st.columns(3)
                m1.metric("Search Pool Evaluated", res.candidates_before_pruning)
                m2.metric("Spatiotemporal Pruning", f"{res.pruning_reduction * 100:.1f}%")
                m3.metric("Ranked Matches", len(res.results))

                if not res.results:
                    st.info("No matching cross-camera observations found within the temporal constraints.")
                else:
                    res_cols = st.columns(len(res.results))
                    for idx, item in enumerate(res.results):
                        with res_cols[idx]:
                            crops = [p for p in item.crop_paths if Path(p).exists()]
                            if crops:
                                img = Image.open(crops[len(crops) // 2])
                                st.image(img, use_container_width=True)
                            st.markdown(f"**Rank #{idx + 1}**")
                            st.caption(f"Camera: `{item.camera_id}`")
                            st.caption(f"Fused Score: `{item.score:.3f}`")
                            st.caption(f"Appearance: `{item.appearance_score:.3f}`")
                            st.caption(f"Observed: `{item.start_time_s:.1f}s`")

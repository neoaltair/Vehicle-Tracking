"""Multi-Camera Incident Investigation Console (Streamlit Web Application).

Accepts:
- 1 Main Video (Incident Scene)
- Multiple Background CCTV Footages (Surrounding Camera Network)
Performs:
- Cross-camera vehicle re-identification and upstream journey reconstruction.
"""

from __future__ import annotations

import json
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
from incident_search.search.journey import JourneyResult  # noqa: E402

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("app")

st.set_page_config(
    page_title="Multi-Camera Incident Search Console",
    page_icon="🚔",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 800;
        color: #0F172A;
        margin-bottom: 0.1rem;
    }
    .sub-title {
        color: #475569;
        font-size: 1.05rem;
        margin-bottom: 1.5rem;
    }
    .incident-badge {
        background-color: #EF4444;
        color: white;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 700;
        font-size: 0.85rem;
    }
    .cctv-badge {
        background-color: #3B82F6;
        color: white;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 700;
        font-size: 0.85rem;
    }
    .journey-timeline {
        background-color: #F8FAFC;
        border-left: 4px solid #3B82F6;
        padding: 12px 18px;
        border-radius: 6px;
        margin-bottom: 15px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="main-title">🚔 Multi-Camera Incident Investigation Console</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">1 Main Incident Feed + Multiple Background CCTV Footages ➔ Cross-Camera Journey Reconstruction.</div>',
    unsafe_allow_html=True,
)


@st.cache_resource
def get_pipeline():
    weights = Path("outputs/weights/veri_sbs_R50-ibn.pth")
    return IncidentSearchPipeline(weights_path=weights)


pipeline = get_pipeline()

# Session State
if "main_video_processed" not in st.session_state:
    st.session_state["main_video_processed"] = False
if "bg_videos_processed" not in st.session_state:
    st.session_state["bg_videos_processed"] = False
if "detected_incidents" not in st.session_state:
    st.session_state["detected_incidents"] = []
if "selected_query_trk" not in st.session_state:
    st.session_state["selected_query_trk"] = None
if "journey_result" not in st.session_state:
    st.session_state["journey_result"] = None

# Sidebar Controls
with st.sidebar:
    st.header("⚙️ Investigation Parameters")
    time_window_s = st.slider(
        "Max Upstream Time Window (s)",
        min_value=30,
        max_value=900,
        value=300,
        step=30,
        help="Search window looking backward in time before the incident.",
    )
    alpha = st.slider(
        "Spatio-Temporal Prior Weight (α)",
        min_value=0.0,
        max_value=0.3,
        value=0.05,
        step=0.01,
        help="Weight assigned to transition graph / travel-time prior.",
    )
    min_match_score = st.slider(
        "Minimum Match Score",
        min_value=0.10,
        max_value=0.80,
        value=0.25,
        step=0.05,
        help="Confidence threshold for displaying background CCTV sightings.",
    )
    max_frames = st.number_input(
        "Max Frames to Ingest per Video",
        min_value=30,
        max_value=1200,
        value=200,
        step=50,
        help="Use 150-300 frames for fast interactive search.",
    )

    st.markdown("---")
    st.subheader("📹 Input Source Selection")
    data_mode = st.radio("Feeds Mode", ["Use Built-in Surveillance Feeds", "Upload Your Own Videos"])

tab1, tab2, tab3 = st.tabs([
    "1️⃣ Ingest Feeds (Main + Background CCTV)",
    "2️⃣ Incident Spotting & Target Selection",
    "3️⃣ Cross-Camera Route & Journey Retrieval",
])

def ensure_sample_video() -> Path:
    sample_dir = Path("data/raw/samples")
    sample_dir.mkdir(parents=True, exist_ok=True)
    sample_file = sample_dir / "demo_traffic.mp4"
    if not sample_file.exists() or sample_file.stat().st_size == 0:
        url = "https://upload.wikimedia.org/wikipedia/commons/6/68/Avtocesta.webm"
        import urllib.request
        try:
            logger.info("Auto-downloading demo traffic video from %s...", url)
            urllib.request.urlretrieve(url, str(sample_file))
        except Exception as e:
            logger.warning("Could not auto-download sample video: %s", e)
    return sample_file


sample_path = ensure_sample_video()

# ==============================================================================
# TAB 1: INGEST FEEDS
# ==============================================================================
with tab1:
    st.markdown("### Step 1: Supply the Main Incident Video & Background CCTV Footages")

    col_main, col_bg = st.columns([1, 1])

    with col_main:
        st.markdown("#### 🚨 Primary Main Video <span class='incident-badge'>INCIDENT SCENE</span>", unsafe_allow_html=True)
        main_cam_id = st.text_input("Main Camera Identifier", value="Main_Cam_Incident")
        main_offset = st.number_input("Main Camera Time Offset (s)", value=0.0, step=1.0)

        main_video_file = None
        if data_mode == "Use Built-in Surveillance Feeds":
            main_video_file = sample_path if sample_path.exists() else None
            st.caption(f"Using sample feed: `{main_video_file}`")
        else:
            up_main = st.file_uploader("Upload Main Video (MP4 / AVI)", type=["mp4", "avi", "mov"], key="up_main")
            if up_main:
                t_main = Path("outputs/uploads/main_video.mp4")
                t_main.parent.mkdir(parents=True, exist_ok=True)
                t_main.write_bytes(up_main.read())
                main_video_file = t_main

        if st.button("Process Main Incident Video", type="primary", key="btn_proc_main"):
            if main_video_file and Path(main_video_file).exists():
                with st.spinner("Tracking vehicles in Main Incident video..."):
                    trks = pipeline.process_camera_video(
                        camera_id=main_cam_id,
                        video_path=main_video_file,
                        time_offset_s=float(main_offset),
                        is_main=True,
                        max_frames=int(max_frames),
                    )
                    st.session_state["main_video_processed"] = True
                    # Run automated incident detection
                    incs = pipeline.detect_incidents_in_camera(main_cam_id)
                    st.session_state["detected_incidents"] = incs
                    st.success(f"Tracked {len(trks)} vehicles in {main_cam_id}! ({len(incs)} incidents detected)")
            else:
                st.error("Main video file not found.")

    with col_bg:
        st.markdown("#### 📹 Background CCTV Network <span class='cctv-badge'>CCTV NETWORK</span>", unsafe_allow_html=True)
        st.write("Upload or configure multiple surrounding camera feeds across the road network.")

        bg_cams_to_run = []

        if data_mode == "Use Built-in Surveillance Feeds":
            st.caption("Configuring 2 surrounding background CCTV cameras using demo stream with time shifts:")
            st.write("• **CCTV_North_Junction**: Offset -45s (Upstream junction)")
            st.write("• **CCTV_Highway_Entry**: Offset -90s (Upstream highway ramp)")
            if sample_path.exists():
                bg_cams_to_run.append(("CCTV_North_Junction", sample_path, -45.0))
                bg_cams_to_run.append(("CCTV_Highway_Entry", sample_path, -90.0))
        else:
            up_bgs = st.file_uploader(
                "Upload Background CCTV Videos (Select Multiple)",
                type=["mp4", "avi", "mov"],
                accept_multiple_files=True,
                key="up_bgs",
            )
            if up_bgs:
                for idx, f in enumerate(up_bgs):
                    t_bg = Path(f"outputs/uploads/cctv_cam_{idx + 1}.mp4")
                    t_bg.parent.mkdir(parents=True, exist_ok=True)
                    t_bg.write_bytes(f.read())
                    cam_id = f"CCTV_Camera_{idx + 1}"
                    offset = -(idx + 1) * 35.0  # Default staggered offset
                    bg_cams_to_run.append((cam_id, t_bg, offset))
                st.caption(f"Loaded {len(bg_cams_to_run)} background CCTV camera files.")

        if st.button("Process & Index Background CCTV Feeds", type="secondary", key="btn_proc_bg"):
            if not bg_cams_to_run:
                st.error("No background CCTV videos available.")
            else:
                with st.spinner(f"Ingesting {len(bg_cams_to_run)} CCTV feeds into multi-camera index..."):
                    total_trks = 0
                    for c_id, v_path, c_offset in bg_cams_to_run:
                        t_list = pipeline.process_camera_video(
                            camera_id=c_id,
                            video_path=v_path,
                            time_offset_s=float(c_offset),
                            is_main=False,
                            max_frames=int(max_frames),
                        )
                        total_trks += len(t_list)
                    st.session_state["bg_videos_processed"] = True
                    st.success(f"Indexed {total_trks} vehicles across {len(bg_cams_to_run)} background CCTV cameras!")

    st.markdown("---")
    st.markdown("#### 📊 Unified Multi-Camera Index Status")
    s1, s2, s3 = st.columns(3)
    main_count = len([t for t in pipeline.tracklets if t.camera_id == pipeline.main_camera_id])
    bg_count = len([t for t in pipeline.tracklets if t.camera_id in pipeline.background_camera_ids])
    s1.metric("Main Video Vehicles", main_count)
    s2.metric("Background CCTV Vehicles", bg_count)
    s3.metric("Registered Background Cameras", len(pipeline.background_camera_ids))

# ==============================================================================
# TAB 2: INCIDENT DETECTION & TARGET SELECTION
# ==============================================================================
with tab2:
    st.markdown("### Step 2: Identify Suspect Vehicle in the Main Video")

    if not st.session_state["main_video_processed"]:
        st.warning("Please process the Main Incident Video in Tab 1 first.")
    else:
        main_trks = [t for t in pipeline.tracklets if t.camera_id == pipeline.main_camera_id and t.crop_paths]

        # 1. Automated Incident Alerts
        incs = st.session_state["detected_incidents"]
        if incs:
            st.error(f"🚨 {len(incs)} Automated Incident(s) Detected in Main Video:")
            for inc in incs:
                st.write(f"• **{inc.incident_id}** at t={inc.time_s:.1f}s: {inc.description} (Confidence: {inc.confidence:.2f})")
                if st.button(f"Investigate Incident {inc.incident_id}", key=f"btn_inc_{inc.incident_id}"):
                    matched_trk = next((t for t in main_trks if t.tracklet_id in inc.involved_tracklet_ids), None)
                    if matched_trk:
                        st.session_state["selected_query_trk"] = matched_trk
        else:
            st.info("No sudden stop or kinematic collision flagged automatically. You can select any target vehicle below.")

        st.markdown("---")
        st.markdown("#### Pick Suspect Vehicle from Main Video:")

        if not main_trks:
            st.warning("No vehicle crops available from Main Video.")
        else:
            q_cols = st.columns(min(len(main_trks), 6))
            for idx, trk in enumerate(main_trks[:6]):
                v_crops = [p for p in trk.crop_paths if Path(p).exists()]
                if v_crops:
                    img = Image.open(v_crops[len(v_crops) // 2])
                    with q_cols[idx]:
                        st.image(img, use_container_width=True)
                        st.caption(f"ID: `{trk.tracklet_id}`\nt={trk.start_time_s:.1f}s")
                        if st.button("Select Car", key=f"sel_main_{trk.tracklet_id}"):
                            st.session_state["selected_query_trk"] = trk

        selected = st.session_state["selected_query_trk"]
        if selected:
            st.success(f"🎯 Target Vehicle Selected: **{selected.tracklet_id}** (First seen at {selected.start_time_s:.1f}s)")

# ==============================================================================
# TAB 3: CROSS-CAMERA RETRIEVAL & JOURNEY
# ==============================================================================
with tab3:
    st.markdown("### Step 3: Cross-Camera Vehicle Journey Reconstruction")
    st.write("Searches across all background CCTV footages backward in time to reconstruct the suspect's path.")

    target = st.session_state["selected_query_trk"]

    if not target:
        st.warning("Please select a target vehicle from the Main Video in Tab 2 first.")
    elif not st.session_state["bg_videos_processed"]:
        st.warning("Please process the Background CCTV Feeds in Tab 1 first.")
    else:
        # Run Journey Reconstruction
        if st.button("🚀 Reconstruct Cross-Camera Journey", type="primary"):
            with st.spinner("Searching background CCTV network and computing spatiotemporal route..."):
                j_res: JourneyResult = pipeline.trigger_incident_search(
                    query_tracklet=target,
                    time_window_s=float(time_window_s),
                    alpha=float(alpha),
                    min_score=float(min_match_score),
                )
                st.session_state["journey_result"] = j_res

        res: JourneyResult = st.session_state["journey_result"]

        if res is not None:
            st.markdown("---")
            st.markdown("### 🗺️ Reconstructed Upstream Route")
            st.markdown(
                f"<div class='journey-timeline'><b>Vehicle Path (Earliest ➔ Incident):</b><br><h3>{res.route_str}</h3></div>",
                unsafe_allow_html=True,
            )

            # Metrics
            m1, m2, m3 = st.columns(3)
            m1.metric("CCTV Candidates Evaluated", res.candidates_evaluated)
            m2.metric("Spatiotemporal Pruning", f"{res.pruning_reduction * 100:.1f}%")
            m3.metric("CCTV Cameras with Confirmed Matches", len(res.sightings))

            st.markdown("---")
            st.markdown("### 🔍 Sighting Verification Cards")

            col_q, col_matches = st.columns([1, 2])

            with col_q:
                st.markdown("#### 🚨 Query Vehicle (Main Video)")
                if res.query_crop_path and Path(res.query_crop_path).exists():
                    st.image(Image.open(res.query_crop_path), use_container_width=True)
                st.write(f"• **Camera:** `{res.incident.camera_id}`")
                st.write(f"• **Tracklet:** `{res.query_tracklet.tracklet_id}`")
                st.write(f"• **Incident Time:** `{res.incident.time_s:.1f}s`")
                st.write(f"• **Class:** `{res.query_tracklet.vehicle_class}`")

            with col_matches:
                st.markdown("#### 📸 Retrieved Sightings in Surrounding CCTV Cameras")
                if not res.sightings:
                    st.info("No matching vehicles found in background CCTV cameras within the temporal search window.")
                else:
                    for s_idx, sighting in enumerate(res.sightings):
                        st.markdown(f"##### Sighting #{s_idx + 1} — Camera: `{sighting.camera_id}`")
                        sc1, sc2 = st.columns([1, 2])
                        with sc1:
                            if sighting.crop_path and Path(sighting.crop_path).exists():
                                st.image(Image.open(sighting.crop_path), use_container_width=True)
                        with sc2:
                            st.write(f"• **Match Score:** `{sighting.score:.3f}` (Appearance: {sighting.appearance_score:.3f})")
                            st.write(f"• **Observation Timestamp:** `{sighting.timestamp_s:.1f}s`")
                            st.write(f"• **Time Delta to Incident:** `{sighting.time_delta_s:+.1f}s` upstream")
                            st.write(f"• **Tracklet ID:** `{sighting.tracklet.tracklet_id}`")
                        st.markdown("---")

            # JSON Dossier Export
            dossier = {
                "incident_id": res.incident.incident_id,
                "incident_camera": res.incident.camera_id,
                "incident_time_s": res.incident.time_s,
                "query_tracklet_id": res.query_tracklet.tracklet_id,
                "reconstructed_route": res.route_str,
                "sightings": [
                    {
                        "camera_id": s.camera_id,
                        "timestamp_s": s.timestamp_s,
                        "time_delta_s": s.time_delta_s,
                        "match_score": round(s.score, 3),
                        "appearance_score": round(s.appearance_score, 3),
                        "tracklet_id": s.tracklet.tracklet_id,
                    }
                    for s in res.sightings
                ],
            }

            st.download_button(
                label="📥 Export Investigation Case Dossier (JSON)",
                data=json.dumps(dossier, indent=2),
                file_name=f"investigation_dossier_{res.incident.incident_id}.json",
                mime="application/json",
            )

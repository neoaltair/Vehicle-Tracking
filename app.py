"""Industrial Multi-Camera Incident Surveillance & Journey Reconstruction Console.

Accepts:
- 1 Main Video (Incident Scene)
- Multiple Background CCTV Footages (Surrounding Camera Network)
Performs:
- Synchronized multi-camera vehicle tracking with H.264 annotated video replay.
- Automated kinematic incident detection (emergency deceleration / vehicle stalls).
- Deep metric vehicle re-identification (FastReID SBS ResNet-50-IBN).
- Spatio-temporal upstream journey reconstruction and forensic side-by-side verification.
"""

from __future__ import annotations

import json
import logging
import sys
import tarfile
import urllib.request
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
    page_title="SENTINEL | Multi-Camera Incident Reconstruction Console",
    page_icon="🚔",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Command-Center Styling
st.markdown(
    """
    <style>
    .reportview-container, .main {
        background-color: #0B0F19;
    }
    .header-box {
        background: linear-gradient(135deg, #0F172A 0%, #1E293B 100%);
        border: 1px solid #334155;
        border-radius: 10px;
        padding: 20px 24px;
        margin-bottom: 20px;
    }
    .main-title {
        font-size: 2.1rem;
        font-weight: 800;
        color: #F8FAFC;
        margin: 0;
        letter-spacing: -0.5px;
    }
    .main-subtitle {
        color: #94A3B8;
        font-size: 1.0rem;
        margin-top: 6px;
        margin-bottom: 0px;
    }
    .status-badge {
        display: inline-block;
        padding: 3px 9px;
        border-radius: 4px;
        font-size: 0.75rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .badge-incident { background-color: #DC2626; color: #FFFFFF; }
    .badge-cctv { background-color: #2563EB; color: #FFFFFF; }
    .badge-success { background-color: #059669; color: #FFFFFF; }
    .badge-clock { background-color: #334155; color: #E2E8F0; font-family: monospace; }

    .timeline-card {
        background-color: #1E293B;
        border: 1px solid #3B82F6;
        border-radius: 8px;
        padding: 16px 20px;
        margin: 15px 0px;
    }
    .sighting-card {
        background-color: #0F172A;
        border: 1px solid #334155;
        border-radius: 8px;
        padding: 14px;
        margin-bottom: 12px;
    }
    .metric-value {
        font-size: 1.4rem;
        font-weight: 700;
        color: #F8FAFC;
    }
    .metric-label {
        font-size: 0.8rem;
        color: #94A3B8;
        text-transform: uppercase;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def format_surveillance_clock(seconds: float, base_hour: int = 14, base_minute: int = 30) -> str:
    """Format elapsed seconds into synchronized surveillance clock string HH:MM:SS.f."""
    total_sec = max(0.0, seconds)
    mins, secs = divmod(total_sec, 60.0)
    hours, mins = divmod(mins, 60.0)
    h = int((base_hour + hours) % 24)
    m = int((base_minute + mins) % 60)
    s = int(secs)
    cs = int((secs - s) * 10)
    return f"{h:02d}:{m:02d}:{s:02d}.{cs:01d}"


def format_time_delta(delta_s: float) -> str:
    """Format time difference relative to incident."""
    sign = "-" if delta_s < 0 else "+"
    sec = abs(delta_s)
    m, s = divmod(sec, 60)
    if m > 0:
        return f"{sign}{int(m)}m {s:.1f}s"
    return f"{sign}{s:.1f}s"


def ensure_cityflow_adjacent_pair() -> tuple[Path, Path]:
    """Download and unpack official CityFlow adjacent corridor CCTV pair (c003 & c004)."""
    dest_dir = Path("data/raw/mtmct-demo")
    dest_dir.mkdir(parents=True, exist_ok=True)
    c3_path = dest_dir / "c003.mp4"
    c4_path = dest_dir / "c004.mp4"

    if not c3_path.exists() or not c4_path.exists() or c3_path.stat().st_size == 0:
        tar_dest = Path("data/raw/mtmct-demo.tar")
        url = "https://paddledet.bj.bcebos.com/data/mot/demo/mtmct-demo.tar"
        logger.info("Downloading official CityFlow adjacent CCTV pair from %s...", url)
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Sentinel-Console"})
            data = urllib.request.urlopen(req, timeout=60).read()
            tar_dest.write_bytes(data)
            with tarfile.open(str(tar_dest)) as tar:
                tar.extractall("data/raw/")
        except Exception as e:
            logger.warning("Could not auto-download CityFlow pair: %s", e)
    return c3_path, c4_path


@st.cache_resource
def get_pipeline():
    weights = Path("outputs/weights/veri_sbs_R50-ibn.pth")
    return IncidentSearchPipeline(weights_path=weights)


pipeline = get_pipeline()

# Session State Initialization
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
if "active_cctv_view" not in st.session_state:
    st.session_state["active_cctv_view"] = None

# Header Banner
st.markdown(
    """
    <div class="header-box">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <div>
                <h1 class="main-title">🚨 SENTINEL | Multi-Camera Incident Investigation Console</h1>
                <p class="main-subtitle">Automated Kinematic Incident Detection ➔ Multi-CCTV Vehicle Search ➔ Journey Reconstruction</p>
            </div>
            <div style="text-align: right;">
                <span class="status-badge badge-clock">SYNC TIME: 14:30:00 EST</span><br>
                <span class="status-badge badge-incident" style="margin-top:6px;">INVESTIGATION ACTIVE</span>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Sidebar: Investigation Controls & Surveillance Feed Setup
with st.sidebar:
    st.header("⚙️ Surveillance Control")
    feed_mode = st.radio(
        "Surveillance Feed Mode",
        [
            "CityFlow Adjacent Road CCTV Feeds (Coherent Corridor Stream)",
            "Local / Drive CityFlow Dataset Folder",
            "Upload Custom Camera Videos",
        ],
        index=0,
    )

    st.markdown("---")
    st.subheader("⏱️ Spatio-Temporal Parameters")
    lookback_window = st.slider(
        "Max Upstream Search Window (s)",
        min_value=30,
        max_value=900,
        value=300,
        step=15,
        help="Search window looking backward in time prior to incident.",
    )
    prior_weight = st.slider(
        "Spatio-Temporal Prior Weight (α)",
        min_value=0.0,
        max_value=0.30,
        value=0.05,
        step=0.01,
        help="Weight for camera transition graph & travel-time prior.",
    )
    match_threshold = st.slider(
        "Re-ID Similarity Threshold",
        min_value=0.10,
        max_value=0.85,
        value=0.25,
        step=0.05,
        help="Minimum confidence required to confirm a background camera sighting.",
    )
    max_frames_to_process = st.number_input(
        "Max Frames to Ingest per Camera",
        min_value=30,
        max_value=1200,
        value=120,
        step=30,
        help="Use 80-150 frames for fast interactive search.",
    )

    st.markdown("---")
    st.caption("Backend: YOLO11 + ByteTrack + FastReID SBS (VeRi-776)")


# Tabs for 3-Step Investigation Workflow
tab1, tab2, tab3 = st.tabs([
    "1️⃣ Surveillance Wall & Feeds (YOLO Replay)",
    "2️⃣ Incident Spotting & Suspect Lock",
    "3️⃣ Cross-Camera Route & Forensic Verification",
])

# ==============================================================================
# TAB 1: SURVEILLANCE WALL & VIDEO INGESTION
# ==============================================================================
with tab1:
    st.markdown("### Step 1: Multi-Camera Feed Ingestion & Tracking Overlays")
    st.write(
        "Process surveillance feeds from neighbouring roads along the traffic network. "
        "Generates YOLO tracking bounding boxes, unique vehicle IDs, and synchronized timecode overlays."
    )

    col_main_setup, col_bg_setup = st.columns([1, 1])

    main_video_path = None
    main_cam_id = "c003_IncidentScene"
    bg_cams_to_run = []

    if feed_mode == "CityFlow Adjacent Road CCTV Feeds (Coherent Corridor Stream)":
        c3_file, c4_file = ensure_cityflow_adjacent_pair()
        main_video_path = c3_file
        main_cam_id = "c003_Intersection"
        bg_cams_to_run = [
            ("c004_UpstreamRoad", c4_file, -15.0, "Bellevue NE 148th Adjacent Road (180m Upstream)"),
        ]

        with col_main_setup:
            st.markdown(
                "#### 🚨 Primary Incident Feed <span class='status-badge badge-incident'>INCIDENT SCENE</span>",
                unsafe_allow_html=True,
            )
            st.info(
                f"**Camera:** `{main_cam_id}`\n\n"
                "**Location:** Bellevue NE 148th Intersection (Incident Location)\n\n"
                "**Footage:** Coherent CityFlow Corridor Stream (`c003.mp4`)"
            )

        with col_bg_setup:
            st.markdown(
                "#### 📹 Background CCTV Network <span class='status-badge badge-cctv'>NEIGHBOURING ROAD</span>",
                unsafe_allow_html=True,
            )
            st.info(
                "**Camera:** `c004_UpstreamRoad`\n\n"
                "**Location:** Bellevue NE 148th Adjacent Road (180m Upstream)\n\n"
                "**Synchronization Offset:** `-15.0s` (Vehicles drive from c004 into c003)"
            )

    elif feed_mode == "Local / Drive CityFlow Dataset Folder":
        with col_main_setup:
            st.markdown("#### 🚨 CityFlow Dataset Directory", unsafe_allow_html=True)
            ds_dir = st.text_input("CityFlow Scenario Directory Path", value="data/raw/cityflowv2/train/S01")
            p_dir = Path(ds_dir)
            if p_dir.exists():
                c_dirs = sorted([d.name for d in p_dir.iterdir() if d.is_dir() and (d / "vdo.avi").exists()])
                if len(c_dirs) >= 2:
                    sel_main = st.selectbox("Select Incident Camera:", c_dirs, index=0)
                    main_cam_id = sel_main
                    main_video_path = p_dir / sel_main / "vdo.avi"

                    with col_bg_setup:
                        st.markdown("#### 📹 Surrounding Cameras", unsafe_allow_html=True)
                        bg_choices = [c for c in c_dirs if c != sel_main]
                        sel_bgs = st.multiselect("Select Background CCTV Cameras:", bg_choices, default=bg_choices[:2])
                        for idx, b_name in enumerate(sel_bgs):
                            bg_cams_to_run.append((b_name, p_dir / b_name / "vdo.avi", -(idx + 1) * 20.0, f"Camera {b_name}"))
                else:
                    st.warning(f"Found fewer than 2 camera folders with vdo.avi in {ds_dir}")
            else:
                st.info("Enter path to CityFlow scenario folder containing camera directories (e.g. c001, c002).")

    else:  # Custom Uploads
        with col_main_setup:
            st.markdown("#### 🚨 Upload Main Incident Video", unsafe_allow_html=True)
            main_cam_id = st.text_input("Main Camera ID", value="CAM_01_INCIDENT")
            up_main = st.file_uploader("Upload Main Video (MP4/AVI)", type=["mp4", "avi", "mov"], key="up_main")
            if up_main:
                t_main = Path("outputs/uploads/main_incident.mp4")
                t_main.parent.mkdir(parents=True, exist_ok=True)
                t_main.write_bytes(up_main.read())
                main_video_path = t_main

        with col_bg_setup:
            st.markdown("#### 📹 Upload Surrounding CCTV Feeds", unsafe_allow_html=True)
            up_bgs = st.file_uploader(
                "Upload Background CCTV Videos (Multiple)",
                type=["mp4", "avi", "mov"],
                accept_multiple_files=True,
                key="up_bgs",
            )
            if up_bgs:
                for idx, f in enumerate(up_bgs):
                    t_bg = Path(f"outputs/uploads/cctv_cam_{idx + 1}.mp4")
                    t_bg.parent.mkdir(parents=True, exist_ok=True)
                    t_bg.write_bytes(f.read())
                    bg_cams_to_run.append((f"CCTV_CAM_{idx + 1:02d}", t_bg, -(idx + 1) * 25.0, f"CCTV Node {idx + 1}"))

    # Tracking Action Buttons
    col_btn_m, col_btn_b = st.columns([1, 1])
    with col_btn_m:
        if st.button("▶ Process & Track Main Incident Video", type="primary", key="btn_main_track"):
            if main_video_path and Path(main_video_path).exists():
                with st.spinner(f"Running YOLO11 + ByteTrack on {main_cam_id}..."):
                    trks = pipeline.process_camera_video(
                        camera_id=main_cam_id,
                        video_path=main_video_path,
                        time_offset_s=0.0,
                        is_main=True,
                        max_frames=int(max_frames_to_process),
                    )
                    st.session_state["main_video_processed"] = True
                    incs = pipeline.detect_incidents_in_camera(main_cam_id)
                    st.session_state["detected_incidents"] = incs
                    st.success(f"Tracked {len(trks)} vehicles in {main_cam_id}! ({len(incs)} incidents detected)")
            else:
                st.error("Main video file not found.")

    with col_btn_b:
        if st.button("▶ Process & Index Background CCTV Feeds", type="secondary", key="btn_bg_track"):
            if not bg_cams_to_run:
                st.error("No background CCTV videos available.")
            else:
                with st.spinner(f"Tracking and indexing {len(bg_cams_to_run)} background CCTV streams..."):
                    total_trks = 0
                    for c_id, v_path, c_offset, _ in bg_cams_to_run:
                        t_list = pipeline.process_camera_video(
                            camera_id=c_id,
                            video_path=v_path,
                            time_offset_s=float(c_offset),
                            is_main=False,
                            max_frames=int(max_frames_to_process),
                        )
                        total_trks += len(t_list)
                    st.session_state["bg_videos_processed"] = True
                    st.success(f"Indexed {total_trks} vehicles across {len(bg_cams_to_run)} background CCTV cameras!")

    # Multi-Camera Video Replay Wall
    st.markdown("---")
    st.markdown("### 🖥️ Surveillance Video Player & Tracking Replay")
    st.caption("Inspect full-motion video feeds with YOLO bounding boxes, vehicle classes, and synchronized timecode watermarks.")

    all_cams = []
    if pipeline.main_camera_id:
        all_cams.append(pipeline.main_camera_id)
    all_cams.extend([c for c in pipeline.background_camera_ids if c not in all_cams])

    if not all_cams:
        st.info("No camera feeds processed yet. Click the tracking buttons above to render annotated feeds.")
    else:
        selected_cam_to_view = st.selectbox(
            "Select Surveillance Camera Feed to Play:",
            all_cams,
            format_func=lambda cid: f"🚨 {cid} (Main Incident Feed)" if cid == pipeline.main_camera_id else f"📹 {cid} (CCTV Node)",
        )

        v_path = pipeline.annotated_videos.get(selected_cam_to_view)
        if v_path and Path(v_path).exists():
            v_col, stat_col = st.columns([2, 1])
            with v_col:
                st.video(str(v_path))
            with stat_col:
                cam_trks = [t for t in pipeline.tracklets if t.camera_id == selected_cam_to_view]
                st.markdown(f"#### Camera Status: `{selected_cam_to_view}`")
                st.write(f"• **Type:** {'Main Incident Feed' if selected_cam_to_view == pipeline.main_camera_id else 'Surrounding CCTV Feed'}")
                st.write(f"• **Vehicles Tracked:** `{len(cam_trks)}`")
                offset = pipeline.camera_offsets.get(selected_cam_to_view, 0.0)
                st.write(f"• **Sync Time Origin:** `{format_surveillance_clock(offset)}`")
                if cam_trks:
                    classes = {}
                    for t in cam_trks:
                        classes[t.vehicle_class] = classes.get(t.vehicle_class, 0) + 1
                    breakdown_str = ", ".join([f"{count} {c}s" for c, count in classes.items()])
                    st.write(f"• **Classification Breakdown:** {breakdown_str}")
        else:
            st.warning(f"Annotated video for {selected_cam_to_view} is not available or processing incomplete.")

# ==============================================================================
# TAB 2: INCIDENT DETECTION & SUSPECT VEHICLE LOCK
# ==============================================================================
with tab2:
    st.markdown("### Step 2: Automated Incident Pinpointing & Target Vehicle Selection")
    st.write("Review kinematic incident alerts or visually lock the suspect vehicle observed in the Main Video.")

    if not st.session_state["main_video_processed"]:
        st.warning("Please process the Main Incident Video in Tab 1 first.")
    else:
        main_trks = [t for t in pipeline.tracklets if t.camera_id == pipeline.main_camera_id and t.crop_paths]

        # 1. Kinematic Incident Alerts
        incs = st.session_state["detected_incidents"]
        if incs:
            st.error(f"🚨 {len(incs)} Automated Kinematic Incident Alert(s) Detected:")
            for inc in incs:
                ic1, ic2 = st.columns([3, 1])
                with ic1:
                    st.write(
                        f"• **Incident `{inc.incident_id}`** at "
                        f"🕒 `{format_surveillance_clock(inc.time_s)}`: "
                        f"**{inc.description}** (Confidence: `{inc.confidence * 100:.1f}%`)"
                    )
                with ic2:
                    if st.button(f"Lock Target from {inc.incident_id}", key=f"btn_inc_{inc.incident_id}"):
                        matched = next((t for t in main_trks if t.tracklet_id in inc.involved_tracklet_ids), None)
                        if matched:
                            st.session_state["selected_query_trk"] = matched
        else:
            st.info("No sudden stop or collision flagged automatically. Select suspect vehicle manually below.")

        st.markdown("---")
        st.markdown("#### Visual Suspect Vehicle Selector (Main Video)")
        st.caption("Click any tracked vehicle from the incident scene to set it as the primary cross-camera query target.")

        if not main_trks:
            st.warning("No vehicle crops available from Main Video.")
        else:
            # Display vehicle crops in responsive grid
            n_cols = min(max(len(main_trks), 1), 5)
            q_cols = st.columns(n_cols)
            for idx, trk in enumerate(main_trks[:10]):
                col_idx = idx % n_cols
                v_crops = [p for p in trk.crop_paths if Path(p).exists()]
                if v_crops:
                    img = Image.open(v_crops[len(v_crops) // 2])
                    with q_cols[col_idx]:
                        st.image(img, use_container_width=True)
                        st.caption(
                            f"**ID:** `{trk.tracklet_id}`\n\n"
                            f"**Class:** `{trk.vehicle_class.upper()}`\n\n"
                            f"🕒 `{format_surveillance_clock(trk.start_time_s)}`"
                        )
                        if st.button("🎯 Lock This Car", key=f"sel_main_{trk.tracklet_id}"):
                            st.session_state["selected_query_trk"] = trk

        selected = st.session_state["selected_query_trk"]
        if selected:
            st.markdown("---")
            st.markdown(
                f"""
                <div class="timeline-card">
                    <span class="status-badge badge-incident">QUERY TARGET LOCKED</span><br>
                    <h3 style="margin: 8px 0px 4px 0px; color:#F8FAFC;">Suspect Target: {selected.tracklet_id} ({selected.vehicle_class.upper()})</h3>
                    <p style="color:#94A3B8; margin:0;">
                        Observed at <b>{selected.camera_id}</b> | Timestamp: <b>{format_surveillance_clock(selected.start_time_s)} EST</b> | Duration: <b>{selected.end_time_s - selected.start_time_s:.1f}s</b>
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )

# ==============================================================================
# TAB 3: SPATIO-TEMPORAL JOURNEY RECONSTRUCTION & FORENSIC VERIFICATION
# ==============================================================================
with tab3:
    st.markdown("### Step 3: Spatio-Temporal Journey Reconstruction & Verification")
    st.write(
        "Searches upstream across the entire background CCTV network. Computes visual Re-ID cosine similarity "
        "and filters candidates with topological travel-time constraints."
    )

    target = st.session_state["selected_query_trk"]

    if not target:
        st.warning("Please select and lock a suspect vehicle in Tab 2 first.")
    elif not st.session_state["bg_videos_processed"]:
        st.warning("Please process background CCTV feeds in Tab 1 first.")
    else:
        # Run Journey Reconstruction Trigger
        if st.button("🚀 Reconstruct Upstream Vehicle Journey", type="primary", key="btn_run_journey"):
            with st.spinner("Searching background CCTV network and computing spatio-temporal route..."):
                j_res: JourneyResult = pipeline.trigger_incident_search(
                    query_tracklet=target,
                    time_window_s=float(lookback_window),
                    alpha=float(prior_weight),
                    min_score=float(match_threshold),
                )
                st.session_state["journey_result"] = j_res

        res: JourneyResult = st.session_state["journey_result"]

        if res is not None:
            st.markdown("---")
            # Chronological Route Breadcrumb Bar
            st.markdown("#### 🗺️ Reconstructed Chronological Vehicle Route")
            st.markdown(
                f"""
                <div class="timeline-card">
                    <div class="metric-label" style="color: #60A5FA;">EARLIEST PERIMETER SIGHTING ────▶ INCIDENT SCENE:</div>
                    <div style="font-size: 1.35rem; font-weight: 800; color: #FFFFFF; margin-top: 6px;">
                        {res.route_str}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Quantitative Efficiency Metrics
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("CCTV Candidates Evaluated", res.candidates_evaluated)
            m2.metric("Spatio-Temporal Pruning", f"{res.pruning_reduction * 100:.1f}%")
            m3.metric("Confirmed Sightings", len(res.sightings))
            m4.metric("Incident Timestamp", format_surveillance_clock(res.incident.time_s))

            st.markdown("---")
            st.markdown("### 🔍 Forensic Side-by-Side Sighting Verification Center")
            st.caption("Verify each confirmed background CCTV observation directly against the incident query vehicle.")

            if not res.sightings:
                st.info("No matching vehicles found in background CCTV cameras within the temporal search window.")
            else:
                for s_idx, sighting in enumerate(res.sightings):
                    st.markdown(
                        f"""
                        <div class="sighting-card">
                            <span class="status-badge badge-cctv">SIGHTING #{s_idx + 1}</span>
                            <span class="status-badge badge-clock">CAMERA: {sighting.camera_id}</span>
                            <span class="status-badge badge-success">MATCH SCORE: {sighting.score:.3f}</span>
                            <span class="status-badge badge-clock">DELTA: {format_time_delta(sighting.time_delta_s)} UPSTREAM</span>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                    s_col_query, s_col_candidate, s_col_cctv_video = st.columns([1, 1, 1.4])

                    # 1. Query Vehicle
                    with s_col_query:
                        st.markdown("##### 🚨 Incident Query Vehicle")
                        if res.query_crop_path and Path(res.query_crop_path).exists():
                            st.image(Image.open(res.query_crop_path), use_container_width=True)
                        st.caption(
                            f"**Camera:** `{res.incident.camera_id}`\n\n"
                            f"**Track:** `{res.query_tracklet.tracklet_id}`\n\n"
                            f"🕒 `{format_surveillance_clock(res.incident.time_s)}`"
                        )

                    # 2. Retrieved CCTV Candidate
                    with s_col_candidate:
                        st.markdown("##### 📸 Retrieved CCTV Candidate")
                        if sighting.crop_path and Path(sighting.crop_path).exists():
                            st.image(Image.open(sighting.crop_path), use_container_width=True)
                        st.caption(
                            f"**Camera:** `{sighting.camera_id}`\n\n"
                            f"**Track:** `{sighting.tracklet.tracklet_id}`\n\n"
                            f"🕒 `{format_surveillance_clock(sighting.timestamp_s)}`"
                        )
                        st.write(
                            f"• **Visual Similarity:** `{sighting.appearance_score:.3f}`\n\n"
                            f"• **Spatio-Temporal Prior:** `{sighting.score - sighting.appearance_score:.3f}`"
                        )

                    # 3. Context & CCTV Video Snippet
                    with s_col_cctv_video:
                        st.markdown("##### 📹 CCTV Stream Playback")
                        cam_v_path = pipeline.annotated_videos.get(sighting.camera_id)
                        if cam_v_path and Path(cam_v_path).exists():
                            st.video(str(cam_v_path))
                        else:
                            st.info("CCTV video stream available in Tab 1.")

                    st.markdown("---")

            # Structured Police / Legal Case Dossier Export
            dossier = {
                "case_id": "CASE-2026-INCIDENT-SEARCH",
                "incident": {
                    "incident_id": res.incident.incident_id,
                    "camera_id": res.incident.camera_id,
                    "timestamp": format_surveillance_clock(res.incident.time_s),
                    "timestamp_raw_s": res.incident.time_s,
                    "description": res.incident.description,
                },
                "query_target": {
                    "tracklet_id": res.query_tracklet.tracklet_id,
                    "vehicle_class": res.query_tracklet.vehicle_class,
                },
                "reconstructed_journey_route": res.route_str,
                "sightings_audit_trail": [
                    {
                        "sighting_index": idx + 1,
                        "camera_id": s.camera_id,
                        "timestamp": format_surveillance_clock(s.timestamp_s),
                        "time_delta_to_incident": format_time_delta(s.time_delta_s),
                        "composite_match_score": round(s.score, 4),
                        "appearance_cosine_score": round(s.appearance_score, 4),
                        "tracklet_id": s.tracklet.tracklet_id,
                    }
                    for idx, s in enumerate(res.sightings)
                ],
            }

            st.download_button(
                label="📥 Export Forensic Investigation Case Dossier (JSON)",
                data=json.dumps(dossier, indent=2),
                file_name=f"investigation_dossier_{res.incident.incident_id}.json",
                mime="application/json",
                type="primary",
            )

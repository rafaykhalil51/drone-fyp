"""
dashboard.py
------------
Streamlit Dashboard for Student & Accessory Vision Analytics.

Layout:
  - Left Panel : Video / Media viewer + File Uploader + Process Controls
  - Right Panel: Live Stats (Total Persons, Caps, Masks, Glasses, Headphones) +
                 Interactive Table + Download buttons for CSV & JSON reports.

Run:
    streamlit run dashboard.py
"""

import os
import cv2
import json
import time
import tempfile
import pandas as pd
import streamlit as st
from pathlib import Path

# Pipeline Modules
from video_source import VideoSource
from tracker import PersonTracker
from detector import PersonDetector
from accessory_detector import AccessoryDetector
from association import associate_accessories_to_tracks
from state_manager import StateManager
from counter import LineCounter, AccessoryCounter
from visualization import Visualizer
from exporter import Exporter

# Streamlit Page Config
st.set_page_layout = "wide"
st.set_page_config(
    page_title="Student & Accessory Vision Analytics",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Custom CSS styling
st.markdown("""
<style>
    .reportview-container { background: #0b0f19; }
    .metric-card {
        background-color: #1e293b;
        border: 1px solid #334155;
        padding: 16px;
        border-radius: 12px;
        text-align: center;
    }
    .metric-val { font-size: 28px; font-weight: bold; }
    .stButton>button { width: 100%; border-radius: 8px; font-weight: 600; }
</style>
""", unsafe_allow_html=True)

st.title("🎓 Student & Accessory Vision Analytics Dashboard")
st.caption("AI-Powered Multi-Object Person Tracking & Temporal Accessory Intelligence")

# Layout: 2 Columns (Left: Video/Upload, Right: Stats/Reports)
left_col, right_col = st.columns([7, 5], gap="large")

with left_col:
    st.subheader("📹 Video / Media Panel")

    uploaded_file = st.file_uploader(
        "Upload Video for Analytics (MP4, AVI, MOV)",
        type=["mp4", "avi", "mov", "jpg", "jpeg", "png"],
        help="Select a classroom video or photo to track and analyze."
    )

    use_sample = False
    if not uploaded_file:
        if st.button("🎬 Load Sample Video (input.mp4)"):
            use_sample = True

    # Controls
    col_ctrl1, col_ctrl2 = st.columns(2)
    with col_ctrl1:
        conf_thresh = st.slider("Detection Confidence", min_value=0.10, max_value=0.90, value=0.30, step=0.05)
    with col_ctrl2:
        model_path = st.text_input("Accessory Model Path", value="accessory_best.pt")

    process_btn = st.button("⚡ Run Video Analytics Pipeline", type="primary")

    # Video display placeholder
    video_placeholder = st.empty()
    progress_placeholder = st.empty()
    status_placeholder = st.empty()

    if not uploaded_file and not use_sample:
        if os.path.exists("output_annotated.mp4"):
            st.info("Displaying latest processed video:")
            st.video("output_annotated.mp4")
        elif os.path.exists("input.mp4"):
            st.info("Sample video available (input.mp4). Click 'Run Video Analytics Pipeline' to process.")

with right_col:
    st.subheader("📊 Live Stats & Accessory Breakdown")

    # Metric placeholders
    m1, m2, m3 = st.columns(3)
    m4, m5, m6 = st.columns(3)

    metric_total = m1.empty()
    metric_cap = m2.empty()
    metric_mask = m3.empty()
    metric_glasses = m4.empty()
    metric_headphones = m5.empty()
    metric_none = m6.empty()

    # Initial metric display
    metric_total.metric("Total Persons", 0)
    metric_cap.metric("🧢 Caps", 0)
    metric_mask.metric("😷 Masks", 0)
    metric_glasses.metric("👓 Glasses", 0)
    metric_headphones.metric("🎧 Headphones", 0)
    metric_none.metric("👤 Plain", 0)

    st.markdown("---")
    st.subheader("📋 Student Details Table")
    table_placeholder = st.empty()

    st.markdown("---")
    st.subheader("📥 Download Reports")
    down_col1, down_col2 = st.columns(2)
    csv_down_placeholder = down_col1.empty()
    json_down_placeholder = down_col2.empty()


# Processing Routine
if process_btn or use_sample:
    # 1. Determine input file path
    temp_input_path = None
    if uploaded_file is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix)
        tfile.write(uploaded_file.read())
        temp_input_path = tfile.name
    elif os.path.exists("input.mp4"):
        temp_input_path = "input.mp4"
    else:
        st.error("Please upload a video file or provide input.mp4 in the workspace directory.")
        st.stop()

    status_placeholder.info("Initializing YOLOv8 + BotSORT & AccessoryDetector models...")

    # 2. Initialize Pipeline Modules
    try:
        source = VideoSource(temp_input_path)
    except Exception as e:
        st.error(f"Error loading video source: {e}")
        st.stop()

    tracker = PersonTracker(
        model_path="yolov8n.pt",
        confidence=conf_thresh,
        iou_threshold=0.50,
        tracker_config="botsort.yaml",
        persist=True,
        person_class_id=0,
    )
    acc_detector = AccessoryDetector(model_path=model_path, confidence=conf_thresh)
    state_mgr = StateManager()
    acc_counter = AccessoryCounter()
    visualizer = Visualizer()
    exporter = Exporter(
        csv_path="tracks.csv",
        summary_path="summary.json",
        final_report_csv="final_report.csv",
        final_report_json="final_report.json",
    )

    out_temp = "output_annotated.mp4"
    writer = cv2.VideoWriter(
        out_temp,
        cv2.VideoWriter_fourcc(*"mp4v"),
        source.fps,
        (source.width, source.height)
    )

    total_frames = max(source.total_frames, 1)
    frame_idx = 0
    t0 = time.time()

    status_placeholder.info("Processing video frames...")
    prog_bar = progress_placeholder.progress(0)

    try:
        with source:
            for frame in source:
                # Step 1: Detect & Track Persons
                tracks = tracker.track(frame)

                # Step 2: Accessory Detection
                person_boxes = [t["xyxy"] for t in tracks] if tracks else []
                raw_accs = acc_detector.detect(frame, person_boxes=person_boxes) if tracks else []

                # Step 3: Spatial Association (upper 40% head/shoulder region)
                acc_map = associate_accessories_to_tracks(tracks, raw_accs, head_fraction=0.40)
                for t in tracks:
                    t["accessories"] = acc_map.get(t["track_id"], [])

                # Step 4: State Manager & Temporal Voting
                state_mgr.update(frame_idx, tracks)
                for t in tracks:
                    tid = t["track_id"]
                    if tid >= 0:
                        state_mgr.update_state(tid, t.get("accessories", []), frame_idx)
                        state_mgr.apply_temporal_voting(tid, window=30, threshold=0.60)

                # Step 5: Live Metrics Calculation
                all_st = list(state_mgr.all_states().values())
                tot_p = state_mgr.total_unique
                tot_cap = sum(1 for s in all_st if s.final_cap)
                tot_mask = sum(1 for s in all_st if s.final_mask)
                tot_glass = sum(1 for s in all_st if s.final_glasses)
                tot_head = sum(1 for s in all_st if s.final_headphones)
                tot_none = sum(1 for s in all_st if not (s.final_cap or s.final_mask or s.final_glasses or s.final_headphones))

                # Step 6: Update Live Stats Panel
                metric_total.metric("Total Persons", tot_p)
                metric_cap.metric("🧢 Caps", tot_cap)
                metric_mask.metric("😷 Masks", tot_mask)
                metric_glasses.metric("👓 Glasses", tot_glass)
                metric_headphones.metric("🎧 Headphones", tot_head)
                metric_none.metric("👤 Plain", tot_none)

                # Step 7: Annotate and Write Video
                live_totals = {
                    "persons": tot_p,
                    "caps": tot_cap,
                    "masks": tot_mask,
                    "glasses": tot_glass,
                    "headphones": tot_head,
                }
                annotated = visualizer.draw(frame, tracks, live_totals=live_totals)
                writer.write(annotated)

                # Update progress
                frame_idx += 1
                if frame_idx % 10 == 0:
                    prog = min(frame_idx / total_frames, 1.0)
                    prog_bar.progress(prog)
                    # Live video frame display (every 10 frames)
                    rgb_frame = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                    video_placeholder.image(rgb_frame, channels="RGB", use_container_width=True)

    finally:
        writer.release()
        exporter.save(state_mgr)

    prog_bar.progress(1.0)
    elapsed = time.time() - t0
    status_placeholder.success(f"✅ Completed! {frame_idx} frames processed in {elapsed:.1f}s ({frame_idx/elapsed:.1f} FPS)")

    # Step 8: Compute Final Aggregate Reports
    acc_counter.compute(state_mgr)
    exporter.save_final_report(
        state_manager=state_mgr,
        acc_counter=acc_counter,
    )

    # Step 9: Render Student Details Table
    table_rows = []
    for tid, s in sorted(state_mgr.all_states().items()):
        accs = []
        if s.final_cap: accs.append("🧢 Cap")
        if s.final_mask: accs.append("😷 Mask")
        if s.final_glasses: accs.append("👓 Glasses")
        if s.final_headphones: accs.append("🎧 Headphones")

        table_rows.append({
            "Student ID": f"#{tid}",
            "First Seen": f"Frame {s.first_frame}",
            "Last Seen": f"Frame {s.last_frame}",
            "Frames Observed": s.frame_count,
            "Confirmed Accessories": ", ".join(accs) if accs else "None",
            "Status": "✅ Verified" if accs else "👤 Regular"
        })

    if table_rows:
        df = pd.DataFrame(table_rows)
        table_placeholder.dataframe(df, use_container_width=True)
    else:
        table_placeholder.info("No tracks observed.")

    # Step 10: Enable CSV & JSON Download Buttons
    if os.path.exists("final_report.csv"):
        with open("final_report.csv", "r") as f:
            csv_down_placeholder.download_button(
                label="📥 Download final_report.csv",
                data=f.read(),
                file_name="final_report.csv",
                mime="text/csv",
                use_container_width=True,
            )

    if os.path.exists("final_report.json"):
        with open("final_report.json", "r") as f:
            json_down_placeholder.download_button(
                label="📥 Download final_report.json",
                data=f.read(),
                file_name="final_report.json",
                mime="application/json",
                use_container_width=True,
            )

    # Display final annotated video
    if os.path.exists(out_temp):
        video_placeholder.video(out_temp)

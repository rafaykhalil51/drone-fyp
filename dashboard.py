"""
dashboard.py
------------
Futuristic Sci-Fi / Cyberpunk HUD Video & Image Analytics Dashboard
Powered by Streamlit, YOLOv8, BotSORT & YOLO-World.
"""

import os
import cv2
import json
import time
import tempfile
import pandas as pd
import numpy as np
import plotly.graph_objects as go
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

# Page Configuration
st.set_page_config(
    page_title="A.E.G.I.S. // Vision Intelligence",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Futuristic Cyberpunk / Sci-Fi HUD Theme CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;600;800;900&family=Rajdhani:wght@500;600;700&family=Inter:wght@300;400;600&display=swap');

    .stApp {
        background-color: #060913;
        background-image: 
            radial-gradient(at 0% 0%, rgba(0, 242, 254, 0.08) 0px, transparent 50%),
            radial-gradient(at 100% 0%, rgba(157, 78, 221, 0.08) 0px, transparent 50%),
            radial-gradient(at 50% 100%, rgba(0, 255, 135, 0.05) 0px, transparent 50%);
        font-family: 'Rajdhani', sans-serif;
        color: #e2e8f0;
    }

    h1, h2, h3, h4, .orbitron {
        font-family: 'Orbitron', sans-serif !important;
        letter-spacing: 1.5px;
    }

    .cyber-card {
        background: rgba(13, 20, 36, 0.75);
        backdrop-filter: blur(16px);
        border: 1px solid rgba(0, 242, 254, 0.2);
        border-radius: 14px;
        padding: 18px;
        box-shadow: 0 0 25px rgba(0, 242, 254, 0.05), inset 0 0 15px rgba(0, 242, 254, 0.02);
        position: relative;
        overflow: hidden;
    }
    .cyber-card::before {
        content: '';
        position: absolute;
        top: 0; left: 0; width: 100%; height: 2px;
        background: linear-gradient(90deg, transparent, #00f2fe, #9d4edd, transparent);
    }

    .hud-metric {
        background: rgba(15, 23, 42, 0.85);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 14px;
        position: relative;
        overflow: hidden;
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .hud-metric:hover {
        transform: translateY(-2px);
        border-color: rgba(0, 242, 254, 0.4);
    }
    .hud-metric-cyan { border-top: 3px solid #00f2fe; }
    .hud-metric-amber { border-top: 3px solid #ffb703; }
    .hud-metric-orange { border-top: 3px solid #fb8500; }
    .hud-metric-purple { border-top: 3px solid #c77dff; }
    .hud-metric-green { border-top: 3px solid #00ff87; }
    .hud-metric-slate { border-top: 3px solid #64748b; }

    .hud-val {
        font-family: 'Orbitron', sans-serif;
        font-size: 28px;
        font-weight: 800;
        line-height: 1.2;
    }
    .hud-label {
        font-size: 11px;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        color: #94a3b8;
        font-weight: 600;
    }
    .hud-sub {
        font-size: 10px;
        color: #64748b;
        margin-top: 4px;
    }

    .stButton>button {
        background: linear-gradient(135deg, #00f2fe 0%, #4facfe 50%, #6b21a8 100%) !important;
        color: #ffffff !important;
        font-family: 'Orbitron', sans-serif !important;
        font-weight: 700 !important;
        letter-spacing: 1.5px !important;
        border: none !important;
        border-radius: 10px !important;
        box-shadow: 0 0 20px rgba(0, 242, 254, 0.3) !important;
        transition: all 0.3s ease !important;
        padding: 10px 20px !important;
    }
    .stButton>button:hover {
        box-shadow: 0 0 35px rgba(0, 242, 254, 0.6) !important;
        transform: scale(1.01) !important;
    }

    .telemetry-pill {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 4px 12px;
        background: rgba(0, 242, 254, 0.08);
        border: 1px solid rgba(0, 242, 254, 0.25);
        border-radius: 999px;
        font-size: 11px;
        font-family: 'Orbitron', monospace;
        color: #00f2fe;
    }
    .pulse-dot {
        width: 8px; height: 8px;
        background-color: #00ff87;
        border-radius: 50%;
        box-shadow: 0 0 10px #00ff87;
        animation: pulse 1.5s infinite;
    }
    @keyframes pulse {
        0% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(0, 255, 135, 0.7); }
        70% { transform: scale(1); box-shadow: 0 0 0 8px rgba(0, 255, 135, 0); }
        100% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(0, 255, 135, 0); }
    }
</style>
""", unsafe_allow_html=True)

# ── TOP FUTURISTIC HEADER ───────────────────────────────────────────────────
st.markdown("""
<div style="display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid rgba(0, 242, 254, 0.2); padding-bottom: 14px; margin-bottom: 18px;">
    <div style="display: flex; align-items: center; gap: 14px;">
        <div style="width: 46px; height: 46px; background: linear-gradient(135deg, #00f2fe, #9d4edd); border-radius: 12px; display: flex; align-items: center; justify-content: center; box-shadow: 0 0 20px rgba(0, 242, 254, 0.4); font-size: 22px;">
            ⚡
        </div>
        <div>
            <div style="font-family: 'Orbitron', sans-serif; font-size: 22px; font-weight: 900; background: linear-gradient(90deg, #ffffff, #00f2fe, #c77dff); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
                A.E.G.I.S. // VISION INTELLIGENCE
            </div>
            <div style="font-size: 12px; color: #94a3b8; letter-spacing: 1px;">
                AUTONOMOUS MULTI-TARGET TRACKING & CRANIAL/FACIAL ACCESSORY TELEMETRY
            </div>
        </div>
    </div>
    <div style="display: flex; gap: 10px; align-items: center;">
        <div class="telemetry-pill">
            <div class="pulse-dot"></div>
            SYS: ACTIVE // LIVE HUD
        </div>
        <div class="telemetry-pill" style="color: #c77dff; border-color: rgba(199, 125, 255, 0.3); background: rgba(199, 125, 255, 0.08);">
            CORE: YOLOv8 + Accessory YOLO
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# ── TOP HUD METRIC CARDS ────────────────────────────────────────────────────
m_cols = st.columns(6)

def render_hud_card(placeholder, title, value, icon, sub, color_class, val_color="#ffffff"):
    html = f"""
    <div class="hud-metric {color_class}">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <span class="hud-label">{title}</span>
            <span style="font-size: 16px;">{icon}</span>
        </div>
        <div class="hud-val" style="color: {val_color}; margin-top: 4px;">{value}</div>
        <div class="hud-sub">{sub}</div>
    </div>
    """
    placeholder.markdown(html, unsafe_allow_html=True)

metric_placeholders = [col.empty() for col in m_cols]

def update_top_metrics(tot_p=0, tot_cap=0, tot_mask=0, tot_glass=0, tot_head=0, tot_none=0):
    pct = lambda v: f"{round((v/tot_p)*100)}%" if tot_p > 0 else "0%"
    render_hud_card(metric_placeholders[0], "TARGETS", tot_p, "👥", "Tracked Persons", "hud-metric-cyan", "#00f2fe")
    render_hud_card(metric_placeholders[1], "CRANIAL", tot_cap, "🧢", f"{pct(tot_cap)} Caps/Hats", "hud-metric-amber", "#ffb703")
    render_hud_card(metric_placeholders[2], "RESPIRATORY", tot_mask, "😷", f"{pct(tot_mask)} Face Masks", "hud-metric-orange", "#fb8500")
    render_hud_card(metric_placeholders[3], "OPTICAL", tot_glass, "👓", f"{pct(tot_glass)} Glasses", "hud-metric-purple", "#c77dff")
    render_hud_card(metric_placeholders[4], "ACOUSTIC", tot_head, "🎧", f"{pct(tot_head)} Headphones", "hud-metric-green", "#00ff87")
    render_hud_card(metric_placeholders[5], "PLAIN", tot_none, "👤", f"{pct(tot_none)} No Gear", "hud-metric-slate", "#94a3b8")

st.markdown("<div style='margin-top: 14px;'></div>", unsafe_allow_html=True)

# ── MAIN 2-COLUMN WORKSPACE ─────────────────────────────────────────────────
left_col, right_col = st.columns([7, 5], gap="large")

with left_col:
    st.markdown("""
    <div style="font-family: 'Orbitron', sans-serif; font-size: 15px; font-weight: 700; color: #00f2fe; margin-bottom: 8px; display: flex; align-items: center; gap: 8px;">
        <span>📹 SURVEILLANCE FEED // TACTICAL VIEWPORT</span>
    </div>
    """, unsafe_allow_html=True)

    video_box = st.empty()
    progress_box = st.empty()
    status_box = st.empty()

    # Upload & Control Bay
    st.markdown("<div class='cyber-card' style='margin-top: 14px;'>", unsafe_allow_html=True)
    st.markdown("<div style='font-family: Orbitron; font-size: 13px; font-weight: 700; color: #94a3b8; margin-bottom: 10px;'>⚡ MISSION CONTROL // INPUT SELECTOR</div>", unsafe_allow_html=True)

    mode_choice = st.radio(
        "Select Input Mode:",
        ["📸 Sample Classroom Photo (Default)", "🎥 Sample Video (input.mp4)", "📁 Upload New Photo/Video"],
        horizontal=True
    )

    uploaded_file = None
    if "Upload" in mode_choice:
        uploaded_file = st.file_uploader("Choose Media File", type=["mp4", "avi", "mov", "jpg", "jpeg", "png", "webp"])

    c_cfg1, c_cfg2 = st.columns(2)
    with c_cfg1:
        conf_thresh = st.slider("AI Confidence Gate (NMS Threshold)", 0.10, 0.90, 0.25, 0.05)
    with c_cfg2:
        model_path = st.text_input("Neural Weights Identifier", value="accessory_best.pt")

    process_btn = st.button("🚀 RE-RUN VISION ANALYTICS ENGINE")
    st.markdown("</div>", unsafe_allow_html=True)

with right_col:
    st.markdown("""
    <div style="font-family: 'Orbitron', sans-serif; font-size: 15px; font-weight: 700; color: #c77dff; margin-bottom: 8px; display: flex; align-items: center; gap: 8px;">
        <span>📊 RADAR TELEMETRY & TARGET MATRIX</span>
    </div>
    """, unsafe_allow_html=True)

    chart_placeholder = st.empty()

    def render_cyber_chart(tot_cap=0, tot_mask=0, tot_glass=0, tot_head=0, tot_none=0):
        categories = ['Caps 🧢', 'Masks 😷', 'Glasses 👓', 'Headphones 🎧', 'Plain 👤']
        values = [tot_cap, tot_mask, tot_glass, tot_head, tot_none]
        
        fig = go.Figure(data=[go.Pie(
            labels=categories,
            values=values,
            hole=.65,
            marker=dict(colors=['#ffb703', '#fb8500', '#c77dff', '#00ff87', '#64748b'],
                        line=dict(color='#060913', width=3)),
            textinfo='label+percent',
            hoverinfo='label+value',
            textfont=dict(family='Orbitron', size=11, color='#ffffff')
        )])

        fig.update_layout(
            paper_bgcolor='rgba(13, 20, 36, 0.7)',
            plot_bgcolor='rgba(0,0,0,0)',
            margin=dict(t=20, b=20, l=20, r=20),
            height=220,
            showlegend=False,
            annotations=[dict(
                text='GEAR<br>SPECTRUM',
                x=0.5, y=0.5,
                font=dict(family='Orbitron', size=11, color='#00f2fe'),
                showarrow=False
            )]
        )
        chart_placeholder.plotly_chart(fig, use_container_width=True)

    st.markdown("<div style='font-family: Orbitron; font-size: 13px; font-weight: 700; color: #00ff87; margin: 12px 0 6px 0;'>🎯 ACTIVE TARGET REGISTER</div>", unsafe_allow_html=True)
    table_placeholder = st.empty()

    st.markdown("<div style='font-family: Orbitron; font-size: 13px; font-weight: 700; color: #ffb703; margin: 14px 0 6px 0;'>💾 INTELLIGENCE DOSSIER EXPORT</div>", unsafe_allow_html=True)
    down_col1, down_col2 = st.columns(2)
    csv_down_placeholder = down_col1.empty()
    json_down_placeholder = down_col2.empty()


# ── RESOLVE INPUT PATH ───────────────────────────────────────────────────────
temp_input_path = None

if uploaded_file is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix)
    tfile.write(uploaded_file.read())
    temp_input_path = tfile.name
elif "Video" in mode_choice and os.path.exists("input.mp4"):
    temp_input_path = "input.mp4"
elif os.path.exists("uploads/classroom_sample.jpg"):
    temp_input_path = "uploads/classroom_sample.jpg"
elif os.path.exists("input.mp4"):
    temp_input_path = "input.mp4"

if temp_input_path is None:
    status_box.error("Media input unavailable. Upload a photo or video to begin.")
    st.stop()

file_ext = Path(temp_input_path).suffix.lower()
is_image = file_ext in [".jpg", ".jpeg", ".png", ".webp", ".bmp"]

# ── EXECUTION ENGINE (AUTO-RUNS ON LOAD OR CHANGE) ──────────────────────────
acc_detector = AccessoryDetector(model_path=model_path, confidence=conf_thresh)
state_mgr = StateManager()
acc_counter = AccessoryCounter()
visualizer = Visualizer(vis_cfg={"box_thickness": 2, "text_scale": 0.65, "show_confidence": True})
exporter = Exporter(
    csv_path="tracks.csv",
    summary_path="summary.json",
    final_report_csv="final_report.csv",
    final_report_json="final_report.json",
)

t0 = time.time()

if is_image:
    # ── PHOTO PROCESSING (AUTO ON LOAD) ──────────────────────────────────────
    frame = cv2.imread(temp_input_path)
    if frame is None:
        status_box.error("Could not decode image file.")
        st.stop()

    # Detect Persons
    p_detector = PersonDetector("yolov8n.pt", confidence=conf_thresh, iou_threshold=0.50)
    person_dets = p_detector.detect(frame)
    tracks = [{
        "track_id": i + 1,
        "xyxy": p["xyxy"],
        "confidence": p["confidence"],
        "accessories": []
    } for i, p in enumerate(person_dets)]

    # Detect Accessories
    raw_accs = acc_detector.detect(frame, person_boxes=[t["xyxy"] for t in tracks])

    # Spatial Association (upper 45% head region)
    acc_map = associate_accessories_to_tracks(tracks, raw_accs, head_fraction=0.45)
    for t in tracks:
        t["accessories"] = acc_map.get(t["track_id"], [])

    # Update State
    state_mgr.update(0, tracks)
    for t in tracks:
        tid = t["track_id"]
        state_mgr.update_state(tid, t.get("accessories", []), 0)
        state_mgr.apply_temporal_voting(tid, window=1, threshold=0.50)

    # Compute metrics
    all_st = list(state_mgr.all_states().values())
    tot_p = len(tracks)
    tot_cap = sum(1 for s in all_st if s.final_cap)
    tot_mask = sum(1 for s in all_st if s.final_mask)
    tot_glass = sum(1 for s in all_st if s.final_glasses)
    tot_head = sum(1 for s in all_st if s.final_headphones)
    tot_none = sum(1 for s in all_st if not (s.final_cap or s.final_mask or s.final_glasses or s.final_headphones))

    # Update Top Metrics & Donut Chart
    update_top_metrics(tot_p, tot_cap, tot_mask, tot_glass, tot_head, tot_none)
    render_cyber_chart(tot_cap, tot_mask, tot_glass, tot_head, tot_none)

    # Draw annotated image & display
    live_totals = {"persons": tot_p, "caps": tot_cap, "masks": tot_mask, "glasses": tot_glass, "headphones": tot_head}
    annotated = visualizer.draw(frame, tracks, live_totals=live_totals)
    rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
    video_box.image(rgb, channels="RGB", use_container_width=True)

    elapsed = time.time() - t0
    status_box.markdown(f"""
    <div class="telemetry-pill" style="color: #00ff87; border-color: #00ff87; font-size: 12px;">
        ⚡ IMAGE SCAN COMPLETE: {elapsed:.2f}s INFERENCE // {tot_p} STUDENTS DETECTED // ACCESSORIES MAPPED
    </div>
    """, unsafe_allow_html=True)

else:
    # ── VIDEO PROCESSING ─────────────────────────────────────────────────────
    try:
        source = VideoSource(temp_input_path)
    except Exception as e:
        status_box.error(f"Stream failure: {e}")
        st.stop()

    tracker = PersonTracker(
        model_path="yolov8n.pt",
        confidence=conf_thresh,
        iou_threshold=0.50,
        tracker_config="botsort.yaml",
        persist=True,
        person_class_id=0,
    )

    out_temp = "output_annotated.mp4"
    writer = cv2.VideoWriter(out_temp, cv2.VideoWriter_fourcc(*"mp4v"), source.fps, (source.width, source.height))

    total_frames = max(source.total_frames, 1)
    frame_idx = 0
    prog_bar = progress_box.progress(0)

    try:
        with source:
            for frame in source:
                tracks = tracker.track(frame)
                person_boxes = [t["xyxy"] for t in tracks] if tracks else []
                raw_accs = acc_detector.detect(frame, person_boxes=person_boxes) if tracks else []

                acc_map = associate_accessories_to_tracks(tracks, raw_accs, head_fraction=0.40)
                for t in tracks:
                    t["accessories"] = acc_map.get(t["track_id"], [])

                state_mgr.update(frame_idx, tracks)
                for t in tracks:
                    tid = t["track_id"]
                    if tid >= 0:
                        state_mgr.update_state(tid, t.get("accessories", []), frame_idx)
                        state_mgr.apply_temporal_voting(tid, window=30, threshold=0.60)

                all_st = list(state_mgr.all_states().values())
                tot_p = state_mgr.total_unique
                tot_cap = sum(1 for s in all_st if s.final_cap)
                tot_mask = sum(1 for s in all_st if s.final_mask)
                tot_glass = sum(1 for s in all_st if s.final_glasses)
                tot_head = sum(1 for s in all_st if s.final_headphones)
                tot_none = sum(1 for s in all_st if not (s.final_cap or s.final_mask or s.final_glasses or s.final_headphones))

                live_totals = {"persons": tot_p, "caps": tot_cap, "masks": tot_mask, "glasses": tot_glass, "headphones": tot_head}
                annotated = visualizer.draw(frame, tracks, live_totals=live_totals)
                writer.write(annotated)

                frame_idx += 1
                if frame_idx % 12 == 0:
                    prog_bar.progress(min(frame_idx / total_frames, 1.0))
                    update_top_metrics(tot_p, tot_cap, tot_mask, tot_glass, tot_head, tot_none)
                    render_cyber_chart(tot_cap, tot_mask, tot_glass, tot_head, tot_none)
                    rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                    video_box.image(rgb, channels="RGB", use_container_width=True)

    finally:
        writer.release()
        exporter.save(state_mgr)

    prog_bar.progress(1.0)
    elapsed = time.time() - t0
    status_box.markdown(f"""
    <div class="telemetry-pill" style="color: #00ff87; border-color: #00ff87; font-size: 12px;">
        ⚡ MISSION COMPLETE: {frame_idx} FRAMES // {elapsed:.1f}s ELAPSED // {frame_idx/elapsed:.1f} FPS // {state_mgr.total_unique} TARGETS IDENTIFIED
    </div>
    """, unsafe_allow_html=True)

    if os.path.exists(out_temp):
        video_box.video(out_temp)

# ── COMPUTE FINAL REPORTS & POPULATE TABLE ──────────────────────────────────
acc_counter.compute(state_mgr)
exporter.save_final_report(state_manager=state_mgr, acc_counter=acc_counter)

table_rows = []
for tid, s in sorted(state_mgr.all_states().items()):
    accs = []
    if s.final_cap: accs.append("🧢 Cap")
    if s.final_mask: accs.append("😷 Mask")
    if s.final_glasses: accs.append("👓 Glasses")
    if s.final_headphones: accs.append("🎧 Headphones")

    table_rows.append({
        "Target ID": f"TRK-{tid:03d}",
        "Inception": f"F:{s.first_frame:03d}",
        "Latest": f"F:{s.last_frame:03d}",
        "Frames": s.frame_count,
        "Gear Verified": ", ".join(accs) if accs else "None",
        "Classification": "SECURED // ACTIVE" if accs else "STANDARD TARGET"
    })

if table_rows:
    df = pd.DataFrame(table_rows)
    table_placeholder.dataframe(df, use_container_width=True)

# Download Buttons
if os.path.exists("final_report.csv"):
    with open("final_report.csv", "r") as f:
        csv_down_placeholder.download_button(
            "📥 EXPORT CSV DOSSIER",
            data=f.read(),
            file_name="final_report.csv",
            mime="text/csv",
            use_container_width=True
        )

if os.path.exists("final_report.json"):
    with open("final_report.json", "r") as f:
        json_down_placeholder.download_button(
            "📥 EXPORT JSON TELEMETRY",
            data=f.read(),
            file_name="final_report.json",
            mime="application/json",
            use_container_width=True
        )

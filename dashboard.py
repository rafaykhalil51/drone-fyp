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

# ── PAGE CONFIG ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="A.E.G.I.S. // Vision Intelligence",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── CACHED MODEL LOADING ─────────────────────────────────────────────────────
@st.cache_resource
def load_person_detector(conf):
    return PersonDetector("yolov8n.pt", confidence=conf, iou_threshold=0.50)

@st.cache_resource
def load_accessory_detector(model_path, conf):
    return AccessoryDetector(model_path=model_path, confidence=conf)

# ── FUTURISTIC CSS ───────────────────────────────────────────────────────────
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
        background: rgba(13, 20, 36, 0.85);
        backdrop-filter: blur(16px);
        border: 1px solid rgba(0, 242, 254, 0.25);
        border-radius: 14px;
        padding: 18px;
        box-shadow: 0 0 25px rgba(0, 242, 254, 0.06), inset 0 0 15px rgba(0, 242, 254, 0.02);
        margin-top: 10px;
    }
    .hud-metric {
        background: rgba(15, 23, 42, 0.9);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px; padding: 14px;
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .hud-metric:hover { transform: translateY(-2px); border-color: rgba(0, 242, 254, 0.4); }
    .hud-metric-cyan   { border-top: 3px solid #00f2fe; }
    .hud-metric-amber  { border-top: 3px solid #ffb703; }
    .hud-metric-orange { border-top: 3px solid #fb8500; }
    .hud-metric-purple { border-top: 3px solid #c77dff; }
    .hud-metric-green  { border-top: 3px solid #00ff87; }
    .hud-metric-slate  { border-top: 3px solid #64748b; }
    .hud-val { font-family: 'Orbitron', sans-serif; font-size: 28px; font-weight: 800; line-height: 1.2; }
    .hud-label { font-size: 11px; text-transform: uppercase; letter-spacing: 1.5px; color: #94a3b8; font-weight: 600; }
    .hud-sub { font-size: 10px; color: #64748b; margin-top: 4px; }

    .stButton>button {
        background: linear-gradient(135deg, #00f2fe 0%, #4facfe 50%, #6b21a8 100%) !important;
        color: #ffffff !important;
        font-family: 'Orbitron', sans-serif !important;
        font-weight: 700 !important; letter-spacing: 1.2px !important;
        border: none !important; border-radius: 10px !important;
        box-shadow: 0 0 20px rgba(0, 242, 254, 0.3) !important;
        transition: all 0.3s ease !important; padding: 10px 18px !important;
    }
    .stButton>button:hover { box-shadow: 0 0 35px rgba(0, 242, 254, 0.6) !important; transform: scale(1.01) !important; }

    .telemetry-pill {
        display: inline-flex; align-items: center; gap: 6px;
        padding: 6px 14px;
        background: rgba(0, 242, 254, 0.08);
        border: 1px solid rgba(0, 242, 254, 0.3);
        border-radius: 999px; font-size: 11px;
        font-family: 'Orbitron', monospace; color: #00f2fe;
    }
    .pulse-dot {
        width: 8px; height: 8px; background-color: #00ff87;
        border-radius: 50%; box-shadow: 0 0 10px #00ff87;
        animation: pulse 1.5s infinite;
    }
    @keyframes pulse {
        0%   { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(0, 255, 135, 0.7); }
        70%  { transform: scale(1);    box-shadow: 0 0 0 8px rgba(0, 255, 135, 0);  }
        100% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(0, 255, 135, 0);  }
    }
    /* Tab Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 12px;
    }
    .stTabs [data-baseweb="tab"] {
        font-family: 'Orbitron', sans-serif !important;
        font-size: 13px !important;
        font-weight: 700 !important;
        border-radius: 8px !important;
        padding: 10px 18px !important;
        background: rgba(15, 23, 42, 0.6) !important;
        border: 1px solid rgba(0, 242, 254, 0.2) !important;
        color: #94a3b8 !important;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, rgba(0, 242, 254, 0.2), rgba(157, 78, 221, 0.2)) !important;
        border-color: #00f2fe !important;
        color: #00f2fe !important;
    }
</style>
""", unsafe_allow_html=True)

# ── HEADER ───────────────────────────────────────────────────────────────────
st.markdown("""
<div style="display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid rgba(0, 242, 254, 0.2); padding-bottom: 14px; margin-bottom: 18px;">
    <div style="display: flex; align-items: center; gap: 14px;">
        <div style="width: 46px; height: 46px; background: linear-gradient(135deg, #00f2fe, #9d4edd); border-radius: 12px; display: flex; align-items: center; justify-content: center; box-shadow: 0 0 20px rgba(0, 242, 254, 0.4); font-size: 22px;">⚡</div>
        <div>
            <div style="font-family: 'Orbitron', sans-serif; font-size: 22px; font-weight: 900; background: linear-gradient(90deg, #ffffff, #00f2fe, #c77dff); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
                A.E.G.I.S. // VISION INTELLIGENCE
            </div>
            <div style="font-size: 12px; color: #94a3b8; letter-spacing: 1px;">
                AUTONOMOUS MULTI-TARGET TRACKING &amp; CRANIAL/FACIAL ACCESSORY TELEMETRY
            </div>
        </div>
    </div>
    <div style="display: flex; gap: 10px; align-items: center;">
        <div class="telemetry-pill"><div class="pulse-dot"></div> SYS: ACTIVE // LIVE HUD</div>
        <div class="telemetry-pill" style="color: #c77dff; border-color: rgba(199, 125, 255, 0.3); background: rgba(199, 125, 255, 0.08);">CORE: YOLOv8 + BotSORT</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ── HELPER: HUD METRIC CARD ─────────────────────────────────────────────────
def render_hud_card(placeholder, title, value, icon, sub, color_class, val_color="#ffffff"):
    placeholder.markdown(f"""
    <div class="hud-metric {color_class}">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <span class="hud-label">{title}</span>
            <span style="font-size: 16px;">{icon}</span>
        </div>
        <div class="hud-val" style="color: {val_color}; margin-top: 4px;">{value}</div>
        <div class="hud-sub">{sub}</div>
    </div>
    """, unsafe_allow_html=True)

# ── TOP HUD METRIC ROW ──────────────────────────────────────────────────────
m_cols = st.columns(6)
metric_ph = [col.empty() for col in m_cols]

def update_top_metrics(tp=0, tc=0, tm=0, tg=0, th=0, tn=0):
    pct = lambda v: f"{round((v/tp)*100)}%" if tp > 0 else "0%"
    render_hud_card(metric_ph[0], "TARGETS",     tp, "👥", "Tracked Persons",       "hud-metric-cyan",   "#00f2fe")
    render_hud_card(metric_ph[1], "CRANIAL",     tc, "🧢", f"{pct(tc)} Caps/Hats",  "hud-metric-amber",  "#ffb703")
    render_hud_card(metric_ph[2], "RESPIRATORY", tm, "😷", f"{pct(tm)} Face Masks",  "hud-metric-orange", "#fb8500")
    render_hud_card(metric_ph[3], "OPTICAL",     tg, "👓", f"{pct(tg)} Glasses",     "hud-metric-purple", "#c77dff")
    render_hud_card(metric_ph[4], "ACOUSTIC",    th, "🎧", f"{pct(th)} Headphones",  "hud-metric-green",  "#00ff87")
    render_hud_card(metric_ph[5], "PLAIN",       tn, "👤", f"{pct(tn)} No Gear",     "hud-metric-slate",  "#94a3b8")

st.markdown("<div style='margin-top: 14px;'></div>", unsafe_allow_html=True)

# ── 2-COLUMN WORKSPACE ──────────────────────────────────────────────────────
left_col, right_col = st.columns([7, 5], gap="large")

with left_col:
    st.markdown("""<div style="font-family: 'Orbitron', sans-serif; font-size: 15px; font-weight: 700; color: #00f2fe; margin-bottom: 8px;">📹 SURVEILLANCE FEED // TACTICAL VIEWPORT</div>""", unsafe_allow_html=True)
    viewport_box = st.empty()
    progress_box = st.empty()
    status_box   = st.empty()

    # Dedicated 2 Tabs: Upload Video (Default) vs Upload Photo
    st.markdown("<div class='cyber-card'>", unsafe_allow_html=True)
    st.markdown("<div style='font-family: Orbitron; font-size: 14px; font-weight: 700; color: #00f2fe; margin-bottom: 10px;'>⚡ INPUT CONTROL BAY</div>", unsafe_allow_html=True)

    tab_video, tab_photo = st.tabs(["🎥 1. UPLOAD VIDEO", "📷 2. UPLOAD PHOTO"])

    with tab_video:
        uploaded_video = st.file_uploader("Upload Video File (MP4, AVI, MOV, MKV, WEBM, M4V)", type=["mp4", "avi", "mov", "mkv", "webm", "m4v", "wmv", "flv"], key="video_uploader")
        btn_sample_video = st.button("🎯 LOAD SAMPLE VIDEO (input.mp4)", key="btn_sample_video")

    with tab_photo:
        uploaded_photo = st.file_uploader("Upload Image File (JPG, PNG, WEBP, BMP, TIFF)", type=["jpg", "jpeg", "png", "webp", "bmp", "tiff"], key="photo_uploader")
        btn_sample_photo = st.button("🎯 LOAD SAMPLE PHOTO", key="btn_sample_photo")

    c1, c2 = st.columns(2)
    with c1:
        conf_thresh = st.slider("AI Confidence Gate", 0.10, 0.90, 0.25, 0.05, key="conf_slider")
    with c2:
        model_path = st.text_input("Neural Weights", value="accessory_best.pt", key="model_input")

    st.markdown("</div>", unsafe_allow_html=True)

with right_col:
    st.markdown("""<div style="font-family: 'Orbitron', sans-serif; font-size: 15px; font-weight: 700; color: #c77dff; margin-bottom: 8px;">📊 RADAR TELEMETRY &amp; TARGET MATRIX</div>""", unsafe_allow_html=True)
    chart_container = st.empty()
    st.markdown("<div style='font-family: Orbitron; font-size: 13px; font-weight: 700; color: #00ff87; margin: 12px 0 6px 0;'>🎯 ACTIVE TARGET REGISTER</div>", unsafe_allow_html=True)
    table_container = st.empty()
    st.markdown("<div style='font-family: Orbitron; font-size: 13px; font-weight: 700; color: #ffb703; margin: 14px 0 6px 0;'>💾 INTELLIGENCE DOSSIER EXPORT</div>", unsafe_allow_html=True)
    dc1, dc2 = st.columns(2)
    csv_ph  = dc1.empty()
    json_ph = dc2.empty()
    video_dl_ph = st.empty()

# ── HELPER: RENDER PIE / DONUT CHART ─────────────────────────────────────────
_chart_count = 0
def render_chart(tc=0, tm=0, tg=0, th=0, tn=0):
    global _chart_count
    _chart_count += 1
    total = tc + tm + tg + th + tn
    if total == 0:
        labels = ['Awaiting Detection']
        values = [1]
        colors = ['rgba(100, 116, 139, 0.4)']
        textinfo = 'label'
        center_text = 'STANDBY<br>READY'
    else:
        raw_cats = [
            ('Caps 🧢', tc, '#ffb703'),
            ('Masks 😷', tm, '#fb8500'),
            ('Glasses 👓', tg, '#c77dff'),
            ('Headphones 🎧', th, '#00ff87'),
            ('Plain 👤', tn, '#64748b'),
        ]
        active_cats = [c for c in raw_cats if c[1] > 0]
        if not active_cats:
            active_cats = raw_cats
        labels = [c[0] for c in active_cats]
        values = [c[1] for c in active_cats]
        colors = [c[2] for c in active_cats]
        textinfo = 'label+percent'
        center_text = f'{total}<br>GEAR'

    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        hole=.65,
        marker=dict(colors=colors, line=dict(color='#060913', width=2)),
        textinfo=textinfo,
        hoverinfo='label+value',
        textfont=dict(family='Orbitron', size=11, color='#ffffff')
    )])
    fig.update_layout(
        paper_bgcolor='rgba(13, 20, 36, 0.7)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(t=15, b=15, l=15, r=15),
        height=220,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.2,
            xanchor="center",
            x=0.5,
            font=dict(family='Rajdhani', size=11, color='#cbd5e1')
        ),
        annotations=[dict(text=center_text, x=0.5, y=0.5,
                          font=dict(family='Orbitron', size=11, color='#00f2fe'), showarrow=False)]
    )
    chart_container.plotly_chart(fig, key=f"gear_chart_{_chart_count}")

# ── RESOLVE INPUT FILE & MODE ────────────────────────────────────────────────
input_mode = "video" # default home mode is Video
target_media_path = None

if uploaded_video is not None:
    input_mode = "video"
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_video.name).suffix)
    tfile.write(uploaded_video.read())
    tfile.flush()
    target_media_path = tfile.name
elif btn_sample_video and os.path.exists("input.mp4"):
    input_mode = "video"
    target_media_path = "input.mp4"
elif uploaded_photo is not None:
    input_mode = "photo"
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_photo.name).suffix)
    tfile.write(uploaded_photo.read())
    tfile.flush()
    target_media_path = tfile.name
elif btn_sample_photo:
    input_mode = "photo"
    if os.path.exists("uploads/classroom_sample.jpg"):
        target_media_path = "uploads/classroom_sample.jpg"
elif os.path.exists("input.mp4"):
    # Default Home Page: load and run video
    input_mode = "video"
    target_media_path = "input.mp4"
elif os.path.exists("uploads/classroom_sample.jpg"):
    input_mode = "photo"
    target_media_path = "uploads/classroom_sample.jpg"

# ── INITIALIZE DETECTION & STATE ENGINES ─────────────────────────────────────
acc_detector = load_accessory_detector(model_path, conf_thresh)
state_mgr    = StateManager()
acc_counter  = AccessoryCounter()
visualizer   = Visualizer(vis_cfg={"box_thickness": 2, "text_scale": 0.65, "show_confidence": True})
exporter     = Exporter(csv_path="tracks.csv", summary_path="summary.json",
                        final_report_csv="final_report.csv", final_report_json="final_report.json")

t0 = time.time()

if input_mode == "photo" and target_media_path is not None:
    # ── 1. PHOTO EXECUTION ───────────────────────────────────────────────────
    frame = cv2.imread(target_media_path)
    if frame is None:
        status_box.error("Could not decode image file.")
        st.stop()

    p_detector  = load_person_detector(conf_thresh)
    person_dets = p_detector.detect(frame, imgsz=480)
    tracks = [{"track_id": i+1, "xyxy": p["xyxy"], "confidence": p["confidence"], "accessories": []}
              for i, p in enumerate(person_dets)]

    raw_accs = acc_detector.detect(frame, person_boxes=[t["xyxy"] for t in tracks], imgsz=384)
    acc_map  = associate_accessories_to_tracks(tracks, raw_accs, head_fraction=0.52)
    for t in tracks:
        t["accessories"] = acc_map.get(t["track_id"], [])

    state_mgr.update(0, tracks)
    for t in tracks:
        tid = t["track_id"]
        state_mgr.update_state(tid, t.get("accessories", []), 0)
        state_mgr.apply_temporal_voting(tid, window=1, threshold=0.50, latch=True)

    all_st   = list(state_mgr.all_states().values())
    tot_p    = len(tracks)
    tot_cap  = sum(1 for s in all_st if s.final_cap)
    tot_mask = sum(1 for s in all_st if s.final_mask)
    tot_gls  = sum(1 for s in all_st if s.final_glasses)
    tot_hd   = sum(1 for s in all_st if s.final_headphones)
    tot_none = sum(1 for s in all_st if not (s.final_cap or s.final_mask or s.final_glasses or s.final_headphones))

    update_top_metrics(tot_p, tot_cap, tot_mask, tot_gls, tot_hd, tot_none)
    render_chart(tot_cap, tot_mask, tot_gls, tot_hd, tot_none)

    live_totals = {"persons": tot_p, "caps": tot_cap, "masks": tot_mask, "glasses": tot_gls, "headphones": tot_hd}
    annotated = visualizer.draw(frame, tracks, live_totals=live_totals)
    rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
    viewport_box.image(rgb, channels="RGB", width="stretch")

    elapsed = time.time() - t0
    status_box.markdown(f"""
    <div class="telemetry-pill" style="color: #00ff87; border-color: #00ff87; font-size: 12px;">
        ⚡ IMAGE SCAN COMPLETE: {elapsed:.2f}s // {tot_p} STUDENTS DETECTED // ACCESSORIES MAPPED
    </div>
    """, unsafe_allow_html=True)

elif input_mode == "video" and target_media_path is not None:
    # ── 2. VIDEO EXECUTION (WITH LOADING ANIMATION & PLAYER CONTROLS) ────────
    try:
        source = VideoSource(target_media_path)
    except Exception as e:
        status_box.error(f"Stream failure: {e}")
        st.stop()

    target_fps = float(source.fps) if (source.fps and 1.0 <= source.fps <= 120.0) else 30.0
    out_raw  = "output_raw.mp4"
    out_final = "output_annotated.mp4"
    writer = cv2.VideoWriter(out_raw, cv2.VideoWriter_fourcc(*"mp4v"),
                             target_fps, (source.width, source.height))

    total_frames = max(source.total_frames, 1)
    frame_idx = 0
    prog_bar = progress_box.progress(0)

    tracker = PersonTracker(model_path="yolov8n.pt", confidence=conf_thresh,
                            iou_threshold=0.50, tracker_config="botsort.yaml",
                            persist=True, person_class_id=0)

    # Show Loading Animation Spinner while processing frames
    with st.spinner("⚡ HIGH-SPEED NEURAL INFERENCE IN PROGRESS // ANALYZING FRAMES..."):
        last_acc_map = {}
        try:
            with source:
                for frame in source:
                    # 1. Fast Person Tracking at imgsz=480
                    tracks = tracker.track(frame, imgsz=480)
                    
                    # 2. High-precision accessory detection every 2nd frame
                    if tracks and (frame_idx % 2 == 0 or not last_acc_map):
                        person_boxes = [t["xyxy"] for t in tracks]
                        raw_accs = acc_detector.detect(frame, person_boxes=person_boxes, imgsz=384)
                        last_acc_map = associate_accessories_to_tracks(tracks, raw_accs, head_fraction=0.52)

                    for t in tracks:
                        t["accessories"] = last_acc_map.get(t["track_id"], [])

                    state_mgr.update(frame_idx, tracks)
                    for t in tracks:
                        tid = t["track_id"]
                        if tid >= 0:
                            state_mgr.update_state(tid, t.get("accessories", []), frame_idx)
                            state_mgr.apply_temporal_voting(tid, window=30, threshold=0.35, latch=True)

                    all_st   = list(state_mgr.all_states().values())
                    tot_p    = state_mgr.total_unique
                    tot_cap  = sum(1 for s in all_st if s.final_cap)
                    tot_mask = sum(1 for s in all_st if s.final_mask)
                    tot_gls  = sum(1 for s in all_st if s.final_glasses)
                    tot_hd   = sum(1 for s in all_st if s.final_headphones)
                    tot_none = sum(1 for s in all_st if not (s.final_cap or s.final_mask or s.final_glasses or s.final_headphones))

                    live_totals = {"persons": tot_p, "caps": tot_cap, "masks": tot_mask, "glasses": tot_gls, "headphones": tot_hd}
                    annotated = visualizer.draw(frame, tracks, live_totals=live_totals)
                    writer.write(annotated)

                    frame_idx += 1
                    if frame_idx % 12 == 0 or frame_idx == total_frames:
                        pct = min(frame_idx / total_frames, 1.0)
                        prog_bar.progress(pct)
                        fps_now = frame_idx / max(time.time() - t0, 0.01)
                        status_box.markdown(f"""
                        <div class="telemetry-pill" style="color: #ffb703; border-color: #ffb703; font-size: 12px;">
                            ⚡ PROCESSING: {frame_idx}/{total_frames} frames ({pct*100:.0f}%) // AI Inference: {fps_now:.1f} FPS
                        </div>
                        """, unsafe_allow_html=True)
                        update_top_metrics(tot_p, tot_cap, tot_mask, tot_gls, tot_hd, tot_none)
                        render_chart(tot_cap, tot_mask, tot_gls, tot_hd, tot_none)
        finally:
            writer.release()
            exporter.save(state_mgr)

    # Update final HUD metrics & Donut Chart
    update_top_metrics(tot_p, tot_cap, tot_mask, tot_gls, tot_hd, tot_none)
    render_chart(tot_cap, tot_mask, tot_gls, tot_hd, tot_none)

    prog_bar.progress(1.0)
    elapsed = time.time() - t0

    # Convert to browser-compatible normal-speed H.264 MP4
    import subprocess
    try:
        import imageio_ffmpeg
        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        ffmpeg_exe = "ffmpeg"

    play_path = out_raw
    try:
        res = subprocess.run([
            ffmpeg_exe, "-y",
            "-r", f"{target_fps:.2f}",
            "-i", out_raw,
            "-c:v", "libx264",
            "-r", f"{target_fps:.2f}",
            "-preset", "fast",
            "-crf", "22",
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            out_final
        ], capture_output=True, timeout=120)
        if res.returncode == 0 and os.path.exists(out_final) and os.path.getsize(out_final) > 0:
            play_path = out_final
    except Exception as e:
        play_path = out_raw

    status_box.markdown(f"""
    <div class="telemetry-pill" style="color: #00ff87; border-color: #00ff87; font-size: 12px;">
        ⚡ PROCESSING COMPLETE // Video Ready at {target_fps:.1f} FPS (1.0x Normal Speed) // {state_mgr.total_unique} Unique Targets
    </div>
    """, unsafe_allow_html=True)

    # Cleanly mount the interactive HTML5 Video Player with Pause/Play/Seek/Volume/Speed controls
    if os.path.exists(play_path):
        with open(play_path, "rb") as vf:
            video_bytes = vf.read()
        viewport_box.empty()
        time.sleep(0.05)
        viewport_box.video(video_bytes, format="video/mp4", autoplay=True)

# ── REPORTS & TABLE ──────────────────────────────────────────────────────────
acc_counter.compute(state_mgr)
exporter.save_final_report(state_manager=state_mgr, acc_counter=acc_counter)

table_rows = []
for tid, s in sorted(state_mgr.all_states().items()):
    accs = []
    if s.final_cap:        accs.append("🧢 Cap")
    if s.final_mask:       accs.append("😷 Mask")
    if s.final_glasses:    accs.append("👓 Glasses")
    if s.final_headphones: accs.append("🎧 Headphones")
    table_rows.append({
        "Target ID":      f"TRK-{tid:03d}",
        "Inception":      f"F:{s.first_frame:03d}",
        "Latest":         f"F:{s.last_frame:03d}",
        "Frames":         s.frame_count,
        "Gear Verified":  ", ".join(accs) if accs else "None",
        "Classification": "SECURED // ACTIVE" if accs else "STANDARD TARGET"
    })

if table_rows:
    table_container.dataframe(pd.DataFrame(table_rows), width="stretch")

if os.path.exists("final_report.csv"):
    with open("final_report.csv", "r") as f:
        csv_ph.download_button("📥 EXPORT CSV DOSSIER", data=f.read(),
                               file_name="final_report.csv", mime="text/csv", key="dl_csv")

if os.path.exists("final_report.json"):
    with open("final_report.json", "r") as f:
        json_ph.download_button("📥 EXPORT JSON TELEMETRY", data=f.read(),
                                file_name="final_report.json", mime="application/json", key="dl_json")

if input_mode == "video" and os.path.exists("output_annotated.mp4"):
    with open("output_annotated.mp4", "rb") as vf:
        video_dl_ph.download_button(
            "📹 DOWNLOAD ANNOTATED VIDEO (1.0x NORMAL SPEED)",
            data=vf.read(),
            file_name="annotated_video_1x.mp4",
            mime="video/mp4",
            key="dl_vid"
        )

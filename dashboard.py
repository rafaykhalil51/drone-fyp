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
from association import associate_accessories_to_tracks
from state_manager import StateManager
from counter import AccessoryCounter
from visualization import Visualizer
from exporter import Exporter
from app_config import (
    zero_shot_enabled,
    zero_shot_model_path,
)
from model_loader import (
    ACCESSORY_OFFLINE_MESSAGE,
    DEFAULT_CUSTOM_PATH,
    ZERO_SHOT_WARNING,
    accessory_ai_label,
    format_class_validation,
    format_status_banner,
    log_status_banner,
    format_model_report,
    inspect_weight_inventory,
    load_accessory_detector,
    load_person_model,
    match_hud_slot,
    person_class_filter,
    resolve_models,
)
from person_registry import PersonTrackRegistry, reset_tracker
from voting_config import PHOTO_VOTING, VOTING

MODEL_PATH = DEFAULT_CUSTOM_PATH
# Shown in accessory cards when no validated custom model is loaded. Kept
# short on purpose: the full explanation appears once above the cards instead
# of being repeated inside every statistic tile.
OFFLINE_VALUE = "--"
OFFLINE_SUBLABEL = "Model offline"

# Minimum detection confidence for person tracking, independent of the UI
# slider, so weak boxes cannot create throwaway BoT-SORT track IDs.
PERSON_TRACK_MIN_CONF = 0.35


def person_tracks_from_result(result, require_track_id: bool = False) -> list[dict]:
    """Extract person tracks from a YOLO result (dual / person-only mode)."""
    tracks = []
    if result.boxes is None:
        return tracks
    for box in result.boxes:
        cls_id = int(box.cls[0])
        cls_name = result.names[cls_id]
        if match_hud_slot(cls_name) != "person":
            continue
        track_id = int(box.id[0]) if box.id is not None else None
        if require_track_id and track_id is None:
            continue
        tracks.append({
            "track_id": track_id if track_id is not None else len(tracks) + 1,
            "xyxy": box.xyxy[0].cpu().numpy().astype(int).tolist(),
            "confidence": float(box.conf[0]),
            "accessories": [],
        })
    return tracks


def parse_frame_detections(result, require_track_id: bool = False):
    """
    Split YOLO boxes into person tracks and accessory detections.
    Class labels are read from result.names (never hardcoded).
    """
    person_tracks = []
    accessory_dets = []
    frame_class_counts = {}

    if result.boxes is None:
        return person_tracks, accessory_dets, frame_class_counts

    for box in result.boxes:
        cls_id = int(box.cls[0])
        cls_name = result.names[cls_id]
        xyxy = box.xyxy[0].cpu().numpy().astype(int).tolist()
        conf = float(box.conf[0])
        track_id = int(box.id[0]) if box.id is not None else None

        if require_track_id and track_id is None:
            continue

        frame_class_counts[cls_name] = frame_class_counts.get(cls_name, 0) + 1
        slot = match_hud_slot(cls_name)

        if slot == "person":
            person_tracks.append({
                "track_id": track_id if track_id is not None else len(person_tracks) + 1,
                "xyxy": xyxy,
                "confidence": conf,
                "accessories": [],
            })
        elif slot in ("cap", "mask", "glasses", "headphones"):
            accessory_dets.append({
                "class_name": slot,
                "confidence": conf,
                "xyxy": xyxy,
            })

    return person_tracks, accessory_dets, frame_class_counts

# ── PAGE CONFIG ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="A.E.G.I.S. // Vision Intelligence",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed",
)

if "analysis" not in st.session_state:
    st.session_state.analysis = None
if "media_key" not in st.session_state:
    st.session_state.media_key = None
if "track_debug" not in st.session_state:
    st.session_state.track_debug = None
if "observation_report" not in st.session_state:
    st.session_state.observation_report = None

# ── CACHED MODEL LOADING ─────────────────────────────────────────────────────
@st.cache_resource
def load_unified_model(model_path: str):
    return load_person_model(model_path)


@st.cache_resource
def load_dual_models(person_path: str, accessory_path: str):
    person_model = load_person_model(person_path)
    accessory_model = load_accessory_detector(accessory_path)
    return person_model, accessory_model

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
header_ph = st.empty()


def render_header(
    accessory_ai_active: bool = False,
    person_ai_active: bool = True,
    zero_shot: bool = False,
):
    """
    Header with a live AI status pill. Same layout and styling as before; only
    the right-hand status text changes with real model state.

    Zero-shot gets its own amber pill: it is real inference from an untrained
    open-vocabulary model, so it must never read "ACCESSORY AI ACTIVE".
    """
    if accessory_ai_active:
        _pill = (
            '<div class="telemetry-pill" style="color: #00ff87; '
            'border-color: rgba(0, 255, 135, 0.4); background: rgba(0, 255, 135, 0.08);">'
            '<div class="pulse-dot"></div> ACCESSORY AI ACTIVE</div>'
        )
    elif zero_shot:
        _person = "PERSON AI ACTIVE" if person_ai_active else "PERSON AI FAILED"
        _pill = (
            f'<div class="telemetry-pill"><div class="pulse-dot"></div> {_person}</div>'
            '<div class="telemetry-pill" style="color: #ffb703; '
            'border-color: rgba(255, 183, 3, 0.45); background: rgba(255, 183, 3, 0.10);">'
            'ACCESSORY: ZERO-SHOT (EXPERIMENTAL)</div>'
        )
    else:
        _person = "PERSON AI ACTIVE" if person_ai_active else "PERSON AI FAILED"
        _pill = (
            f'<div class="telemetry-pill"><div class="pulse-dot"></div> {_person}</div>'
            '<div class="telemetry-pill" style="color: #ffb703; '
            'border-color: rgba(255, 183, 3, 0.35); background: rgba(255, 183, 3, 0.08);">'
            'ACCESSORY AI OFFLINE</div>'
        )

    header_ph.markdown(f"""
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
        {_pill}
        <div class="telemetry-pill" style="color: #c77dff; border-color: rgba(199, 125, 255, 0.3); background: rgba(199, 125, 255, 0.08);">CORE: YOLOv8 + BotSORT</div>
    </div>
</div>
""", unsafe_allow_html=True)


# Draw immediately so layout is stable; refreshed once models are resolved.
render_header(accessory_ai_active=False)

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

def update_top_metrics(tp=0, tc=0, tm=0, tg=0, th=0, tn=0, accessory_model_active=False,
                       zero_shot=False):
    pct = lambda v: f"{round((v/tp)*100)}%" if tp > 0 else "0%"
    render_hud_card(metric_ph[0], "TARGETS",     tp, "👥", "Tracked Persons",       "hud-metric-cyan",   "#00f2fe")

    if accessory_model_active:
        # Zero-shot numbers are real inference but from an untrained model, so
        # every card says so rather than presenting them as measured counts.
        _tag = " (est.)" if zero_shot else ""
        render_hud_card(metric_ph[1], "CRANIAL",     tc, "🧢", f"{pct(tc)} Caps{_tag}",       "hud-metric-amber",  "#ffb703")
        render_hud_card(metric_ph[2], "RESPIRATORY", tm, "😷", f"{pct(tm)} Masks{_tag}",      "hud-metric-orange", "#fb8500")
        render_hud_card(metric_ph[3], "OPTICAL",     tg, "👓", f"{pct(tg)} Glasses{_tag}",    "hud-metric-purple", "#c77dff")
        render_hud_card(metric_ph[4], "ACOUSTIC",    th, "🎧", f"{pct(th)} Headphones{_tag}", "hud-metric-green",  "#00ff87")
    else:
        # Accessory AI offline: show a compact placeholder instead of pasting a
        # sentence into every card. The reason is stated once, above the cards.
        _dim = "#64748b"
        _off = OFFLINE_SUBLABEL
        render_hud_card(metric_ph[1], "CRANIAL",     OFFLINE_VALUE, "🧢", _off, "hud-metric-amber",  _dim)
        render_hud_card(metric_ph[2], "RESPIRATORY", OFFLINE_VALUE, "😷", _off, "hud-metric-orange", _dim)
        render_hud_card(metric_ph[3], "OPTICAL",     OFFLINE_VALUE, "👓", _off, "hud-metric-purple", _dim)
        render_hud_card(metric_ph[4], "ACOUSTIC",    OFFLINE_VALUE, "🎧", _off, "hud-metric-green",  _dim)

    _plain_sub = (
        f"{pct(tn)} No Gear{' (est.)' if zero_shot else ''}"
        if accessory_model_active else "No detected accessories"
    )
    render_hud_card(metric_ph[5], "PLAIN",       tn, "👤", _plain_sub, "hud-metric-slate",  "#94a3b8")

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
        model_path = st.text_input(
            "Accessory Model Weights",
            value=st.session_state.get("accessory_model_override") or MODEL_PATH,
            key="model_input",
            help=(
                "Trained accessory weights for cap/mask/glasses/headphones. "
                f"Default: {DEFAULT_CUSTOM_PATH}. Person detection uses a "
                "separate pretrained COCO model and does not need this."
            ),
        )

    # Optional: load a trained accessory model without touching config.yaml.
    with st.expander("🧩 ACCESSORY MODEL SELECTION (OPTIONAL)", expanded=False):
        st.caption(
            f"Leave empty to use `{DEFAULT_CUSTOM_PATH}` automatically. "
            "Person detection and tracking work regardless."
        )
        _acc_upload = st.file_uploader(
            "Upload trained accessory model (.pt)",
            type=["pt"],
            key="accessory_model_uploader",
        )
        if _acc_upload is not None:
            # Persist under models/ so the choice survives reruns and the
            # weights are not re-written on every script pass.
            _models_dir = Path("models")
            _models_dir.mkdir(exist_ok=True)
            _dest = _models_dir / _acc_upload.name
            _sig = f"{_acc_upload.name}:{_acc_upload.size}"
            if st.session_state.get("accessory_upload_sig") != _sig:
                _dest.write_bytes(_acc_upload.getvalue())
                st.session_state.accessory_upload_sig = _sig
                st.session_state.accessory_model_override = _dest.as_posix()
                st.success(f"Saved accessory model to `{_dest.as_posix()}`.")
            model_path = st.session_state.get(
                "accessory_model_override", _dest.as_posix()
            )
        if st.session_state.get("accessory_model_override"):
            if st.button("↩ Reset to configured model", key="btn_reset_acc_model"):
                st.session_state.accessory_model_override = None
                st.session_state.accessory_upload_sig = None
                st.rerun()

        # ── Experimental zero-shot fallback (opt-in) ─────────────────────
        st.markdown("---")
        _zs_available = Path(zero_shot_model_path()).exists() or (
            Path.cwd() / zero_shot_model_path()
        ).exists()
        zero_shot_requested = st.checkbox(
            "Enable ZERO-SHOT accessory estimation (EXPERIMENTAL — not a trained model)",
            value=zero_shot_enabled(),
            key="zero_shot_toggle",
            disabled=not _zs_available,
            help=(
                "Uses YOLO-World prompted with text labels instead of a model "
                "trained on your classes. Real inference, but an unvalidated "
                "guess. Measured on this project's own sample media it found "
                "some caps and glasses but zero masks and zero headphones. "
                "Never reported as ACCESSORY AI ACTIVE, and the exported "
                "report records that zero-shot produced the numbers."
            ),
        )
        if not _zs_available:
            st.caption(
                f"`{zero_shot_model_path()}` not present, so zero-shot mode "
                "is unavailable."
            )

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
def render_chart(tc=0, tm=0, tg=0, th=0, tn=0, accessory_model_active=False,
                 zero_shot=False):
    global _chart_count
    _chart_count += 1
    if not accessory_model_active:
        # Person AI is up, accessory AI is not. Say exactly that instead of
        # implying COCO can supply the four accessory classes.
        labels = ['Tracked Persons 👤']
        values = [max(tn, 1)]
        colors = ['#64748b']
        textinfo = 'label'
        center_text = 'ACCESSORY<br>AI OFFLINE'
    else:
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
            center_text = (
                f'{total}<br>GEAR (EST.)' if zero_shot else f'{total}<br>GEAR'
            )

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

def build_table_rows(state_mgr):
    rows = []
    for tid, s in sorted(state_mgr.all_states().items()):
        accs = []
        if s.final_cap:
            accs.append("🧢 Cap")
        if s.final_mask:
            accs.append("😷 Mask")
        if s.final_glasses:
            accs.append("👓 Glasses")
        if s.final_headphones:
            accs.append("🎧 Headphones")
        hits = s.hit_counts()
        ratios = s.ratios()
        rows.append({
            "Target ID":      f"TRK-{tid:03d}",
            "Inception":      f"F:{s.first_frame:03d}",
            "Latest":         f"F:{s.last_frame:03d}",
            "Frames Seen":    s.frames_seen,
            "Gear Verified":  ", ".join(accs) if accs else "None",
            "Cap Hits":       hits["cap"],
            "Cap Ratio":      round(ratios["cap"], 2),
            "Mask Hits":      hits["mask"],
            "Mask Ratio":     round(ratios["mask"], 2),
            "Glasses Hits":   hits["glasses"],
            "Glasses Ratio":  round(ratios["glasses"], 2),
            "Headphone Hits": hits["headphones"],
            "Headphone Ratio": round(ratios["headphones"], 2),
            "Classification": "SECURED // ACTIVE" if accs else "STANDARD TARGET",
        })
    return rows

def save_analysis(tp, tc, tm, tg, th, tn, table_rows, media_key,
                  video_path=None, photo_rgb=None, class_counts=None, person_total=None,
                  accessory_model_active=False, zero_shot=False):
    st.session_state.analysis = {
        "tp": tp, "tc": tc, "tm": tm, "tg": tg, "th": th, "tn": tn,
        "table_rows": table_rows,
        "video_path": video_path,
        "photo_rgb": photo_rgb,
        "class_counts": class_counts or {},
        "person_total": person_total if person_total is not None else tp,
        "accessory_model_active": accessory_model_active,
        # Records which detector produced these numbers so a restored view
        # keeps the zero-shot caveat attached to them.
        "zero_shot": zero_shot,
    }
    st.session_state.media_key = media_key

def _format_class_name(name: str) -> str:
    return name.replace("_", " ").title()

def render_saved_analysis():
    """Restore HUD, chart, table, and viewport from the last completed run."""
    analysis = st.session_state.analysis
    if not analysis:
        update_top_metrics(0, 0, 0, 0, 0, 0, accessory_model_active=False)
        render_chart(0, 0, 0, 0, 0, accessory_model_active=False)
        return

    person_total = analysis.get("person_total", analysis["tp"])
    acc_active = analysis.get("accessory_model_active", False)
    acc_zero_shot = analysis.get("zero_shot", False)
    update_top_metrics(
        person_total, analysis["tc"], analysis["tm"],
        analysis["tg"], analysis["th"], analysis["tn"],
        accessory_model_active=acc_active,
        zero_shot=acc_zero_shot,
    )
    render_chart(
        analysis["tc"], analysis["tm"], analysis["tg"],
        analysis["th"], analysis["tn"],
        accessory_model_active=acc_active,
        zero_shot=acc_zero_shot,
    )

    class_counts = analysis.get("class_counts") or {}
    person_rows = analysis.get("table_rows") or []
    if class_counts:
        # With an accessory model the register lists per-person confirmed gear;
        # otherwise it falls back to the per-class count table.
        if acc_active and person_rows:
            table_container.dataframe(
                pd.DataFrame(person_rows), width="stretch", hide_index=True
            )
        else:
            class_df = pd.DataFrame([
                {"Object Class": _format_class_name(name), "Count": count}
                for name, count in sorted(class_counts.items(), key=lambda x: (-x[1], x[0]))
            ])
            table_container.dataframe(class_df, width="stretch", hide_index=True)

        summary_lines = "<br>".join(
            f"<span style='color:#00f2fe;font-family:Orbitron,monospace;'>"
            f"{_format_class_name(name)}:</span> "
            f"<span style='color:#e2e8f0;'>{count}</span>"
            for name, count in sorted(class_counts.items(), key=lambda x: (-x[1], x[0]))
        )
        if not acc_active:
            summary_lines += (
                "<br><br><span style='color:#64748b;font-size:12px;'>"
                "Cap / Mask / Glasses / Headphones: accessory AI offline</span>"
            )
        status_box.markdown(f"""
        <div class="telemetry-pill" style="color: #00ff87; border-color: #00ff87; font-size: 12px; margin-bottom: 10px;">
            ⚡ ANALYSIS COMPLETE // {person_total} unique person(s) tracked
        </div>
        <div class="cyber-card" style="font-family: Rajdhani, sans-serif; font-size: 14px; line-height: 1.8;">
            <div style="font-family: Orbitron, sans-serif; font-size: 13px; font-weight: 700;
                        color: #c77dff; margin-bottom: 8px;">📋 DETECTION SUMMARY</div>
            {summary_lines}
        </div>
        """, unsafe_allow_html=True)
    elif analysis.get("table_rows"):
        table_container.dataframe(pd.DataFrame(analysis["table_rows"]), width="stretch")

    if analysis.get("photo_rgb") is not None:
        viewport_box.image(analysis["photo_rgb"], channels="RGB", width="stretch")
    elif analysis.get("video_path") and os.path.exists(analysis["video_path"]):
        with open(analysis["video_path"], "rb") as vf:
            viewport_box.video(vf.read(), format="video/mp4", autoplay=False)

# ── RESOLVE INPUT FILE & MODE ────────────────────────────────────────────────
# Streamlit keeps an uploaded file in the widget for the whole session, so both
# uploaders report a value on every rerun regardless of the visible tab. Listing
# candidates and picking the first one whose key differs from the last processed
# key means a fresh photo upload or a sample-button click is not shadowed by a
# video that was uploaded earlier and already analysed.
_candidates = []          # (mode, path_or_None, media_key, uploaded_file_or_None)

if btn_sample_video and os.path.exists("input.mp4"):
    _candidates.append(("video", "input.mp4", "sample:video:input.mp4", None))
if btn_sample_photo and os.path.exists("uploads/classroom_sample.jpg"):
    _candidates.append(
        ("photo", "uploads/classroom_sample.jpg", "sample:photo:classroom_sample.jpg", None)
    )
if uploaded_photo is not None:
    _candidates.append(
        ("photo", None, f"photo:{uploaded_photo.name}:{uploaded_photo.size}", uploaded_photo)
    )
if uploaded_video is not None:
    _candidates.append(
        ("video", None, f"video:{uploaded_video.name}:{uploaded_video.size}", uploaded_video)
    )

_last_key = st.session_state.media_key
_selected = next((c for c in _candidates if c[2] != _last_key), None)

input_mode = _selected[0] if _selected else "video"
media_key = _selected[2] if _selected else None
should_process = _selected is not None

# Remove the temp file kept from a previous rerun before creating another, so a
# long session cannot accumulate a copy of every uploaded video in %TEMP%.
_stale_tmp = st.session_state.get("tmp_media_path")
if _stale_tmp and os.path.exists(_stale_tmp):
    try:
        os.unlink(_stale_tmp)
    except OSError:
        pass                      # still held by a reader; retried next rerun
    st.session_state.tmp_media_path = None

target_media_path = None
if should_process:
    _mode, _path, _key, _upload = _selected
    if _upload is not None:
        # .getvalue() rather than .read(): the read pointer sits at EOF after the
        # first rerun, which would silently write a 0-byte file.
        _suffix = Path(_upload.name).suffix or (".mp4" if _mode == "video" else ".jpg")
        _tf = tempfile.NamedTemporaryFile(delete=False, suffix=_suffix)
        try:
            _tf.write(_upload.getvalue())
        finally:
            _tf.close()           # Windows will not let cv2 reopen a held handle
        target_media_path = _tf.name
        st.session_state.tmp_media_path = target_media_path
    else:
        target_media_path = _path

    if not target_media_path or not os.path.exists(target_media_path):
        st.error(f"Input file is unavailable: `{target_media_path}`")
        should_process = False
    elif os.path.getsize(target_media_path) == 0:
        st.error("Uploaded file is empty (0 bytes). Please re-upload.")
        should_process = False

render_saved_analysis()

# ── INITIALIZE DETECTION & STATE ENGINES ─────────────────────────────────────
model_resolution = resolve_models(model_path)
person_weights_missing = not os.path.exists(model_resolution.person_path)

if person_weights_missing and should_process:
    st.error(
        f"Person model weights not found: `{model_resolution.person_path}`. "
        f"Place {model_resolution.person_path} in the project folder."
    )
    st.stop()
elif person_weights_missing:
    st.warning(
        f"Person model weights not found: `{model_resolution.person_path}`. "
        "Upload media after placing weights in the project folder."
    )

yolo_engine = None
accessory_detector = None
use_unified = model_resolution.mode == "unified"

# Accessory AI counts as active only when a real custom model loaded AND
# resolved all four required classes. Never inferred from the UI rendering.
accessory_model_active = False

if not person_weights_missing:
    try:
        if model_resolution.mode == "dual":
            yolo_engine, accessory_detector = load_dual_models(
                model_resolution.person_path,
                model_resolution.accessory_path,
            )
            accessory_detector.confidence = min(conf_thresh, accessory_detector.confidence)
            accessory_model_active = bool(
                accessory_detector is not None
                and accessory_detector.available
                and getattr(accessory_detector, "usable", True)
                and model_resolution.accessory_ai_active
            )
        else:
            yolo_engine = load_unified_model(model_resolution.person_path)
            accessory_model_active = model_resolution.accessory_ai_active
    except Exception as exc:
        st.error(
            f"Model loading failed ({type(exc).__name__}: {exc}). "
            "Check that the weight files are valid YOLO models."
        )
        st.stop()

# ── Experimental zero-shot fallback, only when explicitly opted in ──────────
# Never engaged when a trained model is available, and never allowed to set
# accessory_model_active, which is reserved for a validated trained model.
trained_unusable = bool(
    accessory_detector is not None
    and accessory_detector.available
    and not getattr(accessory_detector, "usable", True)
)

zero_shot_active = False
# Use zero-shot when the user opts in, OR when a trained .pt loaded but
# cannot emit any boxes (the 6-photo model). Otherwise the dashboard
# claims ACCESSORY AI ACTIVE and then detects nothing.
_use_zero_shot = (
    (not accessory_model_active)
    and not person_weights_missing
    and (zero_shot_requested or trained_unusable)
)
if _use_zero_shot:
    _zs = None
    try:
        from zero_shot_accessory_detector import ZeroShotAccessoryDetector
        from app_config import zero_shot_confidence, zero_shot_imgsz
        # Cached in session state: building CLIP text embeddings is slow and
        # must not repeat on every Streamlit rerun.
        _zs = st.session_state.get("_zs_detector")
        if _zs is None or not getattr(_zs, "available", False):
            _zs = ZeroShotAccessoryDetector(
                model_path=zero_shot_model_path(),
                confidence=zero_shot_confidence(),
                imgsz=zero_shot_imgsz(),
            )
            st.session_state._zs_detector = _zs      # cache across reruns
    except Exception as exc:
        st.warning(f"Zero-shot mode could not start ({type(exc).__name__}: {exc}).")
        _zs = None

    if _zs is not None and _zs.available:
        accessory_detector = _zs
        zero_shot_active = True

# Whether accessory numbers exist at all, from either source. Kept separate
# from accessory_model_active, which stays reserved for a validated trained
# model and is what the status pill and the manifest key off.
accessory_values_available = accessory_model_active or zero_shot_active

# render_saved_analysis() runs before models are resolved, so on a fresh load
# it draws the chart as "ACCESSORY AI OFFLINE". Redraw it once the accessory
# source is known, otherwise the chart contradicts the header pill.
if st.session_state.get("analysis") is None and accessory_values_available:
    update_top_metrics(0, 0, 0, 0, 0, 0,
                       accessory_model_active=True,
                       zero_shot=zero_shot_active)
    render_chart(0, 0, 0, 0, 0,
                 accessory_model_active=accessory_values_available,
                 zero_shot=zero_shot_active)

# ── ONE clear accessory status message (not repeated in every card) ──────────
if trained_unusable:
    st.warning(
        "A file is at `models/accessory_best.pt` and it has the right class "
        "names, but it produces **no detections** — even on the classroom "
        "photos it was trained on. Six images is not enough to teach YOLO. "
        "The portal is using zero-shot detection so accessories can still "
        "appear on your video. Add many more labelled photos and retrain "
        "before this custom model can work on its own."
    )
if zero_shot_active:
    st.warning(ZERO_SHOT_WARNING)
elif not accessory_model_active:
    _acc_status = model_resolution.accessory_status
    if _acc_status == "invalid":
        st.warning(
            "**Accessory model loaded but required classes are missing.**\n\n"
            "Accessory detection is disabled to avoid reporting incorrect "
            "statistics. Person detection and tracking are unaffected."
        )
        st.code(
            format_class_validation(model_resolution.accessory_classes),
            language="text",
        )
    elif _acc_status == "disabled":
        st.info(
            "Accessory detection is disabled in `config.yaml` "
            "(`accessories.enabled: false`). Person detection and tracking "
            "remain active."
        )
    else:
        st.warning(ACCESSORY_OFFLINE_MESSAGE)
        st.caption(
            f"Expected location: `{DEFAULT_CUSTOM_PATH}` "
            f"(project root: `{Path.cwd().as_posix()}`)"
        )

# Print the startup diagnostic once per session, not on every rerun.
if not st.session_state.get("status_banner_logged"):
    log_status_banner(
        model_resolution,
        person_loaded=not person_weights_missing,
        tracking_active=not person_weights_missing,
    )
    st.session_state.status_banner_logged = True

# Header pill now reflects verified model state.
render_header(
    accessory_ai_active=accessory_model_active,
    person_ai_active=not person_weights_missing,
    zero_shot=zero_shot_active,
)

with st.expander("🧠 NEURAL MODEL MANIFEST", expanded=True):
    st.code(format_model_report(model_resolution), language="text")
    st.code(
        format_status_banner(
            model_resolution,
            person_loaded=not person_weights_missing,
            tracking_active=not person_weights_missing,
        ),
        language="text",
    )
    if zero_shot_active:
        st.code(
            "Accessory source: ZERO-SHOT (EXPERIMENTAL)\n"
            f"  model   : {accessory_detector.model_path}\n"
            f"  conf    : {accessory_detector.confidence}\n"
            f"  imgsz   : {accessory_detector.imgsz}\n"
            f"  prompts : {len(accessory_detector.vocabulary)} text labels\n"
            "  trained on cap/mask/glasses/headphones: NO",
            language="text",
        )
    if accessory_model_active and accessory_detector is not None:
        st.code(
            "Accessory model loaded:\n"
            f"  {accessory_detector.model_path}\n\n"
            "Classes:\n"
            f"  {accessory_detector.class_names}\n\n"
            + format_class_validation(accessory_detector.class_names),
            language="text",
        )
    inventory = inspect_weight_inventory()
    if inventory:
        inv_lines = []
        for item in inventory:
            slots = ", ".join(item["accessory_slots"]) if item["accessory_slots"] else "—"
            inv_lines.append(
                f"{item['path']} | person={item['has_person']} | accessory slots={slots} | classes={len(item['classes'])}"
            )
        st.caption("Discovered weights:\n" + "\n".join(inv_lines))

t0 = time.time()

if should_process and input_mode == "photo" and target_media_path is not None:
    state_mgr    = StateManager()
    acc_counter  = AccessoryCounter()
    visualizer   = Visualizer(vis_cfg={"box_thickness": 2, "text_scale": 0.65, "show_confidence": True})
    exporter     = Exporter(csv_path="tracks.csv", summary_path="summary.json",
                            final_report_csv="final_report.csv", final_report_json="final_report.json")
    # ── 1. PHOTO EXECUTION ───────────────────────────────────────────────────
    frame = cv2.imread(target_media_path)
    if frame is None:
        status_box.error("Could not decode image file.")
        st.stop()

    yolo = yolo_engine
    if use_unified:
        results = yolo.predict(
            frame, conf=conf_thresh, iou=0.50, imgsz=480, verbose=False,
        )
        result = results[0]
        tracks, accessory_dets, frame_class_counts = parse_frame_detections(result)
    else:
        results = yolo.predict(
            frame,
            conf=conf_thresh,
            iou=0.50,
            imgsz=480,
            classes=person_class_filter(yolo),
            verbose=False,
        )
        result = results[0]
        tracks = person_tracks_from_result(result)
        accessory_dets = (
            accessory_detector.detect(frame)
            if accessory_detector is not None and accessory_detector.available
            else []
        )
        frame_class_counts = {"person": len(tracks)}
        for det in accessory_dets:
            frame_class_counts[det["class_name"]] = frame_class_counts.get(det["class_name"], 0) + 1

    acc_map = associate_accessories_to_tracks(tracks, accessory_dets, head_fraction=0.52)
    for t in tracks:
        t["accessories"] = acc_map.get(t["track_id"], [])

    if tracks:
        state_mgr.update(0, tracks)
        for t in tracks:
            tid = t["track_id"]
            state_mgr.update_state(tid, t.get("accessories", []), 0)
            # A still image has a single frame of evidence, so the multi-frame
            # minimums that guard the video path do not apply here.
            state_mgr.apply_temporal_voting(tid, PHOTO_VOTING)

        all_st   = list(state_mgr.all_states().values())
        tot_cap  = sum(1 for s in all_st if s.final_cap)
        tot_mask = sum(1 for s in all_st if s.final_mask)
        tot_gls  = sum(1 for s in all_st if s.final_glasses)
        tot_hd   = sum(1 for s in all_st if s.final_headphones)
        tot_none = sum(
            1 for s in all_st
            if not (s.final_cap or s.final_mask or s.final_glasses or s.final_headphones)
        )
    else:
        tot_cap = tot_mask = tot_gls = tot_hd = 0
        tot_none = 0

    tot_p = len(tracks)

    update_top_metrics(tot_p, tot_cap, tot_mask, tot_gls, tot_hd, tot_none,
                       accessory_model_active=accessory_values_available,
                       zero_shot=zero_shot_active)
    render_chart(tot_cap, tot_mask, tot_gls, tot_hd, tot_none,
                 accessory_model_active=accessory_values_available,
                 zero_shot=zero_shot_active)

    live_totals = {"persons": tot_p, "caps": tot_cap, "masks": tot_mask, "glasses": tot_gls, "headphones": tot_hd}
    if use_unified:
        annotated = result.plot()
        visualizer._draw_hud(annotated, live_totals, None)
    else:
        annotated = visualizer.draw(frame.copy(), tracks, live_totals=live_totals)
        associated = {id(acc) for accs in acc_map.values() for acc in accs}
        for acc in accessory_dets:
            if id(acc) not in associated:
                visualizer._draw_accessory(annotated, acc)
    rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
    viewport_box.image(rgb, channels="RGB", width="stretch")

    elapsed = time.time() - t0
    acc_msg = ("CUSTOM MODEL" if accessory_model_active
               else "ZERO-SHOT EST." if zero_shot_active else "DETECTION ONLY")
    status_box.markdown(f"""
    <div class="telemetry-pill" style="color: #00ff87; border-color: #00ff87; font-size: 12px;">
        ⚡ IMAGE SCAN COMPLETE: {elapsed:.2f}s // {tot_p} PERSONS DETECTED // {acc_msg}
    </div>
    """, unsafe_allow_html=True)

    if tracks:
        acc_counter.compute(state_mgr)
        table_rows = build_table_rows(state_mgr)
    else:
        acc_counter.total_unique_persons = 0
        acc_counter.totals = {"cap": 0, "mask": 0, "glasses": 0, "headphones": 0}
        acc_counter._computed = True
        table_rows = []

    class_counts_photo = {
        _format_class_name(k): v for k, v in frame_class_counts.items()
    }
    exporter.save_final_report(
        state_manager=state_mgr,
        acc_counter=acc_counter,
        accessory_source=(
            "trained:" + str(model_resolution.accessory_path)
            if accessory_model_active
            else "zero-shot:yolo-world (EXPERIMENTAL, untrained)"
            if zero_shot_active
            else "none (accessory AI offline)"
        ),
    )
    save_analysis(
        tot_p, tot_cap, tot_mask, tot_gls, tot_hd, tot_none,
        table_rows, media_key, photo_rgb=rgb,
        class_counts=class_counts_photo, person_total=tot_p,
        accessory_model_active=accessory_values_available,
        zero_shot=zero_shot_active,
    )
    render_saved_analysis()

elif should_process and input_mode == "video" and target_media_path is not None:
    # ── 2. VIDEO EXECUTION (WITH LOADING ANIMATION & PLAYER CONTROLS) ────────
    state_mgr    = StateManager()
    acc_counter  = AccessoryCounter()
    visualizer   = Visualizer(vis_cfg={"box_thickness": 2, "text_scale": 0.65, "show_confidence": True})
    exporter     = Exporter(csv_path="tracks.csv", summary_path="summary.json",
                            final_report_csv="final_report.csv", final_report_json="final_report.json")

    try:
        source = VideoSource(target_media_path)
    except Exception as e:
        status_box.error(f"Stream failure: {e}")
        st.stop()

    target_fps = float(source.fps) if (source.fps and 1.0 <= source.fps <= 120.0) else 30.0

    # A container can report a valid stream but zero geometry; VideoWriter would
    # then open on some backends and silently emit an unusable file.
    if source.width <= 0 or source.height <= 0:
        source.release()
        status_box.error(
            f"Video reports an invalid frame size ({source.width}x{source.height}). "
            "The file may be corrupt or use an unsupported container."
        )
        st.stop()

    out_raw  = "output_raw.mp4"
    out_final = "output_annotated.mp4"

    # mp4v is the usual choice but is not guaranteed to be present in every
    # OpenCV build, so fall back rather than dead-ending the whole run.
    writer = None
    for _fourcc, _path in (("mp4v", out_raw), ("avc1", out_raw), ("MJPG", "output_raw.avi")):
        _w = cv2.VideoWriter(_path, cv2.VideoWriter_fourcc(*_fourcc),
                             target_fps, (source.width, source.height))
        if _w.isOpened():
            writer, out_raw = _w, _path
            break
        _w.release()

    if writer is None:
        # Release the capture too, or Windows keeps the input file locked.
        source.release()
        status_box.error(
            "Failed to initialize video writer for processed output "
            f"({source.width}x{source.height} @ {target_fps:.1f}fps). "
            "No usable codec found (tried mp4v, avc1, MJPG)."
        )
        st.stop()

    total_frames = max(source.total_frames, 1)
    frame_idx = 0
    prog_bar = progress_box.progress(0)

    yolo = yolo_engine

    # The model is cached across reruns, so clear any tracker state left by a
    # previous video; otherwise track IDs keep climbing between uploads.
    reset_tracker(yolo)

    unique_tracks_by_class = {}   # model class name -> set of persistent track IDs
    # Raw BoT-SORT IDs are not people: a track must survive several frames
    # before it counts toward the unique-person total.
    person_registry = PersonTrackRegistry()
    tot_p = tot_cap = tot_mask = tot_gls = tot_hd = tot_none = 0

    # Person tracking needs a firmer confidence gate than the UI slider default
    # so flickering low-score boxes stop spawning throwaway track IDs.
    person_conf = max(conf_thresh, PERSON_TRACK_MIN_CONF)

    # Optimize PyTorch CPU parallelism
    try:
        import torch
        torch.set_num_threads(max(os.cpu_count() or 4, 4))
    except Exception:
        pass

    # Show Loading Animation Spinner while processing frames
    with st.spinner("⚡ HIGH-SPEED NEURAL INFERENCE IN PROGRESS // ANALYZING FRAMES..."):
        try:
            with source:
                for frame in source:
                    # 1-4. Person-only detection + BoT-SORT persistent tracking
                    if use_unified:
                        results = yolo.track(
                            frame,
                            conf=person_conf,
                            iou=0.50,
                            imgsz=480,
                            tracker="botsort.yaml",
                            persist=True,
                            verbose=False,
                        )
                        result = results[0]
                        person_tracks, accessory_dets, _ = parse_frame_detections(
                            result, require_track_id=True
                        )
                    else:
                        results = yolo.track(
                            frame,
                            conf=person_conf,
                            iou=0.50,
                            imgsz=480,
                            classes=person_class_filter(yolo),
                            tracker="botsort.yaml",
                            persist=True,
                            verbose=False,
                        )
                        result = results[0]
                        person_tracks = person_tracks_from_result(result, require_track_id=True)
                        accessory_dets = (
                            accessory_detector.detect(frame)
                            if accessory_detector is not None and accessory_detector.available
                            else []
                        )

                    # 5-7. Only tracks that survive several frames become people.
                    for t in person_tracks:
                        person_registry.observe(
                            t["track_id"], frame_idx, t.get("confidence", 0.0)
                        )
                    person_tracks = [
                        t for t in person_tracks
                        if person_registry.is_confirmed(t["track_id"])
                    ]

                    if use_unified and result.boxes is not None:
                        # Non-person COCO classes are tallied separately and never
                        # contribute to the person total.
                        for box in result.boxes:
                            if box.id is None:
                                continue
                            cls_name = result.names[int(box.cls[0])]
                            if match_hud_slot(cls_name) == "person":
                                continue
                            unique_tracks_by_class.setdefault(cls_name, set()).add(
                                int(box.id[0])
                            )
                    acc_map = associate_accessories_to_tracks(
                        person_tracks, accessory_dets, head_fraction=0.52
                    )
                    for t in person_tracks:
                        t["accessories"] = acc_map.get(t["track_id"], [])

                    # 5. Store accessory observations per persistent track ID.
                    #    Provisional voting only drives the live HUD; the
                    #    authoritative decision happens after the last frame.
                    if person_tracks:
                        state_mgr.update(frame_idx, person_tracks)
                        for t in person_tracks:
                            tid = t["track_id"]
                            state_mgr.update_state(tid, t.get("accessories", []), frame_idx)

                        if accessory_values_available:
                            for t in person_tracks:
                                state_mgr.apply_temporal_voting(t["track_id"], VOTING)

                    tot_p = person_registry.unique_person_count
                    if accessory_values_available:
                        all_st = list(state_mgr.all_states().values())
                        tot_cap = sum(1 for s in all_st if s.final_cap)
                        tot_mask = sum(1 for s in all_st if s.final_mask)
                        tot_gls = sum(1 for s in all_st if s.final_glasses)
                        tot_hd = sum(1 for s in all_st if s.final_headphones)
                        tot_none = sum(
                            1 for s in all_st
                            if not (s.final_cap or s.final_mask or s.final_glasses or s.final_headphones)
                        )
                    else:
                        tot_cap = tot_mask = tot_gls = tot_hd = 0
                        tot_none = tot_p

                    # 2. Draw bounding boxes, track IDs, class labels, then HUD overlay
                    if use_unified:
                        annotated = result.plot()
                        live_totals = {
                            "persons": tot_p,
                            "caps": tot_cap,
                            "masks": tot_mask,
                            "glasses": tot_gls,
                            "headphones": tot_hd,
                        }
                        visualizer._draw_hud(annotated, live_totals, None)
                    else:
                        live_totals = {
                            "persons": tot_p,
                            "caps": tot_cap,
                            "masks": tot_mask,
                            "glasses": tot_gls,
                            "headphones": tot_hd,
                        }
                        annotated = visualizer.draw(frame.copy(), person_tracks, live_totals=live_totals)
                        # Keep accessory boxes that matched no person visible too
                        associated = {
                            id(acc) for accs in acc_map.values() for acc in accs
                        }
                        for acc in accessory_dets:
                            if id(acc) not in associated:
                                visualizer._draw_accessory(annotated, acc)

                    # 3. Write processed frame to output video
                    writer.write(annotated)

                    frame_idx += 1
                    if frame_idx % 4 == 0 or frame_idx == total_frames:
                        pct = min(frame_idx / total_frames, 1.0)
                        prog_bar.progress(pct)
                        fps_now = frame_idx / max(time.time() - t0, 0.01)
                        det_count = len(result.boxes) if result.boxes is not None else 0
                        status_box.markdown(f"""
                        <div class="telemetry-pill" style="color: #00ff87; border-color: #00ff87; font-size: 12px;">
                            ⚡ FRAME {frame_idx}/{total_frames} ({pct*100:.0f}%) // {det_count} in frame // {tot_p} unique persons // {fps_now:.1f} FPS
                        </div>
                        """, unsafe_allow_html=True)
                        update_top_metrics(tot_p, tot_cap, tot_mask, tot_gls, tot_hd, tot_none,
                                           accessory_model_active=accessory_values_available,
                                           zero_shot=zero_shot_active)
                        render_chart(tot_cap, tot_mask, tot_gls, tot_hd, tot_none,
                                     accessory_model_active=accessory_values_available,
                                     zero_shot=zero_shot_active)
                        rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                        viewport_box.image(rgb, channels="RGB", width="stretch")
        finally:
            writer.release()

    tot_p = person_registry.unique_person_count
    # Copy, not alias: this dict is later reduced to counts and must not keep a
    # live reference to the registry's mutable confirmed-ID set.
    unique_tracks_by_class["person"] = set(person_registry.confirmed_ids)

    # Update final HUD metrics & Donut Chart
    update_top_metrics(tot_p, tot_cap, tot_mask, tot_gls, tot_hd, tot_none,
                       accessory_model_active=accessory_values_available,
                       zero_shot=zero_shot_active)
    render_chart(tot_cap, tot_mask, tot_gls, tot_hd, tot_none,
                 accessory_model_active=accessory_values_available,
                 zero_shot=zero_shot_active)

    person_registry.log_debug_report()
    st.session_state.track_debug = person_registry.debug_report()

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
        else:
            # Raw mp4v/MJPG often will not play in a browser, so say so rather
            # than leaving the user with a silently blank player.
            _err = (res.stderr or b"").decode("utf-8", "replace").strip().splitlines()
            st.warning(
                "H.264 conversion failed, showing the raw encoder output instead — "
                "your browser may not be able to play it. "
                f"ffmpeg exit code {res.returncode}."
                + (f"\n\n`{_err[-1]}`" if _err else "")
            )
    except Exception as e:
        play_path = out_raw
        st.warning(
            f"H.264 conversion unavailable ({type(e).__name__}: {e}). "
            "Showing the raw encoder output, which may not play in the browser."
        )

    status_box.markdown(f"""
    <div class="telemetry-pill" style="color: #00ff87; border-color: #00ff87; font-size: 12px;">
        ⚡ PROCESSING COMPLETE // {frame_idx} frames processed
    </div>
    """, unsafe_allow_html=True)

    unique_class_counts = {
        cls: len(ids) for cls, ids in unique_tracks_by_class.items() if ids
    }
    tot_p = person_registry.unique_person_count
    if accessory_values_available:
        # 7/8. Decide each unique person's gear from their full observation
        #      history (never from a single frame). Thresholds: voting_config.py
        state_mgr.finalize_accessories(VOTING)

        all_states = list(state_mgr.all_states().values())
        tot_cap = sum(1 for s in all_states if s.final_cap)
        tot_mask = sum(1 for s in all_states if s.final_mask)
        tot_gls = sum(1 for s in all_states if s.final_glasses)
        tot_hd = sum(1 for s in all_states if s.final_headphones)
        tot_none = sum(
            1 for s in all_states
            if not (s.final_cap or s.final_mask or s.final_glasses or s.final_headphones)
        )
        acc_counter.compute(state_mgr)
        if model_resolution.mode == "dual":
            unique_class_counts = {"person": tot_p}
            for key, val in acc_counter.totals.items():
                if val > 0:
                    unique_class_counts[key] = val
        # Per-track hits/ratios/decisions, recorded after the final vote
        st.session_state.observation_report = state_mgr.format_observation_report(VOTING)
    else:
        tot_cap = tot_mask = tot_gls = tot_hd = 0
        tot_none = tot_p
        acc_counter.total_unique_persons = tot_p
        acc_counter.totals = {"cap": 0, "mask": 0, "glasses": 0, "headphones": 0}
        acc_counter._computed = True
        st.session_state.observation_report = None

    exporter.save_final_report(
        state_manager=state_mgr,
        acc_counter=acc_counter,
        accessory_source=(
            "trained:" + str(model_resolution.accessory_path)
            if accessory_model_active
            else "zero-shot:yolo-world (EXPERIMENTAL, untrained)"
            if zero_shot_active
            else "none (accessory AI offline)"
        ),
    )

    if accessory_values_available:
        # Per-unique-person gear register (cap / mask / glasses / headphones / none)
        register_rows = build_table_rows(state_mgr)
    else:
        register_rows = [
            {"Object Class": _format_class_name(name), "Count": count}
            for name, count in sorted(unique_class_counts.items(), key=lambda x: (-x[1], x[0]))
        ]
    save_analysis(
        tot_p, tot_cap, tot_mask, tot_gls, tot_hd, tot_none,
        register_rows, media_key, video_path=play_path,
        class_counts=unique_class_counts, person_total=tot_p,
        accessory_model_active=accessory_values_available,
        zero_shot=zero_shot_active,
    )
    render_saved_analysis()

elif not should_process and st.session_state.analysis is None:
    status_box.info("Upload a video or photo, or click **Load Sample**, to begin analysis.")

# ── TRACKING DIAGNOSTICS ─────────────────────────────────────────────────────
_dbg = st.session_state.get("track_debug")
if _dbg:
    with st.expander("🛰️ TRACKING DIAGNOSTICS", expanded=False):
        st.code(
            f"Raw track IDs created:     {_dbg['raw_track_ids_created']}\n"
            f"Confirmed track IDs:       {_dbg['confirmed_track_ids']}\n"
            f"Final unique person count: {_dbg['final_unique_person_count']}\n"
            f"\n"
            f"Discarded short tracks:    {_dbg['discarded_short_tracks']}\n"
            f"Highest raw track ID:      {_dbg['max_raw_track_id']}  (not used as a count)\n"
            f"Confirm rule:              seen in >= {_dbg['min_track_frames']} frames "
            f"and conf >= {_dbg['min_confidence']:.2f}",
            language="text",
        )
        st.caption(f"Confirmed IDs: {_dbg['confirmed_id_list']}")
        st.caption(f"Discarded IDs: {_dbg['discarded_id_list']}")

_obs = st.session_state.get("observation_report")
if _obs:
    with st.expander("👓 ACCESSORY TEMPORAL VOTING", expanded=False):
        st.code(VOTING.describe(), language="text")
        st.caption("Thresholds are configured in `voting_config.py`.")
        st.code(_obs, language="text")
        st.caption(
            "ratio = hits / frames_seen. A person is classified as wearing an "
            "accessory only when the ratio passes its threshold, so a "
            "single-frame detection can never decide the outcome."
        )

# ── REPORTS & DOWNLOADS ──────────────────────────────────────────────────────
# Only offer exports once THIS session has produced an analysis. Keying off
# file existence alone served a previous run's final_report as if it were the
# current result.
_has_analysis = st.session_state.get("analysis") is not None

if _has_analysis and os.path.exists("final_report.csv"):
    with open("final_report.csv", "r") as f:
        csv_ph.download_button("📥 EXPORT CSV DOSSIER", data=f.read(),
                               file_name="final_report.csv", mime="text/csv", key="dl_csv")

if _has_analysis and os.path.exists("final_report.json"):
    with open("final_report.json", "r") as f:
        json_ph.download_button("📥 EXPORT JSON TELEMETRY", data=f.read(),
                                file_name="final_report.json", mime="application/json", key="dl_json")

# Offer exactly the file the viewport is playing. Keying off a fixed filename
# served a previous run's output whenever the H.264 transcode failed, because
# output_annotated.mp4 is overwritten rather than deleted.
_dl_analysis = st.session_state.get("analysis") or {}
_dl_video = _dl_analysis.get("video_path")
if _has_analysis and input_mode == "video" and _dl_video and os.path.exists(_dl_video):
    # Match the extension to the real file: the MJPG codec fallback writes .avi.
    _dl_ext = Path(_dl_video).suffix.lower() or ".mp4"
    with open(_dl_video, "rb") as vf:
        video_dl_ph.download_button(
            "📹 DOWNLOAD ANNOTATED VIDEO (1.0x NORMAL SPEED)",
            data=vf.read(),
            file_name=f"annotated_video_1x{_dl_ext}",
            mime="video/mp4" if _dl_ext == ".mp4" else "video/x-msvideo",
            key="dl_vid"
        )

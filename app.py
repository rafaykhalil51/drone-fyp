"""
app.py
------
Localhost Web UI and REST API server for Student & Accessory Vision Analytics.
Hosts on http://127.0.0.1:5000
"""

import os
import cv2
import json
import logging
import numpy as np
from pathlib import Path
from flask import Flask, render_template, request, jsonify, send_file, send_from_directory

from detector import PersonDetector
from mock_accessory_detector import MockAccessoryDetector
from real_accessory_detector import RealAccessoryDetector
from association import associate_accessories_to_tracks
from visualization import Visualizer
from state_manager import StateManager
from counter import AccessoryCounter
from exporter import Exporter

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("web_app")

app = Flask(__name__, template_folder="templates", static_folder="static")
app.config["UPLOAD_FOLDER"] = "uploads"
app.config["STATIC_FOLDER"] = "static"
Path("uploads").mkdir(exist_ok=True)
Path("static").mkdir(exist_ok=True)

# Preload models
logger.info("Preloading PersonDetector and RealAccessoryDetector...")
person_detector = PersonDetector("yolov8n.pt", confidence=0.30, iou_threshold=0.5)
real_acc_detector = RealAccessoryDetector("yolov8s-world.pt", confidence=0.20)
mock_acc_detector = MockAccessoryDetector(seed=42, max_per_person=2)
visualizer = Visualizer(vis_cfg={"box_thickness": 2, "text_scale": 0.65, "show_confidence": True})


def analyze_image_file(image_path: str, mode: str = "real"):
    """Run detection and association on a single image and generate results."""
    frame = cv2.imread(image_path)
    if frame is None:
        raise ValueError(f"Could not load image from {image_path}")

    # 1. Detect Persons
    person_dets = person_detector.detect(frame)
    tracks = []
    for i, p in enumerate(person_dets):
        tracks.append({
            "track_id": i + 1,
            "xyxy": p["xyxy"],
            "confidence": p["confidence"],
            "accessories": []
        })

    # 2. Detect Accessories
    if mode == "real":
        raw_accs = real_acc_detector.detect([t["xyxy"] for t in tracks], frame=frame)
    else:
        raw_accs = mock_acc_detector.detect([t["xyxy"] for t in tracks])

    # 3. Associate accessories with upper 45% head region
    acc_map = associate_accessories_to_tracks(tracks, raw_accs, head_fraction=0.45)
    for t in tracks:
        t["accessories"] = acc_map.get(t["track_id"], [])

    # 4. Generate annotated output image
    annotated = visualizer.draw(frame, tracks)
    out_img_name = "annotated_latest.jpg"
    out_img_path = os.path.join("static", out_img_name)
    cv2.imwrite(out_img_path, annotated)

    # 5. Build summary metrics
    track_summaries = []
    totals = {"cap": 0, "mask": 0, "glasses": 0, "headphones": 0}
    none_count = 0

    for t in tracks:
        acc_names = {a["class_name"] for a in t["accessories"]}
        has_cap = "cap" in acc_names
        has_mask = "mask" in acc_names
        has_glasses = "glasses" in acc_names
        has_headphones = "headphones" in acc_names

        if has_cap: totals["cap"] += 1
        if has_mask: totals["mask"] += 1
        if has_glasses: totals["glasses"] += 1
        if has_headphones: totals["headphones"] += 1
        if not (has_cap or has_mask or has_glasses or has_headphones):
            none_count += 1

        track_summaries.append({
            "track_id": t["track_id"],
            "first_seen": 0,
            "last_seen": 0,
            "frames_observed": 1,
            "confidence": t["confidence"],
            "xyxy": t["xyxy"],
            "cap": has_cap,
            "mask": has_mask,
            "glasses": has_glasses,
            "headphones": has_headphones,
            "accessories_list": list(acc_names)
        })

    aggregate = {
        "total_unique_persons": len(tracks),
        "total_cap": totals["cap"],
        "total_mask": totals["mask"],
        "total_glasses": totals["glasses"],
        "total_headphones": totals["headphones"],
        "total_none": none_count
    }

    # Save to final_report.json and final_report.csv
    with open("final_report.json", "w") as f:
        json.dump({"aggregate": aggregate, "tracks": track_summaries}, f, indent=2)

    import csv
    with open("final_report.csv", "w", newline="") as f:
        fields = ["track_id", "first_seen", "last_seen", "frames_observed", "cap", "mask", "glasses", "headphones"]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in track_summaries:
            w.writerow({k: r[k] for k in fields})

    return {
        "status": "success",
        "mode": mode,
        "image_url": f"/static/{out_img_name}",
        "aggregate": aggregate,
        "tracks": track_summaries
    }


# ── ROUTES ─────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/analyze/sample")
def analyze_sample():
    mode = request.args.get("mode", "real")
    sample_path = "uploads/classroom_sample.jpg"
    if not os.path.exists(sample_path):
        return jsonify({"error": "Sample image not found"}), 404
    res = analyze_image_file(sample_path, mode=mode)
    return jsonify(res)


@app.route("/api/analyze/upload", methods=["POST"])
def analyze_upload():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    mode = request.form.get("mode", "real")
    save_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
    file.save(save_path)

    # Check if image or video
    ext = Path(save_path).suffix.lower()
    if ext in [".jpg", ".jpeg", ".png", ".bmp", ".webp"]:
        res = analyze_image_file(save_path, mode=mode)
        return jsonify(res)
    else:
        # Video file processing
        return jsonify({"error": "Video processing via Web UI can be launched or run directly using python main.py."}), 400


@app.route("/api/report/csv")
def download_csv():
    if os.path.exists("final_report.csv"):
        return send_file("final_report.csv", as_attachment=True, download_name="final_report.csv")
    return jsonify({"error": "No CSV report available yet"}), 404


@app.route("/api/report/json")
def download_json():
    if os.path.exists("final_report.json"):
        return send_file("final_report.json", as_attachment=True, download_name="final_report.json")
    return jsonify({"error": "No JSON report available yet"}), 404


if __name__ == "__main__":
    print("\n=======================================================")
    print("  Student & Accessory Vision Analytics Web UI Online!")
    print("  Open in Browser: http://127.0.0.1:5000")
    print("=======================================================\n")
    app.run(host="0.0.0.0", port=5000, debug=False)

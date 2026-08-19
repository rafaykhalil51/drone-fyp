"""
main.py
-------
Entry point for the modular video-analytics pipeline.

Per-frame pipeline sequence:
  1. VideoSource  -> Read next frame
  2. Detector/Tracker -> PersonTracker (YOLOv8 + BotSORT) to track persons with persistent IDs
  3. Accessory Detector -> Mock/Real detector to find cap, mask, glasses, headphones
  4. Association  -> associate_accessories_to_tracks (upper 40% head/shoulder region test)
  5. StateManager -> update_state(track_id, accessories, frame_idx)
  6. Temporal Voting -> apply_temporal_voting(track_id, window, threshold)
  7. Visualizer   -> Draw bounding boxes, IDs, accessories, and Live Totals HUD overlay
  8. VideoWriter  -> Write annotated frame to output MP4

At video end:
  9. AccessoryCounter -> Compute total unique persons and confirmed per-accessory counts
 10. Exporter        -> Save final_report.csv, final_report.json, tracks.csv, summary.json

Usage:
    python main.py [--config config.yaml]
"""

import argparse
import logging
import sys
import time
import cv2
import yaml

from video_source            import VideoSource
from tracker                 import PersonTracker
from detector                import PersonDetector
from state_manager           import StateManager
from counter                 import LineCounter, AccessoryCounter
from visualization           import Visualizer
from exporter                import Exporter
from association             import associate_accessories_to_tracks


def setup_logging(level: str):
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
    )


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_writer(path: str, fps: float, w: int, h: int) -> cv2.VideoWriter:
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    if not writer.isOpened():
        raise RuntimeError(f"Cannot open VideoWriter for '{path}'")
    return writer


def main(config_path: str = "config.yaml"):
    cfg = load_config(config_path)
    setup_logging(cfg.get("logging", {}).get("level", "INFO"))
    logger = logging.getLogger("main")

    vid_cfg = cfg["video"]
    det_cfg = cfg["detection"]
    trk_cfg = cfg["tracking"]
    cnt_cfg = cfg["counting"]
    vis_cfg = cfg["visualization"]
    exp_cfg = cfg["export"]
    acc_cfg = cfg.get("accessories", {})

    vote_window = acc_cfg.get("vote_window", 30)
    vote_threshold = acc_cfg.get("vote_threshold", 0.60)
    head_fraction = acc_cfg.get("head_fraction", 0.40)

    logger.info("Initialising pipeline components...")
    source = VideoSource(vid_cfg["source"])
    fps = float(vid_cfg["fps_override"] or source.fps)

    # 1. Accessory Detector (Mock by default, or Real if configured)
    accessory_detector = None
    if acc_cfg.get("use_mock_accessories", True):
        from mock_accessory_detector import MockAccessoryDetector
        accessory_detector = MockAccessoryDetector(
            seed=acc_cfg.get("mock_seed", 42),
            max_per_person=acc_cfg.get("mock_max_per_person", 2),
        )
        logger.info("Mock Accessory Detector ENABLED (seed=%d)", acc_cfg.get("mock_seed", 42))
    elif acc_cfg.get("use_real_accessories", False):
        from real_accessory_detector import RealAccessoryDetector
        accessory_detector = RealAccessoryDetector(
            model_path=acc_cfg.get("real_model", "yolov8s-world.pt"),
            confidence=acc_cfg.get("confidence", 0.25)
        )
        logger.info("Real Accessory Detector ENABLED (YOLO-World)")

    # 2. Person Tracker (YOLOv8 + BotSORT)
    tracker = PersonTracker(
        model_path=det_cfg["model_path"],
        confidence=det_cfg["confidence"],
        iou_threshold=det_cfg["iou_threshold"],
        tracker_config=trk_cfg["tracker_config"],
        persist=trk_cfg["persist"],
        person_class_id=det_cfg["person_class_id"],
    )

    # 3. State & Analytics Managers
    state_mgr = StateManager()
    line_counter = LineCounter(cnt_cfg["line"], state_mgr) if cnt_cfg.get("enabled") else None
    acc_counter = AccessoryCounter()
    visualizer = Visualizer(vis_cfg, cnt_cfg if cnt_cfg.get("enabled") else None)
    exporter = Exporter(
        csv_path=exp_cfg["csv_path"],
        summary_path=exp_cfg["summary_path"],
        final_report_csv=exp_cfg.get("final_report_csv"),
        final_report_json=exp_cfg.get("final_report_json"),
    )
    writer = build_writer(vid_cfg["output"], fps, source.width, source.height)

    logger.info("Starting Processing: '%s' -> '%s'", vid_cfg["source"], vid_cfg["output"])
    t0 = time.time()
    frame_idx = 0

    try:
        with source:
            for frame in source:
                # Step 1: Detect & Track Persons with persistent IDs
                tracks = tracker.track(frame)

                # Step 2: Detect Accessories (Mock or Real)
                if accessory_detector is not None and tracks:
                    person_boxes = [t["xyxy"] for t in tracks]
                    if hasattr(accessory_detector, "detect"):
                        # Support both mock and real detector signatures
                        try:
                            all_accs = accessory_detector.detect(person_boxes, frame=frame)
                        except TypeError:
                            all_accs = accessory_detector.detect(person_boxes)
                    else:
                        all_accs = []

                    # Step 3: Spatial Association (upper 40% head/shoulder region)
                    acc_map = associate_accessories_to_tracks(
                        tracks, all_accs, head_fraction=head_fraction
                    )
                    for t in tracks:
                        t["accessories"] = acc_map.get(t["track_id"], [])
                else:
                    for t in tracks:
                        t.setdefault("accessories", [])

                # Step 4: Update State Manager tracking history
                state_mgr.update(frame_idx, tracks)

                # Step 5: Update State & Apply Temporal Voting per track
                for t in tracks:
                    tid = t["track_id"]
                    if tid < 0:
                        continue
                    state_mgr.update_state(tid, t.get("accessories", []), frame_idx)
                    state_mgr.apply_temporal_voting(
                        tid,
                        window=vote_window,
                        threshold=vote_threshold,
                    )

                # Step 6: Update Line Counter (if enabled)
                if line_counter:
                    line_counter.update(tracks)

                # Step 7: Log per-frame rows to exporter
                exporter.log_frame(frame_idx, tracks)

                # Step 8: Compute Live Totals for HUD Overlay
                all_st = state_mgr.all_states().values()
                live_totals = {
                    "persons": state_mgr.total_unique,
                    "caps": sum(1 for s in all_st if s.final_cap),
                    "masks": sum(1 for s in all_st if s.final_mask),
                    "glasses": sum(1 for s in all_st if s.final_glasses),
                    "headphones": sum(1 for s in all_st if s.final_headphones),
                }

                # Step 9: Visualize with bounding boxes, labels, and Live Totals HUD
                annotated = visualizer.draw(frame, tracks, counter=line_counter, live_totals=live_totals)
                writer.write(annotated)

                frame_idx += 1
                if frame_idx % 50 == 0:
                    logger.info("Frame %d | Active Tracks: %d | Total Seen: %d", frame_idx, len(tracks), state_mgr.total_unique)

    finally:
        writer.release()
        exporter.save(state_mgr, line_counter)

    elapsed = time.time() - t0
    logger.info(
        "Processing complete. %d frames processed in %.1fs (%.1f fps avg). Unique persons: %d",
        frame_idx, elapsed, frame_idx / elapsed if elapsed > 0 else 0, state_mgr.total_unique
    )

    if line_counter:
        logger.info("Line Crossing: IN=%d  OUT=%d  TOTAL=%d", line_counter.count_in, line_counter.count_out, line_counter.total)

    # Final Temporal Voting Summary
    vote_summary = state_mgr.accessory_summary()
    if vote_summary:
        logger.info("Accessory Summary (Confirmed via %d-frame window @ %.0f%%):", vote_window, vote_threshold * 100)
        for tid, accs in sorted(vote_summary.items()):
            logger.info("  Student #%-3d : %s", tid, ", ".join(accs))
    else:
        logger.info("Accessory Summary: No accessories confirmed above %.0f%% threshold across temporal window.", vote_threshold * 100)

    # Step 10: Call counter.py and exporter.py to produce final reports
    acc_counter.compute(state_mgr)
    exporter.save_final_report(
        state_manager=state_mgr,
        line_counter=line_counter,
        acc_counter=acc_counter,
    )
    logger.info(
        "Final Aggregate Totals: Persons=%d | Caps=%d | Masks=%d | Glasses=%d | Headphones=%d",
        acc_counter.total_unique_persons,
        acc_counter.totals["cap"],
        acc_counter.totals["mask"],
        acc_counter.totals["glasses"],
        acc_counter.totals["headphones"],
    )
    logger.info("Generated Outputs: '%s', '%s', '%s', '%s'", exp_cfg["csv_path"], exp_cfg["summary_path"], exp_cfg.get("final_report_csv"), exp_cfg.get("final_report_json"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Student & Accessory Video Analytics Pipeline")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    args = parser.parse_args()
    main(args.config)

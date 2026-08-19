import argparse, logging, sys, time
import cv2, yaml

from video_source            import VideoSource
from tracker                 import PersonTracker
from detector                import PersonDetector
from state_manager           import StateManager
from counter                 import LineCounter
from visualization           import Visualizer
from exporter                import Exporter

def setup_logging(level):
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S", stream=sys.stdout)

def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)

def build_writer(path, fps, w, h):
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    if not writer.isOpened():
        raise RuntimeError(f"Cannot open VideoWriter for '{path}'")
    return writer

def main(config_path="config.yaml"):
    cfg     = load_config(config_path)
    setup_logging(cfg.get("logging", {}).get("level", "INFO"))
    logger  = logging.getLogger("main")

    vid_cfg = cfg["video"]
    det_cfg = cfg["detection"]
    trk_cfg = cfg["tracking"]
    cnt_cfg = cfg["counting"]
    vis_cfg = cfg["visualization"]
    exp_cfg = cfg["export"]
    acc_cfg = cfg.get("accessories", {})

    logger.info("Initialising pipeline ...")
    source  = VideoSource(vid_cfg["source"])
    fps     = float(vid_cfg["fps_override"] or source.fps)

    # Optionally wire mock accessory detector
    accessory_detector = None
    if acc_cfg.get("use_mock_accessories", False):
        from mock_accessory_detector import MockAccessoryDetector
        accessory_detector = MockAccessoryDetector(
            seed=acc_cfg.get("mock_seed", 42),
            max_per_person=acc_cfg.get("mock_max_per_person", 2),
        )
        logger.info("Mock accessory detector ENABLED")

    tracker = PersonTracker(
        model_path=det_cfg["model_path"],
        confidence=det_cfg["confidence"],
        iou_threshold=det_cfg["iou_threshold"],
        tracker_config=trk_cfg["tracker_config"],
        persist=trk_cfg["persist"],
        person_class_id=det_cfg["person_class_id"],
    )

    # PersonDetector is used for standalone detect() + accessory attachment
    # (tracker.track() handles detection+tracking; accessories are attached
    #  post-hoc from the tracker results by a thin wrapper below)
    state_mgr  = StateManager()
    counter    = LineCounter(cnt_cfg["line"], state_mgr) if cnt_cfg["enabled"] else None
    visualizer = Visualizer(vis_cfg, cnt_cfg if cnt_cfg["enabled"] else None)
    exporter   = Exporter(exp_cfg["csv_path"], exp_cfg["summary_path"])
    writer     = build_writer(vid_cfg["output"], fps, source.width, source.height)

    logger.info("Processing: '%s' -> '%s'", vid_cfg["source"], vid_cfg["output"])
    t0 = time.time(); frame_idx = 0

    try:
        with source:
            for frame in source:
                tracks = tracker.track(frame)

                # Attach accessory detections to each track result
                if accessory_detector is not None and tracks:
                    person_boxes = [t["xyxy"] for t in tracks]
                    all_accs = accessory_detector.detect(person_boxes)
                    # Map accessories back onto their owning track by centre-point
                    for t in tracks:
                        t.setdefault("accessories", [])
                    for acc in all_accs:
                        ax1, ay1, ax2, ay2 = acc["xyxy"]
                        acx = (ax1 + ax2) / 2
                        acy = (ay1 + ay2) / 2
                        for t in tracks:
                            dx1, dy1, dx2, dy2 = t["xyxy"]
                            if dx1 <= acx <= dx2 and dy1 <= acy <= dy2:
                                t["accessories"].append(acc)
                                break
                else:
                    for t in tracks:
                        t.setdefault("accessories", [])

                state_mgr.update(frame_idx, tracks)
                if counter: counter.update(tracks)
                exporter.log_frame(frame_idx, tracks)
                annotated = visualizer.draw(frame, tracks, counter)
                writer.write(annotated)
                frame_idx += 1
                if frame_idx % 50 == 0:
                    logger.info("Frame %d  active_tracks=%d", frame_idx, len(tracks))
    finally:
        writer.release()
        exporter.save(state_mgr, counter)

    elapsed = time.time() - t0
    logger.info("Done. %d frames in %.1fs  unique_persons=%d  output='%s'",
                frame_idx, elapsed, state_mgr.total_unique, vid_cfg["output"])
    if counter:
        logger.info("Line count  IN=%d  OUT=%d  TOTAL=%d",
                    counter.count_in, counter.count_out, counter.total)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config.yaml")
    args = p.parse_args()
    main(args.config)

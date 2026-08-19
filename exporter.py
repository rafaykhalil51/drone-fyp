import csv, json, time
from pathlib import Path

class Exporter:
    def __init__(self, csv_path, summary_path):
        self.csv_path     = Path(csv_path)
        self.summary_path = Path(summary_path)
        self._rows = []
        self._start = time.time()

    def log_frame(self, frame_idx, tracks):
        for t in tracks:
            self._rows.append({
                "frame": frame_idx,
                "track_id": t["track_id"],
                "x1": t["xyxy"][0], "y1": t["xyxy"][1],
                "x2": t["xyxy"][2], "y2": t["xyxy"][3],
                "confidence": round(t["confidence"], 4),
            })

    def save(self, state_manager=None, counter=None):
        if self._rows:
            with self.csv_path.open("w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=["frame","track_id","x1","y1","x2","y2","confidence"])
                writer.writeheader(); writer.writerows(self._rows)
            print(f"Saved {self.csv_path}  ({len(self._rows)} rows)")
        elapsed = time.time() - self._start
        summary = {
            "duration_seconds": round(elapsed, 2),
            "total_frames_logged": len({r["frame"] for r in self._rows}),
            "unique_track_ids": len({r["track_id"] for r in self._rows if r["track_id"] >= 0}),
        }
        if state_manager: summary["state_unique_persons"] = state_manager.total_unique
        if counter:
            summary["count_in"]  = counter.count_in
            summary["count_out"] = counter.count_out
            summary["count_total"] = counter.total
        with self.summary_path.open("w") as f:
            json.dump(summary, f, indent=2)
        print(f"Saved {self.summary_path}")

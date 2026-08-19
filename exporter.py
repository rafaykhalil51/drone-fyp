import csv, json, logging, time
from pathlib import Path
from state_manager import ACCESSORY_KEYS

logger = logging.getLogger(__name__)


class Exporter:
    """
    Two-tier export:
      1. tracks.csv / summary.json   – per-frame track log  (existing)
      2. final_report.csv / .json    – per-track summary with accessory
                                       final flags and aggregate totals (new)
    """

    def __init__(self, csv_path, summary_path,
                 final_report_csv=None, final_report_json=None):
        self.csv_path          = Path(csv_path)
        self.summary_path      = Path(summary_path)
        self.final_report_csv  = Path(final_report_csv)  if final_report_csv  else None
        self.final_report_json = Path(final_report_json) if final_report_json else None
        self._rows  = []          # per-frame track rows
        self._start = time.time()

    # ── existing per-frame logging ─────────────────────────────────────────
    def log_frame(self, frame_idx, tracks):
        """Append one row per track per frame to the in-memory log."""
        for t in tracks:
            self._rows.append({
                "frame":      frame_idx,
                "track_id":   t["track_id"],
                "x1": t["xyxy"][0], "y1": t["xyxy"][1],
                "x2": t["xyxy"][2], "y2": t["xyxy"][3],
                "confidence": round(t["confidence"], 4),
            })

    # ── existing end-of-run save ───────────────────────────────────────────
    def save(self, state_manager=None, counter=None):
        """Write tracks.csv and summary.json (unchanged behaviour)."""
        if self._rows:
            fields = ["frame", "track_id", "x1", "y1", "x2", "y2", "confidence"]
            with self.csv_path.open("w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=fields)
                w.writeheader()
                w.writerows(self._rows)
            logger.info("Saved %s  (%d rows)", self.csv_path, len(self._rows))

        elapsed = time.time() - self._start
        summary = {
            "duration_seconds":     round(elapsed, 2),
            "total_frames_logged":  len({r["frame"] for r in self._rows}),
            "unique_track_ids":     len({r["track_id"] for r in self._rows
                                         if r["track_id"] >= 0}),
        }
        if state_manager:
            summary["state_unique_persons"] = state_manager.total_unique
        if counter:
            summary["count_in"]    = counter.count_in
            summary["count_out"]   = counter.count_out
            summary["count_total"] = counter.total

        with self.summary_path.open("w") as f:
            json.dump(summary, f, indent=2)
        logger.info("Saved %s", self.summary_path)

    # ── new final-report writers ───────────────────────────────────────────
    def save_final_report(self, state_manager, line_counter=None,
                          acc_counter=None):
        """
        Write final_report.csv and final_report.json.

        final_report.csv columns
        ------------------------
        track_id, first_seen, last_seen, frames_observed,
        mask, cap, glasses, headphones

        final_report.json structure
        ---------------------------
        {
          "aggregate": {
            "total_unique_persons": int,
            "total_mask": int,
            "total_cap": int,
            "total_glasses": int,
            "total_headphones": int,
            "line_count_in": int,       # if line_counter provided
            "line_count_out": int,
            "line_count_total": int
          },
          "tracks": [
            { "track_id": int, "first_seen": int, "last_seen": int,
              "frames_observed": int,
              "mask": bool, "cap": bool, "glasses": bool, "headphones": bool },
            ...
          ]
        }
        """
        states = state_manager.all_states()

        # ── build per-track rows ────────────────────────────────────────────
        track_rows = []
        for tid in sorted(states):
            s = states[tid]
            flags = s.final_flags()
            row = {
                "track_id":       tid,
                "first_seen":     s.first_frame,
                "last_seen":      s.last_frame,
                "frames_observed": s.frame_count,
            }
            for key in ACCESSORY_KEYS:
                row[key] = flags.get(key, False)
            track_rows.append(row)

        # ── aggregate totals ────────────────────────────────────────────────
        aggregate = {
            "total_unique_persons": state_manager.total_unique,
        }
        # Per-accessory counts from AccessoryCounter (if provided) or
        # recompute inline as a fallback so this method is self-contained.
        if acc_counter is not None:
            aggregate.update({f"total_{k}": acc_counter.totals[k]
                               for k in ACCESSORY_KEYS})
        else:
            for key in ACCESSORY_KEYS:
                aggregate[f"total_{key}"] = sum(
                    1 for r in track_rows if r[key]
                )
        if line_counter is not None:
            aggregate["line_count_in"]    = line_counter.count_in
            aggregate["line_count_out"]   = line_counter.count_out
            aggregate["line_count_total"] = line_counter.total

        # ── write CSV ───────────────────────────────────────────────────────
        if self.final_report_csv:
            fields = (["track_id", "first_seen", "last_seen", "frames_observed"]
                      + list(ACCESSORY_KEYS))
            with self.final_report_csv.open("w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=fields)
                w.writeheader()
                w.writerows(track_rows)
            logger.info(
                "Saved %s  (%d tracks)",
                self.final_report_csv, len(track_rows),
            )

        # ── write JSON ──────────────────────────────────────────────────────
        if self.final_report_json:
            # Serialise booleans as Python bool (json.dump handles them)
            json_tracks = [
                {k: (bool(v) if isinstance(v, (bool,)) else v)
                 for k, v in r.items()}
                for r in track_rows
            ]
            payload = {"aggregate": aggregate, "tracks": json_tracks}
            with self.final_report_json.open("w") as f:
                json.dump(payload, f, indent=2)
            logger.info("Saved %s", self.final_report_json)

        return aggregate

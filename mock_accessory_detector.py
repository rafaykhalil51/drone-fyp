import random
import logging

logger = logging.getLogger(__name__)

ACCESSORIES = ["cap", "mask", "glasses", "headphones"]

# Colour per accessory class (BGR) used by visualization
ACCESSORY_COLORS = {
    "cap":        (0,   215, 255),   # gold
    "mask":       (255, 100,   0),   # blue-orange
    "glasses":    (180,   0, 255),   # purple
    "headphones": (0,   255, 128),   # spring-green
}


class MockAccessoryDetector:
    """
    Simulates accessory detections for each person bounding box.

    For every person bbox it randomly assigns 0-2 accessories drawn
    from {cap, mask, glasses, headphones}, using a fixed random seed
    so results are reproducible across runs.

    Return format (list of dicts) mirrors what a real detector would
    produce:
        {
            "class_name":  str,          # e.g. "cap"
            "confidence":  float,        # 0.50 - 0.98
            "xyxy":        [x1,y1,x2,y2] # sub-bbox inside person bbox
        }
    """

    def __init__(self, seed: int = 42, max_per_person: int = 2):
        self._rng = random.Random(seed)
        self.max_per_person = max_per_person
        logger.info(
            "MockAccessoryDetector ready  seed=%d  max_per_person=%d",
            seed, max_per_person,
        )

    # ── public ────────────────────────────────────────────────────────────
    def detect(self, person_boxes: list[list[int]]) -> list[dict]:
        """
        Parameters
        ----------
        person_boxes : list of [x1, y1, x2, y2] person bounding boxes

        Returns
        -------
        list of accessory detection dicts, one entry per detected accessory
        (multiple entries possible per person).
        """
        all_accessories: list[dict] = []

        for xyxy in person_boxes:
            x1, y1, x2, y2 = xyxy
            pw = x2 - x1          # person width
            ph = y2 - y1          # person height

            # Decide how many accessories this person gets (0, 1 or 2)
            n = self._rng.randint(0, self.max_per_person)
            chosen = self._rng.sample(ACCESSORIES, k=min(n, len(ACCESSORIES)))

            for acc in chosen:
                confidence = round(self._rng.uniform(0.50, 0.98), 2)
                # Place accessory bbox in the upper-quarter of the person box
                # (head/upper-body region) with a small random offset
                ax1 = x1 + int(pw * self._rng.uniform(0.05, 0.20))
                ax2 = x2 - int(pw * self._rng.uniform(0.05, 0.20))
                ay1 = y1 + int(ph * self._rng.uniform(0.00, 0.10))
                ay2 = y1 + int(ph * self._rng.uniform(0.20, 0.35))

                # Clamp so bbox stays inside person box
                ax1 = max(x1, ax1); ax2 = min(x2, ax2)
                ay1 = max(y1, ay1); ay2 = min(y2, ay2)

                if ax2 <= ax1 or ay2 <= ay1:
                    continue   # skip degenerate boxes

                all_accessories.append({
                    "class_name": acc,
                    "confidence": confidence,
                    "xyxy": [ax1, ay1, ax2, ay2],
                })
                logger.debug(
                    "Mock accessory: %s  conf=%.2f  bbox=%s",
                    acc, confidence, [ax1, ay1, ax2, ay2],
                )

        return all_accessories

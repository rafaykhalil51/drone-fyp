import logging
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def box_center(xyxy: list) -> tuple:
    x1, y1, x2, y2 = xyxy
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def box_bottom_center(xyxy: list) -> tuple:
    x1, y1, x2, y2 = xyxy
    return (x1 + x2) / 2.0, float(y2)


def iou(box_a: list, box_b: list) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / union if union > 0 else 0.0


def side_of_line(point: tuple, line_start: tuple, line_end: tuple) -> int:
    px, py = point
    x1, y1 = line_start
    x2, y2 = line_end
    cross = (x2 - x1) * (py - y1) - (y2 - y1) * (px - x1)
    return 1 if cross >= 0 else -1


# ---------------------------------------------------------------------------
# Accessory <-> person track association
# ---------------------------------------------------------------------------

def upper_region(xyxy: list, fraction: float = 0.4) -> tuple:
    """
    Return the bounding box of the upper *fraction* of a person box.

    The 'head/shoulder' region is the top portion of the person's
    bounding box.  fraction=0.4 means the top 40% of the height.

    Parameters
    ----------
    xyxy     : [x1, y1, x2, y2]  person bounding box
    fraction : float in (0, 1]   fraction of height to keep from the top

    Returns
    -------
    (x1, y1, x2, y_cut)  where y_cut = y1 + fraction * height
    """
    x1, y1, x2, y2 = xyxy
    y_cut = y1 + (y2 - y1) * fraction
    return x1, y1, x2, y_cut


def point_in_box(px: float, py: float, box: tuple) -> bool:
    """Return True if (px, py) lies strictly inside *box* (x1,y1,x2,y2)."""
    x1, y1, x2, y2 = box
    return x1 <= px <= x2 and y1 <= py <= y2


def associate_accessories_to_tracks(
    tracks: list[dict[str, Any]],
    accessories: list[dict[str, Any]],
    head_fraction: float = 0.4,
) -> dict[int, list[dict[str, Any]]]:
    """
    For each accessory detection decide which (if any) person track owns it,
    by testing whether the accessory's *centre point* falls inside the
    upper *head_fraction* of a track's bounding box.

    When multiple person boxes all contain the accessory centre (overlap),
    the accessory is assigned to the track whose centre is *closest* to the
    accessory centre (nearest-neighbour tie-break).

    Parameters
    ----------
    tracks       : list of track dicts, each with keys 'track_id' and 'xyxy'
    accessories  : list of accessory dicts, each with key 'xyxy'
    head_fraction: float in (0, 1] - upper fraction of person box to test
                   (default 0.4 = top 40 % / head+shoulder region)

    Returns
    -------
    dict mapping track_id (int) -> list of accessory dicts that belong to it.
    Track IDs with no accessories are NOT included in the dict.
    """
    result: dict[int, list] = {}

    if not tracks or not accessories:
        logger.debug("associate_accessories_to_tracks: nothing to associate")
        return result

    # Pre-compute head regions and track centres once per call
    track_meta = []
    for t in tracks:
        tid  = t["track_id"]
        xyxy = t["xyxy"]
        head = upper_region(xyxy, head_fraction)
        cx, cy = box_center(xyxy)
        track_meta.append({"track_id": tid, "head_box": head,
                            "cx": cx, "cy": cy})

    for acc in accessories:
        ax1, ay1, ax2, ay2 = acc["xyxy"]
        acx = (ax1 + ax2) / 2.0   # accessory centre x
        acy = (ay1 + ay2) / 2.0   # accessory centre y

        # Collect all tracks whose head-region contains the accessory centre
        candidates = [
            m for m in track_meta
            if point_in_box(acx, acy, m["head_box"])
        ]

        if not candidates:
            logger.debug(
                "Accessory '%s' centre (%.1f, %.1f) matched no person head region",
                acc.get("class_name", "?"), acx, acy,
            )
            continue

        # Nearest-track tie-break
        best = min(
            candidates,
            key=lambda m: (acx - m["cx"]) ** 2 + (acy - m["cy"]) ** 2,
        )
        tid = best["track_id"]
        result.setdefault(tid, []).append(acc)
        logger.debug(
            "Accessory '%s' conf=%.2f -> track_id=%d",
            acc.get("class_name", "?"), acc.get("confidence", 0.0), tid,
        )

    logger.debug(
        "associate_accessories_to_tracks: %d accessories -> %d tracks assigned",
        len(accessories), len(result),
    )
    return result

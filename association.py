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


# Expected body region per accessory class, as fractions of the person box.
#   y_min / y_max : vertical band measured from the top of the person box
#   x_inset       : fraction trimmed from each side, since a head is narrower
#                   than the shoulders/torso the person box spans
#
# A cap sits highest, glasses and headphones sit at eye/ear level, and a mask
# reaches lowest because it covers nose and mouth. Bands overlap and are kept
# generous so seated, partially visible, and close-up subjects still match.
ACCESSORY_REGIONS: dict[str, dict[str, float]] = {
    "cap":        {"y_min": 0.00, "y_max": 0.24, "x_inset": 0.12},
    "glasses":    {"y_min": 0.02, "y_max": 0.30, "x_inset": 0.12},
    "headphones": {"y_min": 0.00, "y_max": 0.48, "x_inset": 0.04},
    "mask":       {"y_min": 0.04, "y_max": 0.38, "x_inset": 0.12},
}

# An accessory wider than this fraction of the person box is implausible
# (usually a detection belonging to a much closer/larger person).
MAX_ACCESSORY_WIDTH_RATIO = 0.95

# An accessory taller than this fraction of the person box is implausible.
MAX_ACCESSORY_HEIGHT_RATIO = 0.55


def accessory_region(
    xyxy: list,
    class_name: str,
    fallback_fraction: float = 0.4,
) -> tuple:
    """
    Return the plausible region box for *class_name* within a person box.

    Falls back to the generic upper-region test for unknown accessory classes,
    preserving the previous behaviour.
    """
    x1, y1, x2, y2 = xyxy
    width = x2 - x1
    height = y2 - y1

    spec = ACCESSORY_REGIONS.get((class_name or "").strip().lower())
    if spec is None:
        return upper_region(xyxy, fallback_fraction)

    inset = width * spec["x_inset"]
    return (
        x1 + inset,
        y1 + height * spec["y_min"],
        x2 - inset,
        y1 + height * spec["y_max"],
    )


def is_plausible_size(acc_xyxy: list, person_xyxy: list) -> bool:
    """Reject accessory boxes too large to belong to this person."""
    ax1, ay1, ax2, ay2 = acc_xyxy
    px1, py1, px2, py2 = person_xyxy

    person_w = px2 - px1
    person_h = py2 - py1
    if person_w <= 0 or person_h <= 0:
        return False

    acc_w = ax2 - ax1
    acc_h = ay2 - ay1
    if acc_w <= 0 or acc_h <= 0:
        return False

    return (
        acc_w <= person_w * MAX_ACCESSORY_WIDTH_RATIO
        and acc_h <= person_h * MAX_ACCESSORY_HEIGHT_RATIO
    )


def associate_accessories_to_tracks(
    tracks: list[dict[str, Any]],
    accessories: list[dict[str, Any]],
    head_fraction: float = 0.4,
) -> dict[int, list[dict[str, Any]]]:
    """
    For each accessory detection decide which (if any) person track owns it.

    An accessory is only accepted when it is spatially plausible: its centre
    point must fall inside the body region expected for that accessory class
    (see ACCESSORY_REGIONS) and its box must not be too large for the person.

    When several person boxes qualify (overlapping crowds), the accessory goes
    to the person whose head-region centre is closest to the accessory centre.
    Each accessory detection is assigned to at most one person.

    Parameters
    ----------
    tracks       : list of track dicts, each with keys 'track_id' and 'xyxy'
    accessories  : list of accessory dicts with keys 'xyxy' and 'class_name'
    head_fraction: fallback upper-region fraction for unrecognised classes

    Returns
    -------
    dict mapping track_id (int) -> list of accessory dicts that belong to it.
    Track IDs with no accessories are NOT included in the dict.
    """
    result: dict[int, list] = {}

    if not tracks or not accessories:
        logger.debug("associate_accessories_to_tracks: nothing to associate")
        return result

    for acc in accessories:
        acc_box = acc["xyxy"]
        cls_name = acc.get("class_name", "")
        acx, acy = box_center(acc_box)

        candidates = []
        for t in tracks:
            person_box = t["xyxy"]

            # 4. Plausibility: correct body region for this accessory class...
            region = accessory_region(person_box, cls_name, head_fraction)
            if not point_in_box(acx, acy, region):
                continue

            # ...and a size consistent with this person.
            if not is_plausible_size(acc_box, person_box):
                continue

            rcx, rcy = box_center(list(region))
            candidates.append({
                "track_id": t["track_id"],
                "dist_sq": (acx - rcx) ** 2 + (acy - rcy) ** 2,
            })

        if not candidates:
            logger.debug(
                "Accessory '%s' centre (%.1f, %.1f) matched no plausible person region",
                cls_name or "?", acx, acy,
            )
            continue

        # 3. Closest appropriate person wins; assigned to exactly one track.
        best = min(candidates, key=lambda m: m["dist_sq"])
        result.setdefault(best["track_id"], []).append(acc)
        logger.debug(
            "Accessory '%s' conf=%.2f -> track_id=%d (%d candidate(s))",
            cls_name or "?", acc.get("confidence", 0.0),
            best["track_id"], len(candidates),
        )

    logger.debug(
        "associate_accessories_to_tracks: %d accessories -> %d tracks assigned",
        len(accessories), len(result),
    )
    return result

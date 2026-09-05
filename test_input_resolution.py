"""
Regression tests for dashboard.py input resolution.

Streamlit keeps an uploaded file in its widget for the whole session, so both
file_uploader calls return a value on every rerun no matter which tab is open.
The old if/elif chain therefore let an already-processed video permanently
shadow a new photo upload and both sample buttons. It also rebuilt the temp
file on every rerun using .read(), which returns 0 bytes after the first call.
"""
import os
import sys

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.getcwd())
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

FAILS = []


def check(name, actual, expected):
    ok = actual == expected
    if not ok:
        FAILS.append(name)
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: got={actual} expected={expected}")


class Upload:
    """Stands in for streamlit's UploadedFile."""

    def __init__(self, name, size, data=b"x"):
        self.name = name
        self.size = size
        self._data = data * size
        self._pos = 0

    def read(self):                       # pointer advances, EOF on 2nd call
        out = self._data[self._pos:]
        self._pos = len(self._data)
        return out

    def getvalue(self):                   # stable across reruns
        return self._data


def resolve(uploaded_video, uploaded_photo, btn_sample_video, btn_sample_photo,
            last_key, sample_video_exists=True, sample_photo_exists=True):
    """Mirror of the candidate-selection logic now in dashboard.py."""
    candidates = []
    if btn_sample_video and sample_video_exists:
        candidates.append(("video", "input.mp4", "sample:video:input.mp4", None))
    if btn_sample_photo and sample_photo_exists:
        candidates.append(
            ("photo", "uploads/classroom_sample.jpg", "sample:photo:classroom_sample.jpg", None)
        )
    if uploaded_photo is not None:
        candidates.append(
            ("photo", None, f"photo:{uploaded_photo.name}:{uploaded_photo.size}", uploaded_photo)
        )
    if uploaded_video is not None:
        candidates.append(
            ("video", None, f"video:{uploaded_video.name}:{uploaded_video.size}", uploaded_video)
        )
    selected = next((c for c in candidates if c[2] != last_key), None)
    return {
        "mode": selected[0] if selected else "video",
        "key": selected[2] if selected else None,
        "should_process": selected is not None,
        "upload": selected[3] if selected else None,
        "path": selected[1] if selected else None,
    }


vid = Upload("clip.mp4", 1000)
pho = Upload("class.jpg", 500)

# ── 1. First video upload processes ──────────────────────────────────────
r = resolve(vid, None, False, False, last_key=None)
check("new video processes", (r["should_process"], r["mode"]), (True, "video"))
check("video key", r["key"], "video:clip.mp4:1000")

# ── 2. Same video, later rerun (slider moved) must NOT reprocess ─────────
r = resolve(vid, None, False, False, last_key="video:clip.mp4:1000")
check("same video does not reprocess", r["should_process"], False)
check("no temp file needed on idle rerun", r["upload"], None)

# ── 3. THE BUG: photo uploaded after a video was already processed ───────
r = resolve(vid, pho, False, False, last_key="video:clip.mp4:1000")
check("photo after video is not shadowed", (r["should_process"], r["mode"]), (True, "photo"))
check("photo key selected", r["key"], "photo:class.jpg:500")

# ── 4. THE BUG: sample video button after a video was processed ──────────
r = resolve(vid, None, True, False, last_key="video:clip.mp4:1000")
check("sample video button works after upload",
      (r["should_process"], r["path"]), (True, "input.mp4"))

# ── 5. THE BUG: sample photo button after a video was processed ──────────
r = resolve(vid, None, False, True, last_key="video:clip.mp4:1000")
check("sample photo button works after upload",
      (r["should_process"], r["path"]), (True, "uploads/classroom_sample.jpg"))

# ── 6. Sample click wins over a stale upload in the same rerun ──────────
r = resolve(vid, pho, True, False, last_key=None)
check("explicit sample click takes priority", r["path"], "input.mp4")

# ── 7. Missing sample asset is skipped, not selected as a dead path ──────
r = resolve(None, None, True, False, last_key=None, sample_video_exists=False)
check("missing sample video not selected", r["should_process"], False)
r = resolve(None, None, False, True, last_key=None, sample_photo_exists=False)
check("missing sample photo not selected", r["should_process"], False)

# ── 8. Nothing uploaded at all ───────────────────────────────────────────
r = resolve(None, None, False, False, last_key=None)
check("no input -> no processing", r["should_process"], False)

# ── 9. Switching between two different videos reprocesses ───────────────
vid2 = Upload("other.mp4", 2000)
r = resolve(vid2, None, False, False, last_key="video:clip.mp4:1000")
check("different video reprocesses", (r["should_process"], r["key"]),
      (True, "video:other.mp4:2000"))

# ── 10. getvalue() survives reruns where read() would return 0 bytes ────
u = Upload("clip.mp4", 1000)
first_read = len(u.read())
second_read = len(u.read())
check("read() is empty on second rerun", (first_read, second_read), (1000, 0))
check("getvalue() stable across reruns",
      (len(u.getvalue()), len(u.getvalue())), (1000, 1000))

print()
if FAILS:
    print(f"{len(FAILS)} FAILED: {FAILS}")
    sys.exit(1)
print("ALL INPUT RESOLUTION CHECKS PASSED")

"""
Temp-file lifecycle regression test.

Replays the dashboard's input-resolution block across many simulated Streamlit
reruns with a file retained in the uploader widget, and asserts that temp files
do not accumulate.

Old behaviour: a NamedTemporaryFile was created on EVERY rerun, written with
.read() (0 bytes after the first call), never closed and never deleted.
"""
import glob
import os
import sys
import tempfile
from pathlib import Path

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.getcwd())
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

FAILS = []
TAG = "audittmp_"


def check(name, actual, expected):
    ok = actual == expected
    if not ok:
        FAILS.append(name)
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: got={actual} expected={expected}")


class Upload:
    """streamlit UploadedFile stand-in: read() drains, getvalue() does not."""

    def __init__(self, name, payload):
        self.name = name
        self.size = len(payload)
        self._data = payload
        self._pos = 0

    def read(self):
        out = self._data[self._pos:]
        self._pos = len(self._data)
        return out

    def getvalue(self):
        return self._data


def count_temps():
    return len(glob.glob(os.path.join(tempfile.gettempdir(), TAG + "*")))


def cleanup():
    for p in glob.glob(os.path.join(tempfile.gettempdir(), TAG + "*")):
        try:
            os.unlink(p)
        except OSError:
            pass


# ══════════════════════════════════════════════════════════════════════════
# OLD behaviour, reproduced to show the leak this test guards against
# ══════════════════════════════════════════════════════════════════════════
def old_rerun(upload, session):
    """The original if/elif block: temp file built unconditionally."""
    tf = tempfile.NamedTemporaryFile(delete=False, prefix=TAG,
                                     suffix=Path(upload.name).suffix)
    tf.write(upload.read())        # drains after first call
    tf.flush()                     # never closed, never deleted
    key = f"video:{upload.name}:{upload.size}"
    return {"path": tf.name, "should_process": key != session.get("media_key")}


cleanup()
up = Upload("clip.mp4", b"\x00" * 4096)
sess = {"media_key": None}
old_sizes = []
for i in range(8):
    r = old_rerun(up, sess)
    old_sizes.append(os.path.getsize(r["path"]))
    if r["should_process"]:
        sess["media_key"] = f"video:{up.name}:{up.size}"

print("--- OLD behaviour (for contrast) ---")
check("old: temp file per rerun", count_temps(), 8)
check("old: reruns 2+ wrote 0-byte files", old_sizes, [4096] + [0] * 7)
cleanup()


# ══════════════════════════════════════════════════════════════════════════
# NEW behaviour, mirroring the current dashboard block
# ══════════════════════════════════════════════════════════════════════════
def new_rerun(uploaded_video, uploaded_photo, btn_sample_video, btn_sample_photo,
              session, sample_video_exists=True, sample_photo_exists=True):
    candidates = []
    if btn_sample_video and sample_video_exists:
        candidates.append(("video", "input.mp4", "sample:video:input.mp4", None))
    if btn_sample_photo and sample_photo_exists:
        candidates.append(("photo", "uploads/classroom_sample.jpg",
                           "sample:photo:classroom_sample.jpg", None))
    if uploaded_photo is not None:
        candidates.append(("photo", None,
                           f"photo:{uploaded_photo.name}:{uploaded_photo.size}",
                           uploaded_photo))
    if uploaded_video is not None:
        candidates.append(("video", None,
                           f"video:{uploaded_video.name}:{uploaded_video.size}",
                           uploaded_video))

    selected = next((c for c in candidates if c[2] != session.get("media_key")), None)
    should_process = selected is not None

    # sweep the temp file kept from the previous rerun
    stale = session.get("tmp_media_path")
    if stale and os.path.exists(stale):
        try:
            os.unlink(stale)
        except OSError:
            pass
        session["tmp_media_path"] = None

    path = None
    if should_process:
        mode, spath, key, upload = selected
        if upload is not None:
            suffix = Path(upload.name).suffix or ".mp4"
            tf = tempfile.NamedTemporaryFile(delete=False, prefix=TAG, suffix=suffix)
            try:
                tf.write(upload.getvalue())
            finally:
                tf.close()
            path = tf.name
            session["tmp_media_path"] = path
        else:
            path = spath
        if path and os.path.exists(path) and os.path.getsize(path) == 0:
            should_process = False

    return {"path": path, "should_process": should_process,
            "key": selected[2] if selected else None}


print("\n--- NEW behaviour ---")
up = Upload("clip.mp4", b"\x00" * 4096)
sess = {"media_key": None, "tmp_media_path": None}

r = new_rerun(up, None, False, False, sess)
check("rerun 1 processes the upload", r["should_process"], True)
check("rerun 1 wrote full bytes", os.path.getsize(r["path"]), 4096)
sess["media_key"] = r["key"]
peak = count_temps()

# 20 idle reruns (slider moves, expander toggles, etc.)
counts_seen = []
written_sizes = []
for i in range(20):
    r = new_rerun(up, None, False, False, sess)
    counts_seen.append(count_temps())
    if r["path"]:
        written_sizes.append(os.path.getsize(r["path"]))

check("idle reruns never reprocess", r["should_process"], False)
check("idle reruns create no temp file", r["path"], None)
check("idle reruns write nothing at all", written_sizes, [])
check("temp count never grows over 20 reruns", max(counts_seen) <= peak, True)
check("temp count stays bounded (<=1)", max(counts_seen) <= 1, True)

# switching to a second video reuses one slot, not one per rerun
up2 = Upload("other.mp4", b"\x11" * 2048)
r2 = new_rerun(up2, None, False, False, sess)
check("second video processes", r2["should_process"], True)
check("second video wrote full bytes", os.path.getsize(r2["path"]), 2048)
sess["media_key"] = r2["key"]
check("temp count still bounded after switch", count_temps() <= 2, True)

# photo after video is honoured AND materialised correctly
pho = Upload("class.jpg", b"\x22" * 1024)
r3 = new_rerun(up2, pho, False, False, sess)
check("photo after video processes", (r3["should_process"], r3["key"]),
      (True, "photo:class.jpg:1024"))
check("photo wrote full bytes", os.path.getsize(r3["path"]), 1024)
check("photo temp keeps .jpg suffix", Path(r3["path"]).suffix, ".jpg")
sess["media_key"] = r3["key"]

# final sweep leaves at most the one live file
for _ in range(3):
    new_rerun(up2, pho, False, False, sess)
check("steady state holds at most 1 temp file", count_temps() <= 1, True)

cleanup()
check("cleanup leaves nothing behind", count_temps(), 0)

print()
if FAILS:
    print(f"{len(FAILS)} FAILED: {FAILS}")
    sys.exit(1)
print("ALL TEMP LIFECYCLE CHECKS PASSED")

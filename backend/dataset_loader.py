"""DriveGuard dataset loader: inspect real files, never synthesize.

Recursively scans the raw dataset directory (media files placed by the user or
downloaded via training/01_download_dataset.py) and reports counts, extensions
and sample paths without modifying anything. Also summarizes the processed
feature store (the derived numeric cues actually used for training). When no
real data is present the status says so explicitly.
"""
import glob
import json
import os
import time

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
RAW_DIR = os.path.join(DATA, "dataset", "raw")
PROCESSED_DIR = os.path.join(DATA, "dataset", "processed")
MANIFEST_PATH = os.path.join(DATA, "dataset", "utarldd_source_manifest.json")

MEDIA_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v", ".mpg", ".mpeg", ".wmv"}
MAX_SAMPLES = 8


def scan_raw():
    files = []
    for root, _dirs, names in os.walk(RAW_DIR):
        for n in names:
            p = os.path.join(root, n)
            if os.path.splitext(n)[1].lower() in MEDIA_EXTS:
                try:
                    files.append({"path": os.path.relpath(p, RAW_DIR), "bytes": os.path.getsize(p)})
                except OSError:
                    pass
    exts = {}
    for f in files:
        ext = os.path.splitext(f["path"])[1].lower().lstrip(".")
        exts[ext] = exts.get(ext, 0) + 1
    return {
        "root": "backend/data/dataset/raw",
        "files": len(files),
        "total_bytes": sum(f["bytes"] for f in files),
        "extensions": exts,
        "sample_paths": [f["path"] for f in files[:MAX_SAMPLES]],
        "scanned_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }


def scan_processed():
    streams = []
    for f in sorted(glob.glob(os.path.join(PROCESSED_DIR, "utarldd_*.npz"))):
        try:
            import numpy as np
            d = np.load(f, allow_pickle=True)
            m = json.loads(str(d["meta"]))
            streams.append({
                "file": os.path.basename(f),
                "subject": m["subject"], "fold": m["fold"], "state": m["state"],
                "state_code": m["state_code"], "split": m["split"],
                "frames": int(d["cues"].shape[0]),
                "face_coverage": m.get("face_coverage"),
                "fps_native": m.get("fps_native"),
                "ear_ref_final": m.get("ear_ref_final"),
                "blinks_detected": m.get("blinks_detected"),
                "microsleeps_detected": m.get("microsleeps_detected"),
            })
        except Exception:
            continue
    windows = 0
    win_path = os.path.join(PROCESSED_DIR, "test_windows.npz")
    if os.path.exists(win_path):
        import numpy as np
        windows = int(np.load(win_path)["windows"].shape[0])
    per_state = {}
    for s in streams:
        per_state[s["state"]] = per_state.get(s["state"], 0) + 1
    return {
        "root": "backend/data/dataset/processed",
        "subject_states": len(streams),
        "subjects": sorted({s["subject"] for s in streams}),
        "per_state": per_state,
        "frames_total": sum(s["frames"] for s in streams),
        "demo_windows": windows,
        "splits": {sp: sorted({s["subject"] for s in streams if s["split"] == sp})
                   for sp in ("train", "val", "test") if any(s["split"] == sp for s in streams)},
        "source": {
            "name": "UTA-RLDD (Real-Life Drowsiness Dataset)",
            "citation": "Ghoddoosian, Galib, Athitsos (2019), arXiv:1904.07312",
            "url": "https://sites.google.com/view/utarldd/home",
            "note": "Derived numeric cue features only; raw media is obtained from the "
                    "authors' distribution and is not redistributed here.",
        },
        "streams": streams,
    }


def status():
    st = {"raw": scan_raw(), "processed": scan_processed()}
    if os.path.exists(MANIFEST_PATH):
        st["source_manifest_present"] = True
    return st


def save_upload(filename, data: bytes):
    os.makedirs(RAW_DIR, exist_ok=True)
    safe = os.path.basename(filename)
    dest = os.path.join(RAW_DIR, f"uploaded_{int(time.time())}_{safe}")
    with open(dest, "wb") as f:
        f.write(data)
    return dest

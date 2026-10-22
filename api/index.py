"""DriveGuard serverless API for Vercel.

Same contract as backend/main.py (Pattern B), packaged as a single Python
Serverless Function: the Vercel runtime mounts this FastAPI ASGI app for
every /api/* route (see vercel.json rewrites). The trained ONNX model, the
extracted dataset features and the telemetry replays are bundled alongside
this file under api/_data/ so the deployment needs no external services.

Local check:  python -m uvicorn api.index:app --port 8021   (from repo root)

Note: /api/dataset/upload is intentionally disabled here -- the serverless
filesystem is read-only. Use the local run.sh deployment for that.
"""
import glob
import hashlib
import json
import os
import time

import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

APP_VERSION = "1.0.0"
STARTED_AT = time.time()

# ---------------------------------------------------------------- paths
HERE = os.path.dirname(os.path.abspath(__file__))


def _find_data_dir():
    """Locate the bundled data directory (or the repo layout in local dev)."""
    candidates = [
        os.path.join(HERE, "_data"),                                      # api/_data (bundled next to this file)
        os.path.join(os.path.dirname(HERE), "_data"),                     # _data (sibling of api/)
        os.path.normpath(os.path.join(HERE, "..", "backend", "data")),    # repo layout (local dev)
    ]
    for c in candidates:
        if os.path.isdir(os.path.join(c, "model")):
            return c
    return candidates[0]


DATA = _find_data_dir()
MODEL_DIR = os.path.join(DATA, "model")
ONNX_PATH = os.path.join(MODEL_DIR, "driveguard_dwscnn.onnx")
META_PATH = os.path.join(MODEL_DIR, "metadata.json")
MODEL_ASSETS = os.path.join(MODEL_DIR, "assets")
PROCESSED_DIR = os.path.join(DATA, "dataset", "processed")
RAW_DIR = os.path.join(DATA, "dataset", "raw")
MANIFEST_PATH = os.path.join(DATA, "dataset", "utarldd_source_manifest.json")
TELEMETRY_DIR = os.path.join(DATA, "telemetry")

# ---------------------------------------------------------------- constants (mirror of backend/model_loader.py)
WINDOW_T = 64
N_CUES = 16
EAR_COLS = [0, 1, 2, 15]
DEFAULT_EAR_REF = 0.30
LEVEL_NAMES = ["SAFE", "WARNING", "CRITICAL"]

PERCLOS_FIRE = 0.15
BLINK_DUR_FIRE_MS = 400.0
MAR_FIRE = 0.60
NOD_FIRE = 3.0
YAW_FIRE_DEG = 25.0
PITCH_FIRE_DEG = 15.0
MICROSLEEP_FRAMES = 45

MEDIA_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v", ".mpg", ".mpeg", ".wmv"}

# ---------------------------------------------------------------- model load (once per function instance)
session = None
meta = None
sha256 = None
if os.path.exists(ONNX_PATH) and os.path.exists(META_PATH):
    with open(META_PATH) as f:
        meta = json.load(f)
    session = ort.InferenceSession(ONNX_PATH, providers=["CPUExecutionProvider"])
    h = hashlib.sha256()
    with open(ONNX_PATH, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    sha256 = h.hexdigest()

available = session is not None

app = FastAPI(title="DriveGuard API (Vercel)", version=APP_VERSION,
              description="Edge-deployable driver drowsiness detection -- evaluation API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@app.middleware("http")
async def _strip_api_prefix(request, call_next):
    """The Vercel rewrite forwards the original path (e.g. /api/predict).
    Routes here are defined without the prefix; strip it when present so the
    app behaves identically behind the rewrite and standalone."""
    if request.scope["path"].startswith("/api/"):
        request.scope["path"] = request.scope["path"][4:]
    return await call_next(request)


# ---------------------------------------------------------------- request models
class PredictRequest(BaseModel):
    window: list = Field(..., description="64x16 cue window (raw, browser-computed)")
    ear_ref: float | None = Field(None, description="session open-eye EAR reference (calibration)")


# ---------------------------------------------------------------- dataset helpers (mirror of backend/dataset_loader.py)
def scan_raw():
    files = []
    for root, _dirs, names in os.walk(RAW_DIR):
        for n in names:
            if os.path.splitext(n)[1].lower() in MEDIA_EXTS:
                p = os.path.join(root, n)
                try:
                    files.append((os.path.relpath(p, RAW_DIR), os.path.getsize(p)))
                except OSError:
                    pass
    exts = {}
    for rel, _ in files:
        ext = os.path.splitext(rel)[1].lower().lstrip(".")
        exts[ext] = exts.get(ext, 0) + 1
    return {
        "root": "backend/data/dataset/raw",
        "files": len(files),
        "total_bytes": sum(sz for _, sz in files),
        "extensions": exts,
        "sample_paths": [rel for rel, _ in files[:8]],
        "scanned_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }


def scan_processed():
    streams = []
    for f in sorted(glob.glob(os.path.join(PROCESSED_DIR, "utarldd_*.npz"))):
        try:
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


def telemetry_docs():
    docs = []
    for f in glob.glob(os.path.join(TELEMETRY_DIR, "replay-*.json")):
        try:
            docs.append((os.path.getmtime(f), f, json.load(open(f))))
        except Exception:
            continue
    docs.sort(key=lambda x: -x[0])
    return docs


def health_payload():
    tdocs = telemetry_docs()
    return {
        "status": "ok",
        "app_version": APP_VERSION,
        "uptime_s": round(time.time() - STARTED_AT, 1),
        "components": {
            "model": {"loaded": available,
                      **({"artifact": "driveguard_dwscnn.onnx",
                          "sha256": sha256} if available else {})},
            "dataset": scan_processed()["subject_states"],
            "telemetry_sessions": len(tdocs),
        },
    }


# ---------------------------------------------------------------- inference (mirror of backend/model_loader.py)
def cues_fired(window):
    f = window[-1]
    fired = []
    if f[8] >= PERCLOS_FIRE:
        fired.append("PERCLOS")
    if f[7] >= BLINK_DUR_FIRE_MS:
        fired.append("BLINK_DURATION")
    if f[3] >= MAR_FIRE:
        fired.append("YAWN")
    if f[13] >= NOD_FIRE:
        fired.append("HEAD_NOD")
    if abs(f[11]) >= YAW_FIRE_DEG or abs(f[10]) >= PITCH_FIRE_DEG:
        fired.append("HEAD_POSE")
    run = 0
    for i in range(len(window) - 1, -1, -1):
        if window[i][4] >= 0.5:
            run += 1
        else:
            break
    if run >= MICROSLEEP_FRAMES:
        fired.append("MICROSLEEP")
    return fired


def predict(window, ear_ref=None):
    if not available:
        raise RuntimeError("model artifact unavailable")
    w = np.asarray(window, dtype=np.float32)
    if w.shape != (WINDOW_T, N_CUES):
        raise ValueError(f"window must be {WINDOW_T}x{N_CUES}, got {w.shape}")
    if not np.isfinite(w).all():
        raise ValueError("window contains non-finite values")
    ear_ref = float(ear_ref) if ear_ref else DEFAULT_EAR_REF
    w = w.copy()
    w[:, EAR_COLS] /= max(ear_ref, 1e-3)
    mu = np.asarray(meta["norm"]["mean"], dtype=np.float32)
    sd = np.asarray(meta["norm"]["std"], dtype=np.float32)
    x = np.clip((w - mu) / sd, -6.0, 6.0).astype(np.float32)[None, ...]
    t0 = time.perf_counter()
    logits, score01 = session.run(None, {"cue_window": x})
    latency_ms = (time.perf_counter() - t0) * 1000.0
    logits = logits[0]
    probs = np.exp(logits - logits.max())
    probs = probs / probs.sum()
    pred = int(probs.argmax())
    cues = {
        "perclos": round(float(window[-1][8]), 4),
        "blink_rate_per_min": round(float(window[-1][6]), 2),
        "mean_blink_duration_ms": round(float(window[-1][7]), 1),
        "ear_left": round(float(window[-1][0]), 4),
        "ear_right": round(float(window[-1][1]), 4),
        "mar": round(float(window[-1][3]), 4),
        "head_pitch_deg": round(float(window[-1][10]), 2),
        "head_yaw_deg": round(float(window[-1][11]), 2),
        "head_nod_rate_per_min": round(float(window[-1][13]), 2),
    }
    return {
        "drowsiness_score": round(float(score01[0]) * 100.0, 1),
        "level": LEVEL_NAMES[pred],
        "level_probs": {LEVEL_NAMES[i]: round(float(probs[i]), 4) for i in range(3)},
        "confidence": round(float(probs[pred]), 3),
        "cues": cues,
        "cues_fired": cues_fired(window),
        "latency_ms": round(latency_ms, 2),
        "model_sha256": sha256[:12],
    }


# ---------------------------------------------------------------- routes
@app.get("/health")
def health():
    return health_payload()


@app.get("/dataset/status")
def dataset_status():
    st = {"raw": scan_raw(), "processed": scan_processed()}
    if os.path.exists(MANIFEST_PATH):
        st["source_manifest_present"] = True
    return st


@app.post("/dataset/upload")
async def dataset_upload():
    raise HTTPException(501, "dataset upload is disabled on the serverless deployment -- "
                             "use the local run.sh deployment, which stores uploads on disk")


@app.get("/model/status")
def model_status():
    if not available:
        return {"present": False,
                "reason": "trained artifact not found under api/_data/model -- "
                          "the API refuses to fabricate predictions without it"}
    return {
        "present": True,
        "file": "driveguard_dwscnn.onnx",
        "sha256": sha256,
        "size_bytes": os.path.getsize(ONNX_PATH),
        "input_shape": meta["input"]["shape"],
        "cue_names": meta["input"]["cue_names"],
        "frame_rate_hz": meta["input"]["frame_rate_hz"],
        "classes": meta["classes"],
        "parameters": meta["parameters"],
        "trained_at": meta["export"]["trained_at"],
        "runtime": meta["export"]["runtime"],
        "normalization": meta["input"]["normalization"],
    }


@app.get("/model/metrics")
def model_metrics():
    if not available:
        raise HTTPException(503, "model artifact unavailable -- train first")
    return {
        "metrics": meta["metrics"],
        "dataset": meta["dataset"],
        "training": meta["training"],
        "score_target": meta["score_target"],
        "architecture": meta["architecture"],
        "state_map": meta["state_map"],
    }


@app.get("/model/assets/{name}")
def model_asset(name: str):
    if "/" in name or ".." in name or not name.endswith((".png", ".json")):
        raise HTTPException(400, "invalid asset name")
    p = os.path.join(MODEL_ASSETS, name)
    if not os.path.exists(p):
        raise HTTPException(404, "asset not found")
    return FileResponse(p, media_type="application/json" if name.endswith(".json") else "image/png")


@app.post("/predict")
def predict_route(req: PredictRequest):
    if not available:
        raise HTTPException(503, "model artifact unavailable -- no prediction is fabricated")
    try:
        return predict(req.window, req.ear_ref)
    except (ValueError, RuntimeError) as e:
        raise HTTPException(422, str(e))


@app.get("/demo/random")
def demo_random():
    """One random REAL raw input from the held-out subject archive, for the
    random-input demonstration. The frontend feeds this window to /api/predict."""
    p = os.path.join(PROCESSED_DIR, "test_windows.npz")
    if not os.path.exists(p):
        raise HTTPException(503, "held-out window archive not present")
    d = np.load(p, allow_pickle=True)
    n = d["windows"].shape[0]
    if n == 0:
        raise HTTPException(503, "held-out window archive empty")
    i = int(np.random.default_rng().integers(0, n))
    label_names = {0: "alert", 1: "low_vigilant", 2: "drowsy"}
    level_names = {0: "SAFE", 1: "WARNING", 2: "CRITICAL"}
    return {
        "window": d["windows"][i].round(5).tolist(),
        "ear_ref": round(float(d["ear_refs"][i]), 4),
        "label": int(d["labels"][i]),
        "label_name": label_names[int(d["labels"][i])],
        "expected_level": level_names[int(d["labels"][i])],
        "subject": str(d["subjects"][i]),
        "state": str(d["states"][i]),
        "perclos_anchor_score": round(float(d["anchor_scores"][i]), 1),
        "dataset": "UTA-RLDD (held-out subject, real extracted cues)",
        "shape": [64, 16],
    }


@app.get("/telemetry/sessions")
def telemetry_sessions():
    docs = telemetry_docs()
    return {"sessions": [
        {"session_id": d["session_id"], "source": d["source"], "summary": d["summary"],
         "started_at_ms": d["started_at_ms"], "alarm_events": d.get("alarm_events", [])}
        for _, _, d in docs]}


@app.get("/telemetry/sessions/{session_id}")
def telemetry_session(session_id: str):
    if "/" in session_id or ".." in session_id:
        raise HTTPException(400, "invalid id")
    p = os.path.join(TELEMETRY_DIR, f"{session_id}.json")
    if not os.path.exists(p):
        raise HTTPException(404, "session not found")
    return json.load(open(p))


@app.get("/telemetry/latest")
def telemetry_latest():
    docs = telemetry_docs()
    if not docs:
        raise HTTPException(404, "no telemetry exports yet")
    return docs[0][2]

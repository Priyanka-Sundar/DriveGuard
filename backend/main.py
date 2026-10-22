"""DriveGuard backend API (Pattern B: Dataset + Model -> Backend API -> Frontend).

Serves the real trained model artifact, the real dataset status, and real
telemetry exports. When an artifact is missing the corresponding endpoints
report it honestly instead of inventing values.

Run:  uvicorn main:app --host 0.0.0.0 --port 8000   (from backend/)
"""
import glob
import json
import os
import time

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

import dataset_loader
import model_loader

APP_VERSION = "1.0.0"
STARTED_AT = time.time()
HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
TELEMETRY_DIR = os.path.join(DATA, "telemetry")
MODEL_ASSETS = os.path.join(DATA, "model", "assets")
PROCESSED_DIR = os.path.join(DATA, "dataset", "processed")
FRONTEND_DIST = os.path.normpath(os.path.join(HERE, "..", "frontend", "dist"))

app = FastAPI(title="DriveGuard API", version=APP_VERSION,
              description="Edge-deployable driver drowsiness detection -- evaluation API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

loader = model_loader.ModelLoader()


# ------------------------------------------------------------------ models
class PredictRequest(BaseModel):
    window: list = Field(..., description="64x16 cue window (raw, browser-computed)")
    ear_ref: float | None = Field(None, description="session open-eye EAR reference (calibration)")


# ------------------------------------------------------------------ helpers
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
            "model": {"loaded": loader.available,
                      **({"artifact": "driveguard_dwscnn.onnx",
                          "sha256": loader.sha256} if loader.available else {})},
            "dataset": dataset_loader.scan_processed()["subject_states"],
            "telemetry_sessions": len(tdocs),
        },
    }


# ------------------------------------------------------------------ routes
@app.get("/api/health")
def health():
    return health_payload()


@app.get("/api/dataset/status")
def dataset_status():
    return dataset_loader.status()


@app.post("/api/dataset/upload")
async def dataset_upload(file: UploadFile = File(...)):
    """Accept a real dataset archive/file supplied by the developer. Stored
    verbatim; nothing is synthesized. Media files become visible in status."""
    data = await file.read()
    if len(data) > 8 << 30:
        raise HTTPException(413, "file too large")
    dest = dataset_loader.save_upload(file.filename or "upload.bin", data)
    return {"saved": os.path.basename(dest), "bytes": len(data), "status": dataset_loader.status()}


@app.get("/api/model/status")
def model_status():
    return loader.status()


@app.get("/api/model/metrics")
def model_metrics():
    m = loader.metrics()
    if m is None:
        raise HTTPException(503, "model artifact unavailable -- train first")
    return m


@app.get("/api/model/assets/{name}")
def model_asset(name: str):
    if "/" in name or ".." in name or not name.endswith((".png", ".json")):
        raise HTTPException(400, "invalid asset name")
    p = os.path.join(MODEL_ASSETS, name)
    if not os.path.exists(p):
        raise HTTPException(404, "asset not found")
    return FileResponse(p, media_type="application/json" if name.endswith(".json") else "image/png")


@app.post("/api/predict")
def predict(req: PredictRequest):
    if not loader.available:
        raise HTTPException(503, "model artifact unavailable -- no prediction is fabricated")
    try:
        return loader.predict(req.window, req.ear_ref)
    except (ValueError, RuntimeError) as e:
        raise HTTPException(422, str(e))


@app.get("/api/demo/random")
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


@app.get("/api/telemetry/sessions")
def telemetry_sessions():
    docs = telemetry_docs()
    return {"sessions": [
        {"session_id": d["session_id"], "source": d["source"], "summary": d["summary"],
         "started_at_ms": d["started_at_ms"], "alarm_events": d.get("alarm_events", [])}
        for _, _, d in docs]}


@app.get("/api/telemetry/sessions/{session_id}")
def telemetry_session(session_id: str):
    if "/" in session_id or ".." in session_id:
        raise HTTPException(400, "invalid id")
    p = os.path.join(TELEMETRY_DIR, f"{session_id}.json")
    if not os.path.exists(p):
        raise HTTPException(404, "session not found")
    return json.load(open(p))


@app.get("/api/telemetry/latest")
def telemetry_latest():
    docs = telemetry_docs()
    if not docs:
        raise HTTPException(404, "no telemetry exports yet")
    return docs[0][2]


@app.exception_handler(404)
def spa_fallback(request, exc):  # serve index.html for client-side routes
    if not str(request.url.path).startswith("/api"):
        idx = os.path.join(FRONTEND_DIST, "index.html")
        if os.path.exists(idx):
            return FileResponse(idx)
    return JSONResponse({"detail": getattr(exc, "detail", "not found")}, status_code=404)


# production: serve the built frontend from the same origin
if os.path.isdir(FRONTEND_DIST):
    app.mount("/", StaticFiles(directory=FRONTEND_DIST, html=True), name="frontend")

"""DriveGuard model loader: ONNX inference + explainability rules (contract v1.0).

Loads the real trained artifact (data/model/driveguard_dwscnn.onnx) together
with its training metadata (normalization statistics, class map). If the
artifact is absent, inference is refused -- the system never fabricates a
prediction.
"""
import hashlib
import json
import os
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(HERE, "data", "model")
ONNX_PATH = os.path.join(MODEL_DIR, "driveguard_dwscnn.onnx")
META_PATH = os.path.join(MODEL_DIR, "metadata.json")

WINDOW_T = 64
N_CUES = 16
EAR_COLS = [0, 1, 2, 15]          # ocular cues normalized by the session ear reference
DEFAULT_EAR_REF = 0.30
LEVEL_NAMES = ["SAFE", "WARNING", "CRITICAL"]

# cues_fired thresholds (mirrored from training/cue_config.py and the browser engine)
PERCLOS_FIRE = 0.15
BLINK_DUR_FIRE_MS = 400.0
MAR_FIRE = 0.60
NOD_FIRE = 3.0
YAW_FIRE_DEG = 25.0
PITCH_FIRE_DEG = 15.0
MICROSLEEP_FRAMES = 45           # 1.5 s at 30 Hz
FRAME_MS = 1000.0 / 30.0


class ModelLoader:
    def __init__(self):
        self.session = None
        self.meta = None
        self.sha256 = None
        self.load()

    def load(self):
        if not (os.path.exists(ONNX_PATH) and os.path.exists(META_PATH)):
            return
        with open(META_PATH) as f:
            self.meta = json.load(f)
        import onnxruntime as ort
        self.session = ort.InferenceSession(ONNX_PATH, providers=["CPUExecutionProvider"])
        h = hashlib.sha256()
        with open(ONNX_PATH, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        self.sha256 = h.hexdigest()

    @property
    def available(self):
        return self.session is not None

    def status(self):
        if not self.available:
            return {"present": False,
                    "reason": "trained artifact not found under backend/data/model -- "
                              "run backend/training/03_train_model.py; the API refuses to "
                              "fabricate predictions without it"}
        return {
            "present": True,
            "file": "driveguard_dwscnn.onnx",
            "sha256": self.sha256,
            "size_bytes": os.path.getsize(ONNX_PATH),
            "input_shape": self.meta["input"]["shape"],
            "cue_names": self.meta["input"]["cue_names"],
            "frame_rate_hz": self.meta["input"]["frame_rate_hz"],
            "classes": self.meta["classes"],
            "parameters": self.meta["parameters"],
            "trained_at": self.meta["export"]["trained_at"],
            "runtime": self.meta["export"]["runtime"],
            "normalization": self.meta["input"]["normalization"],
        }

    def metrics(self):
        if not self.available:
            return None
        return {
            "metrics": self.meta["metrics"],
            "dataset": self.meta["dataset"],
            "training": self.meta["training"],
            "score_target": self.meta["score_target"],
            "architecture": self.meta["architecture"],
            "state_map": self.meta["state_map"],
        }

    # ------------------------------------------------------------------
    @staticmethod
    def cues_fired(window):
        """Deterministic explainability rules over the raw cue window."""
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

    def predict(self, window, ear_ref=None):
        """window: 64x16 raw cue tensor (browser-computed, unnormalized)."""
        if not self.available:
            raise RuntimeError("model artifact unavailable")
        w = np.asarray(window, dtype=np.float32)
        if w.shape != (WINDOW_T, N_CUES):
            raise ValueError(f"window must be {WINDOW_T}x{N_CUES}, got {w.shape}")
        if not np.isfinite(w).all():
            raise ValueError("window contains non-finite values")
        ear_ref = float(ear_ref) if ear_ref else DEFAULT_EAR_REF
        w = w.copy()
        w[:, EAR_COLS] /= max(ear_ref, 1e-3)
        mu = np.asarray(self.meta["norm"]["mean"], dtype=np.float32)
        sd = np.asarray(self.meta["norm"]["std"], dtype=np.float32)
        x = np.clip((w - mu) / sd, -6.0, 6.0).astype(np.float32)[None, ...]
        t0 = time.perf_counter()
        logits, score01 = self.session.run(None, {"cue_window": x})
        latency_ms = (time.perf_counter() - t0) * 1000.0
        logits = logits[0]
        probs = np.exp(logits - logits.max())
        probs = probs / probs.sum()
        pred = int(probs.argmax())
        f = w[-1]
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
            "cues_fired": self.cues_fired(window),
            "latency_ms": round(latency_ms, 2),
            "model_sha256": self.sha256[:12],
        }

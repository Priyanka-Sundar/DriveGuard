"""DriveGuard step 4: generate real telemetry sessions + held-out demo windows.

Replays the held-out test subjects through the deployed ONNX model at 1 Hz
(exactly the live-monitor data path) and applies the contract alarm policy,
producing session JSON files for Demo Replay, plus a window archive that the
/api/demo/random endpoint samples from. Every value comes from the real model
running on real extracted cues -- nothing is synthesized.

Usage: python 04_export_telemetry.py
"""
import glob
import json
import os
import time
import uuid

import numpy as np
import onnxruntime as ort

from cue_config import (ALARM, LEVEL_NAMES, STATE_NAMES, TARGET_FPS, WINDOW_T,
                        BLINK_DUR_FIRE_MS, MAR_FIRE, NOD_FIRE, PERCLOS_FIRE,
                        PITCH_FIRE_DEG, YAW_FIRE_DEG)

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.normpath(os.path.join(HERE, "..", "data"))
MODEL_DIR = os.path.join(DATA, "model")
TELEMETRY_DIR = os.path.join(DATA, "telemetry")
PROCESSED = os.path.join(DATA, "dataset", "processed")

MICROSLEEP_FRAMES = int(1.5 * TARGET_FPS)  # 45 frames ~ 1.5 s


def cues_fired_for(window):
    """Deterministic explainability rules from the raw cue window (last frame)."""
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
    # microsleep: trailing run of eye-closed frames >= 1.5 s
    run = 0
    for i in range(len(window) - 1, -1, -1):
        if window[i][4] >= 0.5:
            run += 1
        else:
            break
    if run >= MICROSLEEP_FRAMES:
        fired.append("MICROSLEEP")
    return fired


def cue_snapshot(f):
    return {
        "perclos": round(float(f[8]), 4),
        "blink_rate_per_min": round(float(f[6]), 2),
        "mean_blink_duration_ms": round(float(f[7]), 1),
        "ear_left": round(float(f[0]), 4),
        "ear_right": round(float(f[1]), 4),
        "mar": round(float(f[3]), 4),
        "head_pitch_deg": round(float(f[10]), 2),
        "head_yaw_deg": round(float(f[11]), 2),
        "head_nod_rate_per_min": round(float(f[13]), 2),
    }


class AlarmEngine:
    """Contract alarm policy (whitepaper Table 3.3), default profile."""

    def __init__(self, profile=None):
        p = dict(ALARM)
        if profile:
            p.update(profile)
        self.p = p
        self.state = "NONE"
        self.events = []
        self._above_since = None   # score >= warning_raise since
        self._below_since = None   # score < warning_clear since
        self._crit_since = None
        self._last_microsleep_ms = None
        self._cooldown_until = 0
        self._microsleep_fired = False

    def step(self, t_ms, score, microsleep, face_lost):
        p = self.p
        ev = None
        if face_lost:
            # FACE_LOST is an advisory level, not an alarm; state unchanged
            return self.state, self.events

        if microsleep:
            now_crit = (self._last_microsleep_ms is not None and
                        t_ms - self._last_microsleep_ms <= p["microsleep_window_s"] * 1000)
            self._last_microsleep_ms = t_ms
            if self.state == "NONE" and t_ms >= self._cooldown_until:
                ev = {"type": "WARNING_RAISED", "at_ms": t_ms, "triggered_by": ["MICROSLEEP"]}
                self.state = "WARNING"
            elif now_crit and t_ms >= self._cooldown_until:
                ev = {"type": "CRITICAL_RAISED", "at_ms": t_ms,
                      "triggered_by": ["MICROSLEEP"]}
                self.state = "CRITICAL"
                self._cooldown_until = t_ms + p["cooldown_s"] * 1000
        if ev is None and self.state == "NONE" and t_ms >= self._cooldown_until:
            if score >= p["warning_raise"]:
                if self._above_since is None:
                    self._above_since = t_ms
                if t_ms - self._above_since >= p["warning_raise_sustain_s"] * 1000:
                    ev = {"type": "WARNING_RAISED", "at_ms": t_ms, "triggered_by": ["SCORE"]}
                    self.state = "WARNING"
            else:
                self._above_since = None
        if ev is None and self.state == "WARNING":
            if score >= p["critical_raise"]:
                if self._crit_since is None:
                    self._crit_since = t_ms
                if t_ms - self._crit_since >= p["critical_raise_sustain_s"] * 1000:
                    ev = {"type": "CRITICAL_RAISED", "at_ms": t_ms, "triggered_by": ["SCORE"]}
                    self.state = "CRITICAL"
                    self._cooldown_until = t_ms + p["cooldown_s"] * 1000
            else:
                self._crit_since = None
                if score < p["warning_clear"]:
                    if self._below_since is None:
                        self._below_since = t_ms
                    if t_ms - self._below_since >= p["warning_clear_sustain_s"] * 1000:
                        ev = {"type": "WARNING_CLEARED", "at_ms": t_ms, "triggered_by": []}
                        self.state = "NONE"
                        self._above_since = None
                else:
                    self._below_since = None
        if ev:
            self.events.append(ev)
        return self.state, self.events


def main():
    os.makedirs(TELEMETRY_DIR, exist_ok=True)
    meta = json.load(open(os.path.join(MODEL_DIR, "metadata.json")))
    mu = np.asarray(meta["norm"]["mean"], dtype=np.float32)
    sd = np.asarray(meta["norm"]["std"], dtype=np.float32)
    sess = ort.InferenceSession(os.path.join(MODEL_DIR, "driveguard_dwscnn.onnx"),
                                providers=["CPUExecutionProvider"])
    test_subjects = set(meta["dataset"]["subjects_test"])

    streams = []
    for f in sorted(glob.glob(os.path.join(PROCESSED, "utarldd_*.npz"))):
        d = np.load(f, allow_pickle=True)
        m = json.loads(str(d["meta"]))
        subject = f"{m['fold']}:s{m['subject']}"
        if subject not in test_subjects:
            continue
        streams.append({"cues": d["cues"].astype(np.float32), "meta": m, "subject": subject,
                        "ear_ref": float(m.get("ear_ref_final", 0.0)) or 0.30})

    # -------- held-out window archive for /api/demo/random --------
    win, labels, subjects, states, ear_refs, anchors = [], [], [], [], [], []
    from cue_config import perclos_to_score
    for st in streams:
        cues = st["cues"]
        n = cues.shape[0]
        for end in range(WINDOW_T, n + 1, 15):
            a = end - WINDOW_T
            w = cues[a:end].copy()
            w[:, [0, 1, 2, 15]] /= max(st["ear_ref"], 1e-3)
            win.append(w)
            labels.append({0: 0, 5: 1, 10: 2}[st["meta"]["state_code"]])
            subjects.append(st["subject"])
            states.append(st["meta"]["state"])
            ear_refs.append(st["ear_ref"])
            anchors.append(perclos_to_score(float(cues[end - 1, 8])))
    np.savez_compressed(
        os.path.join(PROCESSED, "test_windows.npz"),
        windows=np.asarray(win, dtype=np.float32),
        labels=np.asarray(labels, dtype=np.int64),
        subjects=np.asarray(subjects), states=np.asarray(states),
        ear_refs=np.asarray(ear_refs, dtype=np.float32),
        anchor_scores=np.asarray(anchors, dtype=np.float32))
    print(f"saved {len(win)} held-out windows for the random-input demo")

    # -------- telemetry sessions (1 Hz, contract objects) --------
    index = []
    for st in streams:
        cues = st["cues"]
        ear_ref = st["ear_ref"]
        alarm = AlarmEngine()
        session_id = f"replay-{st['subject'].split(':')[1].lstrip('s')}-{st['meta']['state']}-{uuid.uuid4().hex[:8]}"
        frames = []
        t0_wall = time.time()
        for k, end in enumerate(range(WINDOW_T, cues.shape[0] + 1, int(TARGET_FPS))):  # 1 Hz
            a = end - WINDOW_T
            w = cues[a:end].copy()
            w[:, [0, 1, 2, 15]] /= max(ear_ref, 1e-3)
            x = np.clip((w - mu) / sd, -6.0, 6.0).astype(np.float32)[None, ...]
            logits, score01 = sess.run(None, {"cue_window": x})
            logits = logits[0]
            probs = np.exp(logits - logits.max())
            probs = probs / probs.sum()
            pred = int(probs.argmax())
            fired = cues_fired_for(cues[a:end])
            t_ms = int(k * 1000)
            microsleep = "MICROSLEEP" in fired
            alarm_state, _ = alarm.step(t_ms, float(score01[0]) * 100.0, microsleep, False)
            frames.append({
                "schema_version": "1.0",
                "session_id": session_id,
                "timestamp_ms": t_ms,
                "drowsiness_score": round(float(score01[0]) * 100.0, 1),
                "level": LEVEL_NAMES[pred],
                "confidence": round(float(probs[pred]), 3),
                "cues": cue_snapshot(cues[end - 1]),
                "cues_fired": fired,
                "alarm_state": alarm_state,
            })
        scores = [f["drowsiness_score"] for f in frames]
        summary = {
            "duration_s": len(frames),
            "frames": len(frames),
            "mean_score": round(float(np.mean(scores)), 1),
            "peak_score": round(float(np.max(scores)), 1),
            "alarm_events": len(alarm.events),
            "level_counts": {lv: sum(1 for f in frames if f["level"] == lv) for lv in LEVEL_NAMES},
        }
        doc = {
            "schema_version": "1.0",
            "session_id": session_id,
            "source": {
                "kind": "model-replay",
                "dataset": "UTA-RLDD",
                "subject": st["subject"],
                "state": st["meta"]["state"],
                "split": "test",
                "note": "Real model output replayed over real extracted cues from a held-out subject.",
            },
            "started_at_ms": int(t0_wall * 1000),
            "alarm_events": alarm.events,
            "frames": frames,
            "summary": summary,
        }
        path = os.path.join(TELEMETRY_DIR, f"{session_id}.json")
        with open(path, "w") as f:
            json.dump(doc, f)
        index.append({"session_id": session_id, "source": doc["source"], "summary": summary,
                      "file": os.path.basename(path)})
        print(f"  session {session_id}: {len(frames)} frames, peak={summary['peak_score']}, "
              f"alarms={summary['alarm_events']}, levels={summary['level_counts']}")
    with open(os.path.join(TELEMETRY_DIR, "index.json"), "w") as f:
        json.dump({"generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "sessions": index}, f, indent=1)
    print("done")


if __name__ == "__main__":
    main()

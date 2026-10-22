"""DriveGuard Stage-1/2 feature extraction: MediaPipe FaceLandmarker -> 16-cue stream.

The CueEngine is a causal (streaming) implementation: every cue at time t is
computed only from frames at or before t, so the exact same algorithm runs in
the browser during live monitoring (mirrored in frontend/src/lib/cueEngine.ts).
"""
import bisect
import json
import math

import cv2
import numpy as np

from cue_config import (
    ALARM, AUTO_REF_WINDOW_S, BLINK_MIN_MS, CLOSED_RATIO, DEBOUNCE_FRAMES, DEFAULT_EAR_REF,
    EAR_L_PTS, EAR_R_PTS, EAR_VEL_EMA, MAR_H, MAR_V1, MAR_V2, NOSE, CHIN,
    CHEEK_L, CHEEK_R, FOREHEAD, MICROSLEEP_MS, NOD_DROP_DEG_PER_FRAME,
    NOD_MAX_S, NOD_RECOVER_DEG, PITCH_DIV, PITCH_EMA, PITCH_NEUTRAL, SEC_SINCE_BLINK_DEFAULT,
    SEC_SINCE_BLINK_MAX, TARGET_FPS, YAW_DIV, CUE_NAMES,
)

_ = (ALARM, CUE_NAMES)  # imported for provenance; alarm used by telemetry step


class CueEngine:
    """Causal multi-cue computation over a landmark stream (30 Hz target)."""

    def __init__(self, ear_ref=None):
        self.ear_ref_override = ear_ref
        self.ear_ref = ear_ref if ear_ref else DEFAULT_EAR_REF
        self.reset()

    def reset(self):
        if self.ear_ref_override:
            self.ear_ref = self.ear_ref_override
        else:
            self.ear_ref = DEFAULT_EAR_REF
        self.t_hist, self.ear_hist, self.closed_hist, self.pitch_hist = [], [], [], []
        self.blinks = []        # (t_end_ms, duration_ms)
        self.microsleeps = []   # (t_end_ms, duration_ms)
        self.nods = []          # t_ms of nod events
        self._ref_window = []
        self._closed = False
        self._cand_start = None
        self._deb_close = 0
        self._deb_open = 0
        self._last_blink_end = None
        self._prev_ear = None
        self._ear_vel = 0.0
        self._pitch_s = None
        self._prev_pitch_s = None
        self._nod = None

    # -- geometry helpers -------------------------------------------------
    @staticmethod
    def _ear(lm, w, h, pts):
        p1, p2, p3, p4, p5, p6 = pts

        def d(a, b):
            return math.hypot((lm[a].x - lm[b].x) * w, (lm[a].y - lm[b].y) * h)

        return (d(p2, p6) + d(p3, p5)) / (2.0 * max(d(p1, p4), 1e-6))

    def _head_pose(self, lm, w, h):
        def d(a, b):
            return math.hypot((lm[a].x - lm[b].x) * w, (lm[a].y - lm[b].y) * h)

        fw = max(d(CHEEK_L, CHEEK_R), 1e-6)
        fh = max(d(FOREHEAD, CHIN), 1e-6)
        ex_r, ey_r = lm[33].x * w, lm[33].y * h
        ex_l, ey_l = lm[263].x * w, lm[263].y * h
        roll = math.degrees(math.atan2(ey_r - ey_l, ex_l - ex_r))
        mid_x = (lm[CHEEK_L].x + lm[CHEEK_R].x) * 0.5 * w
        yr = (lm[NOSE].x * w - mid_x) / fw
        yaw = math.degrees(math.asin(max(-1.0, min(1.0, yr / YAW_DIV))))
        emy = (ey_r + ey_l) * 0.5
        pr = ((lm[NOSE].y * h - emy) / fh) - PITCH_NEUTRAL
        pitch = math.degrees(math.asin(max(-1.0, min(1.0, pr / PITCH_DIV))))
        return pitch, yaw, roll

    # -- window statistics --------------------------------------------------
    def _trailing(self, events, t_ms, window_ms):
        """Events (t_end, ...) fully inside the trailing window."""
        lo = bisect.bisect_left(events, (t_ms - window_ms, -1))
        return events[lo:]

    def _frac_closed(self, t_ms, window_ms):
        lo = bisect.bisect_left(self.t_hist, t_ms - window_ms)
        hi = len(self.t_hist)
        if hi - lo < 2:
            return 0.0
        return float(np.mean(self.closed_hist[lo:hi]))

    # -- main step ------------------------------------------------------------
    def process(self, lm, w, h, t_ms):
        """Process one frame from landmarks. lm: iterable with .x/.y/.z."""
        ear_r = self._ear(lm, w, h, EAR_R_PTS)
        ear_l = self._ear(lm, w, h, EAR_L_PTS)

        def dd(a, b):
            return math.hypot((lm[a].x - lm[b].x) * w, (lm[a].y - lm[b].y) * h)

        mar = (dd(*MAR_V1) + dd(*MAR_V2)) / (2.0 * max(dd(*MAR_H), 1e-6))
        pitch, yaw, roll = self._head_pose(lm, w, h)
        return self.step((ear_l, ear_r, mar, pitch, yaw, roll), t_ms)

    def step(self, prim, t_ms):
        """Advance the engine from per-frame primitives.

        prim: (ear_left, ear_right, mar, pitch_deg, yaw_deg, roll_deg).
        Split from process() so stored features can be replayed with new
        constants without touching the raw videos (see recompute_derived.py).
        """
        ear_l, ear_r, mar, pitch, yaw, roll = prim
        ear = (ear_l + ear_r) * 0.5
        pitch_s = pitch if self._pitch_s is None else PITCH_EMA * pitch + (1 - PITCH_EMA) * self._pitch_s

        # automatic open-eye reference (rolling P95 of EAR) when not calibrated
        if not self.ear_ref_override:
            self._ref_window.append((t_ms, ear))
            cutoff = t_ms - int(AUTO_REF_WINDOW_S * 1000)
            while self._ref_window and self._ref_window[0][0] < cutoff:
                self._ref_window.pop(0)
            if (len(self._ref_window) >= 150 and
                    (self._ref_window[-1][0] - self._ref_window[0][0]) >= 8_000):
                self.ear_ref = float(np.percentile([e for _, e in self._ref_window], 95))

        closed_now = ear < CLOSED_RATIO * self.ear_ref

        # debounced closure state machine (blink / microsleep events)
        if not self._closed:
            self._deb_close = self._deb_close + 1 if closed_now else 0
            if self._deb_close >= DEBOUNCE_FRAMES:
                self._closed = True
                self._cand_start = t_ms - (DEBOUNCE_FRAMES - 1) * (1000.0 / TARGET_FPS)
                self._deb_open = 0
        else:
            self._deb_open = self._deb_open + 1 if not closed_now else 0
            if self._deb_open >= DEBOUNCE_FRAMES:
                dur = t_ms - self._cand_start
                self._closed = False
                if dur >= MICROSLEEP_MS:
                    self.microsleeps.append((t_ms, dur))
                    self._last_blink_end = t_ms
                elif dur >= BLINK_MIN_MS:
                    self.blinks.append((t_ms, dur))
                    self._last_blink_end = t_ms

        # head-nod detection on smoothed pitch
        vel = 0.0 if self._prev_pitch_s is None else pitch_s - self._prev_pitch_s
        if self._nod is None:
            if vel <= NOD_DROP_DEG_PER_FRAME:
                self._nod = (t_ms, pitch_s)
        else:
            t0, min_p = self._nod
            min_p = min(min_p, pitch_s)
            if pitch_s - min_p >= NOD_RECOVER_DEG:
                self.nods.append((t_ms, 0.0))
                self._nod = None
            elif t_ms - t0 > NOD_MAX_S * 1000:
                self._nod = None
            else:
                self._nod = (t0, min_p)
        self._prev_pitch_s = pitch_s

        # histories
        self.t_hist.append(t_ms)
        self.ear_hist.append(ear)
        self.closed_hist.append(1.0 if self._closed else 0.0)
        self.pitch_hist.append(pitch_s)

        # smoothed EAR velocity (per second)
        if self._prev_ear is not None:
            inst = (ear - self._prev_ear) * TARGET_FPS
            self._ear_vel = EAR_VEL_EMA * inst + (1 - EAR_VEL_EMA) * self._ear_vel
        self._prev_ear = ear

        # trailing-window statistics (counts are normalized to the shorter of
        # elapsed time vs the full window, so early-session rates are unbiased)
        elapsed_s = (t_ms - (self.t_hist[0] if self.t_hist else t_ms)) / 1000.0
        rate_div = min(max(elapsed_s, 1.0), 60.0) / 60.0  # fraction of a full window
        win_blinks = self._trailing(self.blinks, t_ms, 60_000)
        blink_rate = len(win_blinks) / rate_div
        mean_blink_ms = float(np.mean([d for _, d in win_blinks])) if win_blinks else 0.0
        perclos60 = self._frac_closed(t_ms, 60_000)
        perclos10 = self._frac_closed(t_ms, 10_000)
        nod_rate = len(self._trailing(self.nods, t_ms, 60_000)) / rate_div
        ssb = ((t_ms - self._last_blink_end) / 1000.0
               if self._last_blink_end is not None else SEC_SINCE_BLINK_DEFAULT)
        ssb = min(max(ssb, 0.0), SEC_SINCE_BLINK_MAX)

        return [
            ear_l, ear_r, ear, mar,
            1.0 if closed_now else 0.0,
            1.0 if self._closed else 0.0,
            float(blink_rate), float(mean_blink_ms), float(perclos60), float(perclos10),
            pitch, yaw, roll, float(nod_rate), float(ssb), float(self._ear_vel),
        ]

    # ------------------------------------------------------------------


_mp_state = {"ts": 0}  # landmarker timestamps must be globally monotonic across videos


def extract_video(video_path, landmarker, seg_s=150.0, max_side=480):
    """Extract the middle segment of one video into a 30 Hz cue stream.

    Returns dict with cues (N,16) float32, t_ms (N,), face_ok (N,), meta.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    if fps <= 1 or fps > 240:
        fps = 30.0
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    dur_s = n_frames / fps
    start_s = max(0.0, (dur_s - seg_s) * 0.5)
    end_s = min(dur_s, start_s + seg_s)

    engine = CueEngine()
    raw_t, raw_cues, raw_ok = [], [], []
    # seek near the segment start (keyframe-accurate; frames before start are skipped)
    cap.set(cv2.CAP_PROP_POS_MSEC, start_s * 1000.0 - 500 if start_s > 1 else 0)
    last_mp_ts = 0
    last_t = 0.0
    frame_i = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        t_raw = cap.get(cv2.CAP_PROP_POS_MSEC)
        # sanitize decoder timestamps: some codecs stall or jump POS_MSEC
        if not np.isfinite(t_raw) or t_raw <= last_t:
            t_raw = last_t + 1000.0 / fps
        t_ms = float(t_raw)
        last_t = t_ms
        frame_i += 1
        if t_ms / 1000.0 < start_s:
            continue
        if t_ms / 1000.0 > end_s:
            break
        h, w = frame.shape[:2]
        scale = max_side / float(max(h, w))
        if scale < 0.99:
            frame = cv2.resize(frame, (int(round(w * scale)), int(round(h * scale))))
            h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        import mediapipe as mp  # local import to keep module import light
        _mp_state["ts"] += 1
        mp_ts = _mp_state["ts"]  # strictly increasing, decoupled from decoder time
        res = landmarker.detect_for_video(
            mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), mp_ts)
        if res.face_landmarks:
            cues = engine.process(res.face_landmarks[0], w, h, t_ms)
            raw_t.append(t_ms)
            raw_cues.append(cues)
            raw_ok.append(True)
        else:
            raw_t.append(t_ms)
            raw_cues.append(None)
            raw_ok.append(False)
    cap.release()

    # resample onto a strict 30 Hz grid (nearest sample by time; low-fps
    # sources duplicate samples onto the grid -- this matches the model's
    # 30-frame-per-second input contract)
    grid = []
    t0 = start_s * 1000.0
    n_grid = int((end_s - start_s) * TARGET_FPS)
    raw_t_arr = np.asarray(raw_t, dtype=np.float64)
    for i in range(n_grid):
        gt = t0 + i * (1000.0 / TARGET_FPS)
        j = int(np.searchsorted(raw_t_arr, gt))
        best, bestd = None, 1e9
        for jj in (j - 1, j):
            if 0 <= jj < len(raw_t_arr):
                d = abs(raw_t_arr[jj] - gt)
                if d < bestd:
                    bestd, best = d, jj
        if best is not None:
            grid.append((raw_cues[best], raw_ok[best]))
        else:
            grid.append((None, False))

    # forward-fill gaps (face lost); count real coverage
    cues_out, ok_out = [], []
    last = [0.0] * 16
    for c, ok in grid:
        if c is not None:
            last = c
        cues_out.append(last)
        ok_out.append(bool(ok and c is not None))
    t_out = (t0 + np.arange(len(grid)) * (1000.0 / TARGET_FPS)).astype(np.int64)

    face_pct = float(np.mean(ok_out)) if ok_out else 0.0
    meta = {
        "fps_native": round(float(fps), 3),
        "duration_s": round(float(dur_s), 2),
        "segment_s": [round(start_s, 2), round(end_s, 2)],
        "face_coverage": round(face_pct, 4),
        "ear_ref_final": round(float(engine.ear_ref), 4),
        "blinks_detected": len(engine.blinks),
        "microsleeps_detected": len(engine.microsleeps),
        "nods_detected": len(engine.nods),
    }
    return {
        "cues": np.asarray(cues_out, dtype=np.float32),
        "t_ms": t_out,
        "face_ok": np.asarray(ok_out, dtype=bool),
        "meta": meta,
        "engine": engine,
    }

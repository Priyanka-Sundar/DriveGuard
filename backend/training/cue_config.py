"""DriveGuard shared cue / alarm constants (contract v1.0) -- Python side.

This file is the single source of truth for the Python pipeline. The frontend
mirrors these constants in frontend/src/lib/cueConfig.ts -- keep both in sync.

Cue vector layout (F = 16), computed per frame at 30 Hz:
 idx  name               meaning
  0   ear_left           eye aspect ratio, subject's left eye
  1   ear_right          eye aspect ratio, subject's right eye
  2   ear_mean           mean of both EARs
  3   mar                mouth aspect ratio (yawn proxy)
  4   eye_closed         1.0 if ear_mean < CLOSED_RATIO * ear_ref
  5   blink_active       1.0 while inside a debounced closure episode
  6   blink_rate_1m      blinks per minute, trailing 60 s
  7   mean_blink_ms_1m   mean blink duration (ms), trailing 60 s
  8   perclos_60         fraction of closed frames, trailing 60 s
  9   perclos_10         fraction of closed frames, trailing 10 s
 10   head_pitch_deg     approximate head pitch (negative = looking down)
 11   head_yaw_deg       approximate head yaw
 12   head_roll_deg      approximate head roll
 13   nod_rate_1m        head-nod events per minute, trailing 60 s
 14   sec_since_blink    seconds since last completed blink (clipped 0-20)
 15   ear_vel            smoothed d(ear_mean)/dt, per second
"""
import bisect
import math

CUE_NAMES = [
    "ear_left", "ear_right", "ear_mean", "mar", "eye_closed", "blink_active",
    "blink_rate_1m", "mean_blink_ms_1m", "perclos_60", "perclos_10",
    "head_pitch_deg", "head_yaw_deg", "head_roll_deg", "nod_rate_1m",
    "sec_since_blink", "ear_vel",
]

TARGET_FPS = 30.0
WINDOW_T = 64        # frames per inference window (~2.13 s @ 30 Hz)
WINDOW_STRIDE = 15    # training-window stride (0.5 s)

# MediaPipe FaceLandmarker landmark indices (468-point base topology)
EAR_R_PTS = (33, 160, 158, 133, 153, 144)     # subject's right eye
EAR_L_PTS = (362, 385, 387, 263, 373, 380)   # subject's left eye
MAR_V1, MAR_V2, MAR_H = (13, 14), (82, 87), (61, 291)
NOSE, FOREHEAD, CHIN, CHEEK_L, CHEEK_R = 1, 10, 152, 234, 456

# EAR reference (open-eye) handling
DEFAULT_EAR_REF = 0.30      # population default when no calibration exists
CLOSED_RATIO = 0.55         # closed if ear_mean < ratio * ear_ref (approximates the P80 convention)
AUTO_REF_WINDOW_S = 120.0   # rolling window for automatic ear_ref estimate (long: must track the OPEN-eye level, not sink with closures)
AUTO_REF_MIN_S = 8.0        # minimum data before auto estimate replaces default
DEBOUNCE_FRAMES = 2         # ~60 ms @ 30 Hz
BLINK_MIN_MS = 40.0
MICROSLEEP_MS = 1500.0
EAR_VEL_EMA = 0.35
PITCH_EMA = 0.40
NOD_DROP_DEG_PER_FRAME = -0.4
NOD_RECOVER_DEG = 6.0
NOD_MAX_S = 2.5
SEC_SINCE_BLINK_MAX = 20.0
SEC_SINCE_BLINK_DEFAULT = 5.0

# Head-pose geometric approximation constants (documented approximation, not PnP)
YAW_DIV = 0.52
PITCH_NEUTRAL = 0.32
PITCH_DIV = 0.40

# v1.0 PERCLOS -> score anchor (piecewise linear). Knots align WARNING=40 @
# PERCLOS 0.15 and CRITICAL=70 @ PERCLOS 0.35 with the default alarm policy.
SCORE_ANCHOR_KNOTS = [
    (0.00, 5.0), (0.08, 22.0), (0.15, 40.0), (0.25, 55.0),
    (0.35, 70.0), (0.50, 85.0), (0.80, 100.0),
]


def perclos_to_score(p):
    """Piecewise-linear anchor from PERCLOS (0-1) to drowsiness score (0-100)."""
    p = min(max(float(p), 0.0), 1.0)
    k = SCORE_ANCHOR_KNOTS
    if p <= k[0][0]:
        return k[0][1]
    for (x0, y0), (x1, y1) in zip(k, k[1:]):
        if p <= x1:
            return y0 + (y1 - y0) * (p - x0) / (x1 - x0)
    return k[-1][1]


# cues_fired thresholds (deterministic explainability rules)
PERCLOS_FIRE = 0.15
BLINK_DUR_FIRE_MS = 400.0
MAR_FIRE = 0.60
NOD_FIRE = 3.0
YAW_FIRE_DEG = 25.0
PITCH_FIRE_DEG = 15.0

# Alarm policy, default profile (whitepaper Table 3.3). Hysteresis and
# confirmation windows are contract-mandatory.
ALARM = {
    "warning_raise": 40.0, "warning_raise_sustain_s": 5.0,
    "warning_clear": 32.0, "warning_clear_sustain_s": 10.0,
    "critical_raise": 70.0, "critical_raise_sustain_s": 2.0,
    "microsleep_window_s": 60.0, "cooldown_s": 30.0, "face_lost_s": 10.0,
}
SENSITIVITY_PROFILES = {
    "commute": {"warning_raise": 40.0, "critical_raise": 70.0},
    "night":   {"warning_raise": 35.0, "critical_raise": 60.0},
    "truck":   {"warning_raise": 30.0, "critical_raise": 55.0},
}

LEVEL_NAMES = ["SAFE", "WARNING", "CRITICAL"]          # class index order
# UTA-RLDD state code -> class index
STATE_TO_CLASS = {0: 0, 5: 1, 10: 2}
STATE_NAMES = {0: "alert", 5: "low_vigilant", 10: "drowsy"}

"""Recompute derived cues for all stored feature files.

The extraction step stores per-frame primitives (ear_left, ear_right, mar,
pitch, yaw, roll -- cue columns 0,1,3,10,11,12). Every derived cue (closure
flags, blink/microsleep events, PERCLOS, rates, nod rate, ...) is a causal
function of those primitives plus the shared constants, so tuning a constant
does not require the raw videos again: this script replays the primitives
through CueEngine.step() and rewrites the cue matrices.

Usage: python recompute_derived.py
"""
import glob
import json
import os
import time

import numpy as np

from cue_config import STATE_NAMES
from feature_extraction import CueEngine

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.normpath(os.path.join(HERE, "..", "data", "dataset", "processed"))


def recompute_file(path):
    d = np.load(path, allow_pickle=True)
    cues = d["cues"].copy()
    t_ms = d["t_ms"]
    meta = json.loads(str(d["meta"]))
    engine = CueEngine()
    n = cues.shape[0]
    for i in range(n):
        prim = (float(cues[i, 0]), float(cues[i, 1]), float(cues[i, 3]),
                float(cues[i, 10]), float(cues[i, 11]), float(cues[i, 12]))
        out = engine.step(prim, float(t_ms[i]))
        cues[i] = out
    meta.update({
        "ear_ref_final": round(float(engine.ear_ref), 4),
        "blinks_detected": len(engine.blinks),
        "microsleeps_detected": len(engine.microsleeps),
        "nods_detected": len(engine.nods),
        "recomputed_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    })
    np.savez_compressed(path, cues=cues, t_ms=t_ms, face_ok=d["face_ok"],
                        meta=json.dumps(meta))
    return meta, cues


def main():
    files = sorted(glob.glob(os.path.join(OUT, "utarldd_*.npz")))
    print(f"{'file':<40} {'state':<14} {'blinks':>6} {'mslp':>5} {'nods':>5} {'perclos mean/max':>18}")
    for f in files:
        meta, cues = recompute_file(f)
        p60 = cues[:, 8]
        print(f"{os.path.basename(f):<40} {meta['state']:<14} {meta['blinks_detected']:>6} "
              f"{meta['microsleeps_detected']:>5} {meta['nods_detected']:>5} "
              f"{p60.mean():>8.3f}/{p60.max():>7.3f}")


if __name__ == "__main__":
    main()

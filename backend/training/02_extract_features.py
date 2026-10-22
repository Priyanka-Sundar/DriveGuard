"""DriveGuard dataset step 2: extract the 16-cue stream from each raw video.

For every planned subject-state this worker:
  1. ensures the raw video exists (downloads it first if missing),
  2. runs MediaPipe FaceLandmarker over the middle segment at 30 Hz,
  3. computes the causal 16-cue stream (see feature_extraction.CueEngine),
  4. saves one .npz per subject-state and deletes the raw video.

Fully resumable and time-budget aware, so it can be run repeatedly until all
items are processed. Raw videos are deleted after extraction to bound disk
usage; nothing is modified inside the source archives.

Usage:
  python 02_extract_features.py --max-seconds 540
"""
import argparse
import json
import os
import threading
import time

import numpy as np

from cue_config import STATE_NAMES
from feature_extraction import extract_video
import importlib
dl = importlib.import_module("01_download_dataset")

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.normpath(os.path.join(HERE, "..", "data", "dataset", "raw"))
OUT = os.path.normpath(os.path.join(HERE, "..", "data", "dataset", "processed"))


def out_npz(item):
    return os.path.join(OUT, f"utarldd_{item['fold']}_s{item['subject']}_{item['state']}.npz")


def find_video(item):
    for ext in (".mp4", ".mov", ".avi", ".mkv"):
        p = os.path.join(RAW, f"utarldd_{item['fold']}_s{item['subject']}_{item['state']}{ext}")
        if os.path.exists(p):
            return p
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-seconds", type=float, default=1e9)
    ap.add_argument("--seg-s", type=float, default=150.0)
    args = ap.parse_args()

    os.makedirs(RAW, exist_ok=True)
    os.makedirs(OUT, exist_ok=True)
    plan = json.load(open(os.path.join(HERE, "00_plan.json")))
    items = plan["items"]
    manifest = dl.load_manifest()
    t0 = time.time()

    # prefetch thread walks ahead of the main cursor and downloads next items
    lock = threading.Lock()
    state = {"cursor": 0, "stop": False, "inflight": None}
    dead_folds = set()  # archives whose download quota is exhausted this run

    # pre-probe every fold once so quota-dead archives are skipped without retries
    for fold, fid in dl.FOLD_IDS.items():
        if not any(it["fold"] == fold for it in items):
            continue
        try:
            dl.fetch_range(dl.url_for(fid), 0, 1023)
        except Exception:
            dead_folds.add(fold)
    if dead_folds:
        print(f"[probe] quota-dead this run: {sorted(dead_folds)}")

    def prefetch():
        i = 0
        while not state["stop"]:
            with lock:
                if state["stop"]:
                    return
                # advance past items already extracted or being processed
                while i < len(items) and (os.path.exists(out_npz(items[i])) or i <= state["cursor"]
                                          or items[i]["fold"] in dead_folds):
                    i += 1
                if i >= len(items):
                    return
                item, idx = items[i], i
                i += 1
                state["inflight"] = idx
            try:
                dl.download_item(manifest, item, RAW)
            except RuntimeError as e:
                if "non-206" in str(e):
                    with lock:
                        dead_folds.add(item["fold"])
                    print(f"  [quota] {item['fold']} quota exhausted; skipping its remaining items this run")
                else:
                    print(f"  [prefetch error] {item['fold']}/{item['subject']}/{item['state']}: {e}")
                time.sleep(1)
            except Exception as e:
                print(f"  [prefetch error] {item['fold']}/{item['subject']}/{item['state']}: {e}")
                time.sleep(1)
            finally:
                with lock:
                    state["inflight"] = None

    th = threading.Thread(target=prefetch, daemon=True)
    th.start()

    # lazily create the FaceLandmarker (mediapipe import is heavy)
    landmarker = None

    def get_landmarker():
        nonlocal landmarker
        if landmarker is None:
            import mediapipe as mp
            from mediapipe.tasks import python as mp_tasks
            from mediapipe.tasks.python import vision
            model = os.path.normpath(os.path.join(HERE, "..", "data", "model", "face_landmarker.task"))
            if not os.path.exists(model):  # zip dedup: fall back to the bundled frontend copy
                model = os.path.normpath(os.path.join(HERE, "..", "..", "frontend", "dist", "models", "mediapipe", "face_landmarker.task"))
            opts = vision.FaceLandmarkerOptions(
                base_options=mp_tasks.BaseOptions(model_asset_path=model),
                running_mode=vision.RunningMode.VIDEO, num_faces=1)
            landmarker = vision.FaceLandmarker.create_from_options(opts)
        return landmarker

    summary_path = os.path.join(OUT, "_summary.json")
    summary = {"files": {}}
    if os.path.exists(summary_path):
        summary = json.load(open(summary_path))

    done = 0
    for idx, item in enumerate(items):
        if time.time() - t0 > args.max_seconds:
            print(f"[budget] stopping after {done} extractions this run")
            break
        if os.path.exists(out_npz(item)):
            done += 1
            continue
        video = find_video(item)
        waited = 0.0
        while video is None and waited < 900:
            with lock:
                state["cursor"] = idx
            time.sleep(2)
            waited += 2
            video = find_video(item)
            if video is None and waited > 30:  # prefetcher hasn't reached it; download inline
                with lock:
                    inflight = state["inflight"]
                if inflight == idx:
                    pass  # prefetcher is actively fetching this item; keep waiting
                elif item["fold"] in dead_folds:
                    break
                else:
                    print(f"  [inline download] {item['fold']}/{item['subject']}/{item['state']}")
                    try:
                        video = dl.download_item(manifest, item, RAW)
                    except RuntimeError as e:
                        if "non-206" in str(e):
                            with lock:
                                dead_folds.add(item["fold"])
                            print(f"  [quota] {item['fold']} quota exhausted; skipping its remaining items this run")
                        else:
                            print(f"  [download error] {e}")
                        break
                    except Exception as e:
                        print(f"  [download error] {e}")
                        break
        if video is None:
            print(f"  [skip] no video for {item}")
            continue
        t1 = time.time()
        print(f"[extract] {item['fold']}/s{item['subject']}/{item['state']} "
              f"({STATE_NAMES[item['state']]})")
        try:
            res = extract_video(video, get_landmarker(), seg_s=args.seg_s)
        except Exception as e:
            print(f"  [extract error] {e}")
            continue
        meta = {
            "dataset": "UTA-RLDD",
            "fold": item["fold"], "subject": str(item["subject"]),
            "state_code": item["state"], "state": STATE_NAMES[item["state"]],
            "split": item["split"],
            "source_member": f"{item['fold']}/{item['subject']}/{item['state']}",
            "video_bytes": os.path.getsize(video),
            "extracted_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            **res["meta"],
        }
        np.savez_compressed(
            out_npz(item), cues=res["cues"], t_ms=res["t_ms"], face_ok=res["face_ok"],
            meta=json.dumps(meta))
        summary["files"][os.path.basename(out_npz(item))] = meta
        json.dump(summary, open(summary_path, "w"), indent=1)
        os.remove(video)
        done += 1
        print(f"  ok: {res['cues'].shape[0]} frames, face={res['meta']['face_coverage']*100:.1f}%, "
              f"blinks={res['meta']['blinks_detected']}, microsleeps={res['meta']['microsleeps_detected']}, "
              f"({time.time()-t1:.0f}s)")

    with lock:
        state["stop"] = True
    n_total = len(items)
    n_done = sum(1 for it in items if os.path.exists(out_npz(it)))
    print(f"[progress] {n_done}/{n_total} subject-states complete")
    if n_done == n_total:
        print("[progress] ALL DONE")


if __name__ == "__main__":
    main()

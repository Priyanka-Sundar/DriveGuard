"""DriveGuard step 3: train the temporal DWS-CNN (whitepaper Table 4.1) and export ONNX.

Modes:
  deploy -- train the artifact that ships with the application.
           train = 3 subjects, window-disjoint val for early stopping,
           subject-disjoint test for reported metrics.
  loso   -- leave-one-subject-out cross-validation (one fold per subject);
           pooled confusion + per-fold metrics give the honest small-n
           generalization estimate reported in the model card.

All metrics in metadata.json are real measurements from these runs.

Usage: python 03_train_model.py [--mode deploy|loso]
"""
import argparse
import glob
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from cue_config import (CUE_NAMES, LEVEL_NAMES, SCORE_ANCHOR_KNOTS, STATE_TO_CLASS,
                         TARGET_FPS, WINDOW_STRIDE, WINDOW_T, perclos_to_score)

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.normpath(os.path.join(HERE, "..", "data"))
MODEL_DIR = os.path.join(DATA, "model")
ASSETS = os.path.join(MODEL_DIR, "assets")
WARMUP_S = 60.0           # trailing-window statistics need 60 s to fill
MIN_FACE_FRAC = 0.90      # windows below this face coverage are dropped
CLASS_CENTER = {0: 12.0, 1: 42.0, 2: 78.0}  # score prior per labelled state
SCORE_BLEND = 0.5         # target = blend * perclos_anchor + (1-blend) * class_center
VAL_FRAC = 0.2            # window-disjoint early-stopping split (intra-subject)


# ---------------------------------------------------------------- windows
def load_streams():
    files = sorted(glob.glob(os.path.join(DATA, "dataset", "processed", "utarldd_*.npz")))
    streams = []
    for f in files:
        d = np.load(f, allow_pickle=True)
        meta = json.loads(str(d["meta"]))
        subject = f"{meta['fold']}:s{meta['subject']}"
        # subject invariance: normalize ocular cues by the session's open-eye
        # reference (mirrors the 30 s calibration wizard / rolling P95 auto-ref)
        ear_ref = float(meta.get("ear_ref_final", 0.0)) or 0.30
        cues_n = d["cues"].astype(np.float32).copy()
        cues_n[:, [0, 1, 2, 15]] /= max(ear_ref, 1e-3)
        streams.append({"path": f, "cues": cues_n, "face_ok": d["face_ok"], "meta": meta,
                        "cls": STATE_TO_CLASS[meta["state_code"]],
                        "subject": subject})
    return streams


def make_windows(stream):
    cues, face = stream["cues"], stream["face_ok"]
    warm = int(WARMUP_S * TARGET_FPS)
    n = cues.shape[0]
    out = []
    for end in range(WINDOW_T, n + 1, WINDOW_STRIDE):
        start = end - WINDOW_T
        if start < warm:
            continue
        if face[start:end].mean() < MIN_FACE_FRAC:
            continue
        out.append((start, end))
    return out


def build_windows(streams, subjects=None):
    X, y, anchors, src = [], [], [], []
    for st in streams:
        if subjects is not None and st["subject"] not in subjects:
            continue
        for (a, b) in make_windows(st):
            X.append(st["cues"][a:b])
            y.append(st["cls"])
            anchors.append(perclos_to_score(float(st["cues"][b - 1, 8])))
            src.append((st["subject"], st["meta"]["state"], a, b))
    return (np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.int64),
            np.asarray(anchors, dtype=np.float32), src)


# ---------------------------------------------------------------- model
class DWSBlock(nn.Module):
    def __init__(self, cin, cout, k=5, pool=False):
        super().__init__()
        self.dw = nn.Conv1d(cin, cin, k, padding=k // 2, groups=cin)
        self.pw = nn.Conv1d(cin, cout, 1)
        self.bn = nn.BatchNorm1d(cout)
        self.pool = nn.MaxPool1d(2) if pool else nn.Identity()

    def forward(self, x):
        return self.pool(F.relu(self.bn(self.pw(self.dw(x)))))


class DriveGuardNet(nn.Module):
    """Temporal depthwise-separable CNN, T x F = 64 x 16 (whitepaper Table 4.1)."""

    def __init__(self, n_cues=16, n_classes=3):
        super().__init__()
        self.body = nn.Sequential(
            DWSBlock(n_cues, 32, 5, pool=True),   # 64x16 -> 32x32
            DWSBlock(32, 48, 5, pool=True),       # 32x32 -> 16x48
            DWSBlock(48, 64, 5, pool=False),      # 16x48 -> 16x64
        )
        self.head_cls = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, n_classes))
        self.head_score = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1), nn.Sigmoid())

    def forward(self, x):                    # x: (B, T, F)
        z = self.body(x.transpose(1, 2))     # (B, C, T)
        z = z.mean(dim=2)                    # global average pool -> (B, 64)
        return self.head_cls(z), self.head_score(z).squeeze(-1)


# ---------------------------------------------------------------- training
def focal_loss(logits, target, gamma=2.0):
    logp = F.log_softmax(logits, dim=1)
    p = logp.gather(1, target.unsqueeze(1)).squeeze(1)
    with torch.no_grad():
        w = torch.bincount(target, minlength=3).float().clamp(min=1)
        w = (w.sum() / w).sqrt()
        w = w / w.sum() * 3
    loss = -(1 - p).pow(gamma) * p
    return (loss * w[target]).mean()


def augment(x, std, p=0.5):
    """Vectorized deployment-noise augmentation: gaussian cue noise, frame
    dropout, cue dropout, EAR scale jitter, and targeted dropout of
    subject-fragile blink statistics."""
    B, T, C = x.shape
    out = x + torch.randn_like(x) * (std * 0.08)
    if torch.rand(1).item() < p:
        shift = torch.full((B, T, 1), -1)
        drop = torch.rand(B, T, 1) < 0.10
        idx = (torch.arange(T).view(1, T, 1) + torch.where(drop, shift, torch.zeros_like(shift))).clamp(0, T - 1)
        out = torch.gather(out, 1, idx.expand(B, T, C))
    if torch.rand(1).item() < p:
        bsel = torch.rand(B) < 0.10
        if bsel.any():
            csel = torch.randint(0, C, (B,))
            rows = torch.arange(B)[bsel]
            out[rows, :, csel[rows]] = out[rows, :, csel[rows]].mean(dim=1, keepdim=True)
    if torch.rand(1).item() < 0.8:  # EAR scale jitter (reference mismatch)
        fac = 0.8 + 0.45 * torch.rand(B, 1, 1)
        out[:, :, [0, 1, 2, 15]] = out[:, :, [0, 1, 2, 15]] * fac
    m = torch.rand(B) < 0.5  # blink-rate/duration/gap statistics vary per subject
    for c in (6, 7, 14):
        out[m, :, c] = out[m, :, c].mean(dim=1, keepdim=True)
    if torch.rand(1).item() < 0.4:
        out[:, :, 15] = out[:, :, 15].mean(dim=1, keepdim=True)
    return out


def train_model(Xtr, ytr, ttr, mu, sd, epochs=110, batch=64, seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)

    def norm(X):
        return np.clip((X - mu) / sd, -6.0, 6.0).astype(np.float32)

    tr_full = torch.from_numpy(norm(Xtr))
    ytr_full = torch.from_numpy(ytr)
    ttr_full = torch.from_numpy(ttr)
    # window-disjoint validation split, randomized across subjects (not positional)
    rng = np.random.default_rng(seed)
    perm0 = rng.permutation(len(tr_full))
    nva = max(1, int(len(tr_full) * VAL_FRAC))
    va_idx = torch.from_numpy(perm0[-nva:])
    tr_idx = torch.from_numpy(perm0[:-nva])
    tr, ytr_t, ttr_t = tr_full[tr_idx], ytr_full[tr_idx], ttr_full[tr_idx]
    sd_t = torch.from_numpy(sd)
    model = DriveGuardNet()
    opt = torch.optim.AdamW(model.parameters(), lr=2.5e-3, weight_decay=1e-4)
    hist = []
    best = {"f1": -1.0, "epoch": 0, "state": None}
    for ep in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(len(tr))
        for i in range(0, len(tr), batch):
            idx = perm[i:i + batch]
            logits, score = model(augment(tr[idx], sd_t))
            loss = focal_loss(logits, ytr_t[idx]) + 0.4 * F.l1_loss(score, ttr_t[idx])
            opt.zero_grad(); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            logits, score = model(tr_full[va_idx])
            pred = logits.argmax(1)
            yva_t = ytr_full[va_idx]
            f1s = []
            for c in range(3):
                tp = ((pred == c) & (yva_t == c)).sum().item()
                fp = ((pred == c) & (yva_t != c)).sum().item()
                fn = ((pred != c) & (yva_t == c)).sum().item()
                f1s.append(0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn))
            macro = float(np.mean(f1s))
            vloss = (focal_loss(logits, yva_t) + 0.4 * F.l1_loss(score, ttr_full[va_idx])).item()
        hist.append({"epoch": ep, "train_loss": None, "val_loss": vloss, "val_macro_f1": macro,
                     "val_f1_per_class": f1s})
        if macro >= best["f1"]:
            best = {"f1": macro, "epoch": ep, "state": {k: v.clone() for k, v in model.state_dict().items()}}
        if ep - best["epoch"] >= 18:
            break
    model.load_state_dict(best["state"])
    return model, hist, best


def evaluate(model, X, mu, sd):
    model.eval()

    def norm(Xa):
        return np.clip((Xa - mu) / sd, -6.0, 6.0).astype(np.float32)

    with torch.no_grad():
        logits, score = model(torch.from_numpy(norm(X)))
        prob = F.softmax(logits, 1)
        pred = logits.argmax(1)
    return pred.numpy(), score.numpy() * 100.0, prob.numpy()


def prf(y, pred):
    cm = np.zeros((3, 3), dtype=int)
    for i in range(len(y)):
        cm[y[i], pred[i]] += 1
    per_class = {}
    f1s = []
    for c in range(3):
        tp = cm[c, c]; fp = cm[:, c].sum() - tp; fn = cm[c, :].sum() - tp
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * tp / (2 * tp + fp + fn) if tp else 0.0
        f1s.append(f1)
        per_class[LEVEL_NAMES[c]] = {"precision": round(float(prec), 4), "recall": round(float(rec), 4),
                                     "f1": round(float(f1), 4), "support": int(cm[c].sum())}
    return float((pred == y).mean()), float(np.mean(f1s)), per_class, cm


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["deploy", "loso"], default="deploy")
    args = ap.parse_args()
    torch.set_num_threads(min(8, os.cpu_count() or 2))
    os.makedirs(ASSETS, exist_ok=True)
    streams = load_streams()
    subjects = sorted({s["subject"] for s in streams})
    for st in streams:
        print(f"  {st['subject']:<16} {st['meta']['state']:<14} frames={st['cues'].shape[0]}")

    # ---------------------------------------------------------------- LOSO
    loso_metrics = None
    if args.mode == "loso":
        pooled_y, pooled_pred, pooled_score = [], [], []
        fold_acc, fold_f1 = [], []
        for held in subjects:
            tr_subj = [s for s in subjects if s != held]
            Xtr, ytr, atr, _ = build_windows(streams, set(tr_subj))
            Xte, yte, ate, _ = build_windows(streams, {held})
            mu = Xtr.reshape(-1, 16).mean(0)
            sd = Xtr.reshape(-1, 16).std(0).clip(min=1e-4)
            cctr = np.array([CLASS_CENTER[c] for c in ytr], dtype=np.float32)
            ttr = (SCORE_BLEND * atr + (1 - SCORE_BLEND) * cctr) / 100.0
            model, _, _ = train_model(Xtr, ytr, ttr, mu, sd, seed=42)
            pred, score100, _ = evaluate(model, Xte, mu, sd)
            acc, f1, _, _ = prf(yte, pred)
            print(f"  LOSO fold held-out={held}: n={len(yte)} acc={acc:.3f} macroF1={f1:.3f}")
            fold_acc.append(acc); fold_f1.append(f1)
            pooled_y.extend(yte.tolist()); pooled_pred.extend(pred.tolist())
            pooled_score.extend(score100.tolist())
        pooled_y = np.asarray(pooled_y); pooled_pred = np.asarray(pooled_pred)
        acc, f1, per_class, cm = prf(pooled_y, pooled_pred)
        loso_metrics = {
            "protocol": "leave-one-subject-out (4 folds, subject-disjoint by construction)",
            "folds": [{"held_out_subject": s, "accuracy": round(a, 4), "macro_f1": round(b, 4)}
                      for s, a, b in zip(subjects, fold_acc, fold_f1)],
            "mean_accuracy": round(float(np.mean(fold_acc)), 4),
            "std_accuracy": round(float(np.std(fold_acc)), 4),
            "mean_macro_f1": round(float(np.mean(fold_f1)), 4),
            "std_macro_f1": round(float(np.std(fold_f1)), 4),
            "pooled_accuracy": round(acc, 4), "pooled_macro_f1": round(f1, 4),
            "pooled_per_class": per_class, "pooled_confusion": cm.tolist(),
        }
        print(f"  LOSO mean acc={loso_metrics['mean_accuracy']} mean macroF1={loso_metrics['mean_macro_f1']}")
        with open(os.path.join(MODEL_DIR, "loso_metrics.json"), "w") as f:
            json.dump(loso_metrics, f, indent=1)
        return

    # ---------------------------------------------------------------- deploy
    test_subjects = [os.environ.get("DG_TEST_SUBJECT", "Fold2_part1:s17")]
    train_subjects = [s for s in subjects if s not in test_subjects]
    Xtr, ytr, atr, _ = build_windows(streams, set(train_subjects))
    Xte, yte, ate, src_te = build_windows(streams, set(test_subjects))
    print(f"windows: train={len(Xtr)} test={len(Xte)}")
    mu = Xtr.reshape(-1, 16).mean(0)
    sd = Xtr.reshape(-1, 16).std(0).clip(min=1e-4)

    def norm(Xa):
        return np.clip((Xa - mu) / sd, -6.0, 6.0).astype(np.float32)

    cctr = np.array([CLASS_CENTER[c] for c in ytr], dtype=np.float32)
    ttr = (SCORE_BLEND * atr + (1 - SCORE_BLEND) * cctr) / 100.0
    model, hist, best = train_model(Xtr, ytr, ttr, mu, sd)
    print(f"best val (window-disjoint) macroF1={best['f1']:.3f} @ep{best['epoch']}")

    pred, score100, prob = evaluate(model, Xte, mu, sd)
    acc, macro_f1, per_class, cm = prf(yte, pred)
    ccte = np.array([CLASS_CENTER[c] for c in yte])
    tte = SCORE_BLEND * ate + (1 - SCORE_BLEND) * ccte
    score_mae = float(np.abs(score100 - tte).mean())
    r = float(np.corrcoef(score100, tte)[0, 1]) if len(score100) > 2 else float("nan")
    print(f"TEST (subject-disjoint): acc={acc:.3f} macroF1={macro_f1:.3f} scoreMAE={score_mae:.1f} r={r:.3f}")
    print("confusion (rows=true SAFE/WARNING/CRITICAL):")
    print(cm)

    by_subj = {}
    for subj_state in sorted({f"{s}|{st}" for s, st, _, _ in src_te}):
        m = np.array([f"{s}|{st}" == subj_state for s, st, _, _ in src_te])
        by_subj[subj_state] = {"n": int(m.sum()),
                               "acc": round(float((pred[m] == yte[m]).mean()), 4),
                               "mean_score": round(float(score100[m].mean()), 1)}

    torch.onnx.export(
        model, (torch.from_numpy(norm(Xte[:1])),),
        os.path.join(MODEL_DIR, "driveguard_dwscnn.onnx"),
        input_names=["cue_window"], output_names=["level_logits", "score01"],
        dynamic_axes={"cue_window": {0: "batch"}, "level_logits": {0: "batch"},
                      "score01": {0: "batch"}})

    import onnxruntime as ort
    sess = ort.InferenceSession(os.path.join(MODEL_DIR, "driveguard_dwscnn.onnx"),
                                providers=["CPUExecutionProvider"])
    lg, sc = sess.run(None, {"cue_window": norm(Xte[:8])})
    par = float(np.abs(np.asarray(sc).reshape(-1) * 100.0
                       - evaluate(model, Xte[:8], mu, sd)[1]).max())
    print(f"onnx parity: score maxdiff={par:.5f}")
    ts = []
    for _ in range(200):
        t0 = time.perf_counter()
        sess.run(None, {"cue_window": norm(Xte[:1])})
        ts.append((time.perf_counter() - t0) * 1000)
    lat = float(np.median(ts))
    print(f"onnx latency single window: median={lat:.2f}ms")

    # ---------------- artifacts ----------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.font_manager as fm
    import matplotlib.pyplot as plt
    font_path = os.path.normpath(os.path.join(HERE, "..", "..", "frontend", "public", "fonts",
                                              "PixelifySans-Variable.ttf"))
    if os.path.exists(font_path):
        fm.fontManager.addfont(font_path)
    plt.rcParams["font.family"] = ["Pixelify Sans", "DejaVu Sans"]
    plt.rcParams["axes.edgecolor"] = "#2A2A38"
    plt.rcParams["text.color"] = "#F5F7FA"
    plt.rcParams["axes.labelcolor"] = "#F5F7FA"
    plt.rcParams["xtick.color"] = "#8B8F9A"
    plt.rcParams["ytick.color"] = "#8B8F9A"

    fig, ax = plt.subplots(1, 2, figsize=(9, 3.2), facecolor="#0A0A0F", constrained_layout=True)
    ax[0].plot([h["epoch"] for h in hist], [h["val_loss"] for h in hist], color="#3B82F6", lw=1.5)
    ax[0].set_title("validation loss"); ax[0].set_yscale("log")
    ax[1].plot([h["epoch"] for h in hist], [h["val_macro_f1"] for h in hist], color="#22C55E", lw=1.5)
    ax[1].set_title("validation macro-F1")
    for a in ax:
        a.set_facecolor("#0A0A0F"); a.grid(color="#1E1E2A", lw=0.5)
    fig.savefig(os.path.join(ASSETS, "training_curves.png"), dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(4.2, 3.6), facecolor="#0A0A0F", constrained_layout=True)
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(3), LEVEL_NAMES); ax.set_yticks(range(3), LEVEL_NAMES)
    ax.set_xlabel("predicted"); ax.set_ylabel("true")
    for i in range(3):
        for j in range(3):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="#F5F7FA" if cm[i, j] < cm.max() / 2 else "#050507")
    ax.set_facecolor("#0A0A0F"); fig.colorbar(im)
    fig.savefig(os.path.join(ASSETS, "confusion_matrix.png"), dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(4.6, 3.4), facecolor="#0A0A0F", constrained_layout=True)
    for c, col in ((0, "#22C55E"), (1, "#F59E0B"), (2, "#EF4444")):
        m = yte == c
        if m.any():
            ax.scatter(tte[m], score100[m], s=8, alpha=0.5, color=col, label=LEVEL_NAMES[c])
    ax.plot([0, 100], [0, 100], color="#3B82F6", lw=1, ls="--")
    ax.set_xlabel("target score"); ax.set_ylabel("predicted score")
    ax.set_facecolor("#0A0A0F"); ax.grid(color="#1E1E2A", lw=0.5)
    ax.legend(frameon=False)
    fig.savefig(os.path.join(ASSETS, "score_scatter.png"), dpi=150)
    plt.close(fig)

    import hashlib
    onnx_path = os.path.join(MODEL_DIR, "driveguard_dwscnn.onnx")
    sha = hashlib.sha256(open(onnx_path, "rb").read()).hexdigest()
    n_params = sum(p.numel() for p in model.parameters())
    n_body = sum(p.numel() for p in model.body.parameters())
    loso = {}
    loso_path = os.path.join(MODEL_DIR, "loso_metrics.json")
    if os.path.exists(loso_path):
        loso = json.load(open(loso_path))
    meta = {
        "model": "DriveGuard DWS-CNN v1.0",
        "task": "driver drowsiness level (3-class) + continuous 0-100 score",
        "input": {"shape": [WINDOW_T, len(CUE_NAMES)], "cue_names": CUE_NAMES,
                  "frame_rate_hz": TARGET_FPS,
                  "normalization": "ear cues (0,1,2,15) divided by session ear_ref (open-eye P95, default 0.30), then z-score with train-set statistics, clipped to +/-6"},
        "norm": {"mean": [round(float(v), 5) for v in mu], "std": [round(float(v), 5) for v in sd]},
        "classes": LEVEL_NAMES,
        "state_map": {"alert": "SAFE", "low_vigilant": "WARNING", "drowsy": "CRITICAL"},
        "score_target": {
            "definition": f"{SCORE_BLEND} * perclos_anchor(perclos_60) + {1-SCORE_BLEND} * class_center",
            "class_center": CLASS_CENTER, "perclos_anchor_knots": SCORE_ANCHOR_KNOTS,
            "note": "PERCLOS anchor provides within-class gradation; labelled state centres keep "
                    "the continuous score consistent for drowsiness that shows posturally "
                    "rather than ocularly (eyes-open drowsiness)."},
        "architecture": "3x depthwise-separable 1-D conv blocks (16-32-48-64, k=5) + BN + ReLU "
                        "+ maxpool(2) on first two, global average pooling, dual head "
                        "(FC 64-32-3 softmax level, FC 64-32-1 sigmoid score x100)",
        "parameters": {"total": n_params, "conv_body": n_body},
        "training": {"optimizer": "AdamW lr=2.5e-3 wd=1e-4", "batch": 64, "epochs_run": len(hist),
                     "best_epoch": best["epoch"], "loss": "focal CE (gamma=2) + 0.4 * L1(score)",
                     "augmentation": "gaussian cue noise, frame dropout, cue dropout, EAR scale "
                                     "jitter, targeted dropout of subject-fragile blink statistics",
                     "warmup_s": WARMUP_S, "window_stride_s": WINDOW_STRIDE / TARGET_FPS,
                     "early_stopping": "window-disjoint 20% validation split (intra-subject)"},
        "dataset": {
            "name": "UTA-RLDD",
            "citation": "Ghoddoosian, Galib, Athitsos (2019), arXiv:1904.07312",
            "subjects": subjects,
            "subjects_train": train_subjects, "subjects_test": test_subjects,
            "windows": {"train": int(len(Xtr)), "test": int(len(Xte))},
            "segments_s": 150,
            "notes": "subject-disjoint test; subject 27 contributes alert+low-vigilant only; "
                     "acquisition limited by per-archive download quotas on the public "
                     "distribution -- see model card limitations"},
        "metrics": {
            "test_accuracy": round(acc, 4), "test_macro_f1": round(macro_f1, 4),
            "per_class": per_class, "confusion_matrix": cm.tolist(),
            "score_mae_vs_target": round(score_mae, 2), "score_pearson_r": round(r, 4),
            "per_subject_state_test": by_subj,
            "onnx_latency_ms_single_window_cpu": round(lat, 2),
            "val_macro_f1_best": round(best["f1"], 4),
            **({"loso_cross_validation": loso} if loso else {}),
        },
        "export": {"format": "ONNX", "sha256": sha,
                   "onnx_parity_max_abs_diff": round(par, 6),
                   "runtime": f"onnxruntime {ort.__version__}",
                   "trained_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")},
    }
    with open(os.path.join(MODEL_DIR, "metadata.json"), "w") as f:
        json.dump(meta, f, indent=1)
    json.dump(hist, open(os.path.join(ASSETS, "training_history.json"), "w"))
    print("saved model + metadata + assets")


if __name__ == "__main__":
    main()

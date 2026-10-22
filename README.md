# DriveGuard

**Edge-deployable driver drowsiness detection — real dataset, real trained model, real telemetry.**

DriveGuard watches a single front-facing camera, estimates driver fatigue on a
continuous 0–100 scale anchored to PERCLOS ocular metrics, and raises graduated
warnings (SAFE / WARNING / CRITICAL) with per-cue attribution. Face video is
processed on-device; only derived numeric cues are sent to the inference API.

- Theme: near-black UI, Pixelify Sans, blue actions, red alerts, green status
- Architecture: **Pattern B** — dataset + model served through a FastAPI backend
- Contract: versioned 1 Hz telemetry object (see `backend/training/cue_config.py`)

---

## Deploy to Vercel

The repository deploys to Vercel as a single project: the website is built from
`frontend/` and served statically, and every `/api/*` route runs on one Python
Serverless Function (`api/index.py`) with the trained model bundled in
`api/_data/`. See **DEPLOY_VERCEL.md** for the step-by-step guide (GitHub push
→ import in Vercel → done). `bash vercel-prepare.sh` restores the dev assets
the zip deduplicates before the first push.

## Quick start

### 1. Backend (FastAPI + real model)

```bash
cd backend
python -m venv .venv && source .venv/bin/activate   # Python 3.11+
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8000
```

The backend ships with the trained ONNX artifact, extracted dataset features,
and real telemetry exports under `backend/data/` — it runs immediately:

- API: `http://localhost:8000/api/health`
- If `frontend/dist/` exists (it does), the whole app is served at
  `http://localhost:8000/`

> Camera access requires `localhost` or HTTPS in production.

### Demo videos

The demo page ships with three 60-second clips cut from a real UTA-RLDD test
recording (driver 17 — alert / low-vigilant / drowsy), bundled under
`frontend/dist/demo/`. They are played through the exact on-device pipeline
used for live monitoring. The clips are trimmed excerpts for demonstration
purposes; the dataset and its license belong to the dataset authors
(<https://sites.google.com/view/utarldd/home>).

### 2. Frontend (development mode)

```bash
cd frontend
npm install
npm run dev        # http://localhost:5173 (proxies /api to :8000)
```

Production build: `npm run build` (output served by the backend).

> **Dev-mode vision assets**: the archive deduplicates the 36MB MediaPipe
> bundle and ships it once under `frontend/dist/models/`. `run.sh` restores
> `frontend/public/models/` automatically; manually:
> `cp -r frontend/dist/models frontend/public/`

### One-command start

```bash
bash run.sh        # starts backend, then frontend dev server
```

---

## Routes

| Route | Purpose |
|---|---|
| `/` | Home — hero, Capture → Analyze → Warn, feature grid, privacy strip |
| `/monitor` | Live monitor — camera, 0–100 gauge, level, confidence, cue attribution, alarm banner, measured FPS/latency, pause/end, FACE_LOST state, demo fallback |
| `/calibrate` | 30-second neutral-baseline wizard (personal EAR reference + blink prior) |
| `/session/:id` | Session dashboard — score timeline with alarm bands, cue trajectories, export/delete |
| `/history` | Trip history — local session records, filters, hour-of-day heatmap |
| `/demo` | **Video demo** — a real UTA-RLDD driver recording plays while the live pipeline monitors it side by side; random raw input → output (real test windows through the live API); telemetry replay player |
| `/model` | Model card — architecture, dataset, measured metrics, LOSO, limitations |
| `/settings` | Sensitivity profiles, alarms, privacy, local data, system info |

## Backend API

| Endpoint | Purpose |
|---|---|
| `GET /api/health` | Readiness + component status |
| `GET /api/dataset/status` | Real dataset inspection (raw scan + processed stats) |
| `POST /api/dataset/upload` | Accept a real dataset file (never synthesizes) |
| `GET /api/model/status` | Trained artifact status (sha256, shape, runtime) |
| `GET /api/model/metrics` | Real measured metrics + training metadata |
| `POST /api/predict` | Inference on a 64×16 cue window (+ optional `ear_ref`) |
| `GET /api/demo/random` | One random real held-out window with ground truth |
| `GET /api/telemetry/sessions[/{id}]` | Real telemetry exports |
| `GET /api/telemetry/latest` | Newest export |

## The real pipeline

```
camera frame → MediaPipe FaceLandmarker (browser, on-device)
            → 16 physiological cues @30 Hz (EAR, MAR, PERCLOS, blink events,
              microsleeps, head pose, nods — causal streaming engine)
            → 64×16 window ──POST──▶ FastAPI
                                   → ear_ref normalization → z-score → DWS-CNN (ONNX)
            ◀── score / level / confidence / cues_fired ──┘
            → alarm policy (Table 3.3: hysteresis + confirmation windows)
            → 1 Hz telemetry → local IndexedDB
```

## Dataset & model (honest provenance)

- **Dataset**: UTA-RLDD (Real-Life Drowsiness Dataset), Ghoddoosian, Galib &
  Athitsos, 2019 (arXiv:1904.07312). Real subject videos obtained from the
  authors' public distribution; raw media is **not** redistributed here — only
  derived numeric cue features and a source manifest.
- **Model**: temporal depthwise-separable CNN (whitepaper Table 4.1), dual head
  (3-level softmax + sigmoid score), trained subject-disjoint, exported to ONNX.
- **Metrics**: see `/model` in the app or `GET /api/model/metrics`. Measured
  values only; limitations stated explicitly. The acquisition run was limited
  by per-archive download quotas on the public distribution — the subject count
  is small and cross-subject generalization is the dominant error source.

### Reproducing the training pipeline

```bash
cd backend/training
python 01_download_dataset.py     # fetch selected UTA-RLDD subjects (needs internet)
python 02_extract_features.py     # MediaPipe FaceLandmarker → 16-cue streams
python 03_train_model.py --mode deploy   # train + export ONNX + metrics
python 03_train_model.py --mode loso     # leave-one-subject-out cross-validation
python 04_export_telemetry.py     # real telemetry sessions + demo windows
```

Training dependencies (PyTorch CPU, mediapipe, matplotlib) are separate from
the runtime: `pip install torch --index-url https://download.pytorch.org/whl/cpu
mediapipe matplotlib scikit-learn`. On headless systems mediapipe needs
`libEGL1` (Debian/Ubuntu: `sudo apt install libegl1`). Feature extraction
resolves `face_landmarker.task` from `backend/data/model/`, falling back to the
bundled `frontend/dist/models/mediapipe/` copy.

## Project layout

```
DriveGuard/
├── backend/
│   ├── main.py                  # FastAPI app (all endpoints)
│   ├── dataset_loader.py        # real-file inspection, no synthesis
│   ├── model_loader.py         # ONNX session + explainability rules
│   ├── requirements.txt
│   ├── data/
│   │   ├── dataset/raw/         # place real media here (gitignored)
│   │   ├── dataset/processed/   # extracted features (real, included)
│   │   ├── model/               # ONNX artifact + metadata + assets
│   │   └── telemetry/           # real telemetry exports
│   └── training/                # 01–04 pipeline scripts + shared constants
├── frontend/
│   ├── src/                     # React + TS + Tailwind + Framer Motion + Recharts
│   ├── public/demo/             # demo video clips (restored by run.sh from dist/)
│   ├── public/models/mediapipe/ # dev vision assets (restored by run.sh from dist/)
│   └── dist/                    # prebuilt production bundle + bundled models + demo clips
├── api/
│   ├── index.py                 # Vercel Serverless Function (same API contract as backend/)
│   └── _data/                   # model + features + telemetry bundled for the function
├── docs/                        # screenshots + presentation
├── vercel.json                  # build + routes for the Vercel deployment
├── vercel-prepare.sh            # restores public/{models,demo} before the first git push
├── run.sh
└── README.md
```

## Privacy

Face video is sensitive biometric data. DriveGuard processes every frame
locally in a transient buffer — no storage, no upload. Only numeric cues are
transmitted for inference; session records stay in the browser's IndexedDB.
Delete-all removes every stored session and local key.

## License notes

- Code: MIT-style for this evaluation build.
- Model weights: trained by this project on UTA-RLDD; research use.
- UTA-RLDD videos: property of the dataset authors; obtain from
  <https://sites.google.com/view/utarldd/home> and cite arXiv:1904.07312.
- MediaPipe FaceLandmarker: Apache-2.0 (Google); Pixelify Sans: OFL (Google Fonts).

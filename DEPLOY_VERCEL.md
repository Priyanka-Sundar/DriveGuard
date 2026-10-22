# Deploying DriveGuard to Vercel

This repository deploys to Vercel as **one project with two parts**:

| Part | What serves it | How |
|------|----------------|-----|
| The website (React + Vite) | Vercel static CDN | Built from `frontend/` on every deploy |
| The API (`/api/…`) | One Python Serverless Function | `api/index.py` (FastAPI) + the bundled model in `api/_data/` |

No other services are needed. The trained ONNX model, the extracted dataset
features, and the telemetry replays are all bundled inside the function, so the
deployed site is fully self-contained. The local `bash run.sh` deployment keeps
working unchanged alongside this.

---

## Step 1 — Push the repository to GitHub

The zip deduplicates two folders to stay small (`frontend/public/models` and
`frontend/public/demo` are byte-identical mirrors of `frontend/dist/models` and
`frontend/dist/demo`). Restore them first, then push:

```bash
# after unzipping DriveGuard_Project.zip:
cd DriveGuard

bash vercel-prepare.sh        # restores frontend/public/{models,demo} from dist

git init
git add .
git commit -m "DriveGuard: real-time driver drowsiness monitoring"
git branch -M main

# create an EMPTY repo on github.com first (no README, no .gitignore),
# named for example "driveguard", then:
git remote add origin https://github.com/<your-username>/driveguard.git
git push -u origin main
```

Notes:
- `vercel-prepare.sh` is idempotent — running it twice changes nothing.
- All files are well under GitHub's 100 MB per-file limit (the largest is a
  ~10 MB MediaPipe wasm). The full push is roughly 70–80 MB.

## Step 2 — Import the project in Vercel

With your GitHub account already linked in Vercel:

1. **vercel.com → Add New… → Project**
2. In **Import Git Repository**, find `driveguard` and click **Import**.
3. Configure the project (this is the part that must be right):

   | Setting | Value |
   |---------|-------|
   | Framework Preset | **Other** |
   | Root Directory | leave at the repository root (`./`) — do **not** set it to `frontend/` |
   | Build Command | leave empty — provided by `vercel.json` |
   | Output Directory | leave empty — provided by `vercel.json` |
   | Install Command | leave default |

   The `vercel.json` in the repository root already contains the correct
   build (`cd frontend && npm install && npm run build`), the static output
   (`frontend/dist`), the serverless function (`api/index.py`), and the two
   rewrites (`/api/*` → function, everything else → `index.html` for the
   client-side routes). You should not have to type any command yourself.
4. Click **Deploy** and wait about two minutes (npm install + Vite build +
   Python function bundling).

## Step 3 — Verify

Open the deployment URL (`https://driveguard-xxxx.vercel.app`):

- the site loads with the DriveGuard home page;
- **Model Card** shows the measured metrics (44.5 % accuracy on the new driver,
  0.382 macro-F1 — identical numbers to the local deployment);
- **Demo → Random input from dataset** returns a real held-out window and a
  prediction;
- **Demo → Dataset video** plays the real UTA-RLDD footage and monitors it live;
- direct check: `https://<your-deployment>.vercel.app/api/health` returns
  `"status": "ok"` with the model loaded.

## How the routes resolve on Vercel

```
request                     → served by
─────────────────────────────────────────────────────────────
/assets/*, /models/*,
/demo/*, /fonts/*, files    → static files built from frontend/ (CDN)
/api/*                      → the Python function in api/index.py
/monitor, /history, …       → index.html (client-side routing)
```

The function in `api/index.py` is the same contract as `backend/main.py`:
same normalization, same ONNX session, same responses — verified field by
field against the local backend before shipping.

## Differences vs. the local deployment

- **Cold starts.** The first `/api` request after idle loads Python +
  onnxruntime + the model (a few seconds); subsequent requests are fast.
  The Monitor page reconnects automatically — nothing to do.
- **`/api/dataset/upload` is disabled** (serverless filesystems are
  read-only). The endpoint answers 501 with a note. The local `run.sh`
  deployment keeps the full upload capability.
- **The raw dataset media is never uploaded anyway** (license), same as local.

## Updating the deployment

Every `git push` to `main` redeploys automatically. After retraining the model
(`backend/training/03_train_model.py`), refresh the bundle copies:

```bash
cp backend/data/model/driveguard_dwscnn.onnx      api/_data/model/
cp backend/data/model/driveguard_dwscnn.onnx.data api/_data/model/
cp backend/data/model/metadata.json               api/_data/model/
cp backend/data/model/assets/*                    api/_data/model/assets/
cp backend/data/dataset/processed/*.npz          api/_data/dataset/processed/
cp backend/data/telemetry/replay-*.json           api/_data/telemetry/
git add . && git commit -m "retrained model" && git push
```

## Costs and limits (Hobby plan)

The free tier comfortably runs this project: the static traffic is tiny and
function invocations are far below the monthly allowance. The total static
output is ~39 MB and the function bundle a few MB — both well inside Vercel's
limits. No environment variables are needed.

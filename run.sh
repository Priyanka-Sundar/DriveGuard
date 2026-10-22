#!/usr/bin/env bash
# DriveGuard one-command start: backend (FastAPI) + frontend (Vite dev).
set -e
cd "$(dirname "$0")"

# Restore dev-mode assets. The zip deduplicates the 36MB MediaPipe bundle and the
# demo video clips (public/ mirrors are byte-identical to dist/); run.sh puts
# them back for dev mode.
if [ ! -d frontend/public/models ] && [ -d frontend/dist/models ]; then
  cp -r frontend/dist/models frontend/public/
  echo "restored frontend/public/models (dev vision assets)"
fi
if [ ! -d frontend/public/demo ] && [ -d frontend/dist/demo ]; then
  cp -r frontend/dist/demo frontend/public/
  echo "restored frontend/public/demo (demo video clips)"
fi

# backend
cd backend
if [ ! -d .venv ]; then
  python3 -m venv .venv
  .venv/bin/pip install -q -U pip
  .venv/bin/pip install -q -r requirements.txt
fi
(.venv/bin/uvicorn main:app --port 8000 > /tmp/driveguard-backend.log 2>&1 &)
echo "backend  → http://localhost:8000  (log: /tmp/driveguard-backend.log)"
cd ..

# frontend
cd frontend
[ -d node_modules ] || npm install
echo "frontend → http://localhost:5173"
npm run dev

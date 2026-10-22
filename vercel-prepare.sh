#!/usr/bin/env bash
# DriveGuard → GitHub → Vercel preparation.
#
# The distribution zip deduplicates the 36 MB MediaPipe bundle and the demo
# video clips: frontend/public/{models,demo} are byte-identical mirrors of
# frontend/dist/{models,demo} and only the dist copies ship in the zip.
# Vercel rebuilds the frontend from source, and Vite copies frontend/public/
# into the fresh build -- so the public copies must exist before you push.
# This script restores them. Idempotent: safe to run any number of times.
set -e
cd "$(dirname "$0")"

if [ ! -d frontend/dist/models ]; then
  echo "ERROR: frontend/dist/models not found -- extract the full zip first." >&2
  exit 1
fi
if [ ! -d frontend/public/models ]; then
  cp -r frontend/dist/models frontend/public/
  echo "restored frontend/public/models (MediaPipe vision assets)"
else
  echo "frontend/public/models already present"
fi
if [ ! -d frontend/public/demo ] && [ -d frontend/dist/demo ]; then
  cp -r frontend/dist/demo frontend/public/
  echo "restored frontend/public/demo (demo video clips)"
else
  echo "frontend/public/demo already present"
fi
echo "OK -- the repository is ready to push."

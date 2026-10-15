#!/usr/bin/env bash
# Start the inpainting website: builds the frontend if needed, then serves the
# FastAPI backend and the built site together on one port.
#
#   ./start.sh            # http://localhost:8000
#   ./start.sh --dev      # backend on :8000 + Vite dev server with hot reload on :5173
#   PORT=9000 HOST=0.0.0.0 ./start.sh   # other port / listen on the LAN
#
# Environment variables understood by the backend are listed in backend/app/config.py.
set -euo pipefail
cd "$(dirname "$0")"

PORT="${PORT:-8000}"
HOST="${HOST:-127.0.0.1}"
PY="${PYTHON_BIN:-${VIRTUAL_ENV:+$VIRTUAL_ENV/bin/python}}"
if [ -z "$PY" ] || [ ! -x "$PY" ]; then
  if [ -x ".venv/bin/python" ]; then
    PY=".venv/bin/python"
  else
    PY="$(command -v python3 || command -v python || echo .venv/bin/python)"
  fi
fi

# Prefer a Linux Node.js (the Windows npm on the WSL PATH does not work here).
for d in "$HOME"/.local/opt/node-v*-linux-x64/bin; do [ -x "$d/node" ] && PATH="$d:$PATH"; done
export PATH

[ -x "$PY" ] || { echo "Missing $PY - create the venv and install requirements.txt first (see README)." >&2; exit 1; }
"$PY" -c "import fastapi, uvicorn, multipart" 2>/dev/null || {
  echo "Backend packages missing: $PY -m pip install -r requirements.txt" >&2; exit 1; }
command -v node >/dev/null && node -v >/dev/null 2>&1 || {
  echo "Node.js (Linux build) is required for the frontend; see README." >&2; exit 1; }

if [ ! -d frontend/node_modules ]; then
  (cd frontend && npm ci --no-fund --no-audit)
fi

if [ "${1:-}" = "--dev" ]; then
  "$PY" -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000 --reload --reload-dir backend &
  BACK=$!
  trap 'kill $BACK 2>/dev/null' EXIT
  echo "Backend: http://127.0.0.1:8000   Frontend (dev): http://localhost:5173"
  (cd frontend && npx vite --port 5173)
  exit
fi

# Rebuild when any frontend source is newer than the last build.
if [ ! -f frontend/dist/index.html ] || [ -n "$(find frontend/src frontend/index.html -newer frontend/dist/index.html -print -quit)" ]; then
  echo "Building frontend..."
  (cd frontend && npm run build)
fi

echo "Starting on http://localhost:${PORT}  (Ctrl+C to stop)"
exec "$PY" -m uvicorn backend.app.main:app --host "$HOST" --port "$PORT"

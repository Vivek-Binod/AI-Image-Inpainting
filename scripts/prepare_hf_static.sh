#!/usr/bin/env bash
# Assemble the free static Hugging Face Space (models run in the browser) in build/hf-static:
#   ONNX export -> static frontend build -> copy + Space README.
# Upload:  .venv/bin/hf upload <user>/<space> build/hf-static . --repo-type space
set -euo pipefail
cd "$(dirname "$0")/.."
NODE_BIN="${NODE_BIN:-$HOME/.local/opt/node-v24.21.0-linux-x64/bin}"
export PATH="$NODE_BIN:$PATH"
CUDA_VISIBLE_DEVICES= .venv/bin/python scripts/export_onnx.py
(cd frontend && npm run build:static)
rm -rf build/hf-static
cp -r frontend/dist-static build/hf-static
cp deploy/huggingface-static/README.md build/hf-static/README.md
du -sh build/hf-static

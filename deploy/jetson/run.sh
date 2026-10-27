#!/usr/bin/env bash
set -euo pipefail
project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$project_dir"
export PYTHONUNBUFFERED=1
export YOLO_AUTOINSTALL=false
export YOLOv5_AUTOINSTALL=false
exec "$project_dir/.venv/bin/python" -m uvicorn weed_vision.main:app \
  --host "${WEED_HTTP_HOST:-127.0.0.1}" --port "${WEED_HTTP_PORT:-8000}" --workers 1

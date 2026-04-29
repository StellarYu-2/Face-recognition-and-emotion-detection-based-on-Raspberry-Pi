#!/usr/bin/env bash
set -euo pipefail

device="${1:-/dev/video0}"

echo "[camera] device: $device"

if ! command -v v4l2-ctl >/dev/null 2>&1; then
  echo "[camera] v4l2-ctl not found. Install with:"
  echo "  sudo apt install -y v4l-utils"
  exit 1
fi

echo
echo "[camera] current format:"
v4l2-ctl -d "$device" --get-fmt-video || true

echo
echo "[camera] current stream params:"
v4l2-ctl -d "$device" --get-parm || true

echo
echo "[camera] supported formats:"
v4l2-ctl -d "$device" --list-formats-ext

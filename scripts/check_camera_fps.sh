#!/usr/bin/env bash
set -euo pipefail

device="${1:-/dev/video0}"
width="${2:-320}"
height="${3:-240}"
fps="${4:-30}"
format="${5:-YUYV}"
count="${6:-180}"

echo "[camera-fps] device=$device size=${width}x${height} fps=$fps format=$format count=$count"

if ! command -v v4l2-ctl >/dev/null 2>&1; then
  echo "[camera-fps] v4l2-ctl not found. Install with:"
  echo "  sudo apt install -y v4l-utils"
  exit 1
fi

v4l2-ctl -d "$device" \
  --set-fmt-video="width=${width},height=${height},pixelformat=${format}" \
  --set-parm="$fps" \
  --stream-mmap \
  --stream-count="$count" \
  --stream-to=/dev/null

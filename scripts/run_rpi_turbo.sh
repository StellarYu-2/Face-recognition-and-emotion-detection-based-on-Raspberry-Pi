#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

if [[ ! -x ./build_rpi/asdun_access ]]; then
  echo "[turbo] ./build_rpi/asdun_access not found. Build first:"
  echo "  cmake --build build_rpi -j4"
  exit 1
fi

./scripts/rpi_performance_mode.sh

if command -v vcgencmd >/dev/null 2>&1; then
  echo "[turbo] thermal/throttle status:"
  vcgencmd measure_temp || true
  vcgencmd get_throttled || true
fi

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export OMP_PROC_BIND="${OMP_PROC_BIND:-true}"
export OMP_PLACES="${OMP_PLACES:-cores}"
export OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1

if [[ -z "${DISPLAY:-}" && -z "${WAYLAND_DISPLAY:-}" ]]; then
  if [[ -S /tmp/.X11-unix/X0 ]]; then
    export DISPLAY=:0
    export XAUTHORITY="${XAUTHORITY:-$HOME/.Xauthority}"
    echo "[turbo] DISPLAY was empty; trying local desktop display: $DISPLAY"
  else
    export QT_QPA_PLATFORM="${QT_QPA_PLATFORM:-offscreen}"
    echo "[turbo] no display detected; using QT_QPA_PLATFORM=$QT_QPA_PLATFORM"
    echo "[turbo] no preview window will be visible. Use Ctrl+C to stop, or run from the Pi desktop for UI."
  fi
fi

echo "[turbo] running with OMP_NUM_THREADS=$OMP_NUM_THREADS OMP_PROC_BIND=$OMP_PROC_BIND OMP_PLACES=$OMP_PLACES"
exec ./build_rpi/asdun_access

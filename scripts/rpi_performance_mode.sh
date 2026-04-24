#!/usr/bin/env bash
set -euo pipefail

echo "[perf] switching CPU governor to performance"
for gov in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do
  if [[ -w "$gov" ]]; then
    echo performance > "$gov"
  else
    echo performance | sudo tee "$gov" >/dev/null
  fi
done

echo "[perf] reducing swap tendency"
sudo sysctl -w vm.swappiness=10 >/dev/null

echo "[perf] suggested runtime environment:"
echo "export OMP_NUM_THREADS=4"
echo "export OPENBLAS_NUM_THREADS=1"
echo "export VECLIB_MAXIMUM_THREADS=1"

echo "[perf] done"

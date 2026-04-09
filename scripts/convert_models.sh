#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
MODELS_DIR="${PROJECT_ROOT}/models"

PNNX="${PNNX:-}"
NCNNOPTIMIZE="${NCNNOPTIMIZE:-$HOME/ncnn/build/tools/ncnnoptimize}"

DETECTOR_ONNX="${MODELS_DIR}/version-RFB-320.onnx"
RECOGNIZER_ONNX="${MODELS_DIR}/arcfaceresnet100-8.onnx"
EMOTION_ONNX="${MODELS_DIR}/emotion-ferplus-8.onnx"

DETECTOR_PARAM="${MODELS_DIR}/face_detector.param"
DETECTOR_BIN="${MODELS_DIR}/face_detector.bin"
RECOGNIZER_PARAM="${MODELS_DIR}/face_recognizer.param"
RECOGNIZER_BIN="${MODELS_DIR}/face_recognizer.bin"
EMOTION_PARAM="${MODELS_DIR}/emotion.param"
EMOTION_BIN="${MODELS_DIR}/emotion.bin"

echo "[1/5] Checking prerequisites..."
if [[ -z "${PNNX}" ]]; then
  if command -v pnnx >/dev/null 2>&1; then
    PNNX="$(command -v pnnx)"
  elif [[ -x "$HOME/.venvs/pnnx/bin/pnnx" ]]; then
    PNNX="$HOME/.venvs/pnnx/bin/pnnx"
  elif [[ -x "$HOME/ncnn/tools/pnnx/build/install/bin/pnnx" ]]; then
    PNNX="$HOME/ncnn/tools/pnnx/build/install/bin/pnnx"
  elif [[ -x "$HOME/ncnn/tools/pnnx/build/src/pnnx" ]]; then
    PNNX="$HOME/ncnn/tools/pnnx/build/src/pnnx"
  fi
fi

if [[ ! -x "${PNNX}" ]]; then
  echo "ERROR: pnnx not found at ${PNNX}"
  echo "Hint: run 'pip3 install pnnx' first, or build ncnn/tools/pnnx separately"
  exit 1
fi

for file in "${DETECTOR_ONNX}" "${RECOGNIZER_ONNX}" "${EMOTION_ONNX}"; do
  if [[ ! -f "${file}" ]]; then
    echo "ERROR: missing model file: ${file}"
    exit 1
  fi
done

tmp_dir="${MODELS_DIR}/.pnnx_tmp"
rm -rf "${tmp_dir}"
mkdir -p "${tmp_dir}"

echo "[2/5] Converting detector ONNX -> NCNN with pnnx"
pushd "${tmp_dir}" >/dev/null
"${PNNX}" "${DETECTOR_ONNX}" inputshape="[1,3,240,320]"
popd >/dev/null
mv "${tmp_dir}/version-RFB-320.ncnn.param" "${DETECTOR_PARAM}"
mv "${tmp_dir}/version-RFB-320.ncnn.bin" "${DETECTOR_BIN}"
rm -f "${tmp_dir}/version-RFB-320.pnnx.param" "${tmp_dir}/version-RFB-320.pnnx.bin" "${tmp_dir}/version-RFB-320_pnnx.py" "${tmp_dir}/version-RFB-320_pnnx.onnx"

echo "[3/5] Converting face recognizer ONNX -> NCNN with pnnx"
pushd "${tmp_dir}" >/dev/null
"${PNNX}" "${RECOGNIZER_ONNX}" inputshape="[1,3,112,112]"
popd >/dev/null
mv "${tmp_dir}/arcfaceresnet100-8.ncnn.param" "${RECOGNIZER_PARAM}"
mv "${tmp_dir}/arcfaceresnet100-8.ncnn.bin" "${RECOGNIZER_BIN}"
rm -f "${tmp_dir}/arcfaceresnet100-8.pnnx.param" "${tmp_dir}/arcfaceresnet100-8.pnnx.bin" "${tmp_dir}/arcfaceresnet100-8_pnnx.py" "${tmp_dir}/arcfaceresnet100-8_pnnx.onnx"

echo "[4/5] Converting emotion ONNX -> NCNN with pnnx"
pushd "${tmp_dir}" >/dev/null
"${PNNX}" "${EMOTION_ONNX}" inputshape="[1,1,64,64]"
popd >/dev/null
mv "${tmp_dir}/emotion-ferplus-8.ncnn.param" "${EMOTION_PARAM}"
mv "${tmp_dir}/emotion-ferplus-8.ncnn.bin" "${EMOTION_BIN}"
rm -f "${tmp_dir}/emotion-ferplus-8.pnnx.param" "${tmp_dir}/emotion-ferplus-8.pnnx.bin" "${tmp_dir}/emotion-ferplus-8_pnnx.py" "${tmp_dir}/emotion-ferplus-8_pnnx.onnx"

if [[ -x "${NCNNOPTIMIZE}" ]]; then
  echo "[5/5] Optimizing NCNN models"
  "${NCNNOPTIMIZE}" "${DETECTOR_PARAM}" "${DETECTOR_BIN}" "${DETECTOR_PARAM}.opt.param" "${DETECTOR_BIN}.opt.bin" 0
  mv "${DETECTOR_PARAM}.opt.param" "${DETECTOR_PARAM}"
  mv "${DETECTOR_BIN}.opt.bin" "${DETECTOR_BIN}"

  "${NCNNOPTIMIZE}" "${RECOGNIZER_PARAM}" "${RECOGNIZER_BIN}" "${RECOGNIZER_PARAM}.opt.param" "${RECOGNIZER_BIN}.opt.bin" 0
  mv "${RECOGNIZER_PARAM}.opt.param" "${RECOGNIZER_PARAM}"
  mv "${RECOGNIZER_BIN}.opt.bin" "${RECOGNIZER_BIN}"

  "${NCNNOPTIMIZE}" "${EMOTION_PARAM}" "${EMOTION_BIN}" "${EMOTION_PARAM}.opt.param" "${EMOTION_BIN}.opt.bin" 0
  mv "${EMOTION_PARAM}.opt.param" "${EMOTION_PARAM}"
  mv "${EMOTION_BIN}.opt.bin" "${EMOTION_BIN}"
else
  echo "[5/5] ncnnoptimize not found, skipping optimization"
fi

rm -rf "${tmp_dir}"

echo "Done. Generated files:"
ls -lh "${DETECTOR_PARAM}" "${DETECTOR_BIN}" "${RECOGNIZER_PARAM}" "${RECOGNIZER_BIN}" "${EMOTION_PARAM}" "${EMOTION_BIN}"

#!/usr/bin/env bash
set -euo pipefail

PREFIX="${PREFIX:-/usr/local}"
SRC_DIR="${SRC_DIR:-$HOME/ncnn}"
BUILD_DIR="${BUILD_DIR:-$SRC_DIR/build}"
RPI4_FLAGS="${RPI4_FLAGS:--O3 -DNDEBUG -mcpu=cortex-a72 -mtune=cortex-a72}"

echo "[ncnn] Installing build dependencies..."
sudo apt update
sudo apt install -y \
  build-essential \
  cmake \
  git \
  ninja-build \
  pkg-config

if [[ -d "$SRC_DIR/.git" ]]; then
  echo "[ncnn] Updating existing source: $SRC_DIR"
  git -C "$SRC_DIR" pull --ff-only
else
  echo "[ncnn] Cloning source into: $SRC_DIR"
  git clone --depth=1 https://github.com/Tencent/ncnn.git "$SRC_DIR"
fi

echo "[ncnn] Configuring..."
cmake -S "$SRC_DIR" -B "$BUILD_DIR" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX="$PREFIX" \
  -DCMAKE_C_FLAGS_RELEASE="$RPI4_FLAGS" \
  -DCMAKE_CXX_FLAGS_RELEASE="$RPI4_FLAGS" \
  -DNCNN_VULKAN=OFF \
  -DNCNN_BUILD_TOOLS=OFF \
  -DNCNN_BUILD_EXAMPLES=OFF \
  -DNCNN_BUILD_TESTS=OFF \
  -DNCNN_OPENMP=ON

echo "[ncnn] Building..."
cmake --build "$BUILD_DIR" -j"$(nproc)"

echo "[ncnn] Installing to $PREFIX ..."
sudo cmake --install "$BUILD_DIR"
sudo ldconfig

echo "[ncnn] Verification:"
find "$PREFIX" \( -name "libncnn*" -o -name "ncnnConfig.cmake" -o -name "ncnn-config.cmake" \) -print

echo "[ncnn] Done. Reconfigure this project with:"
echo "  cmake -S . -B build_rpi -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH=$PREFIX"

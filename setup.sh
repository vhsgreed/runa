#!/usr/bin/env bash
# Runa setup: deps, whisper.cpp (Vulkan), models. Idempotent.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WHISPER_DIR="${RUNA_WHISPER_DIR:-$HERE/whisper.cpp}"
MODELS_DIR="${RUNA_MODELS_DIR:-$HERE/models}"

echo "==> [1/4] APT deps (needs sudo)"
SUDO=""
[ "$(id -u)" -eq 0 ] || SUDO="sudo"
$SUDO apt-get update -qq
$SUDO apt-get install -y -qq \
  build-essential cmake git curl ca-certificates pkg-config \
  libsdl2-dev libvulkan-dev vulkan-tools glslc \
  spirv-headers \
  python3 python3-pip python3-venv python3-numpy

echo "==> [2/4] Python deps"
# Debian 13+ is PEP 668: use --break-system-packages for a system-wide CLI tool,
# or set RUNA_PIP_USER=1 for a user install.
PIPFLAGS="--break-system-packages"
[ "${RUNA_PIP_USER:-0}" = "1" ] && PIPFLAGS="--user"
python3 -m pip install $PIPFLAGS "sounddevice>=0.4.6" "numpy>=1.26"

echo "==> [3/4] whisper.cpp (Vulkan)"
if [ ! -d "$WHISPER_DIR/.git" ]; then
  git clone --depth 1 --branch v1.8.6 https://github.com/ggml-org/whisper.cpp "$WHISPER_DIR"
fi
BUILD_OK=0
if command -v vulkaninfo >/dev/null 2>&1 && vulkaninfo --summary >/dev/null 2>&1; then
  echo "    Vulkan ICD detected -> building GGML_VULKAN=1"
  if cmake -S "$WHISPER_DIR" -B "$WHISPER_DIR/build-vulkan" \
      -DGGML_VULKAN=1 -DGGML_CUDA=0 -DWHISPER_SDL2=ON \
      -DCMAKE_BUILD_TYPE=Release && \
     cmake --build "$WHISPER_DIR/build-vulkan" --config Release -j"$(nproc)"; then
    BUILD_OK=1
    echo "    Vulkan build OK"
  else
    echo "    WARNING: Vulkan build FAILED -> falling back to CPU (AVX2)"
  fi
else
  echo "    No Vulkan ICD available -> CPU (AVX2) build"
fi
if [ "$BUILD_OK" -eq 0 ]; then
  cmake -S "$WHISPER_DIR" -B "$WHISPER_DIR/build" \
    -DGGML_VULKAN=0 -DGGML_CUDA=0 -DGGML_NATIVE=1 -DWHISPER_SDL2=ON \
    -DCMAKE_BUILD_TYPE=Release
  cmake --build "$WHISPER_DIR/build" --config Release -j"$(nproc)"
  echo "NOTE: built CPU fallback (AVX2). Vulkan build failed on this host." >> "$HERE/BUILD-MODE.txt"
else
  rm -f "$HERE/BUILD-MODE.txt"
fi

echo "==> [4/4] Models"
mkdir -p "$MODELS_DIR"
# download-ggml-model.sh takes the models dir as a POSITIONAL arg (it does not
# support --outdir; that printed the usage banner and downloaded nothing).
bash "$WHISPER_DIR/models/download-ggml-model.sh" base "$MODELS_DIR"
# optional accuracy tier (recommended on 16GB + Vulkan): RUNA_EXTRA_MODELS="large-v3-turbo"
for m in ${RUNA_EXTRA_MODELS:-}; do
  bash "$WHISPER_DIR/models/download-ggml-model.sh" "$m" "$MODELS_DIR"
done
curl -sL -o "$MODELS_DIR/ggml-silero-v5.1.2.bin" \
  https://huggingface.co/ggml-org/whisper-vad/resolve/main/ggml-silero-v5.1.2.bin

echo
echo "Done. Try: python3 $HERE/runa.py --mode 5s --lang sv"

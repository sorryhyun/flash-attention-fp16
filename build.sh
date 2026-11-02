#!/usr/bin/env bash
# Usage: build.sh <python with the target torch>  (nvcc on PATH, CUDA_HOME set, ninja installed)
# Each cicc takes about 5 GB, so keep MAX_JOBS at 6 on a 64 GB box.
set -euo pipefail
PY=${1:?python}
HERE=$(cd "$(dirname "$0")" && pwd)
WORK=$(mktemp -d)
curl -sSLO --output-dir "$WORK" https://files.pythonhosted.org/packages/3b/b2/8d76c41ad7974ee264754709c22963447f7f8134613fd9ce80984ed0dab7/flash_attn-2.8.3.tar.gz
echo "1e71dd64a9e0280e0447b8a0c2541bad4bf6ac65bdeaa2f90e51a9e57de0370d  $WORK/flash_attn-2.8.3.tar.gz" | sha256sum -c
tar -xzf "$WORK/flash_attn-2.8.3.tar.gz" -C "$WORK"
cd "$WORK/flash_attn-2.8.3"
patch -p1 < "$HERE/patches/sm120.patch"
FLASH_ATTN_CUDA_ARCHS=120 FLASH_ATTENTION_FORCE_BUILD=TRUE BUILD_TARGET=cuda MAX_JOBS=${MAX_JOBS:-6} \
  FLASH_ATTN_LOCAL_VERSION=${LOCAL_VERSION:-sm120.hdim128.cu132torch2.14} \
  "$PY" setup.py bdist_wheel
ls -la "$PWD"/dist/*.whl

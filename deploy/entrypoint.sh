#!/usr/bin/env bash
set -euo pipefail
# CTranslate2 and ONNX Runtime load CUDA libraries outside PyTorch. Expose the
# libraries installed by the matching Torch wheel before starting Python.
cuda_libs=""
for lib_dir in /usr/local/lib/python3.11/site-packages/nvidia/*/lib; do
    if [[ -d "$lib_dir" ]]; then cuda_libs="${cuda_libs}${lib_dir}:"; fi
done
if [[ -n "$cuda_libs" ]]; then export LD_LIBRARY_PATH="${cuda_libs}${LD_LIBRARY_PATH:-}"; fi
exec "$@"

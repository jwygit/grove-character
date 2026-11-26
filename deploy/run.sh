#!/usr/bin/env bash
# Keep /work and /models stable: graph evidence/cache files contain absolute paths.
set -euo pipefail
profile=${1:?usage: bash deploy/run.sh core|visual|asr|character|workbench COMMAND ...}
shift
case "$profile" in core|visual|asr|character|workbench) ;; *) echo 'unknown profile' >&2; exit 2;; esac
repo_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
asset_dir=${GROVE_ASSETS:?export GROVE_ASSETS as the absolute offline asset directory}
if [[ "$asset_dir" != /* || ! -d "$asset_dir/models" ]]; then
    echo 'GROVE_ASSETS/models must exist at an absolute path' >&2
    exit 2
fi
gpu_options=()
if [[ "$profile" == asr || "$profile" == character ]]; then
    : "${GROVE_GPU:?set GROVE_GPU to one physical GPU ID for this process}"
fi
if [[ -n "${GROVE_GPU:-}" ]]; then gpu_options=(--gpus "device=${GROVE_GPU}"); fi
exec docker run --rm -i --network host --ipc=host \
    --user "$(id -u):$(id -g)" "${gpu_options[@]}" \
    -v "$repo_dir:/work" -v "$asset_dir/models:/models:ro" -w /work \
    -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 -e HF_HUB_DISABLE_TELEMETRY=1 \
    -e VLM_API_KEY -e LLM_API_KEY -e EMBEDDING_API_KEY -e GEMINI_API_KEY -e HF_TOKEN \
    "grove-${profile}:offline-v1" "$@"

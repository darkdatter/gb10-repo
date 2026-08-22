#!/usr/bin/env bash
# Prove GPU passthrough into Docker before anything else. Five minutes here
# saves an hour of misdiagnosis later.
#
# DGX OS ships CDI rather than a registered Docker runtime, so `docker info`
# listing only `runc` is NORMAL and --gpus all still works.
set -euo pipefail

IMAGE="${CUDA_IMAGE:-nvidia/cuda:13.0.1-base-ubuntu24.04}"

echo "== host =="
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader || {
  echo "nvidia-smi failed on the host - fix the driver before continuing"; exit 1; }

echo
echo "== CDI devices =="
nvidia-ctk cdi list 2>/dev/null | head -5 || echo "(nvidia-ctk not found)"

echo
echo "== container test: --gpus all =="
if docker run --rm --gpus all "$IMAGE" nvidia-smi -L; then
  echo "OK: --gpus all works"
  exit 0
fi

echo
echo "== fallback: explicit CDI device =="
if docker run --rm --device nvidia.com/gpu=all "$IMAGE" nvidia-smi -L; then
  echo "OK: use --device nvidia.com/gpu=all instead of --gpus all"
  exit 0
fi

cat <<'EOF'

Both failed. Register the nvidia runtime, then re-run this script:

    sudo nvidia-ctk runtime configure --runtime=docker
    sudo systemctl restart docker
EOF
exit 1

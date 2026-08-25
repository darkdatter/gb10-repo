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
  cat <<'EOF'

OK, but only via CDI: --gpus all does NOT work on this host.

This is not automatically actionable. The toolkit launcher hardcodes
`--gpus all` (start.sh), and SparkStation's launcher builds its own docker
arguments, so neither will pick this up. You must apply it yourself:

  standalone toolkit:
      cd Qwen3.8-27B-SGLang-DGX-Spark
      sed -i 's/--gpus all/--device nvidia.com\/gpu=all/' start.sh

  sparkstation:
      edit supervisor/launchers/sglang_launcher.py, replace "--gpus", "all"
      with "--device", "nvidia.com/gpu=all"
      then: cp any edited top-level module into .venv/lib/python3.12/site-packages/

Re-run this script after the edit to confirm.
EOF
  exit 0
fi

cat <<EOF

Both failed. Check which daemon you are actually talking to first — the active
Docker context selects the endpoint, and a remote or rootless context will not
see this host's GPUs:

    docker context show
    docker context ls
    https://docs.docker.com/engine/manage-resources/contexts/

  active context: $(docker context show 2>/dev/null || echo "unknown")

If the context is correct, register the nvidia runtime and re-run this script:

    sudo nvidia-ctk runtime configure --runtime=docker
    sudo systemctl restart docker
EOF
exit 1

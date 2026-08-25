#!/usr/bin/env bash
# Emit the resolved build configuration. Run it next to a benchmark so a later
# comparison has the actual inputs rather than the intended ones.
#
#   ./scripts/build-manifest.sh > results/BUILD-MANIFEST.md
#
# Tags and default branches move; digests and commits do not.
set -uo pipefail

WORKDIR="${GB10_WORKDIR:-$HOME/spark}"
TOOLKIT="$WORKDIR/Qwen3.8-27B-SGLang-DGX-Spark"
HF_HUB="${HF_HOME:-$HOME/.cache/huggingface}/hub"
BASE_IMAGE=lmsysorg/sglang:qwen38-27b
BUILT_IMAGE=lmsysorg/sglang:qwen38-27b-dflash2

snap() {  # resolved snapshot(s) for a cached repo
  local d="$HF_HUB/models--${1//\//--}/snapshots"
  [ -d "$d" ] && ls "$d" 2>/dev/null | paste -sd, || echo "(not cached)"
}
img() { docker inspect "$1" --format "$2" 2>/dev/null || echo "(absent)"; }

cat <<EOF
# Build manifest

Generated $(date -u +%Y-%m-%dT%H:%M:%SZ) on \`$(hostname)\`.

| Input | Resolved value |
|---|---|
| Target revision | \`$(snap RadixArk/Qwen3.8-27B-NVFP4)\` |
| Draft revision | \`$(snap z-lab/Qwen3.8-27B-DFlash2)\` |
| Toolkit commit | \`$(git -C "$TOOLKIT" rev-parse HEAD 2>/dev/null || echo "(no clone)")\` |
| SGLang commit (overlay) | \`$(grep -oE 'full_sha=[0-9a-f]+' "$TOOLKIT/patch/build-dflash2-image.sh" 2>/dev/null | cut -d= -f2 || echo "(unknown)")\` |
| NVFP4 head patch sha256 | \`$(sha256sum "$TOOLKIT/patch/dflash2_nvfp4_head.patch" 2>/dev/null | cut -d' ' -f1 || echo "(absent)")\` |
| Base image digest | \`$(img "$BASE_IMAGE" '{{index .RepoDigests 0}}')\` |
| Built image id | \`$(img "$BUILT_IMAGE" '{{.Id}}')\` |

## Host

| | |
|---|---|
| Kernel | \`$(uname -sr)\` |
| Driver / CUDA | $(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1) / $(nvidia-smi 2>/dev/null | grep -oE 'CUDA Version: [0-9.]+' | cut -d' ' -f3) |
| Docker | $(docker --version 2>/dev/null | grep -oE '[0-9]+\.[0-9]+\.[0-9]+' | head -1) |
| nvidia-container-toolkit | $(nvidia-ctk --version 2>/dev/null | grep -oE '[0-9]+\.[0-9]+\.[0-9]+' | head -1) |

## Serving flags

\`\`\`
$(docker inspect "$(docker ps -q --filter name=sglang --filter name=sparkstation-qwen | head -1)" \
    --format '{{json .Config.Cmd}}' 2>/dev/null | tr ',' '\n' | tr -d '["]' | paste -sd' ' \
  || echo "(no server running)")
\`\`\`

The base image is referenced by **tag**, not digest, by the toolkit build. The
digest above is what it resolved to at build time; a republish changes the image
without any warning during the build.
EOF

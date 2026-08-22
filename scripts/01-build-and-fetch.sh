#!/usr/bin/env bash
# Build the DFlash2 SGLang image and pre-download the weights.
#
# These two run happily in parallel - the build pulls a ~39GB base image while
# the weights pull ~25GB. Budget ~110GB of free disk overall.
#
# WHY A LOCAL BUILD: lmsysorg/sglang:qwen38-27b-dflash2 does NOT exist on
# Docker Hub. SparkStation's models.yaml references it, but no released SGLang
# tag ships DFlash2. The base :qwen38-27b tag is real; the dflash2 tag is
# produced by the toolkit's patch/ script.
set -euo pipefail

WORKDIR="${GB10_WORKDIR:-$HOME/spark}"
TOOLKIT="$WORKDIR/Qwen3.8-27B-SGLang-DGX-Spark"
TARGET_REPO="RadixArk/Qwen3.8-27B-NVFP4"
DRAFT_REPO="z-lab/Qwen3.8-27B-DFlash2"
DRAFT_REV="50307d4c4cde6860d4eee73e2547cd786fe8e8a4"

mkdir -p "$WORKDIR"

# ---- weights ---------------------------------------------------------------
# Deliberately into the REAL ~/.cache/huggingface. Do not symlink hub/ or xet/
# inside that directory: the container bind-mounts it, in-mount symlinks point
# at host-only paths, and it re-downloads then dies with
#   OSError: I/O error: File exists (os error 17)
if [ ! -x "$WORKDIR/venv/bin/hf" ]; then
  python3 -m venv "$WORKDIR/venv"
  "$WORKDIR/venv/bin/pip" -q install -U huggingface_hub pandas pyarrow
fi

echo "== downloading weights (~25GB) =="
"$WORKDIR/venv/bin/hf" download "$TARGET_REPO"
"$WORKDIR/venv/bin/hf" download "$DRAFT_REPO" --revision "$DRAFT_REV"

# ---- image -----------------------------------------------------------------
if [ ! -d "$TOOLKIT/.git" ]; then
  git clone --depth 1 \
    https://github.com/MiaAI-Lab/Qwen3.8-27B-SGLang-DGX-Spark.git "$TOOLKIT"
fi

echo
echo "== building lmsysorg/sglang:qwen38-27b-dflash2 =="
# --full clones SGLang at a pinned commit and applies an NVFP4 lm_head patch.
# That patch is load-bearing: the earlier dequant-everything approach allocated
# ~2.5GB at draft-graph capture and hard-rebooted the machine.
# Use --minimal for a checksum-verified 5-file overlay that needs no network.
( cd "$TOOLKIT" && ./patch/build-dflash2-image.sh --full )

echo
docker images | grep -E "REPOSITORY|sglang"
echo
echo "Next: cd '$TOOLKIT' && cp -n .env.sample .env && \\"
echo "      DF_EXTRA=\"--mem-fraction-static 0.85\" ./start-dflash.sh"

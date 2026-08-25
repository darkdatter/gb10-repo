#!/usr/bin/env bash
# Fetch the DFlash2 SGLang image and pre-download the weights.
#
# Budget ~110GB of free disk.
#
# LMSYS published official DFlash2 images on 2026-08-22, so the local build is
# no longer required and is OFF by default. It remains available because the
# results in results/RESULTS.md were produced with the locally built image:
#
#   BUILD_LOCAL=1 ./scripts/01-build-and-fetch.sh
#
# The local build clones SGLang at a pinned commit and applies an NVFP4 lm_head
# patch. That patch is load-bearing on the old base: the earlier
# dequant-everything approach allocated ~2.5GB at draft-graph capture and
# hard-rebooted the machine. The official image needs no patch.
set -euo pipefail

IMAGE="${IMAGE:-lmsysorg/sglang:dev-cu13-qwen38-27b-dflash2}"
BUILD_LOCAL="${BUILD_LOCAL:-0}"

WORKDIR="${GB10_WORKDIR:-$HOME/spark}"
TOOLKIT="$WORKDIR/Qwen3.8-27B-SGLang-DGX-Spark"
TARGET_REPO="RadixArk/Qwen3.8-27B-NVFP4"
TARGET_REV="${TARGET_REV:-554ebba9b5f1b79dc11246341960360e6ef05ef4}"
DRAFT_REPO="z-lab/Qwen3.8-27B-DFlash2"
DRAFT_REV="${DRAFT_REV:-50307d4c4cde6860d4eee73e2547cd786fe8e8a4}"
# Toolkit commit this recipe was validated against. Its default branch moves.
TOOLKIT_REV="${TOOLKIT_REV:-c90d8c34cf795185ee8de736b7ded9bca3fe0de1}"

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
# BOTH revisions are pinned. Without --revision on the target you get the repo's
# mutable default, which may not be the checkpoint these results were measured
# on. Pass the same revision to the server too (see note at the end).
"$WORKDIR/venv/bin/hf" download "$TARGET_REPO" --revision "$TARGET_REV"
"$WORKDIR/venv/bin/hf" download "$DRAFT_REPO" --revision "$DRAFT_REV"

# ---- image -----------------------------------------------------------------
# Pinned: the toolkit's default branch moves, and its build script selects the
# base image by tag, so an unpinned clone can produce a different image than the
# one benchmarked here.
if [ ! -d "$TOOLKIT/.git" ]; then
  git clone https://github.com/MiaAI-Lab/Qwen3.8-27B-SGLang-DGX-Spark.git "$TOOLKIT"
fi
git -C "$TOOLKIT" fetch --depth 1 origin "$TOOLKIT_REV" 2>/dev/null || true
git -C "$TOOLKIT" checkout -q "$TOOLKIT_REV" || {
  echo "WARNING: could not check out toolkit $TOOLKIT_REV; using $(git -C "$TOOLKIT" rev-parse --short HEAD)" >&2; }

echo
if [ "$BUILD_LOCAL" = "1" ]; then
  echo "== building lmsysorg/sglang:qwen38-27b-dflash2 (BUILD_LOCAL=1) =="
  # Use --minimal for a checksum-verified 5-file overlay that needs no network.
  ( cd "$TOOLKIT" && ./patch/build-dflash2-image.sh --full )
  IMAGE=lmsysorg/sglang:qwen38-27b-dflash2
else
  echo "== pulling $IMAGE =="
  echo "   (official image; set BUILD_LOCAL=1 to build the patched one instead)"
  docker pull "$IMAGE"
fi

echo
docker images | grep -E "REPOSITORY|sglang"
echo
echo "Next: cd '$TOOLKIT' && cp -n .env.sample .env && \\"
echo "      IMAGE=$IMAGE DF_EXTRA=\"--mem-fraction-static 0.85\" ./start-dflash.sh"

cat <<NOTE

Pin the target on the server too — downloading the right revision does not make
SGLang serve it:

    ./start-dflash.sh          # add to EXTRA_ARGS / DF_EXTRA:
        --revision $TARGET_REV

SparkStation users: models.yaml already carries "--revision" in sglang_flags.

Fully-offline caveat: SGLang's early speculative-algorithm probe resolves the
draft config WITHOUT a revision, so a cache holding only the pinned draft
snapshot can still reach for that repo's default ref on first start. Prime the
cache while online, or expect that one lookup.

Record what actually got built:
    ./scripts/build-manifest.sh > results/BUILD-MANIFEST.md
NOTE

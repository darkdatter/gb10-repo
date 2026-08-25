# Build manifest

Resolved build configuration for every number in
[`RESULTS.md`](RESULTS.md). Recorded 2026-08-24 from the machine that produced
them.

Tags and default branches move; digests and commits do not. If your numbers
differ from ours, compare this table first — a mismatch here explains more than
any amount of re-benchmarking.

| Input | Resolved value |
|---|---|
| Target checkpoint | `RadixArk/Qwen3.8-27B-NVFP4` |
| Target revision | `554ebba9b5f1b79dc11246341960360e6ef05ef4` |
| Draft checkpoint | `z-lab/Qwen3.8-27B-DFlash2` |
| Draft revision | `50307d4c4cde6860d4eee73e2547cd786fe8e8a4` |
| Toolkit | [MiaAI-Lab/Qwen3.8-27B-SGLang-DGX-Spark](https://github.com/MiaAI-Lab/Qwen3.8-27B-SGLang-DGX-Spark) |
| Toolkit commit | `c90d8c34cf795185ee8de736b7ded9bca3fe0de1` |
| SGLang commit (overlay) | `c14312a66420b75ca9a11bf1817c4db1fa26b097` |
| NVFP4 head patch sha256 | `5bdce963c535ac46db6ef968ffa0332d1f9439f1ea6ddc9020421dbfec244071` |
| Base image | `lmsysorg/sglang:qwen38-27b` |
| Base image digest | `sha256:febfb971c7352570fc445c466ebd6ffc9d896024958e544a60f2137fd85856b1` |
| Built image | `lmsysorg/sglang:qwen38-27b-dflash2` |
| Built image id | `sha256:c41f8da7497ed198fe5edb738ca684bd11be40ec48678f9371ea3964ff423298` |

## Host

| | |
|---|---|
| Hardware | ASUS DGX Spark, GB10, 128 GB unified |
| OS / kernel | Ubuntu 24.04, 6.17.0-1031-nvidia, aarch64 |
| Driver / CUDA | 580.173.02 / 13.0 |
| Docker | 29.2.1 |
| nvidia-container-toolkit | 1.20.0 (CDI) |

## Which inputs are pinned, and which are not

Pinned by the scripts in this repo: target revision, draft revision, toolkit
commit.

Pinned by the toolkit itself: the SGLang commit and the NVFP4 head patch.

**Not pinned:** the base image is referenced by tag, not digest, by the
toolkit's build script. The digest above is what that tag resolved to on the
build date. If `lmsysorg/sglang:qwen38-27b` is republished, a rebuild produces a
different image than the one benchmarked here, and nothing in the build will
warn you.

Regenerate this table with [`scripts/build-manifest.sh`](../scripts/build-manifest.sh).

#!/usr/bin/env bash
# Download the trained residual-refiner checkpoints from the Hugging Face Hub.
#
# The diffusion backbone (Stable Diffusion 2.1) and RoMa weights download
# automatically on first use, so only the Stage-3 refiner checkpoints are hosted here.
#
# Requires the Hugging Face CLI:  pip install -U huggingface_hub
#
# Usage:
#   bash scripts/download_weights.sh            # all datasets
#   bash scripts/download_weights.sh levir      # one dataset
set -euo pipefail

HF_REPO="${MTT_HF_REPO:-Anita1379m/morphingthroughtime}"
DST="checkpoints"
mkdir -p "$DST"

DATASETS=("${@:-levir whu dsifn}")
for ds in ${DATASETS[@]}; do
  echo ">> $ds"
  hf download "$HF_REPO" "${ds}/refiner.pth" --local-dir "$DST" || true
done

echo "Done. Checkpoints are under ./$DST/"

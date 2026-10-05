#!/usr/bin/env bash
set -eo pipefail
cd "$(dirname "$0")"

CONDA_SH="$(conda info --base 2>/dev/null)/etc/profile.d/conda.sh"
if [[ -f "$CONDA_SH" ]]; then
  source "$CONDA_SH"
  conda activate augenv
else
  source activate augenv
fi

rm -rf text_patches_augmented_betterv3_80
python Augment_patchForOrientation_v4.py

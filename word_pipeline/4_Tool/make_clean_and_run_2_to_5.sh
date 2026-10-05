#!/usr/bin/env bash
# Clean this Writing_Tool folder, then run each flow via its own *.sh so outputs
# stay in that folder (same as running train.sh / test.sh by hand).
#
#   2_search_architecture/train.sh
#   3_Train_augmented_data/train.sh
#   4_Test_tool_with_annotation/test.sh
#   5_Test_tool_without_annotation/test.sh
#
#   bash make_clean_and_run_2_to_5.sh              # clean this folder, then run 2→5
#   bash make_clean_and_run_2_to_5.sh full         # re-run 1_augmentation, clean, run 2→5
#   bash make_clean_and_run_2_to_5.sh clean        # only delete junk here
#   bash make_clean_and_run_2_to_5.sh run          # run 2→5 without deleting
#   bash make_clean_and_run_2_to_5.sh export [DIR] # copy a clean tree elsewhere
#
# Needs: 1_augmentation/augmented_patches_tool_v3
#        4_Test_tool_with_annotation/text_patches_test_20
#        5_Test_tool_without_annotation/ColoredOrnedName_Output2Input  (optional; auto-link from sibling 4_Tool if present)
#        5_Test_tool_without_annotation/ColoredOrnedJson_Output2Input  (optional; same)
# GPU:   CUDA_VISIBLE_DEVICES (default 0)   env: conda open-mmlab

set -euo pipefail

SRC="$(cd "$(dirname "$0")" && pwd)"
MODE="${1:-all}"
DEST="${2:-$(cd "$SRC/.." && pwd)/Writing_Tool_clean}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

CONDA_SH="$(conda info --base 2>/dev/null)/etc/profile.d/conda.sh"
set +u
if [[ -f "$CONDA_SH" ]]; then
  # shellcheck source=/dev/null
  source "$CONDA_SH"
  conda activate open-mmlab
else
  source activate open-mmlab
fi
set -u

need() {
  [[ -e "$1" ]] || { echo "Missing: $1" >&2; exit 1; }
}

# Delete outputs that a fresh 2→5 run will recreate. Keep code + input patches.
clean_workspace() {
  local ROOT="$1"
  echo "=== Clean generated files in $ROOT ==="

  # only the step folders; never touch 0_Crop_Patches or 1_augmentation
  local STEPS=(
    "$ROOT/2_search_architecture"
    "$ROOT/3_Train_augmented_data"
    "$ROOT/4_Test_tool_with_annotation"
    "$ROOT/5_Test_tool_without_annotation"
  )
  find "${STEPS[@]}" -type d -name '__pycache__' -prune -exec rm -rf {} +
  find "${STEPS[@]}" -type f \( \
      -name '*.pt' -o -name '*.npz' -o -name '*.csv' -o -name '*.meta.json' \
      -o -name 'search_log.txt' -o -name '*rankscore_report.txt' \
      -o -name 'confusion_matrix*.png' \
    \) -delete

  rm -rf \
    "$ROOT/2_search_architecture/confusion_matrices_writing_tool" \
    "$ROOT/5_Test_tool_without_annotation/ColoredOrnedTooledJson_Output" \
    "$ROOT/5_Test_tool_without_annotation/ColoredOrnedTooledName_Output"

  # one copy of aug patches; steps 2 and 3 only need a link
  need "$ROOT/1_augmentation/augmented_patches_tool_v3"
  if [[ ! -L "$ROOT/2_search_architecture/augmented_patches_tool_v3" ]]; then
    rm -rf "$ROOT/2_search_architecture/augmented_patches_tool_v3"
    ln -sfn ../1_augmentation/augmented_patches_tool_v3 \
      "$ROOT/2_search_architecture/augmented_patches_tool_v3"
  fi
  if [[ ! -L "$ROOT/3_Train_augmented_data/augmented_patches_tool_v3" ]]; then
    rm -rf "$ROOT/3_Train_augmented_data/augmented_patches_tool_v3"
    ln -sfn ../1_augmentation/augmented_patches_tool_v3 \
      "$ROOT/3_Train_augmented_data/augmented_patches_tool_v3"
  fi

  echo "Left: *.py *.sh + input patches/json. Removed: .pt .npz .csv logs reports old outputs."
}

export_clean() {
  echo "=== Clean copy → $DEST ==="
  need "$SRC/1_augmentation/augmented_patches_tool_v3"
  need "$SRC/4_Test_tool_with_annotation/text_patches_test_20"
  need "$SRC/5_Test_tool_without_annotation/ColoredOrnedName_Output2Input"
  need "$SRC/5_Test_tool_without_annotation/ColoredOrnedJson_Output2Input"

  rm -rf "$DEST"
  mkdir -p "$DEST"

  while IFS= read -r -d '' f; do
    rel="${f#"$SRC"/}"
    mkdir -p "$DEST/$(dirname "$rel")"
    cp -a "$f" "$DEST/$rel"
  done < <(find "$SRC" \( -name '*.py' -o -name '*.sh' \) \
    ! -path '*/__pycache__/*' -print0)

  mkdir -p \
    "$DEST/2_search_architecture" \
    "$DEST/3_Train_augmented_data" \
    "$DEST/4_Test_tool_with_annotation" \
    "$DEST/5_Test_tool_without_annotation"

  rsync -a "$SRC/1_augmentation/augmented_patches_tool_v3/" \
    "$DEST/1_augmentation/augmented_patches_tool_v3/"
  rm -rf "$DEST/2_search_architecture/augmented_patches_tool_v3"
  rm -rf "$DEST/3_Train_augmented_data/augmented_patches_tool_v3"
  ln -sfn ../1_augmentation/augmented_patches_tool_v3 \
    "$DEST/2_search_architecture/augmented_patches_tool_v3"
  ln -sfn ../1_augmentation/augmented_patches_tool_v3 \
    "$DEST/3_Train_augmented_data/augmented_patches_tool_v3"

  n_png="$(find -L "$DEST/2_search_architecture/augmented_patches_tool_v3" \
    -maxdepth 1 -iname '*.png' | wc -l)"
  if [[ "$n_png" -lt 1 ]]; then
    echo "Patch symlink is empty or broken (0 png)." >&2
    exit 1
  fi
  echo "Augmented patches visible from step 2: $n_png"

  rsync -a "$SRC/4_Test_tool_with_annotation/text_patches_test_20/" \
    "$DEST/4_Test_tool_with_annotation/text_patches_test_20/"

  rsync -a \
    "$SRC/5_Test_tool_without_annotation/ColoredOrnedName_Output2Input/" \
    "$DEST/5_Test_tool_without_annotation/ColoredOrnedName_Output2Input/"
  rsync -a \
    "$SRC/5_Test_tool_without_annotation/ColoredOrnedJson_Output2Input/" \
    "$DEST/5_Test_tool_without_annotation/ColoredOrnedJson_Output2Input/"

  echo "Kept: *.py *.sh + aug patches + test patches + unlabeled step-5 inputs"
  echo "Dropped: .pt .npz .csv reports logs ColoredOrnedTooled* outputs"
}

run_folder_sh() {
  local dir="$1"
  local script="$2"
  echo
  echo "======== ${dir} / ${script} ========"
  (
    cd "$dir"
    bash "$script"
  )
  echo "-------- outputs in ${dir} --------"
  ls -lt -- "$dir" | awk 'NR<=12 {print}'
}

run_pipeline() {
  local ROOT="$1"
  cd "$ROOT"
  export PYTHONUNBUFFERED=1

  need "$ROOT/1_augmentation/augmented_patches_tool_v3"
  need "$ROOT/2_search_architecture/augmented_patches_tool_v3"
  need "$ROOT/3_Train_augmented_data/augmented_patches_tool_v3"
  need "$ROOT/4_Test_tool_with_annotation/text_patches_test_20"

  # Optional unlabeled inputs for step 5 (often copied from orientation ColoredOrned*).
  # If missing, try the sibling 4_Tool tree; if still missing, run 2→4 only.
  local STEP5_NAME="$ROOT/5_Test_tool_without_annotation/ColoredOrnedName_Output2Input"
  local STEP5_JSON="$ROOT/5_Test_tool_without_annotation/ColoredOrnedJson_Output2Input"
  local FALLBACK_NAME="$(cd "$ROOT/../.." && pwd)/4_Tool/5_Test_tool_without_annotation/ColoredOrnedName_Output2Input"
  local FALLBACK_JSON="$(cd "$ROOT/../.." && pwd)/4_Tool/5_Test_tool_without_annotation/ColoredOrnedJson_Output2Input"
  if [[ ! -e "$STEP5_NAME" && -e "$FALLBACK_NAME" ]]; then
    ln -sfn "$FALLBACK_NAME" "$STEP5_NAME"
    echo "Linked step-5 patches ← $FALLBACK_NAME"
  fi
  if [[ ! -e "$STEP5_JSON" && -e "$FALLBACK_JSON" ]]; then
    ln -sfn "$FALLBACK_JSON" "$STEP5_JSON"
    echo "Linked step-5 JSON ← $FALLBACK_JSON"
  fi

  run_folder_sh "$ROOT/2_search_architecture" train.sh

  run_folder_sh "$ROOT/3_Train_augmented_data" train.sh

  run_folder_sh "$ROOT/4_Test_tool_with_annotation" test.sh

  if [[ -e "$STEP5_NAME" && -e "$STEP5_JSON" ]]; then
    rm -rf "$ROOT/5_Test_tool_without_annotation/ColoredOrnedTooledJson_Output"
    rm -rf "$ROOT/5_Test_tool_without_annotation/ColoredOrnedTooledName_Output"
    run_folder_sh "$ROOT/5_Test_tool_without_annotation" test.sh
  else
    echo
    echo "Skip step 5: need ColoredOrnedName_Output2Input and ColoredOrnedJson_Output2Input"
    echo "  under 5_Test_tool_without_annotation (or place them in sibling 4_Tool)."
  fi

  echo
  echo "======== results by folder ========"
  echo "[2_search_architecture]"
  if [[ -f "$ROOT/2_search_architecture/writing_tool_mlp_arch_search_V01-V20_rankscore_report.txt" ]]; then
    tail -n 8 "$ROOT/2_search_architecture/writing_tool_mlp_arch_search_V01-V20_rankscore_report.txt"
  fi
  echo
  echo "[4_Test_tool_with_annotation]  (Acc / Macro-F1 printed above by test.sh)"
  ls -1 "$ROOT/4_Test_tool_with_annotation"/confusion_matrix_*.png \
        "$ROOT/4_Test_tool_with_annotation"/test_predictions_*.csv 2>/dev/null || true
  echo
  echo "[5_Test_tool_without_annotation]"
  ls -1 "$ROOT/5_Test_tool_without_annotation"/test_predictions_*.csv \
        "$ROOT/5_Test_tool_without_annotation"/ColoredOrnedTooledJson_Output 2>/dev/null || true
}

case "$MODE" in
  clean)
    clean_workspace "$SRC"
    ;;
  export)
    export_clean
    ;;
  run)
    run_pipeline "$SRC"
    ;;
  full)
    (cd "$SRC/1_augmentation" && bash aug.sh)
    clean_workspace "$SRC"
    run_pipeline "$SRC"
    ;;
  all)
    clean_workspace "$SRC"
    run_pipeline "$SRC"
    ;;
  *)
    echo "Usage: $0 [all|full|clean|run|export] [DEST]" >&2
    exit 1
    ;;
esac

#!/usr/bin/env bash
set -euo pipefail

# =========================================================
# Full testing pipeline for current Submit_Code_All structure
#
# OUTPUT_CONFLICT_MODE:
#   archive -> rename old outputs to *_auto_Vx
#   delete  -> delete old outputs before overwrite
#
# CLEAN_AUTO_VERSIONS:
#   true  -> delete all previously created *_auto_Vx files/folders first
#   false -> keep them
# =========================================================

OUTPUT_CONFLICT_MODE="${OUTPUT_CONFLICT_MODE:-delete}"  # archive | delete
CLEAN_AUTO_VERSIONS="${CLEAN_AUTO_VERSIONS:-true}"      # true | false

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
BASE="${SCRIPT_DIR}"

# ======================= Paths =======================

TD="${BASE}/1_textdetection"

COLOUR="${BASE}/2_Colour/5_Test_Colour_without_annotation"

ORN="${BASE}/3_Orn/5_Test_Orn_without_annotation"
TRAIN_ORN="${BASE}/3_Orn/3_Train_augmented_data"

TOOL="${BASE}/4_Tool/5_Test_tool_without_annotation"
TRAIN_TOOL="${BASE}/4_Tool/3_Train_augmented_data"

VIS="${BASE}/5_visualization"

TEST_JSON="${TD}/data/DLA/instances_test.json"
DLA_IMG_SRC="${TD}/data/DLA/img"

# Output folders
COLOUR_JSON_OUT="${COLOUR}/ColoredJson_Output"
COLOUR_PATCH_OUT="${COLOUR}/ColoredName_CropedPatch_Output"

ORN_JSON_IN="${ORN}/ColoredJson_Output2Input"
ORN_PATCH_IN="${ORN}/ColoredName_CropedPatch_Output2Input"

ORN_JSON_OUT="${ORN}/ColoredOrnedJson_Output"
ORN_NAME_OUT="${ORN}/ColoredOrnedName_Output"

TOOL_JSON_IN="${TOOL}/ColoredOrnedJson_Output2Input"
TOOL_NAME_IN="${TOOL}/ColoredOrnedName_Output2Input"

TOOL_JSON_OUT="${TOOL}/ColoredOrnedTooledJson_Output"

VIS_IMG_DST="${VIS}/img"
VIS_JSON_OUT="${VIS}/ColoredOrnedTooledJson_Output"
VIS_GT_JSON="${VIS}/GT_updated.json"

echo "BASE resolved to: ${BASE}"
echo "OUTPUT_CONFLICT_MODE=${OUTPUT_CONFLICT_MODE}"
echo "CLEAN_AUTO_VERSIONS=${CLEAN_AUTO_VERSIONS}"

# ======================= Validation =======================

if [[ "$OUTPUT_CONFLICT_MODE" != "archive" && "$OUTPUT_CONFLICT_MODE" != "delete" ]]; then
    echo "ERROR: OUTPUT_CONFLICT_MODE must be 'archive' or 'delete'"
    exit 1
fi

if [[ "$CLEAN_AUTO_VERSIONS" != "true" && "$CLEAN_AUTO_VERSIONS" != "false" ]]; then
    echo "ERROR: CLEAN_AUTO_VERSIONS must be 'true' or 'false'"
    exit 1
fi

# ======================= Helpers =======================

archive_if_exists() {
    local path="$1"
    [[ -e "$path" ]] || return 0

    local dir base_name stem ext candidate version
    dir="$(dirname "$path")"
    base_name="$(basename "$path")"

    if [[ -f "$path" && "$base_name" == *.* ]]; then
        stem="${base_name%.*}"
        ext=".${base_name##*.}"
    else
        stem="$base_name"
        ext=""
    fi

    version=1
    while true; do
        candidate="${dir}/${stem}_auto_V${version}${ext}"
        if [[ ! -e "$candidate" ]]; then
            mv "$path" "$candidate"
            echo "Archived old output:"
            echo "  $path"
            echo "  -> $candidate"
            break
        fi
        version=$((version + 1))
    done
}

handle_existing_output() {
    local path="$1"
    [[ -e "$path" ]] || return 0

    if [[ "$OUTPUT_CONFLICT_MODE" == "archive" ]]; then
        archive_if_exists "$path"
    else
        rm -rf "$path"
        echo "Deleted old output:"
        echo "  $path"
    fi
}

copy_dir_safely() {
    local src="$1"
    local dst="$2"

    [[ -d "$src" ]] || {
        echo "ERROR: source directory not found:"
        echo "  $src"
        exit 1
    }

    handle_existing_output "$dst"
    cp -a "$src" "$dst"
}

copy_file_safely() {
    local src="$1"
    local dst="$2"

    [[ -f "$src" ]] || {
        echo "ERROR: source file not found:"
        echo "  $src"
        exit 1
    }

    if [[ "$(realpath "$src")" == "$(realpath -m "$dst")" ]]; then
        echo "Skip copy because source and destination are the same:"
        echo "  $src"
        return 0
    fi

    handle_existing_output "$dst"
    cp -f "$src" "$dst"
}

clean_auto_versions_in_dir() {
    local root="$1"
    [[ -d "$root" ]] || return 0

    echo "Cleaning *_auto_Vx in:"
    echo "  $root"

    find "$root" -mindepth 1 -maxdepth 1 \
        \( -type f -o -type d \) \
        -regextype posix-extended \
        -regex '.*/.*_auto_V[0-9]+(\.[^/]+)?$' \
        -print -exec rm -rf {} +
}

find_first_file() {
    local root="$1"
    shift

    local pattern found
    for pattern in "$@"; do
        found="$(find "$root" -maxdepth 1 -type f -name "$pattern" -print -quit 2>/dev/null || true)"
        if [[ -n "$found" ]]; then
            echo "$found"
            return 0
        fi
    done

    return 1
}

find_latest_npz() {
    local root="$1"

    find "$root" \
        -maxdepth 1 \
        -type f \
        -name 'features_clipL14*.npz' \
        -printf '%T@ %p\n' 2>/dev/null | \
        sort -nr | \
        head -n 1 | \
        cut -d' ' -f2-
}

copy_images_matching_jsons() {
    local json_dir="$1"
    local img_src="$2"
    local img_dst="$3"

    [[ -d "$json_dir" ]] || {
        echo "ERROR: JSON folder not found:"
        echo "  $json_dir"
        exit 1
    }

    [[ -d "$img_src" ]] || {
        echo "ERROR: image source folder not found:"
        echo "  $img_src"
        exit 1
    }

    handle_existing_output "$img_dst"
    mkdir -p "$img_dst"

    local copied=0
    local missing=0
    local json_file stem img ext src_img

    shopt -s nullglob

    for json_file in "$json_dir"/*.json; do
        stem="$(basename "$json_file" .json)"

        src_img=""
        for ext in jpg jpeg png tif tiff JPG JPEG PNG TIF TIFF; do
            img="${img_src}/${stem}.${ext}"
            if [[ -f "$img" ]]; then
                src_img="$img"
                break
            fi
        done

        if [[ -n "$src_img" ]]; then
            cp -f "$src_img" "$img_dst/"
            copied=$((copied + 1))
        else
            echo "WARNING: image not found for JSON:"
            echo "  $json_file"
            missing=$((missing + 1))
        fi
    done

    shopt -u nullglob

    echo "Copied matched images:"
    echo "  copied=${copied}"
    echo "  missing=${missing}"

    if [[ "$copied" -eq 0 ]]; then
        echo "ERROR: no matched images were copied."
        exit 1
    fi
}

run_shell_script_in_dir() {
    local workdir="$1"
    local script_name="$2"

    [[ -d "$workdir" ]] || {
        echo "ERROR: folder not found:"
        echo "  $workdir"
        exit 1
    }

    [[ -f "${workdir}/${script_name}" ]] || {
        echo "ERROR: script not found:"
        echo "  ${workdir}/${script_name}"
        exit 1
    }

    cd "$workdir"
    bash "./${script_name}"
}

# ======================= Auto clean =======================

if [[ "$CLEAN_AUTO_VERSIONS" == "true" ]]; then
    echo "[AUTO-CLEAN] Delete previous *_auto_Vx files/folders"

    clean_auto_versions_in_dir "$COLOUR"
    clean_auto_versions_in_dir "$ORN"
    clean_auto_versions_in_dir "$TOOL"
    clean_auto_versions_in_dir "$VIS"
fi

# ======================= Main pipeline =======================

echo "[0/16] Check required input files"

[[ -f "$TEST_JSON" ]] || {
    echo "ERROR: Test JSON not found:"
    echo "  $TEST_JSON"
    exit 1
}

[[ -d "$DLA_IMG_SRC" ]] || {
    echo "ERROR: DLA image folder not found:"
    echo "  $DLA_IMG_SRC"
    exit 1
}

[[ -d "$COLOUR" ]] || {
    echo "ERROR: Colour test folder not found:"
    echo "  $COLOUR"
    exit 1
}

[[ -d "$ORN" ]] || {
    echo "ERROR: ORN test folder not found:"
    echo "  $ORN"
    exit 1
}

[[ -d "$TOOL" ]] || {
    echo "ERROR: TOOL test folder not found:"
    echo "  $TOOL"
    exit 1
}

[[ -d "$VIS" ]] || {
    echo "ERROR: visualization folder not found:"
    echo "  $VIS"
    exit 1
}

echo "Using test annotation file:"
echo "  $TEST_JSON"

echo "[1/16] Run text detection"
run_shell_script_in_dir "$TD" "test_DLA.sh"

echo "[2/16] Locate text detection output"

DETECTION_DIR="${TD}/show/text_detection"
DETECTION_JSON_DIR="${DETECTION_DIR}/jsons"
DETECTION_IMG_DIR="${DETECTION_DIR}/imgs"

[[ -d "$DETECTION_JSON_DIR" ]] || {
    echo "ERROR: detection JSON folder not found:"
    echo "  $DETECTION_JSON_DIR"
    exit 1
}

[[ -d "$DETECTION_IMG_DIR" ]] || {
    echo "ERROR: detection image folder not found:"
    echo "  $DETECTION_IMG_DIR"
    exit 1
}

echo "Detection output:"
echo "  $DETECTION_DIR"

echo "[3/16] Copy detection JSONs to Colour"

copy_dir_safely "$DETECTION_JSON_DIR" "${COLOUR}/jsons"

echo "[4/16] Copy only matched images to Colour"

copy_images_matching_jsons "${COLOUR}/jsons" "$DLA_IMG_SRC" "${COLOUR}/imgs"

# Compatibility folder if old colour script expects 'img'
copy_images_matching_jsons "${COLOUR}/jsons" "$DLA_IMG_SRC" "${COLOUR}/img"

echo "[5/16] Run colour extraction"

cd "$COLOUR"

COLOUR_SCRIPT="${COLOUR_SCRIPT:-}"

if [[ -z "$COLOUR_SCRIPT" ]]; then
    COLOUR_SCRIPT="$(find_first_file "$COLOUR" \
        'V8_EnhancedBinarization_MultiAlgo_Gmm2Color_UpdatedJson_ProgressBar.py' \
        '*Color*.py' \
        '*Colour*.py' \
        '*.py' || true)"
fi

[[ -n "$COLOUR_SCRIPT" && -f "$COLOUR_SCRIPT" ]] || {
    echo "ERROR: cannot find colour Python script in:"
    echo "  $COLOUR"
    echo "You can manually specify it like:"
    echo "  COLOUR_SCRIPT=/path/to/script.py bash ./Run_pipeline_testing_auto_paper_v2.sh"
    exit 1
}

echo "Colour script:"
echo "  $COLOUR_SCRIPT"

python3 "$COLOUR_SCRIPT"

echo "[6/16] Verify colour outputs"

[[ -d "$COLOUR_JSON_OUT" ]] || {
    echo "ERROR: missing:"
    echo "  $COLOUR_JSON_OUT"
    exit 1
}

[[ -d "$COLOUR_PATCH_OUT" ]] || {
    echo "ERROR: missing:"
    echo "  $COLOUR_PATCH_OUT"
    exit 1
}

echo "[7/16] Copy colour outputs to ORN"

copy_dir_safely "$COLOUR_JSON_OUT" "$ORN_JSON_IN"
copy_dir_safely "$COLOUR_PATCH_OUT" "$ORN_PATCH_IN"

echo "[8/16] Prepare ORN model"

ORN_MODEL_SRC="${ORN_MODEL_SRC:-}"

if [[ -z "$ORN_MODEL_SRC" ]]; then
    ORN_MODEL_SRC="$(find_first_file "$TRAIN_ORN" \
        'Orn_classifier_v09_trained_on_all_augmented_training_data.pt' \
        'Orn_classifier_v09_trained_on_all_augmented_data.pt' \
        'Orn_classifier*.pt' || true)"
fi

if [[ -z "$ORN_MODEL_SRC" ]]; then
    ORN_MODEL_SRC="$(find_first_file "$ORN" \
        'Orn_classifier_v09_trained_on_all_augmented_training_data.pt' \
        'Orn_classifier_v09_trained_on_all_augmented_data.pt' \
        'Orn_classifier*.pt' || true)"
fi

[[ -n "$ORN_MODEL_SRC" && -f "$ORN_MODEL_SRC" ]] || {
    echo "ERROR: ORN model not found."
    echo "Checked:"
    echo "  $TRAIN_ORN"
    echo "  $ORN"
    exit 1
}

ORN_MODEL_DST="${ORN}/$(basename "$ORN_MODEL_SRC")"
copy_file_safely "$ORN_MODEL_SRC" "$ORN_MODEL_DST"

echo "ORN model:"
echo "  $ORN_MODEL_DST"

echo "[9/16] Run ORN test"

run_shell_script_in_dir "$ORN" "test.sh"

echo "[10/16] Find ORN features and copy ORN outputs to TOOL"

ORN_FEATURES="$(find_latest_npz "$ORN" || true)"

[[ -n "$ORN_FEATURES" && -f "$ORN_FEATURES" ]] || {
    echo "ERROR: ORN features not found in:"
    echo "  $ORN"
    exit 1
}

echo "ORN features:"
echo "  $ORN_FEATURES"

copy_file_safely "$ORN_FEATURES" "${TOOL}/$(basename "$ORN_FEATURES")"

[[ -d "$ORN_JSON_OUT" ]] || {
    echo "ERROR: ORN JSON output missing:"
    echo "  $ORN_JSON_OUT"
    exit 1
}

[[ -d "$ORN_NAME_OUT" ]] || {
    echo "ERROR: ORN name output missing:"
    echo "  $ORN_NAME_OUT"
    exit 1
}

copy_dir_safely "$ORN_JSON_OUT" "$TOOL_JSON_IN"
copy_dir_safely "$ORN_NAME_OUT" "$TOOL_NAME_IN"

echo "[11/16] Prepare TOOL model"

TOOL_MODEL_SRC="${TOOL_MODEL_SRC:-}"

if [[ -z "$TOOL_MODEL_SRC" ]]; then
    TOOL_MODEL_SRC="$(find_first_file "$TRAIN_TOOL" \
        'writing_tool_classifier_V16_trained_on_all_training_data.pt' \
        'writing_tool_classifier_V16_trained_on_all_data.pt' \
        'writing_tool_classifier*.pt' || true)"
fi

if [[ -z "$TOOL_MODEL_SRC" ]]; then
    TOOL_MODEL_SRC="$(find_first_file "$TOOL" \
        'writing_tool_classifier_V16_trained_on_all_training_data.pt' \
        'writing_tool_classifier_V16_trained_on_all_data.pt' \
        'writing_tool_classifier*.pt' || true)"
fi

[[ -n "$TOOL_MODEL_SRC" && -f "$TOOL_MODEL_SRC" ]] || {
    echo "ERROR: TOOL model not found."
    echo "Checked:"
    echo "  $TRAIN_TOOL"
    echo "  $TOOL"
    exit 1
}

TOOL_MODEL_DST="${TOOL}/$(basename "$TOOL_MODEL_SRC")"
copy_file_safely "$TOOL_MODEL_SRC" "$TOOL_MODEL_DST"

echo "TOOL model:"
echo "  $TOOL_MODEL_DST"

echo "[12/16] Run TOOL test"

run_shell_script_in_dir "$TOOL" "test.sh"

echo "[13/16] Copy TOOL outputs to visualization"

[[ -d "$TOOL_JSON_OUT" ]] || {
    echo "ERROR: TOOL JSON output missing:"
    echo "  $TOOL_JSON_OUT"
    exit 1
}

copy_dir_safely "$TOOL_JSON_OUT" "$VIS_JSON_OUT"

echo "[14/16] Copy matched images to visualization"

copy_images_matching_jsons "$VIS_JSON_OUT" "$DLA_IMG_SRC" "$VIS_IMG_DST"

echo "[15/16] Copy instances_test.json as GT_updated.json"

copy_file_safely "$TEST_JSON" "$VIS_GT_JSON"

echo "GT JSON for visualization:"
echo "  $VIS_GT_JSON"

echo "[16/16] Run visualization pipeline"

run_shell_script_in_dir "$VIS" "4_visualize_all.sh"

echo
echo "DONE."
echo
echo "Final comparison output:"
echo "  ${VIS}/output_compare_pred_top_gt_bottom"

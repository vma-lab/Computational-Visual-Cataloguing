import os
import cv2
import json
import numpy as np
from tqdm import tqdm
import albumentations as A
from multiprocessing import Pool, cpu_count
from pathlib import Path

# === CONFIG ===
IMG_DIR = "img"
ANNOT_PATH = "Converted_DLA_train.json"
OUTPUT_DIR = "augmented_result/img"
OUTPUT_JSON = "augmented_result/instances_training.json"

AUG_TIMES = 4
NUM_WORKERS = max(1, cpu_count() - 1)

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(Path(OUTPUT_JSON).parent, exist_ok=True)

# === AUGMENTATION PIPELINE ===
transform = A.ReplayCompose([
    A.HorizontalFlip(p=0.3),
    A.VerticalFlip(p=0.15),

    A.Rotate(
        limit=10,
        border_mode=cv2.BORDER_CONSTANT,
        p=0.6
    ),

    A.Affine(
        scale=(0.96, 1.04),
        shear=(-3, 3),
        p=0.6
    ),

    A.Perspective(
        scale=(0.01, 0.02),
        p=0.4
    ),

    A.RandomBrightnessContrast(
        brightness_limit=0.05,
        contrast_limit=0.05,
        p=0.6
    ),

    A.RandomGamma(
        gamma_limit=(95, 105),
        p=0.2
    ),

    A.ColorJitter(
        brightness=0.02,
        contrast=0.02,
        saturation=0.02,
        hue=0.01,
        p=0.2
    ),

    A.GaussianBlur(
        blur_limit=3,
        p=0.3
    ),

    A.ISONoise(
        color_shift=(0.01, 0.02),
        intensity=(0.08, 0.15),
        p=0.2
    ),

    A.MultiplicativeNoise(
        multiplier=(0.96, 1.04),
        per_channel=True,
        p=0.15
    ),

    A.ImageCompression(
        p=0.15
    )
])


# === LOAD COCO JSON ===
with open(ANNOT_PATH, "r", encoding="utf-8") as f:
    coco = json.load(f)

categories = coco["categories"]


# === BUILD ANNOTATION MAP ===
ann_map = {}

for ann in coco["annotations"]:
    ann_map.setdefault(
        ann["image_id"],
        []
    ).append(ann)


# === IMAGE PATH RESOLVER ===
def resolve_image_path(file_name: str) -> str:
    """
    Try several possible locations for the image.

    Priority:
    1. IMG_DIR / file_name
    2. file_name directly
    3. ./img / file_name
    4. directory next to annotation JSON / img / file_name
    """

    candidates = []

    if IMG_DIR:
        candidates.append(
            Path(IMG_DIR) / file_name
        )

    candidates.append(
        Path(file_name)
    )

    candidates.append(
        Path("img") / file_name
    )

    candidates.append(
        Path(ANNOT_PATH).parent / "img" / file_name
    )

    for candidate in candidates:
        if candidate.exists():
            return str(candidate)

    # fallback
    if IMG_DIR:
        return str(
            Path(IMG_DIR) / file_name
        )

    return file_name


# === CLEAN POLYGON ===
def clean_polygon(
    contour,
    img_w,
    img_h,
    max_points=50
):
    """
    Validate and simplify polygons.

    Returns:
        contour, area

    or:
        None
    """

    if contour is None:
        return None

    if len(contour) < 4:
        return None

    if contour.ndim != 2:
        return None

    if contour.shape[1] != 2:
        return None

    # Clip coordinates to image boundaries
    contour[:, 0] = np.clip(
        contour[:, 0],
        0,
        img_w - 1
    )

    contour[:, 1] = np.clip(
        contour[:, 1],
        0,
        img_h - 1
    )

    # Simplify polygon if too many points
    if len(contour) > max_points:

        peri = cv2.arcLength(
            contour,
            True
        )

        epsilon = 0.001 * peri

        simplified = cv2.approxPolyDP(
            contour,
            epsilon,
            True
        )

        while (
            len(simplified) > max_points
            and epsilon < 0.05 * peri
        ):

            epsilon *= 1.2

            simplified = cv2.approxPolyDP(
                contour,
                epsilon,
                True
            )

        contour = simplified.squeeze(1)

    # Validate again after simplification
    if len(contour) < 4:
        return None

    if contour.ndim != 2:
        return None

    if contour.shape[1] != 2:
        return None

    flat = contour.flatten()

    if len(flat) < 8:
        return None

    if len(flat) % 2 != 0:
        return None

    area = cv2.contourArea(
        contour
    )

    # Remove tiny polygons
    if area < 50:
        return None

    return contour, area


# === PROCESS ONE IMAGE ===
def process_image(args):

    (
        img_info,
        anns,
        img_id_start,
        ann_id_start
    ) = args

    # Resolve path
    img_path = resolve_image_path(
        img_info["file_name"]
    )

    # Read original image
    img_orig = cv2.imread(
        img_path
    )

    if img_orig is None:

        print(
            f"⚠️ Cannot read image: "
            f"{img_info['file_name']} "
            f"(tried: {img_path})"
        )

        return [], []

    img_results = []
    ann_results = []

    # Original + AUG_TIMES augmentations
    for aug_idx in range(
        AUG_TIMES + 1
    ):

        # ==========================================
        # ORIGINAL IMAGE
        # ==========================================
        if aug_idx == 0:

            img_aug = img_orig.copy()

            replay = None

            new_filename = Path(
                img_info["file_name"]
            ).name

        # ==========================================
        # AUGMENTED IMAGE
        # ==========================================
        else:

            replayed = transform(
                image=img_orig
            )

            img_aug = replayed[
                "image"
            ]

            replay = replayed[
                "replay"
            ]

            original_name = Path(
                img_info["file_name"]
            ).name

            name_prefix = os.path.splitext(
                original_name
            )[0]

            new_filename = (
                f"{name_prefix}"
                f"_aug_{aug_idx}.jpg"
            )

        # ==========================================
        # SAVE IMAGE
        # ==========================================
        out_path = os.path.join(
            OUTPUT_DIR,
            new_filename
        )

        ok = cv2.imwrite(
            out_path,
            img_aug
        )

        if not ok:

            print(
                f"⚠️ Failed to write: "
                f"{out_path}"
            )

        # ==========================================
        # IMAGE COCO ENTRY
        # ==========================================
        curr_img_id = (
            img_id_start
            + aug_idx
        )

        img_results.append({
            "id": curr_img_id,
            "file_name": new_filename,
            "width": int(
                img_aug.shape[1]
            ),
            "height": int(
                img_aug.shape[0]
            )
        })

        # ==========================================
        # PROCESS ANNOTATIONS
        # ==========================================
        for ann in anns:

            segmentation_data = ann.get(
                "segmentation",
                []
            )

            if not segmentation_data:
                continue

            # Current code assumes first polygon
            polygon_data = segmentation_data[0]

            if len(polygon_data) < 8:
                continue

            try:

                poly = np.array(
                    polygon_data,
                    dtype=np.float32
                ).reshape(
                    (-1, 2)
                )

            except Exception:
                continue

            # ======================================
            # CREATE MASK FROM ORIGINAL POLYGON
            # ======================================
            mask = np.zeros(
                img_orig.shape[:2],
                dtype=np.uint8
            )

            poly_int = np.round(
                poly
            ).astype(
                np.int32
            )

            cv2.fillPoly(
                mask,
                [poly_int],
                color=1
            )

            # ======================================
            # APPLY SAME AUGMENTATION TO MASK
            # ======================================
            if aug_idx == 0:

                mask_aug = mask

            else:

                replay_result = (
                    A.ReplayCompose.replay(
                        replay,
                        image=img_orig,
                        mask=mask
                    )
                )

                mask_aug = replay_result[
                    "mask"
                ]

            # Ensure binary mask
            mask_aug = (
                mask_aug > 0
            ).astype(
                np.uint8
            )

            # ======================================
            # FIND POLYGON AGAIN
            # ======================================
            contours, _ = (
                cv2.findContours(
                    mask_aug,
                    cv2.RETR_EXTERNAL,
                    cv2.CHAIN_APPROX_SIMPLE
                )
            )

            for contour in contours:

                if contour is None:
                    continue

                contour = contour.squeeze(
                    1
                )

                cleaned = clean_polygon(
                    contour,
                    img_aug.shape[1],
                    img_aug.shape[0]
                )

                if cleaned is None:
                    continue

                contour_clean, area = (
                    cleaned
                )

                # ==================================
                # BOUNDING BOX
                # ==================================
                (
                    x,
                    y,
                    w_box,
                    h_box
                ) = cv2.boundingRect(
                    contour_clean
                )

                # ==================================
                # SEGMENTATION
                # ==================================
                segmentation = (
                    contour_clean
                    .flatten()
                    .astype(float)
                    .tolist()
                )

                # ==================================
                # COCO ANNOTATION
                # ==================================
                ann_results.append({
                    "id": ann_id_start,
                    "image_id": curr_img_id,
                    "category_id": ann[
                        "category_id"
                    ],
                    "segmentation": [
                        segmentation
                    ],
                    "bbox": [
                        float(x),
                        float(y),
                        float(w_box),
                        float(h_box)
                    ],
                    "area": float(area),
                    "iscrowd": ann.get(
                        "iscrowd",
                        0
                    )
                })

                ann_id_start += 1

    return (
        img_results,
        ann_results
    )


# ==============================================
# MULTIPROCESSING
# ==============================================
if __name__ == "__main__":

    args_list = []

    start_img_id = 10000
    start_ann_id = 20000

    # Each original image generates:
    # 1 original + AUG_TIMES augmented images
    images_per_original = (
        AUG_TIMES + 1
    )

    for idx, img_info in enumerate(
        coco["images"]
    ):

        anns = ann_map.get(
            img_info["id"],
            []
        )

        args_list.append((
            img_info,
            anns,

            # unique image IDs
            start_img_id
            + idx
            * images_per_original,

            # reserve a large annotation range
            # for each original image
            start_ann_id
            + idx
            * 10000
        ))

    print(
        f"Images: {len(coco['images'])}"
    )

    print(
        f"Annotations: "
        f"{len(coco['annotations'])}"
    )

    print(
        f"Augmentations per image: "
        f"{AUG_TIMES}"
    )

    print(
        f"Expected images after augmentation: "
        f"{len(coco['images']) * images_per_original}"
    )

    print(
        f"Workers: {NUM_WORKERS}"
    )

    # ==========================================
    # RUN MULTIPROCESSING
    # ==========================================
    with Pool(
        NUM_WORKERS
    ) as pool:

        results = list(
            tqdm(
                pool.imap_unordered(
                    process_image,
                    args_list
                ),
                total=len(
                    args_list
                ),
                desc="Augmenting"
            )
        )

    # ==========================================
    # COMBINE RESULTS
    # ==========================================
    new_images = []
    new_annotations = []

    for imgs, anns in results:

        new_images.extend(
            imgs
        )

        new_annotations.extend(
            anns
        )

    # ==========================================
    # SORT RESULTS
    # ==========================================
    # Because imap_unordered returns workers
    # in arbitrary order.

    new_images.sort(
        key=lambda x: x["id"]
    )

    new_annotations.sort(
        key=lambda x: x["id"]
    )

    # ==========================================
    # SAVE TRAIN JSON
    # ==========================================
    train_json = {
        "info": coco.get(
            "info",
            {}
        ),

        "licenses": coco.get(
            "licenses",
            []
        ),

        "images": new_images,

        "annotations": new_annotations,

        "categories": categories
    }

    with open(
        OUTPUT_JSON,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            train_json,
            f,
            indent=2,
            ensure_ascii=False
        )

    # ==========================================
    # SUMMARY
    # ==========================================
    print()
    print(
        "✅ Done: Training augmentation complete."
    )

    print(
        f"Original images: "
        f"{len(coco['images'])}"
    )

    print(
        f"Output images: "
        f"{len(new_images)}"
    )

    print(
        f"Output annotations: "
        f"{len(new_annotations)}"
    )

    print(
        f"Saved images to: "
        f"{OUTPUT_DIR}"
    )

    print(
        f"Saved annotations to: "
        f"{OUTPUT_JSON}"
    )

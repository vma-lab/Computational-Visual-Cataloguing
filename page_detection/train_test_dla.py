import csv
import json
import platform
from pathlib import Path

import numpy as np
import yaml
import torch
from ultralytics import YOLO

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
# Folders produced by the split script (each has images/ and labels/ inside):
TRAIN_DIR = Path(r"E:/Projects_PyCharm/YOLO/data/dla/train")
TEST_DIR = Path(r"E:/Projects_PyCharm/YOLO/data/dla/test")

MODEL = "yolo11s.pt"
CLASS_NAMES = ["page"]
K = 5  # notebook-level folds (80 train -> ~64 train / ~16 val each)

RESULTS = Path("./Results_page_cv_select")  # all runs go here (fresh folder)
SPLITS_DIR = Path("./page_kfold_splits_cv_select")  # generated fold lists + yamls

IMGSZ = 640
BATCH = 16
# Model selection by notebook-level CV: every candidate is trained exactly like the
# final model (fixed number of epochs, no early stopping) and scored at its last epoch.
# Fix this list BEFORE running; do not change it after seeing test results.
CANDIDATE_EPOCHS = [10, 20, 30, 50]
SEED = 42
SAVE_TEST_PREDICTIONS = True  # also dump annotated test images for manual check
# ---------------------------------------------------------------------------

IS_WINDOWS = platform.system() == "Windows"
DEVICE = 0 if torch.cuda.is_available() else "cpu"
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

# Shared training args (same config for CV candidates and the final run).
COMMON = dict(
    imgsz=IMGSZ,
    batch=BATCH,
    rect=False,
    cache=True,
    device=DEVICE,
    workers=0 if IS_WINDOWS else 8,
    seed=SEED,
    amp=True,
    save=True,
    plots=True,
    single_cls=True,  # one class ('page')
    max_det=1,  # one primary page per image: the highest-scoring box is the page
    optimizer="auto",  # Ultralytics selects AdamW for runs this short
    # ---- augmentation: primarily photometric, mild geometric jitter ----
    fliplr=0.5,
    flipud=0.0,
    degrees=0.0,  # rotation would inflate the axis-aligned page boxes
    translate=0.05,
    scale=0.1,  # large zooms cut pages, and cut pages would stay labelled "page"
    mosaic=0.0,  # mosaic cuts pages at tile borders; partial pages must not be detected
    close_mosaic=0,
    hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
    mixup=0.0, shear=0.0, perspective=0.0,
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def list_images(images_dir):
    imgs = sorted(p for p in images_dir.iterdir() if p.suffix.lower() in IMG_EXT)
    if not imgs:
        raise FileNotFoundError(f"No images found in {images_dir}")
    return imgs


def write_list(path, paths):
    path.write_text("\n".join(str(p.resolve()) for p in paths), encoding="utf-8")


def notebook_id(path):
    """HS01309362_0003.jpg -> HS01309362 ; Rilke_nb_T15_Auszug_Page2.jpg -> T15"""
    stem = Path(path).stem
    parts = stem.split("_")
    if stem.startswith("Rilke_nb_") and len(parts) > 2:
        return parts[2]
    return parts[0]


def write_yaml(path, train_txt, val_txt):
    """YOLO finds labels by swapping the last 'images' -> 'labels' in each path,
    so train/images/x.jpg pairs with train/labels/x.txt automatically."""
    data = {
        "path": str(TRAIN_DIR.parent.resolve()),  # harmless; train/val below are absolute
        "train": str(train_txt.resolve()),
        "val": str(val_txt.resolve()),
        "names": {i: n for i, n in enumerate(CLASS_NAMES)},
        "nc": len(CLASS_NAMES),
    }
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)


def read_last(save_dir):
    """(mAP50-95, mAP50) of the LAST epoch in a run's results.csv (no epoch selection)."""
    with open(Path(save_dir) / "results.csv", "r", encoding="utf-8") as f:
        rows = [r for r in csv.reader(f) if r]
    header = [h.strip() for h in rows[0]]
    i95 = next(i for i, h in enumerate(header) if "mAP50-95" in h)
    i50 = next(i for i, h in enumerate(header) if "mAP50(B)" in h and "mAP50-95" not in h)
    return float(rows[-1][i95]), float(rows[-1][i50])


# ---------------------------------------------------------------------------
# stages
# ---------------------------------------------------------------------------
def run_kfold():
    train_imgs = list_images(TRAIN_DIR / "images")
    print(f"TRAIN images: {len(train_imgs)}   K={K}   model={MODEL}")

    # notebook-level folds: all pages of a notebook go into the same fold
    groups = np.array([notebook_id(p) for p in train_imgs])
    test_nbs = {notebook_id(p) for p in list_images(TEST_DIR / "images")}
    if set(groups) & test_nbs:
        raise ValueError(f"Notebooks in both train and test: {sorted(set(groups) & test_nbs)}")
    rng = np.random.default_rng(SEED)
    nbs = sorted(rng.permutation(np.unique(groups)), key=lambda n: -(groups == n).sum())
    fold_of, sizes = {}, [0] * K
    for n in nbs:  # largest notebooks first, each into the currently smallest fold
        j = int(np.argmin(sizes))
        fold_of[n] = j
        sizes[j] += int((groups == n).sum())
    chunks = [np.where([fold_of[g] == j for g in groups])[0] for j in range(K)]
    folds = [
        (np.concatenate([chunks[m] for m in range(K) if m != j]), chunks[j])
        for j in range(K)
    ]
    for j, c in enumerate(chunks):
        print(f"fold{j}: {len(c)} images, notebooks = {sorted(str(g) for g in set(groups[c]))}")

    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    for k, (tr, va) in enumerate(folds):  # write the fold lists once
        tr_txt = SPLITS_DIR / f"fold{k}_train.txt"
        va_txt = SPLITS_DIR / f"fold{k}_val.txt"
        write_list(tr_txt, [train_imgs[i] for i in tr])
        write_list(va_txt, [train_imgs[i] for i in va])
        write_yaml(SPLITS_DIR / f"fold{k}.yaml", tr_txt, va_txt)

    # model selection: number of training epochs, chosen by mean CV mAP50-95
    scores = {}
    for E in CANDIDATE_EPOCHS:
        scores[E] = []
        for k in range(K):
            print(f"\n===== Candidate {E} epochs | fold {k + 1}/{K} =====")
            model = YOLO(MODEL)  # fresh weights every fold
            model.train(
                data=str(SPLITS_DIR / f"fold{k}.yaml"),
                project=str(RESULTS / "kfold"),
                name=f"e{E}_fold{k}",
                epochs=E,
                patience=E + 1,  # no early stopping: identical recipe to the final run
                **COMMON,
            )
            scores[E].append(read_last(model.trainer.save_dir))  # actual run folder
        m95 = np.array([s[0] for s in scores[E]])
        print(f"Candidate {E} epochs: CV mAP50-95 = {m95.mean():.4f} +/- {m95.std():.4f}")

    # ties -> fewer epochs (list order)
    best_e = max(CANDIDATE_EPOCHS, key=lambda e: np.mean([s[0] for s in scores[e]]))
    m95 = np.array([s[0] for s in scores[best_e]])
    m50 = np.array([s[1] for s in scores[best_e]])
    cv = {
        "selected_epochs": best_e,
        "candidates_mean_map5095": {str(e): float(np.mean([s[0] for s in scores[e]]))
                                    for e in CANDIDATE_EPOCHS},
        "folds": [{"map5095": a, "map50": b} for a, b in scores[best_e]],
        "mean_map5095": float(m95.mean()), "std_map5095": float(m95.std()),
        "mean_map50": float(m50.mean()), "std_map50": float(m50.std()),
    }
    print("\n================ K-FOLD SUMMARY (TRAIN) ================")
    for e in CANDIDATE_EPOCHS:
        print(f"  {e:>3} epochs: mean mAP50-95 = {cv['candidates_mean_map5095'][str(e)]:.4f}")
    print(f"  selected: {best_e} epochs -> mAP50-95 = {cv['mean_map5095']:.4f} +/- {cv['std_map5095']:.4f}, "
          f"mAP50 = {cv['mean_map50']:.4f} +/- {cv['std_map50']:.4f}")
    return train_imgs, best_e, cv


def train_final(train_imgs, final_epochs):
    """Train the deployment model on ALL train images with the selected number of epochs."""
    all_txt = SPLITS_DIR / "all_train.txt"
    all_yaml = SPLITS_DIR / "all_train.yaml"
    write_list(all_txt, train_imgs)
    write_yaml(all_yaml, all_txt, all_txt)  # val == train; early stopping off below

    print(f"\n############ FINAL TRAIN on all {len(train_imgs)} train images "
          f"for {final_epochs} epochs ############")
    model = YOLO(MODEL)
    model.train(
        data=str(all_yaml),
        project=str(RESULTS),
        name="final_all_train",
        epochs=final_epochs,
        patience=final_epochs + 1,  # no early stopping
        **COMMON,
    )
    weights = Path(model.trainer.save_dir) / "weights" / "last.pt"  # actual run folder
    print(f"Final weights: {weights}  (fully trained, use this)")
    return weights


def evaluate_test(weights):
    """Run the final model on the TEST set and report mAP (test set has labels)."""
    test_imgs = list_images(TEST_DIR / "images")
    test_txt = SPLITS_DIR / "test.txt"
    test_yaml = SPLITS_DIR / "test.yaml"
    write_list(test_txt, test_imgs)
    write_yaml(test_yaml, test_txt, test_txt)  # val points at the test images

    print(f"\n############ TEST evaluation on {len(test_imgs)} images ############")
    model = YOLO(weights)
    metrics = model.val(
        data=str(test_yaml),
        imgsz=IMGSZ, batch=BATCH, device=DEVICE, max_det=1,
        project=str(RESULTS), name="test_eval", plots=True,
    )
    test_m95 = float(metrics.box.map)  # mAP50-95
    test_m50 = float(metrics.box.map50)  # mAP50
    print(f"TEST mAP50-95 = {test_m95:.4f}")
    print(f"TEST mAP50    = {test_m50:.4f}")

    if SAVE_TEST_PREDICTIONS:
        model.predict(
            source=str(TEST_DIR / "images"),
            imgsz=IMGSZ, conf=0.001, max_det=1, device=DEVICE,  # always draw the top box
            save=True, project=str(RESULTS), name="test_predictions",
        )
        print(f"Annotated predictions saved under {RESULTS / 'test_predictions'}")

    return test_m95, test_m50


# ---------------------------------------------------------------------------
def main():
    torch.cuda.empty_cache()
    train_imgs, final_epochs, cv = run_kfold()
    weights = train_final(train_imgs, final_epochs)
    test_m95, test_m50 = evaluate_test(weights)

    print("\n==================== OVERALL ====================")
    print(f"CV (train, {K}-fold, selected {final_epochs} epochs): "
          f"mAP50-95={cv['mean_map5095']:.4f}+/-{cv['std_map5095']:.4f}  "
          f"mAP50={cv['mean_map50']:.4f}+/-{cv['std_map50']:.4f}")
    print(f"TEST (held-out)     : mAP50-95={test_m95:.4f}  mAP50={test_m50:.4f}")

    RESULTS.mkdir(parents=True, exist_ok=True)
    summary = {
        "model": MODEL, "k": K, "imgsz": IMGSZ, "final_epochs": final_epochs,
        "cv": cv,
        "test_map5095": test_m95, "test_map50": test_m50,
        "final_weights": str(weights),
    }
    with open(RESULTS / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary written to {RESULTS / 'summary.json'}")


if __name__ == "__main__":
    main()

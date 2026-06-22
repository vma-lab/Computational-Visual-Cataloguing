import csv
import json
import platform
from pathlib import Path

import numpy as np
import yaml
import torch
from ultralytics import YOLO

# ---------------------------------------------------------------------------
# CONFIG  -- edit these to match your machine
# ---------------------------------------------------------------------------
# Folders produced by the split script (each has images/ and labels/ inside):
TRAIN_DIR = Path(r"E:/Projects_PyCharm/YOLO/data/dla/train")
TEST_DIR = Path(r"E:/Projects_PyCharm/YOLO/data/dla/test")

MODEL = "yolo11s.pt"
CLASS_NAMES = ["page"]
K = 5  # folds (80 train -> ~64 train / ~16 val each)

RESULTS = Path("./Results_page_smodel")  # all runs go here
SPLITS_DIR = Path("./page_kfold_splits_smodel")  # generated fold lists + yamls

IMGSZ = 640
BATCH = 16
FOLD_EPOCHS = 100  # easy task converges fast; early stopping ends folds sooner
FOLD_PATIENCE = 30
SEED = 42
SAVE_TEST_PREDICTIONS = True  # also dump annotated test images for manual check
# ---------------------------------------------------------------------------

IS_WINDOWS = platform.system() == "Windows"
DEVICE = 0 if torch.cuda.is_available() else "cpu"
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

# Shared training args (same config for folds and the final run).
COMMON = dict(
    imgsz=IMGSZ,
    batch=BATCH,
    rect=False,  # let mosaic work (the printing-block model used rect=True; opposite here)
    cache=True,
    device=DEVICE,
    workers=0 if IS_WINDOWS else 8,
    seed=SEED,
    amp=True,
    save=True,
    plots=True,
    single_cls=True,  # one class ('page')
    optimizer="auto",
    # ---- augmentation: generous, since the single class has no positional meaning ----
    fliplr=0.5,
    flipud=0.0,  # pages aren't upside-down in practice; safe to raise for more regularisation
    degrees=10.0,  # small rotation only; keeps boxes tight
    translate=0.1,
    scale=0.5,
    mosaic=1.0,
    close_mosaic=10,
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


def read_best(csv_path):
    """Best (epoch, mAP50-95, mAP50) row from a run's results.csv."""
    try:
        with open(csv_path, "r", encoding="utf-8") as f:
            rows = list(csv.reader(f))
    except OSError:
        return None
    if not rows:
        return None
    header = [h.strip() for h in rows[0]]

    def find(pred):
        return next((i for i, h in enumerate(header) if pred(h)), None)

    i_ep = find(lambda h: h == "epoch")
    i_m95 = find(lambda h: "mAP50-95" in h)
    i_m50 = find(lambda h: "mAP50(B)" in h and "mAP50-95" not in h)
    if i_m95 is None:
        return None

    best = None
    for r in rows[1:]:
        if not r:
            continue
        try:
            v = float(r[i_m95])
        except (ValueError, IndexError):
            continue
        if best is None or v > best["map5095"]:
            best = {
                "epoch": int(float(r[i_ep])) if i_ep is not None else FOLD_EPOCHS,
                "map5095": v,
                "map50": float(r[i_m50]) if i_m50 is not None else float("nan"),
            }
    return best


# ---------------------------------------------------------------------------
# stages
# ---------------------------------------------------------------------------
def run_kfold():
    train_imgs = list_images(TRAIN_DIR / "images")
    print(f"TRAIN images: {len(train_imgs)}   K={K}   model={MODEL}")

    # random folds (single class -> no stratification needed)
    rng = np.random.default_rng(SEED)
    chunks = np.array_split(rng.permutation(len(train_imgs)), K)
    folds = [
        (np.concatenate([chunks[m] for m in range(K) if m != j]), chunks[j])
        for j in range(K)
    ]

    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    results = []

    for k, (tr, va) in enumerate(folds):
        tr_txt = SPLITS_DIR / f"fold{k}_train.txt"
        va_txt = SPLITS_DIR / f"fold{k}_val.txt"
        y_yaml = SPLITS_DIR / f"fold{k}.yaml"
        write_list(tr_txt, [train_imgs[i] for i in tr])
        write_list(va_txt, [train_imgs[i] for i in va])
        write_yaml(y_yaml, tr_txt, va_txt)

        print(f"\n===== Fold {k + 1}/{K}  (train={len(tr)}, val={len(va)}) =====")
        model = YOLO(MODEL)  # fresh weights every fold
        model.train(
            data=str(y_yaml),
            project=str(RESULTS / "kfold"),
            name=f"fold{k}",
            epochs=FOLD_EPOCHS,
            patience=FOLD_PATIENCE,
            **COMMON,
        )
        best = read_best(RESULTS / "kfold" / f"fold{k}" / "results.csv")
        if best:
            print(f"Fold {k}: mAP50-95={best['map5095']:.4f}  "
                  f"mAP50={best['map50']:.4f}  @ epoch {best['epoch']}")
            results.append(best)

    cv = {"folds": results, "mean_map5095": None, "std_map5095": None,
          "mean_map50": None, "std_map50": None}
    if results:
        m95 = np.array([r["map5095"] for r in results])
        m50 = np.array([r["map50"] for r in results])
        ep = np.array([r["epoch"] for r in results])
        print("\n================ K-FOLD SUMMARY (TRAIN) ================")
        for k, r in enumerate(results):
            print(f"  fold{k}: mAP50-95={r['map5095']:.4f}  "
                  f"mAP50={r['map50']:.4f}  best_epoch={r['epoch']}")
        cv.update(mean_map5095=float(m95.mean()), std_map5095=float(m95.std()),
                  mean_map50=float(m50.mean()), std_map50=float(m50.std()))
        print(f"  mean mAP50-95 = {cv['mean_map5095']:.4f} +/- {cv['std_map5095']:.4f}")
        print(f"  mean mAP50    = {cv['mean_map50']:.4f} +/- {cv['std_map50']:.4f}")
        final_epochs = max(int(round(ep.mean())), 30)  # CV-informed budget; raise if undertrained
    else:
        print("[warn] no fold metrics parsed; using default final epochs.")
        final_epochs = FOLD_EPOCHS

    return train_imgs, final_epochs, cv


def train_final(train_imgs, final_epochs):
    """Train the deployment model on ALL train images for a fixed CV-informed budget."""
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
    wdir = RESULTS / "final_all_train" / "weights"
    print(f"Final weights: {wdir / 'last.pt'}  (fully trained, use this)")
    return wdir / "last.pt"


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
        imgsz=IMGSZ, batch=BATCH, device=DEVICE,
        project=str(RESULTS), name="test_eval", plots=True,
    )
    test_m95 = float(metrics.box.map)  # mAP50-95
    test_m50 = float(metrics.box.map50)  # mAP50
    print(f"TEST mAP50-95 = {test_m95:.4f}")
    print(f"TEST mAP50    = {test_m50:.4f}")

    if SAVE_TEST_PREDICTIONS:
        model.predict(
            source=str(TEST_DIR / "images"),
            imgsz=IMGSZ, conf=0.25, device=DEVICE,
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
    if cv["mean_map5095"] is not None:
        print(f"CV (train, {K}-fold): mAP50-95={cv['mean_map5095']:.4f}+/-{cv['std_map5095']:.4f}  "
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

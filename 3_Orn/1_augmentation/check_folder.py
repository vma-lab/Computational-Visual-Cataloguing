import os

PATCH_FOLDER = "text_patches_train_80"
CLASS_NAMES = ["hor", "vrt", "tlt"]

# === Tracking variables ===
total_files = 0
valid_images = 0

skipped_ext_files = []
skipped_format_files = []
skipped_label_files = []

for fname in sorted(os.listdir(PATCH_FOLDER)):
    total_files += 1

    # Check extension
    if not fname.lower().endswith(('.jpg', '.jpeg', '.png')):
        skipped_ext_files.append(fname)
        continue

    # Check naming format
    parts = fname.split("___")
    if len(parts) < 2:
        skipped_format_files.append(fname)
        continue

    label_parts = parts[1].split("_")
    if len(label_parts) < 2:
        skipped_format_files.append(fname)
        continue

    label = label_parts[1].strip().lower()
    if label not in CLASS_NAMES:
        skipped_label_files.append(fname)
        continue

    valid_images += 1

# === Save skipped filenames ===
def save_list(filename, items):
    if items:
        with open(filename, "w") as f:
            f.write("\n".join(items))
        print(f"📝 Saved {len(items)} skipped files to {filename}")

save_list("skipped_extension.txt", skipped_ext_files)
save_list("skipped_format.txt", skipped_format_files)
save_list("skipped_label.txt", skipped_label_files)

# === Report ===
print("\n🔍 Summary")
print("=" * 40)
print(f"📂 Total files in folder     : {total_files}")
print(f"✅ Valid image patches       : {valid_images}")
print(f"❌ Skipped due to extension  : {len(skipped_ext_files)}")
print(f"❌ Skipped due to format     : {len(skipped_format_files)}")
print(f"❌ Skipped due to bad label  : {len(skipped_label_files)}")
print("=" * 40)


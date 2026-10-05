import os
from collections import Counter

PATCH_FOLDER = "text_patches_train_80"
TOOL_NAMES = ["pcl", "ink"]

# === Tracking variables ===
total_files = 0
valid_images = 0

skipped_ext_files = []
skipped_format_files = []
skipped_tool_files = []

found_tool_labels = []

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
    if len(label_parts) < 3:  # Expected at least: idx_orientation_tool
        skipped_format_files.append(fname)
        continue

    tool = label_parts[2].strip().lower()
    found_tool_labels.append(tool)

    if tool not in TOOL_NAMES:
        skipped_tool_files.append(fname)
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
save_list("skipped_tool.txt", skipped_tool_files)

# === Tool label summary ===
tool_counter = Counter(found_tool_labels)
with open("found_tool_labels.txt", "w") as f:
    for tool, count in sorted(tool_counter.items()):
        f.write(f"{tool}: {count}\n")
print("🔍 Saved all found tool labels to found_tool_labels.txt")

invalid_labels = [lbl for lbl in tool_counter if lbl not in TOOL_NAMES]
if invalid_labels:
    with open("skipped_tool_labels.txt", "w") as f:
        for lbl in invalid_labels:
            f.write(f"{lbl}\n")
    print(f"⚠️ Found unexpected tool labels: {', '.join(invalid_labels)}")
    print("📝 Saved to skipped_tool_labels.txt")

# === Report ===
print("\n🔍 Summary")
print("=" * 40)
print(f"📂 Total files in folder     : {total_files}")
print(f"✅ Valid image patches       : {valid_images}")
print(f"❌ Skipped due to extension  : {len(skipped_ext_files)}")
print(f"❌ Skipped due to format     : {len(skipped_format_files)}")
print(f"❌ Skipped due to bad tool   : {len(skipped_tool_files)}")
print(f"🔎 Unique tool labels found  : {len(tool_counter)} ({', '.join(tool_counter.keys())})")
print("=" * 40)


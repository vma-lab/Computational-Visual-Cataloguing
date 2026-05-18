from ultralytics import YOLO
import torch
import os
import cv2
import numpy as np
import csv
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

# Set CUDA devices visible to PyTorch
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
DEVICE = "cuda:1" if torch.cuda.is_available() else "cpu"

# Clears CUDA memory
torch.cuda.empty_cache()

# Directory to save images with detections and cropped objects
output_dir = r"\\uhhdisk.uni-hamburg.de\daten\gw\CSMC_Projects_5\DLA\cropped_pages"
os.makedirs(output_dir, exist_ok=True)

# Load the YOLO model
model = YOLO('E:/Projects_PyCharm/YOLO/models/DLA_pages2.pt').to(DEVICE)

source_dir = r'\\uhhdisk.uni-hamburg.de\daten\gw\CSMC_Projects_5\DLA\images_png\all_images'

if os.path.exists(source_dir):
    print("Directory exists!")
    print("Contents:", os.listdir(source_dir))
else:
    print("Directory not accessible or does not exist.")

image_extensions = {'.jpg', '.jpeg', '.tif', '.tiff', '.png', '.bmp'}
image_paths = []
for root, _, files in os.walk(source_dir):
    for file in files:
        if os.path.splitext(file)[1].lower() in image_extensions:
            image_paths.append(os.path.join(root, file))


def save_crops(result, model, output_dir, num_detections=1, padding=0):
    """
    Crops and saves detected objects from the result with optional padding.
    :param result: A single YOLO result (from model.predict).
    :param model: The loaded YOLO model (for class names).
    :param output_dir: Directory to save the cropped images.
    :param num_detections: Number of detections to consider (1 => only first detection).
    :param padding: Extra pixels to add around the bounding box.
    """
    if not result.boxes:
        return

    orig_img = result.orig_img
    img_h, img_w = orig_img.shape[:2]  # Get image dimensions
    boxes = result.boxes

    for i, box in enumerate(boxes):
        if i >= num_detections:
            break

        # Extract bounding box coords and class
        x1, y1, x2, y2 = box.xyxy[0].int().tolist()
        class_id = int(box.cls[0])
        class_name = model.names[class_id]

        # Apply padding while keeping within image bounds
        x1 = max(0, x1 - padding)
        y1 = max(0, y1 - padding)
        x2 = min(img_w, x2 + padding)
        y2 = min(img_h, y2 + padding)

        # Crop the detection with padding
        cropped_img = orig_img[y1:y2, x1:x2]

        # Generate crop filename with index
        base_filename = os.path.splitext(os.path.basename(result.path))[0]
        crop_filename = f"{base_filename}_crop_{i + 1}.jpg"
        crop_path = os.path.join(output_dir, crop_filename)

        # Save the cropped detection
        cv2.imwrite(crop_path, cropped_img)


def save_detections_csv(result, csv_writer, num_detections=1):
    """
    Writes up to num_detections boxes from result into an open csv_writer
    as: image_name, class_id, x_center_norm, y_center_norm, width_norm, height_norm
    """
    if not result.boxes:
        return

    img_h, img_w = result.orig_img.shape[:2]
    # sort by confidence descending
    boxes = sorted(result.boxes, key=lambda b: float(b.conf[0]), reverse=True)

    for box in boxes[:num_detections]:
        x1, y1, x2, y2 = box.xyxy[0].int().tolist()
        cls_id = int(box.cls[0])

        x_c = ((x1 + x2) / 2) / img_w
        y_c = ((y1 + y2) / 2) / img_h
        w_n = (x2 - x1) / img_w
        h_n = (y2 - y1) / img_h

        csv_writer.writerow([
            os.path.basename(result.path),
            cls_id,
            x_c, y_c,
            w_n, h_n
        ])


csv_f = open(os.path.join(output_dir, 'detections.csv'), 'w', newline='')
csv_w = csv.writer(csv_f)
csv_w.writerow([
    'image_name', 'class_id',
    'x_center_norm', 'y_center_norm',
    'width_norm', 'height_norm'
])

# Run predictions
for image_path in image_paths:
    results = model.predict(
        source=image_path,
        save=False,
        save_txt=False,
        save_conf=False,
        save_crop=False,
        show_labels=True,
        show_conf=True,
        conf=0.5,
        iou=0.5,
        max_det=3,
        device=DEVICE,
        verbose=True
    )

    # Iterate through results and save images with detections + optional crops
    for result in results:
        if result.boxes:  # Check if there are any detections
            print("Detection is being saved!")
            img_with_detections = result.plot()
            output_path = os.path.join(
                output_dir,
                f"{os.path.basename(os.path.dirname(result.path))}_{os.path.basename(result.path)}"
            )

            # cv2.imwrite(output_path, img_with_detections)  # Save the image with detections

            # Now save cropped detections from this result
            # Set num_detections to 1 if you only want the first detection
            # save_crops(result, model, output_dir, num_detections=1, padding=3)

            save_detections_csv(result, csv_w, num_detections=1)

            # Save YOLO annotations
            # save_yolo_annotations(result, output_dir)

# Iterate through results and save images with detections
# for result in results:
#     if result.boxes:  # Check if there are any detections
#         print("Detection is being saved!")
#         img_with_detections = result.plot()  # Render detections on the image
#         output_path = os.path.join(
#             output_dir,
#             f"{os.path.basename(os.path.dirname(result.path))}_{os.path.basename(result.path)}"
#         )
#
#         cv2.imwrite(output_path, img_with_detections)  # Save the image

# Optionally, display the first result
# if results:
#     cv2.imshow("Detections", results[0].plot())
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()

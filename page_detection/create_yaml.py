import os
import yaml


def generate_yaml(data_dir, output_yaml):
    sections = ['train', 'val', 'test']  # Sections for dataset
    data_dict = {'path': data_dir}
    class_file_path = os.path.join(data_dir, 'classes.txt')

    print("Looking for classes.txt at:", class_file_path)  # Debug print
    if os.path.exists(class_file_path):
        with open(class_file_path, 'r') as class_file:
            classes = [line.strip() for line in class_file if line.strip()]
        print("Read classes from classes.txt:", classes)  # Debug print
    else:
        print("Cannot find classes.txt, checking label files.")  # Debug print
        classes = []

    if not classes:
        class_ids = set()
        for section in sections:
            label_path = os.path.join(data_dir, section, 'labels')
            if os.path.exists(label_path):
                for file in os.listdir(label_path):
                    with open(os.path.join(label_path, file), 'r') as f:
                        for line in f:
                            class_ids.add(line.split()[0])
        classes = sorted(list(class_ids), key=int)
        print("Classes derived from label files:", classes)  # Debug print

    data_dict['nc'] = len(classes)
    data_dict['names'] = classes

    # Use absolute paths for images
    for section in sections:
        image_path = os.path.abspath(os.path.join(data_dir, section, 'images'))
        if os.path.exists(image_path):
            data_dict[section] = image_path.replace('\\', '/')  # Format for YAML compatibility

    with open(output_yaml, 'w') as file:
        yaml.dump(data_dict, file, default_flow_style=False)
    print("YAML configuration written:", data_dict)  # Final debug print


# Example usage
data_path = './YOLO/data/CVC/'
generate_yaml(data_path, os.path.join(data_path, 'CVC.yaml'))

import os
import json
import numpy as np
from PIL import Image

def convert_bbox_to_yolo_format(image_width, image_height, bbox):
    x_min, y_min, x_max, y_max = bbox

    x_center = (x_min + x_max) / 2.0
    y_center = (y_min + y_max) / 2.0
    width = x_max - x_min
    height = y_max - y_min

    # Normalize coordinates by image dimensions
    x_center /= image_width
    y_center /= image_height
    width /= image_width
    height /= image_height

    return x_center, y_center, width, height

def convert_annotations_to_yolo(input_dir, output_dir, label_map, image_prefix="rgb_", label_prefix="bounding_box_2d_tight_labels_", bbox_prefix="bounding_box_2d_tight_"):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for i in range(1000):
        file_index = f"{i:04d}"
        json_filename = f"{label_prefix}{file_index}.json"
        npy_filename = f"{bbox_prefix}{file_index}.npy"
        image_filename = f"{image_prefix}{file_index}.png"
        
        json_path = os.path.join(input_dir, json_filename)
        npy_path = os.path.join(input_dir, npy_filename)
        image_path = os.path.join(input_dir, image_filename)
        
        if not os.path.exists(json_path) or not os.path.exists(npy_path) or not os.path.exists(image_path):
            print(f"Skipping {json_filename}, {npy_filename} or {image_filename} because they do not exist.")
            continue

        with Image.open(image_path) as img:
            image_width, image_height = img.size

        # Load JSON file
        with open(json_path, 'r') as f:
            label_data = json.load(f)

        # Load bounding boxes from npy
        bounding_boxes = np.load(npy_path)

        yolo_labels = []

        # Process bounding boxes and labels
        for bbox in bounding_boxes:
            semantic_id = int(bbox['semanticId'])
            class_info = label_data.get(str(semantic_id), None)
            
            if class_info is None:
                print(f"Semantic ID {semantic_id} not found in JSON labels, skipping...")
                continue

            class_label = class_info['class']
            class_id = label_map.get(class_label.lower(), None)
            if class_id is None:
                print(f"Class '{class_label}' not found in label_map, skipping...")
                continue

            x_min, y_min, x_max, y_max = bbox['x_min'], bbox['y_min'], bbox['x_max'], bbox['y_max']
            x_center, y_center, width, height = convert_bbox_to_yolo_format(image_width, image_height, (x_min, y_min, x_max, y_max))
            yolo_labels.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
        
        if yolo_labels:
            label_filename = f"{file_index}.txt"
            label_path = os.path.join(output_dir, label_filename)

            with open(label_path, 'w') as f:
                f.write("\n".join(yolo_labels))
        else:
            print(f"No labels found for {json_filename}, skipping file.")

def main():
    input_dir = r'C:\Inmind\PROJECT\raw data'
    output_dir = r'C:\Inmind\PROJECT\yolo_labels'
    
    label_map = {
        "forklift": 0,
        "rack": 1,
        "crate": 2,
        "floor": 3,
        "railing": 4,
        "pallet": 5,
        "stillage": 6,
        "iwhub": 7,
        "dolly": 8
    }

    convert_annotations_to_yolo(input_dir, output_dir, label_map)

if __name__ == "__main__":
    main()
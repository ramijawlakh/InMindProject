import os
import json

# Path to your training labels
train_labels_dir = 'C:\\Inmind\\PROJECT\\SegNet\\segnet_dataset\\Labels\\segnet_train'

# Dictionary to store all unique label mappings
label_map = {}
current_index = 0

# Iterate over all label files in the training directory
for label_file in os.listdir(train_labels_dir):
    if label_file.endswith('.json'):
        with open(os.path.join(train_labels_dir, label_file), 'r') as f:
            labels = json.load(f)
            for rgba, class_info in labels.items():
                if rgba not in label_map:
                    label_map[rgba] = current_index
                    current_index += 1

# Path to save the label_map.json
label_map_file = 'C:\\Inmind\\PROJECT\\SegNet\\segnet_dataset\\label_map.json'

# Save the label_map dictionary as a JSON file
with open(label_map_file, 'w') as f:
    json.dump(label_map, f, indent=4)

print(f"label_map.json has been created and saved to {label_map_file}")

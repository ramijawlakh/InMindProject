import os
from PIL import Image
import json
import numpy as np
from torch.utils.data import Dataset

class SegNetDataset(Dataset):
    def __init__(self, image_dir, mask_dir, label_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.label_dir = label_dir
        self.transform = transform
        self.images = sorted(os.listdir(image_dir))
        self.class_mapping = {
            "(0, 0, 0, 0)": 0,  # Background
            "(140, 255, 25, 255)": 1,  # Rack
            "(0, 0, 0, 255)": 2,  # UNLABELLED or another class
            "(140, 25, 255, 255)": 3,  # Crate
            "(255, 197, 25, 255)": 4,  # Forklift
            "(25, 255, 82, 255)": 5,  # iwhub
            "(25, 82, 255, 255)": 6,  # dolly
            "(255, 25, 197, 255)": 7,  # pallet
            "(255, 111, 25, 255)": 8,  # railing
            "(226, 255, 25, 255)": 9,  # floor
            "(54, 255, 25, 255)": 10,  # stillage
        }

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_name = self.images[index]
        img_path = os.path.join(self.image_dir, img_name)
        mask_name = img_name.replace('rgb', 'semantic_segmentation')
        mask_path = os.path.join(self.mask_dir, mask_name)
        label_name = mask_name.replace('.png', '.json')
        label_path = os.path.join(self.label_dir, label_name)

        # Load the image and mask
        image = np.array(Image.open(img_path).convert("RGB"))
        mask = np.array(Image.open(mask_path).convert("RGBA"), dtype=np.float32)

        # Map mask colors to class indices using the JSON label file
        mapped_mask = self._map_mask_to_class_indices(mask, label_path)

        # Apply transformations if any
        if self.transform is not None:
            augmentations = self.transform(image=image, mask=mapped_mask)
            image = augmentations["image"]
            mask = augmentations["mask"]

        return image, mapped_mask

    def _map_mask_to_class_indices(self, mask, label_path):
        mapped_mask = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.int64)

        with open(label_path, 'r') as file:
            label_data = json.load(file)
            for rgba, class_name in label_data.items():
                rgba_value = tuple(map(int, rgba.strip("()").split(",")))
                class_index = self.class_mapping[rgba]
                mask_area = np.all(mask == rgba_value, axis=-1)
                mapped_mask[mask_area] = class_index
        
        return mapped_mask

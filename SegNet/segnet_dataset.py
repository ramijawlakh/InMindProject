import os
import json
import torch
from torch.utils.data import Dataset
from PIL import Image, UnidentifiedImageError
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2

class SegNetDataset(Dataset):
    def __init__(self, image_dir, mask_dir, label_dir, label_map_file, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.label_dir = label_dir
        self.transform = transform

        # Only load valid image files and JSON label files
        self.images = sorted([f for f in os.listdir(image_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])
        self.masks = sorted([f for f in os.listdir(mask_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])
        self.labels = sorted([f for f in os.listdir(label_dir) if f.endswith('.json')])
        
        # Load the label map from JSON
        with open(label_map_file, 'r') as f:
            self.label_map = json.load(f)

        # Convert label map RGBA keys to tuples and map to indices
        self.class_map = {tuple(map(int, key.strip("()").split(", "))): idx for key, idx in self.label_map.items()}

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.images[idx])
        mask_path = os.path.join(self.mask_dir, self.masks[idx])
        label_path = os.path.join(self.label_dir, self.labels[idx])

        try:
            # Load the image
            image = Image.open(img_path).convert("RGB")
        except UnidentifiedImageError as e:
            print(f"Error loading image {img_path}: {e}")
            raise e

        try:
            # Load the mask
            mask = Image.open(mask_path)
        except UnidentifiedImageError as e:
            print(f"Error loading mask {mask_path}: {e}")
            raise e

        try:
            # Load the JSON label
            with open(label_path, 'r') as f:
                label_data = json.load(f)
        except json.JSONDecodeError as e:
            print(f"Error loading label {label_path}: {e}")
            raise e

        # Initialize label indices array
        label_indices = np.zeros(mask.size, dtype=np.uint8)

        # Populate label indices based on JSON data
        for rgba_str, class_info in label_data.items():
            rgba_tuple = tuple(map(int, rgba_str.strip("()").split(", ")))
            class_idx = self.class_map.get(rgba_tuple, 0)
            label_indices[(mask == rgba_tuple).all(axis=-1)] = class_idx

        # Apply transformations
        if self.transform:
            augmented = self.transform(image=np.array(image), mask=np.array(mask), label=label_indices)
            image = augmented['image']
            mask = augmented['mask']
            label_indices = augmented['label']

        return image, mask, label_indices

def get_train_transform():
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])

def get_val_transform():
    return A.Compose([
        A.Resize(height=720, width=1280),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])

def create_dataloaders(train_img_dir, train_mask_dir, train_label_dir, label_map_file,
                       val_img_dir, val_mask_dir, val_label_dir,
                       batch_size=8, num_workers=4):
    train_transform = get_train_transform()
    val_transform = get_val_transform()

    train_dataset = SegNetDataset(train_img_dir, train_mask_dir, train_label_dir, label_map_file, transform=train_transform)
    val_dataset = SegNetDataset(val_img_dir, val_mask_dir, val_label_dir, label_map_file, transform=val_transform)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader

import os
import json
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2

class SegNetDataset(Dataset):
    def __init__(self, image_dir, mask_dir, label_dir, label_map_file, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.label_dir = label_dir
        self.transform = transform

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

        # Debugging: Print the file paths being loaded
        print(f"Loading image: {img_path}")
        print(f"Loading mask: {mask_path}")
        print(f"Loading label: {label_path}")

        try:
            # Ensure only image files are loaded as images
            image = np.array(Image.open(img_path).convert("RGB"))
        except Exception as e:
            print(f"Error loading image: {img_path}, Error: {e}")
            raise e
        
        try:
            mask = np.array(Image.open(mask_path))
        except Exception as e:
            print(f"Error loading mask: {mask_path}, Error: {e}")
            raise e

        try:
            # Load the JSON label file
            with open(label_path, 'r') as f:
                label_data = json.load(f)
        except Exception as e:
            print(f"Error loading label: {label_path}, Error: {e}")
            raise e

        # Create a label array with the same size as the mask
        label_indices = np.zeros(mask.shape, dtype=np.uint8)

        # Convert JSON label data to class indices
        for rgba_str, class_info in label_data.items():
            rgba_tuple = tuple(map(int, rgba_str.strip("()").split(", ")))
            class_idx = self.class_map[rgba_tuple]
            label_indices[mask == rgba_tuple] = class_idx

        if self.transform:
            augmented = self.transform(image=image, mask=mask, label=label_indices)
            image = augmented['image']
            mask = augmented['mask']
            label = augmented['label']

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

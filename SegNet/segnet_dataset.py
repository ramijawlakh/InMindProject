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

        print(f"Loading image: {img_path}")
        print(f"Loading mask: {mask_path}")
        print(f"Loading label: {label_path}")

        try:
            image = np.array(Image.open(img_path).convert("RGB"))
            print(f"Original image shape: {image.shape}")
        except Exception as e:
            print(f"Error loading image: {img_path}, Error: {e}")
            raise e

        try:
            mask = np.array(Image.open(mask_path))
            print(f"Original mask shape: {mask.shape}")
        except Exception as e:
            print(f"Error loading mask: {mask_path}, Error: {e}")
            raise e

        try:
            with open(label_path, 'r') as f:
                label_data = json.load(f)
        except Exception as e:
            print(f"Error loading label: {label_path}, Error: {e}")
            raise e

        label_indices = np.zeros(mask.shape, dtype=np.uint8)

        for rgba_str, class_info in label_data.items():
            rgba_tuple = tuple(map(int, rgba_str.strip("()").split(", ")))
            class_idx = self.class_map.get(rgba_tuple, 0)
            label_indices[mask == rgba_tuple] = class_idx

        if self.transform:
            # Apply transform to both image and label_indices
            augmented = self.transform(image=image, mask=label_indices)
            image = augmented['image']
            label_indices = augmented['mask']

            print(f"Transformed image shape: {image.shape}")
            print(f"Transformed mask shape: {label_indices.shape}")

        # Convert label_indices to a PyTorch tensor with the correct dtype
        label_indices = torch.tensor(label_indices, dtype=torch.long)

        # Ensure the label is a 2D tensor by removing any extra dimensions
        if label_indices.dim() == 3:
            label_indices = label_indices.squeeze(0)  # Remove any singleton dimensions

        return image, label_indices

def get_train_transform():
    return A.Compose([
        A.Resize(height=512, width=512),  # Resize to 512x512
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
        A.Resize(height=512, width=512),  # Resize to 512x512
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

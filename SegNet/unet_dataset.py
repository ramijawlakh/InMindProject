
import os
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2

class SegNetDataset(Dataset):
    def __init__(self, image_dir, mask_dir, label_dir, transform=None):
        """
        Args:
            image_dir (str): Path to the directory with images.
            mask_dir (str): Path to the directory with masks.
            label_dir (str): Path to the directory with labels.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.resize_transform = A.Compose([
    A.Resize(height=720, width=1280),
    A.PadIfNeeded(min_height=720, min_width=1280, border_mode=0, value=0)
])
        self.images = sorted(os.listdir(image_dir))
        self.masks = sorted(os.listdir(mask_dir))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.images[idx])
        mask_path = os.path.join(self.mask_dir, self.masks[idx])

        # Open images, masks, and labels
        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path)

        # Convert to numpy arrays
        image = np.array(image)
        mask = np.array(mask)

        # Apply resize transform
        augmented = self.resize_transform(image=image, mask=mask)
        image = augmented["image"]
        mask = augmented["mask"]

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']

        return image, mask

def get_train_transforms():
    return A.Compose([
        A.HorizontalFlip(p=0.5),
                A.PadIfNeeded(min_height=720, min_width=1280, border_mode=0, value=0),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])

def get_val_transforms():
    return A.Compose([
        A.Resize(height=720, width=1280),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])

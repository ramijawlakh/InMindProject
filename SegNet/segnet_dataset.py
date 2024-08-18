import os
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms

class SegNetDataset(Dataset):
    def __init__(self, image_dir, mask_dir, label_dir=None, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.label_dir = label_dir
        self.transform = transform
        self.images = sorted(os.listdir(image_dir))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.image_dir, img_name)
        mask_name = img_name.replace('rgb', 'semantic_segmentation')
        mask_path = os.path.join(self.mask_dir, mask_name)

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")  # Convert mask to grayscale

        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)

        return image, mask

# Define transformations to resize the images and masks to 1280x720
transform = transforms.Compose([
    transforms.Resize((720, 1280)),  # Resize to 720x1280 (Height x Width)
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Example usage:
# dataset = SegNetDataset(image_dir="path/to/images", mask_dir="path/to/masks", transform=transform)

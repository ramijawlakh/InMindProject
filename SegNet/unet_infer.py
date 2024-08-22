
import torch
import torchvision.transforms.functional as TF
from PIL import Image
import os
from modified_unet_model import UNET
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2

def load_checkpoint(checkpoint, model):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])

def save_predictions_as_imgs(
    loader, model, folder="saved_images/", device="cuda"
):
    model.eval()
    for idx, (x, _) in enumerate(loader):
        x = x.to(device=device)
        with torch.no_grad():
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()
        torchvision.utils.save_image(preds, f"{folder}/pred_{idx}.png")
    model.train()

def infer_image(model, image_path, transform):
    image = Image.open(image_path).convert("RGB")
    image = np.array(image)

    augmented = transform(image=image)
    image = augmented["image"]

    image = image.unsqueeze(0)  # add batch dimension
    model.eval()
    with torch.no_grad():
        pred = torch.sigmoid(model(image))
        pred = (pred > 0.5).float()
    model.train()
    return pred.squeeze(0)  # remove batch dimension

def main():
    IMAGE_HEIGHT = 720
    IMAGE_WIDTH = 1280
    IMAGE_DIR = "C:/Inmind/PROJECT/SegNet/segnet_dataset/Images/segnet_test"
    CHECKPOINT_PATH = "my_checkpoint.pth.tar"

    transform = A.Compose([
        A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])

    model = UNET(in_channels=3, out_channels=1).to("cuda")
    load_checkpoint(torch.load(CHECKPOINT_PATH), model)

    for img_name in os.listdir(IMAGE_DIR):
        img_path = os.path.join(IMAGE_DIR, img_name)
        pred_mask = infer_image(model, img_path, transform)
        torchvision.utils.save_image(pred_mask, f"saved_images/{img_name}")

if __name__ == "__main__":
    main()

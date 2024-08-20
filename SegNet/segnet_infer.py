import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from segnet_model import SegNet
from segnet_utils import load_checkpoint
from PIL import Image
import numpy as np
import cv2
import os

# Define the dataset path in your local machine
DATASET_PATH = r'C:\Inmind\PROJECT\SegNet\segnet_dataset'

# Define the paths for input image and output directory in your local machine
IMAGE_PATH = os.path.join(DATASET_PATH, 'some_test_image.jpg')
OUTPUT_PATH = r'C:\Inmind\PROJECT\SegNet\segnet_inference_output\predicted_mask.png'

# Hyperparameters
DEVICE = "cpu"  # Set to CPU for local execution
IMAGE_HEIGHT = 720  # Same as training height
IMAGE_WIDTH = 1280  # Same as training width
CHECKPOINT_PATH = r'C:\Inmind\PROJECT\SegNet\segnet_model_checkpoint.pth.tar'

def transform_image(image_path):
    transform = A.Compose([
        A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            max_pixel_value=255.0,
        ),
        ToTensorV2(),
    ])
    image = np.array(Image.open(image_path).convert("RGB"))
    image = transform(image=image)["image"]
    return image

def predict(image_path, model):
    model.eval()
    image = transform_image(image_path).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        prediction = model(image)
        prediction = torch.argmax(prediction, dim=1).squeeze(0)
    return prediction.cpu().numpy()

def save_prediction(prediction, output_path):
    prediction_image = (prediction * 255 / prediction.max()).astype(np.uint8)
    cv2.imwrite(output_path, prediction_image)

def main(image_path, output_path):
    model = SegNet(in_channels=3, out_channels=9).to(DEVICE)
    load_checkpoint(torch.load(CHECKPOINT_PATH), model)

    prediction = predict(image_path, model)
    save_prediction(prediction, output_path)

if __name__ == "__main__":
    image_path = r'C:\Inmind\PROJECT\SegNet\some_test_image.jpg'  # Replace with your image path
    output_path = r'C:\Inmind\PROJECT\SegNet\segmentation_mask.png'  # Replace with desired output path
    main(image_path, output_path)
    print(f"Prediction saved to {output_path}")

import os
from PIL import Image

def resize_images_in_folder(input_folder, output_folder, size=(720, 1280)):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(input_folder, filename)
            img = Image.open(img_path)

            # Resize and save the image using LANCZOS resampling
            img_resized = img.resize(size, Image.Resampling.LANCZOS)
            img_resized.save(os.path.join(output_folder, filename))

def preprocess_dataset(train_img_dir, train_mask_dir, val_img_dir, val_mask_dir, output_size=(720, 1280)):
    # Process training images and masks
    resize_images_in_folder(train_img_dir, os.path.join(train_img_dir, "resized"), size=output_size)
    resize_images_in_folder(train_mask_dir, os.path.join(train_mask_dir, "resized"), size=output_size)

    # Process validation images and masks
    resize_images_in_folder(val_img_dir, os.path.join(val_img_dir, "resized"), size=output_size)
    resize_images_in_folder(val_mask_dir, os.path.join(val_mask_dir, "resized"), size=output_size)

if __name__ == "__main__":
    DATASET_PATH = r'C:\Inmind\PROJECT\SegNet\segnet_dataset'

    TRAIN_IMG_DIR = os.path.join(DATASET_PATH, 'Images/segnet_train')
    TRAIN_MASK_DIR = os.path.join(DATASET_PATH, 'Masks/segnet_train')
    VAL_IMG_DIR = os.path.join(DATASET_PATH, 'Images/segnet_val')
    VAL_MASK_DIR = os.path.join(DATASET_PATH, 'Masks/segnet_val')

    preprocess_dataset(TRAIN_IMG_DIR, TRAIN_MASK_DIR, VAL_IMG_DIR, VAL_MASK_DIR, output_size=(720, 1280))

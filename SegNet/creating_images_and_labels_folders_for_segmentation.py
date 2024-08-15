# Python script that copies the images and labels from C:\Inmind\PROJECT\raw data to the respective folders semantic_images and semantic_labels in C:\Inmind\PROJECT:



import os
import shutil

# Define source and destination directories
source_dir = r"C:\Inmind\PROJECT\raw data"
dest_image_dir = r"C:\Inmind\PROJECT\segmentation_masks"
dest_label_dir = r"C:\Inmind\PROJECT\segmentation_labels"

# Create destination directories if they don't exist
os.makedirs(dest_image_dir, exist_ok=True)
os.makedirs(dest_label_dir, exist_ok=True)

# Loop through files in the source directory
for i in range(1000):  # Assuming there are exactly 1000 images and labels
    # Format the file number with leading zeros
    file_num = f"{i:04d}"

    # Define the source paths for image and label
    image_src = os.path.join(source_dir, f"semantic_segmentation_{file_num}.png")
    label_src = os.path.join(source_dir, f"semantic_segmentation_labels_{file_num}.json")

    # Define the destination paths for image and label
    image_dest = os.path.join(dest_image_dir, f"semantic_segmentation_{file_num}.png")
    label_dest = os.path.join(dest_label_dir, f"semantic_segmentation_labels_{file_num}.json")

    # Copy image and label to the respective destination directories
    if os.path.exists(image_src) and os.path.exists(label_src):
        shutil.copy(image_src, image_dest)
        shutil.copy(label_src, label_dest)
    else:
        print(f"File missing: {image_src} or {label_src}")

print("Files copied successfully.")

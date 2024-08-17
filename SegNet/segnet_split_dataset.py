import os
import shutil
import random

# Define paths
base_dir = r"C:\Inmind\PROJECT\SegNet"
image_dir = os.path.join(base_dir, "segmentation_images")
mask_dir = os.path.join(base_dir, "segmentation_masks")
label_dir = os.path.join(base_dir, "segmentation_labels")

# Define the output directories
output_dir = os.path.join(base_dir, "segnet_dataset")
image_output_dir = os.path.join(output_dir, "Images")
mask_output_dir = os.path.join(output_dir, "Masks")
label_output_dir = os.path.join(output_dir, "Labels")

# Create output directories if they don't exist
for folder in [image_output_dir, mask_output_dir, label_output_dir]:
    for subfolder in ["segnet_train", "segnet_val", "segnet_test"]:
        os.makedirs(os.path.join(folder, subfolder), exist_ok=True)

# Get list of all image, mask, and label filenames
image_filenames = sorted(os.listdir(image_dir))
mask_filenames = sorted(os.listdir(mask_dir))
label_filenames = sorted(os.listdir(label_dir))

# Ensure the number of images, masks, and labels match
assert len(image_filenames) == len(mask_filenames) == len(label_filenames), "Mismatch in number of files."

# Shuffle the filenames for random splitting
combined = list(zip(image_filenames, mask_filenames, label_filenames))
random.shuffle(combined)
image_filenames, mask_filenames, label_filenames = zip(*combined)

# Define split sizes
train_split = 0.7
val_split = 0.2
test_split = 0.1

# Calculate split indices
total_files = len(image_filenames)
train_idx = int(total_files * train_split)
val_idx = int(total_files * (train_split + val_split))

# Split filenames into train, val, and test sets
train_images, val_images, test_images = image_filenames[:train_idx], image_filenames[train_idx:val_idx], image_filenames[val_idx:]
train_masks, val_masks, test_masks = mask_filenames[:train_idx], mask_filenames[train_idx:val_idx], mask_filenames[val_idx:]
train_labels, val_labels, test_labels = label_filenames[:train_idx], label_filenames[train_idx:val_idx], label_filenames[val_idx:]

# Function to copy files to their respective directories
def copy_files(file_list, src_dir, dest_dir):
    for filename in file_list:
        shutil.copy(os.path.join(src_dir, filename), os.path.join(dest_dir, filename))

# Copy files to the output directories
copy_files(train_images, image_dir, os.path.join(image_output_dir, "segnet_train"))
copy_files(val_images, image_dir, os.path.join(image_output_dir, "segnet_val"))
copy_files(test_images, image_dir, os.path.join(image_output_dir, "segnet_test"))

copy_files(train_masks, mask_dir, os.path.join(mask_output_dir, "segnet_train"))
copy_files(val_masks, mask_dir, os.path.join(mask_output_dir, "segnet_val"))
copy_files(test_masks, mask_dir, os.path.join(mask_output_dir, "segnet_test"))

copy_files(train_labels, label_dir, os.path.join(label_output_dir, "segnet_train"))
copy_files(val_labels, label_dir, os.path.join(label_output_dir, "segnet_val"))
copy_files(test_labels, label_dir, os.path.join(label_output_dir, "segnet_test"))

print("Dataset successfully split and organized!")

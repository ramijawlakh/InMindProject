import os
import shutil

# Define source and destination directories
source_dir = r"C:\Inmind\PROJECT\raw data"
destination_dir = r"C:\Inmind\PROJECT\SegNet\segmentation_images"

# Create the destination directory if it doesn't exist
os.makedirs(destination_dir, exist_ok=True)

# Loop through the images and copy them to the destination directory
for i in range(1000):
    filename = f"rgb_{i:04}.png"
    source_path = os.path.join(source_dir, filename)
    destination_path = os.path.join(destination_dir, filename)
    
    # Copy the file if it exists
    if os.path.exists(source_path):
        shutil.copy(source_path, destination_path)
        print(f"Copied {filename} to {destination_dir}")
    else:
        print(f"{filename} not found in {source_dir}")

print("Image copying complete.")

import os
import shutil
from random import seed, shuffle

def split_dataset(images_folder, labels_folder, output_base):
    seed(42)  # For reproducibility
    image_files = sorted([f for f in os.listdir(images_folder) if f.endswith('.png')])
    label_files = sorted([f for f in os.listdir(labels_folder) if f.endswith('.txt')])

    # Ensure files match between images and labels
    image_indices = {f.split('_')[-1].split('.')[0]: f for f in image_files}
    label_indices = {f.split('.')[0]: f for f in label_files}
    matched_files = [(image_indices[idx], label_indices[idx]) for idx in image_indices if idx in label_indices]

    shuffle(matched_files)  # Shuffle the matched files

    # Split the files
    total_files = len(matched_files)
    train_end = int(0.7 * total_files)
    val_end = train_end + int(0.2 * total_files)

    # Create directories for the splits
    subsets = ['train', 'val', 'test']
    paths = {}
    for subset in subsets:
        img_dir = os.path.join(output_base, 'images', subset)
        lbl_dir = os.path.join(output_base, 'labels', subset)
        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(lbl_dir, exist_ok=True)
        paths[subset] = (img_dir, lbl_dir)

    # Assign files to train, val, and test
    for i, (img, lbl) in enumerate(matched_files):
        if i < train_end:
            subset = 'train'
        elif i < val_end:
            subset = 'val'
        else:
            subset = 'test'
        
        # Copy image and label to the correct directory
        shutil.copy(os.path.join(images_folder, img), os.path.join(paths[subset][0], img))
        shutil.copy(os.path.join(labels_folder, lbl), os.path.join(paths[subset][1], lbl))
        print(f"Copied {img} and {lbl} to {subset} folders")

def main():
    images_folder = r'C:\Inmind\PROJECT\labeled_images'
    labels_folder = r'C:\Inmind\PROJECT\yolo_labels'
    output_base = r'C:\Inmind\PROJECT\dataset'

    split_dataset(images_folder, labels_folder, output_base)

if __name__ == "__main__":
    main()




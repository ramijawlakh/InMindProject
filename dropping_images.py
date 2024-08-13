import os
import shutil

def copy_unlabeled_images(input_dir, output_dir, skipped_indices):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for i in range(1000):
        file_index = f"{i:04d}"
        if file_index in skipped_indices:
            continue

        image_filename = f"rgb_{file_index}.png"
        image_path = os.path.join(input_dir, image_filename)

        if os.path.exists(image_path):
            shutil.copy(image_path, os.path.join(output_dir, image_filename))
            print(f"Copied {image_filename} to {output_dir}")
        else:
            print(f"Image {image_filename} does not exist in {input_dir}, skipping.")

def main():
    input_dir = r'C:\Inmind\PROJECT\raw data'
    output_dir = r'C:\Inmind\PROJECT\unlabeled_images'

    skipped_files = [
        "0035", "0042", "0044", "0050", "0070", "0079", "0091", "0098", "0101",
        "0112", "0138", "0177", "0188", "0210", "0211", "0216", "0229", "0258",
        "0260", "0263", "0268", "0277", "0315", "0321", "0346", "0353", "0358",
        "0383", "0412", "0414", "0418", "0424", "0429", "0434", "0460", "0463",
        "0469", "0499", "0503", "0508", "0523", "0535", "0540", "0549", "0551",
        "0564", "0604", "0607", "0618", "0635", "0636", "0639", "0674", "0684",
        "0689", "0696", "0726", "0731", "0732", "0740", "0741", "0756", "0779",
        "0805", "0807", "0823", "0896", "0927", "0938", "0949", "0965", "0968", 
        "0976"
    ]

    copy_unlabeled_images(input_dir, output_dir, skipped_files)

if __name__ == "__main__":
    main()

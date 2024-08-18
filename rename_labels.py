import os

# Define the directory containing your label files
labels_dir = 'C:\\Inmind\\PROJECT\\dataset\\labels\\train'

# Rename each label file to include the 'rgb_' prefix
for label_file in os.listdir(labels_dir):
    if label_file.endswith('.txt'):
        base_name = label_file.split('.')[0]
        new_name = f'rgb_{base_name}.txt'
        os.rename(os.path.join(labels_dir, label_file), os.path.join(labels_dir, new_name))

print("Label files renamed successfully.")


# Define the directory containing your label files
labels_dir = 'C:\\Inmind\PROJECT\\dataset\\labels\\val'

# Rename each label file to include the 'rgb_' prefix
for label_file in os.listdir(labels_dir):
    if label_file.endswith('.txt'):
        base_name = label_file.split('.')[0]
        new_name = f'rgb_{base_name}.txt'
        os.rename(os.path.join(labels_dir, label_file), os.path.join(labels_dir, new_name))

print("Label files renamed successfully.")

# Define the directory containing your label files
labels_dir = 'C:\\Inmind\PROJECT\\dataset\\labels\\test'

# Rename each label file to include the 'rgb_' prefix
for label_file in os.listdir(labels_dir):
    if label_file.endswith('.txt'):
        base_name = label_file.split('.')[0]
        new_name = f'rgb_{base_name}.txt'
        os.rename(os.path.join(labels_dir, label_file), os.path.join(labels_dir, new_name))

print("Label files renamed successfully.")


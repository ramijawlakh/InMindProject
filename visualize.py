import os
import cv2
import numpy as np
import argparse
import matplotlib.pyplot as plt

def plot_one_box(box, img, color=[0, 255, 0], label=None, line_thickness=None):
    tl = line_thickness or round(0.002 * max(img.shape[0:2])) + 1  # Line thickness
    c1 = (int(box[0]), int(box[1]))  # Top-left corner
    c2 = (int(box[2]), int(box[3]))  # Bottom-right corner
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # Font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # Filled rectangle for label background
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)

def visualize_annotations(image_path, label_path, class_names):
    image = cv2.imread(image_path)
    labels = np.loadtxt(label_path).reshape(-1, 5)

    for label in labels:
        class_id, x_center, y_center, width, height = label
        img_height, img_width, _ = image.shape
        x1 = int((x_center - width * 0.5) * img_width)
        y1 = int((y_center - height * 0.5) * img_height)
        x2 = int((x_center + width * 0.5) * img_width)
        y2 = int((y_center + height * 0.5) * img_height)
        plot_one_box([x1, y1, x2, y2], image, label=class_names[int(class_id)], line_thickness=2)
    
    # Convert BGR to RGB for displaying with Matplotlib
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Display the image with Matplotlib
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    plt.axis('off')  # Hide axes
    plt.show()

def main():
    parser = argparse.ArgumentParser(description="Visualize labeled images with bounding boxes.")
    parser.add_argument('image_path', type=str, help='Path to the image file')
    parser.add_argument('label_path', type=str, help='Path to the label file')
    parser.add_argument('--class_names', type=str, nargs='+', default=['Forklift', 'Rack', 'Crate', 'Floor', 'Railing', 'Pallet', 'Stillage', 'iwhub', 'dolly'], help='List of class names')
    
    args = parser.parse_args()

    visualize_annotations(args.image_path, args.label_path, args.class_names)

if __name__ == "__main__":
    main()


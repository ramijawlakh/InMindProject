import torch
import argparse
import os
from tqdm import tqdm
import numpy as np

from segnet_model import SegNet
from segnet_dataset import create_dataloaders
from segnet_utils import load_checkpoint, save_predictions_as_imgs

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate SegNet on Testing Dataset")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for evaluation")
    parser.add_argument("--image_height", type=int, default=720, help="Height of input images")
    parser.add_argument("--image_width", type=int, default=1280, help="Width of input images")
    parser.add_argument("--test_img_dir", type=str, required=True, help="Directory with testing images")
    parser.add_argument("--test_mask_dir", type=str, required=True, help="Directory with testing masks")
    parser.add_argument("--test_label_dir", type=str, required=True, help="Directory with testing labels")
    parser.add_argument("--label_map_file", type=str, required=True, help="Path to the label map JSON file")
    parser.add_argument("--checkpoint_file", type=str, default="checkpoint.pth.tar", help="Path to the model checkpoint")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save predictions and metrics")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use for evaluation (cuda or cpu)")
    return parser.parse_args()

def evaluate(loader, model, device):
    model.eval()
    num_correct = 0
    num_pixels = 0
    dice_score = 0
    iou_score = 0
    num_classes = 9  # Adjust based on your number of classes

    class_iou = np.zeros(num_classes)
    class_dice = np.zeros(num_classes)
    class_counts = np.zeros(num_classes)

    with torch.no_grad():
        for data, masks, labels in tqdm(loader, leave=True):
            data = data.to(device)
            labels = labels.to(device)

            outputs = model(data)
            preds = torch.argmax(outputs, dim=1)

            num_correct += (preds == labels).sum().item()
            num_pixels += torch.numel(preds)

            for cls in range(num_classes):
                pred_class = (preds == cls).float()
                label_class = (labels == cls).float()
                intersection = (pred_class * label_class).sum()
                union = pred_class.sum() + label_class.sum() - intersection
                dice = 2 * intersection / (pred_class.sum() + label_class.sum() + 1e-8)
                iou = intersection / (union + 1e-8)

                class_iou[cls] += iou.item()
                class_dice[cls] += dice.item()
                class_counts[cls] += 1

    overall_accuracy = num_correct / num_pixels
    mean_iou = class_iou.sum() / class_counts.sum()
    mean_dice = class_dice.sum() / class_counts.sum()

    print(f"Overall Accuracy: {overall_accuracy:.4f}")
    print(f"Mean IoU: {mean_iou:.4f}")
    print(f"Mean Dice Score: {mean_dice:.4f}")

    return overall_accuracy, mean_iou, mean_dice, class_iou / class_counts, class_dice / class_counts

def main():
    args = parse_args()

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Initialize the model and load the checkpoint
    model = SegNet(in_channels=3, out_channels=9).to(args.device)
    load_checkpoint(torch.load(args.checkpoint_file), model)

    # Create DataLoader for testing dataset
    _, test_loader = create_dataloaders(
        args.test_img_dir,
        args.test_mask_dir,
        args.test_label_dir,
        args.label_map_file,
        val_img_dir=args.test_img_dir,
        val_mask_dir=args.test_mask_dir,
        val_label_dir=args.test_label_dir,
        batch_size=args.batch_size,
        num_workers=4,
    )

    # Evaluate the model
    overall_accuracy, mean_iou, mean_dice, class_iou, class_dice = evaluate(test_loader, model, args.device)

    # Save predictions
    save_predictions_as_imgs(test_loader, model, folder=os.path.join(args.output_dir, "predictions"), device=args.device)

    # Save evaluation metrics
    metrics_path = os.path.join(args.output_dir, "metrics.txt")
    with open(metrics_path, "w") as f:
        f.write(f"Overall Accuracy: {overall_accuracy:.4f}\n")
        f.write(f"Mean IoU: {mean_iou:.4f}\n")
        f.write(f"Mean Dice Score: {mean_dice:.4f}\n")
        for i in range(9):
            f.write(f"Class {i} IoU: {class_iou[i]:.4f}\n")
            f.write(f"Class {i} Dice Score: {class_dice[i]:.4f}\n")
    print(f"Metrics saved to {metrics_path}")

if __name__ == "__main__":
    main()

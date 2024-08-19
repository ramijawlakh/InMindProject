import torch
import argparse
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import os

# Define the dataset path in Google Drive
DATASET_PATH = '/content/drive/MyDrive/segnet_dataset'

# Update paths to your dataset in Google Drive
TRAIN_IMG_DIR = os.path.join(DATASET_PATH, 'Images/segnet_train')
TRAIN_MASK_DIR = os.path.join(DATASET_PATH, 'Masks/segnet_train')
TRAIN_LABEL_DIR = os.path.join(DATASET_PATH, 'Labels/segnet_train')
VAL_IMG_DIR = os.path.join(DATASET_PATH, 'Images/segnet_val')
VAL_MASK_DIR = os.path.join(DATASET_PATH, 'Masks/segnet_val')
VAL_LABEL_DIR = os.path.join(DATASET_PATH, 'Labels/segnet_val')
LABEL_MAP_FILE = os.path.join(DATASET_PATH, 'label_map.json')

# Path to save checkpoints and TensorBoard logs
CHECKPOINT_FILE = '/content/drive/MyDrive/segnet_model_checkpoint.pth.tar'
LOG_DIR = '/content/drive/MyDrive/segnet_tensorboard_logs'

# Create the directory if it doesn't exist
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)

from segnet_model import SegNet  # Import your SegNet model
from segnet_dataset import create_dataloaders  # Import the data loader creation function
from segnet_utils import (
    load_checkpoint,
    save_checkpoint,
    check_accuracy,
    save_predictions_as_imgs,
)

def parse_args():
    parser = argparse.ArgumentParser(description="Train SegNet for Semantic Segmentation")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate for the optimizer")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for training")
    parser.add_argument("--num_epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--image_height", type=int, default=720, help="Height of input images")
    parser.add_argument("--image_width", type=int, default=1280, help="Width of input images")
    parser.add_argument("--train_img_dir", type=str, required=True, help="Directory with training images")
    parser.add_argument("--train_mask_dir", type=str, required=True, help="Directory with training masks")
    parser.add_argument("--train_label_dir", type=str, required=True, help="Directory with training labels")
    parser.add_argument("--val_img_dir", type=str, required=True, help="Directory with validation images")
    parser.add_argument("--val_mask_dir", type=str, required=True, help="Directory with validation masks")
    parser.add_argument("--val_label_dir", type=str, required=True, help="Directory with validation labels")
    parser.add_argument("--label_map_file", type=str, required=True, help="Path to the label map JSON file")
    parser.add_argument("--checkpoint_file", type=str, default="checkpoint.pth.tar", help="Path to save the model checkpoint")
    parser.add_argument("--log_dir", type=str, default="runs", help="Directory for TensorBoard logs")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use for training (cuda or cpu)")
    return parser.parse_args()

def train_fn(loader, model, optimizer, loss_fn, device):
    model.train()
    loop = tqdm(loader, leave=True)
    for batch_idx, (data, targets, labels) in enumerate(loop):
        data = data.to(device)
        labels = labels.to(device)

        # forward
        predictions = model(data)
        loss = loss_fn(predictions, labels)

        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # update tqdm loop
        loop.set_postfix(loss=loss.item())

def main():
    args = parse_args()

    writer = SummaryWriter(log_dir=args.log_dir)

    # Initialize model, loss function, optimizer
    model = SegNet(in_channels=3, out_channels=9).to(args.device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    # Load checkpoint if available
    if os.path.exists(args.checkpoint_file):
        load_checkpoint(torch.load(args.checkpoint_file), model, optimizer)
        print(f"Checkpoint loaded from {args.checkpoint_file}.")
    else:
        print(f"No checkpoint found at {args.checkpoint_file}. Starting training from scratch.")

    # Create DataLoaders
    train_loader, val_loader = create_dataloaders(
        args.train_img_dir,
        args.train_mask_dir,
        args.train_label_dir,
        args.label_map_file,
        args.val_img_dir,
        args.val_mask_dir,
        args.val_label_dir,
        batch_size=args.batch_size,
        num_workers=4,
    )

    for epoch in range(args.num_epochs):
        train_fn(train_loader, model, optimizer, loss_fn, args.device)

        # Check accuracy on validation set
        val_acc = check_accuracy(val_loader, model, device=args.device)
        print(f"Validation Accuracy after epoch {epoch+1}: {val_acc:.2f}%")

        # Ensure checkpoint directory exists
        checkpoint_dir = os.path.dirname(args.checkpoint_file)
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        # Save model checkpoint
        checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        }
        save_checkpoint(checkpoint, filename=args.checkpoint_file)

        # Log to TensorBoard
        writer.add_scalar("Validation Accuracy", val_acc, epoch)

    writer.close()

if __name__ == "__main__":
    main()

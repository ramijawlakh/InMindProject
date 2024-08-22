
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import torchvision
import os
import argparse
from torch.utils.tensorboard import SummaryWriter

from modified_unet_model import UNET
from unet_dataset import SegNetDataset, get_train_transforms, get_val_transforms
from unet_utils import load_checkpoint, save_checkpoint, check_accuracy, save_predictions_as_imgs

def train_fn(loader, model, optimizer, loss_fn, scaler, writer, epoch):
    DEVICE = args.device
    loop = tqdm(loader)
    for batch_idx, (data, targets) in enumerate(loop):
        data = data.to(device=DEVICE)
        targets = targets.to(device=DEVICE)
        
        with torch.cuda.amp.autocast():
            predictions = model(data)
            # Reshape target labels to match predictions
    if targets.dim() == 4 and targets.size(-1) == 4:
        targets = targets.permute(0, 3, 1, 2)  # Convert to [batch_size, channels, height, width]
        targets = targets[:, :1, :, :]  # Select only the first channel if necessary

    loss = loss_fn(predictions, targets)
        
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        loop.set_postfix(loss=loss.item())
        writer.add_scalar("Loss/train", loss.item(), epoch * len(loader) + batch_idx)

def main(args):
    writer = SummaryWriter(log_dir=args.log_dir)
    train_transforms = get_train_transforms()
    val_transforms = get_val_transforms()

    model = UNET(in_channels=3, out_channels=1).to(args.device)
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    train_dataset = SegNetDataset(
        image_dir=args.train_img_dir,
        mask_dir=args.train_mask_dir,
        label_dir=args.label_dir,
        transform=train_transforms,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        shuffle=True,
    )

    val_dataset = SegNetDataset(
        image_dir=args.val_img_dir,
        mask_dir=args.val_mask_dir,
        label_dir=args.label_dir,
        transform=val_transforms,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        shuffle=False,
    )

    if args.load_model:
        load_checkpoint(torch.load(args.checkpoint_path), model)

    scaler = torch.cuda.amp.GradScaler()

    for epoch in range(args.epochs):
        train_fn(train_loader, model, optimizer, loss_fn, scaler, writer, epoch)
        
        checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        }
        save_checkpoint(checkpoint, filename=args.checkpoint_path)

        val_acc, val_dice = check_accuracy(val_loader, model, device=args.device)
        writer.add_scalar("Accuracy/val", val_acc, epoch)
        writer.add_scalar("Dice/val", val_dice, epoch)

        save_predictions_as_imgs(
            val_loader, model, folder="saved_images/", device=args.device
        )

    writer.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the UNet model.")
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs to train for")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for training")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use for training")
    parser.add_argument("--num_workers", type=int, default=2, help="Number of workers for data loading")
    parser.add_argument("--pin_memory", type=bool, default=True, help="Whether to pin memory for data loading")
    parser.add_argument("--load_model", action="store_true", help="Load model from checkpoint")
    parser.add_argument("--checkpoint_path", type=str, default="/content/drive/MyDrive/MyProject/checkpoints/my_checkpoint.pth.tar", help="Path to save/load checkpoint")
    parser.add_argument("--log_dir", type=str, default="/content/drive/MyDrive/MyProject/runs", help="Directory for TensorBoard logs")
    parser.add_argument("--train_img_dir", type=str, default="/content/drive/MyDrive/MyProject/segnet_dataset/Images/segnet_train", help="Directory for training images")
    parser.add_argument("--train_mask_dir", type=str, default="/content/drive/MyDrive/MyProject/segnet_dataset/Masks/segnet_train", help="Directory for training masks")
    parser.add_argument("--val_img_dir", type=str, default="/content/drive/MyDrive/MyProject/segnet_dataset/Images/segnet_val", help="Directory for validation images")
    parser.add_argument("--val_mask_dir", type=str, default="/content/drive/MyDrive/MyProject/segnet_dataset/Masks/segnet_val", help="Directory for validation masks")
    parser.add_argument("--label_dir", type=str, default="/content/drive/MyDrive/MyProject/segnet_dataset/Labels", help="Directory for labels")

    args = parser.parse_args()
    main(args)

import torch
import torchvision
from segnet_dataset import SegNetDataset
from torch.utils.data import DataLoader
import os

# Save checkpoint locally
def save_checkpoint(state, filename="C:\\Inmind\\PROJECT\\SegNet\\segnet_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)

# Load checkpoint
def load_checkpoint(checkpoint, model, optimizer=None, device="cpu"):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])
    if optimizer:
        optimizer.load_state_dict(checkpoint["optimizer"])
    model.to(device)

def get_loaders(
    train_dir,
    train_maskdir,
    train_labeldir,
    val_dir,
    val_maskdir,
    val_labeldir,
    batch_size,
    train_transform,
    val_transform,
    num_workers=4,
    pin_memory=True,
):
    train_ds = SegNetDataset(
        image_dir=train_dir,
        mask_dir=train_maskdir,
        label_dir=train_labeldir,
        transform=train_transform,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=True,
    )

    val_ds = SegNetDataset(
        image_dir=val_dir,
        mask_dir=val_maskdir,
        label_dir=val_labeldir,
        transform=val_transform,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=False,
    )

    return train_loader, val_loader

def check_accuracy(loader, model, device="cpu"):
    num_correct = 0
    num_pixels = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device).unsqueeze(1)
            preds = model(x)
            preds = torch.argmax(preds, dim=1)
            num_correct += (preds == y).sum()
            num_pixels += torch.numel(preds)

    accuracy = num_correct / num_pixels
    print(f"Got {num_correct}/{num_pixels} with acc {accuracy:.2f}")
    model.train()

    return accuracy

def calculate_iou(preds, labels, num_classes):
    iou = 0
    preds = preds.view(-1)
    labels = labels.view(-1)

    for cls in range(num_classes):
        pred_inds = (preds == cls)
        label_inds = (labels == cls)
        intersection = (pred_inds & label_inds).sum().item()
        union = (pred_inds | label_inds).sum().item()
        if union == 0:
            iou += 0
        else:
            iou += intersection / union

    return iou / num_classes

def save_predictions_as_imgs(loader, model, folder="C:\\Inmind\\PROJECT\\SegNet\\saved_images\\", device="cpu"):
    if not os.path.exists(folder):
        os.makedirs(folder)

    model.eval()
    for idx, (x, y) in enumerate(loader):
        x = x.to(device=device)
        with torch.no_grad():
            preds = model(x)
            preds = torch.argmax(preds, dim=1).cpu().numpy()

        # Save predictions as images
        for i in range(preds.shape[0]):
            pred_img = preds[i]
            save_path = os.path.join(folder, f"pred_{idx}_{i}.png")
            torchvision.utils.save_image(torch.tensor(pred_img, dtype=torch.float32).unsqueeze(0), save_path)

    model.train()

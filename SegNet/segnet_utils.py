import torch
import torchvision
from segnet_dataset import SegNetDataset
from torch.utils.data import DataLoader

def save_checkpoint(state, filename="segnet_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)

def load_checkpoint(checkpoint, model):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])

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

def check_accuracy(loader, model, device="cuda"):
    model.eval()
    num_correct = 0
    num_pixels = 0
    iou_score = 0
    num_classes = len(loader.dataset.class_mapping)

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)

            preds = model(x)
            preds = torch.argmax(preds, dim=1)

            num_correct += (preds == y).sum().item()
            num_pixels += torch.numel(preds)
            iou_score += calculate_iou(preds, y, num_classes)

    print(
        f"Got {num_correct}/{num_pixels} with acc {num_correct/num_pixels*100:.2f}"
    )
    print(f"IoU score: {iou_score/len(loader):.4f}")
    model.train()

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

def save_predictions_as_imgs(
    loader, model, folder="saved_images/", device="cuda"
):
    model.eval()
    for idx, (x, y) in enumerate(loader):
        x = x.to(device)
        with torch.no_grad():
            preds = model(x)
            preds = torch.argmax(preds, dim=1)
        torchvision.utils.save_image(preds.unsqueeze(1).float(), f"{folder}/pred_{idx}.png")
        torchvision.utils.save_image(y.unsqueeze(1).float(), f"{folder}/{idx}.png")

    model.train()

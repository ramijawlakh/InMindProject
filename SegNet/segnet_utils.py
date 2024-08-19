import torch
import os
from torchvision.utils import save_image

def save_checkpoint(state, filename="checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)

def load_checkpoint(checkpoint, model):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])

def check_accuracy(loader, model, device="cuda", num_classes=9):
    model.eval()
    correct_pixels = 0
    total_pixels = 0
    class_correct = [0] * num_classes
    class_total = [0] * num_classes
    dice_score = 0

    with torch.no_grad():
        for data, masks, labels in loader:
            data = data.to(device)
            labels = labels.to(device)

            outputs = model(data)
            preds = torch.argmax(outputs, dim=1)

            correct_pixels += (preds == labels).sum().item()
            total_pixels += torch.numel(preds)

            for i in range(num_classes):
                class_correct[i] += ((preds == i) & (labels == i)).sum().item()
                class_total[i] += (labels == i).sum().item()

            dice_score += (2 * (preds * labels).sum()) / ((preds + labels).sum() + 1e-8)

    overall_accuracy = correct_pixels / total_pixels * 100
    class_accuracy = [100 * (c / t) if t != 0 else 0 for c, t in zip(class_correct, class_total)]
    mean_dice_score = dice_score / len(loader)

    print(f"Overall Accuracy: {overall_accuracy:.2f}%")
    for i, acc in enumerate(class_accuracy):
        print(f"Class {i} Accuracy: {acc:.2f}%")
    print(f"Mean Dice Score: {mean_dice_score:.4f}")

    model.train()
    return overall_accuracy

def save_predictions_as_imgs(loader, model, folder="saved_images/", device="cuda"):
    model.eval()
    os.makedirs(folder, exist_ok=True)

    with torch.no_grad():
        for idx, (data, masks, labels) in enumerate(loader):
            data = data.to(device)
            preds = model(data)
            preds = torch.argmax(preds, dim=1).unsqueeze(1)

            save_image(preds, os.path.join(folder, f"pred_{idx}.png"))

    model.train()


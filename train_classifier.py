"""
SynthVision: Transfer Learning & Typology Classifier (ResNet-18)
"""
import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, models
from torchvision.models import ResNet18_Weights
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

def train(data_dir, epochs=50, batch_size=32, lr=0.001, save_path="best_resnet18.pth"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Executing on: {device}")

    weights = ResNet18_Weights.DEFAULT
    transform = weights.transforms()

    train_dir = os.path.join(data_dir, "train")
    val_dir = os.path.join(data_dir, "val")

    train_dataset = datasets.ImageFolder(root=train_dir, transform=transform)
    val_dataset = datasets.ImageFolder(root=val_dir, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True if torch.cuda.is_available() else False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    num_classes = len(train_dataset.classes)
    print(f"Loaded {len(train_dataset)} Train images and {len(val_dataset)} Val images across {num_classes} classes.")

    # Initialize model with frozen backbone
    model = models.resnet18(weights=weights)
    for param in model.parameters():
        param.requires_grad = False

    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.fc.parameters(), lr=lr)

    best_val_loss = float("inf")

    for epoch in range(epochs):
        model.train()
        running_loss, correct, total = 0.0, 0, 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()

        # Validation loop
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
                _, preds = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (preds == labels).sum().item()

        epoch_val_loss = val_loss / len(val_dataset)
        if (epoch + 1) % 5 == 0 or epoch == epochs - 1:
            print(f"Epoch [{epoch+1:02d}/{epochs}] | Train Acc: {(correct/total)*100:.2f}% | Val Acc: {(val_correct/val_total)*100:.2f}% | Val Loss: {epoch_val_loss:.4f}")

        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            torch.save(model.state_dict(), save_path)

    print(f"Training finished. Checkpoint saved to: {save_path}")

def evaluate(test_dir, checkpoint_path="best_resnet18.pth"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weights = ResNet18_Weights.DEFAULT
    transform = weights.transforms()

    test_dataset = datasets.ImageFolder(root=test_dir, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    classes = [c.replace('_', ' ').title() for c in test_dataset.classes]

    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(classes))
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model = model.to(device)
    model.eval()

    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    print("\n--- Test Set Evaluation ---")
    print(classification_report(all_labels, all_preds, target_names=classes, digits=4))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ResNet18 Typology Classifier")
    parser.add_argument("--data", type=str, default="./data", help="Root data folder")
    parser.add_argument("--eval_only", action="store_true", help="Run evaluation on test split")
    args = parser.parse_args()

    if not args.eval_only:
        train(args.data)
    evaluate(os.path.join(args.data, "test"))
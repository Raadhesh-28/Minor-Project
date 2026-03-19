"""
Model training script for Indian traditional painting classification.

This script manages:
- Dataset loading
- Model training and validation
- Metric tracking (loss & accuracy)
- Saving best model checkpoint
- Plotting training curves

Run using:
    python -m src.training.train
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from src.dataset.dataset_loader import create_dataloader
from src.models.cnn_model import PaintingClassifier


def train_model():
    """
    Executes full training pipeline and saves metrics + plots.
    """

    # ---------------------
    # CREATE FOLDERS
    # ---------------------
    os.makedirs("models", exist_ok=True)
    os.makedirs("experiments", exist_ok=True)

    # ---------------------
    # TRACK METRICS
    # ---------------------
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    # ---------------------
    # LOAD DATA
    # ---------------------
    print("Loading datasets...")
    train_loader, val_loader = create_dataloader(
        root_dir="data/raw",
        mapping_file="data/metadata/class_mapping.json",
        batch_size=16,
        num_workers=0
    )

    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")

    # ---------------------
    # DEVICE
    # ---------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True
    print(f"Using device: {device}")

    # ---------------------
    # MODEL
    # ---------------------
    print("Initializing model...")
    model = PaintingClassifier(num_classes=8).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    epochs = 10
    best_val_acc = 0.0

    print("Starting training...")

    # ---------------------
    # TRAIN LOOP
    # ---------------------
    for epoch in range(epochs):

        # ---------------------
        # TRAINING
        # ---------------------
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)

            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

        epoch_train_loss = running_loss / total
        epoch_train_acc = (correct / total) * 100

        # ---------------------
        # VALIDATION
        # ---------------------
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for val_images, val_labels in val_loader:
                val_images, val_labels = val_images.to(device), val_labels.to(device)

                val_outputs = model(val_images)
                loss = criterion(val_outputs, val_labels)

                val_loss += loss.item() * val_images.size(0)

                _, val_predicted = torch.max(val_outputs, 1)
                val_correct += (val_predicted == val_labels).sum().item()
                val_total += val_labels.size(0)

        epoch_val_loss = val_loss / val_total
        epoch_val_acc = (val_correct / val_total) * 100

        # ---------------------
        # STORE METRICS
        # ---------------------
        train_losses.append(epoch_train_loss)
        val_losses.append(epoch_val_loss)
        train_accuracies.append(epoch_train_acc)
        val_accuracies.append(epoch_val_acc)

        print(f"Epoch {epoch+1}/{epochs} | "
              f"Train Loss: {epoch_train_loss:.4f} | "
              f"Train Acc: {epoch_train_acc:.2f}% | "
              f"Val Loss: {epoch_val_loss:.4f} | "
              f"Val Acc: {epoch_val_acc:.2f}%")

        # ---------------------
        # SAVE BEST MODEL
        # ---------------------
        if epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc
            torch.save(model.state_dict(), "models/cnn_classifier.pth")
            print("  [✓] Best model saved")

    # ---------------------
    # PLOT CURVES
    # ---------------------
    epochs_range = range(1, epochs + 1)

    # LOSS CURVE
    plt.figure()
    plt.plot(epochs_range, train_losses, label="Train Loss")
    plt.plot(epochs_range, val_losses, label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training vs Validation Loss")
    plt.legend()
    plt.savefig("experiments/loss_curve.png")
    plt.show()

    # ACCURACY CURVE
    plt.figure()
    plt.plot(epochs_range, train_accuracies, label="Train Accuracy")
    plt.plot(epochs_range, val_accuracies, label="Validation Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy (%)")
    plt.title("Training vs Validation Accuracy")
    plt.legend()
    plt.savefig("experiments/accuracy_curve.png")
    plt.show()

    print("\nTraining complete.")
    print("Plots saved in experiments/")


# ---------------------
# ENTRY POINT
# ---------------------
if __name__ == "__main__":
    train_model()
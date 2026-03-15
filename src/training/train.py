"""
Model training script for Indian traditional painting classification.

This script manages the initialization of the PaintingClassifier model, the training loop,
validation, and the saving of the best model state based on validation accuracy.
Expected to be executed from the project root using: `python -m src.training.train`
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim

from src.dataset.dataset_loader import create_dataloader
from src.models.cnn_model import PaintingClassifier

def train_model():
    """
    Sets up the dataset loaders, model, loss criterion, and optimizer, then executes 
    a training loop for 10 epochs. Tracks accuracy and saves the best model natively.
    """
    print("Loading datasets...")
    train_loader, val_loader = create_dataloader(
        root_dir="data/raw",
        mapping_file="data/metadata/class_mapping.json",
        batch_size=16,
        num_workers=0
    )
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")

    # Allow processing on GPU natively if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True
    print(f"Using device: {device}")

    print("Initializing model...")
    model = PaintingClassifier(num_classes=8).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    epochs = 10
    best_val_acc = 0.0

    print("Starting training...")
    for epoch in range(epochs):
        # ---------------------
        # Training Phase
        # ---------------------
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            # Zero gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward pass & optimization
            loss.backward()
            optimizer.step()

            # Track metrics
            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

        epoch_loss = running_loss / total
        epoch_acc = (correct / total) * 100

        # ---------------------
        # Validation Phase
        # ---------------------
        model.eval()
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for val_images, val_labels in val_loader:
                val_images, val_labels = val_images.to(device), val_labels.to(device)

                val_outputs = model(val_images)
                _, val_predictions = torch.max(val_outputs, 1)

                val_correct += (val_predictions == val_labels).sum().item()
                val_total += val_labels.size(0)

        # Avoid zero division if val_loader happens to be empty during early dev
        val_acc = (val_correct / val_total) * 100 if val_total > 0 else 0.0

        # Output formatting identically matching requested structural style
        print(f"Epoch {epoch + 1}/{epochs} — Loss: {epoch_loss:.2f} — Accuracy: {epoch_acc:.1f}% (Val Acc: {val_acc:.1f}%)")

        # ---------------------
        # Checkpointing
        # ---------------------
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            os.makedirs("models", exist_ok=True)
            torch.save({
                "model_state_dict": model.state_dict(),
                "best_val_accuracy": best_val_acc
            }, "models/painting_classifier.pth")
            print("  [✓] Improved validation accuracy. Saved model checkpoint.")

if __name__ == "__main__":
    train_model()

"""
Hybrid Classification module merging ResNet18 embeddings with handcrafted CSV features.

This robust script manages:
1. Frozen CNN Feature Extraction
2. Tabular Feature Normalization & Precomputation 
3. Deterministic DataLoader construction merging 512 CNN + 7 Math parameters native to RAM.
4. Custom 519-Input MLP ('HybridClassifier')
5. Comprehensive Torch Loop generating Train/Eval metrics and state checkpoints.

Expected Execution:
`python -m src.models.hybrid_model`
"""

import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
import pandas as pd
import numpy as np

def extract_cnn_features(image_path: str, backbone: nn.Module, device: torch.device) -> torch.Tensor:
    """
    Produces standard 512-dimension ResNet features given an image bound sequence.
    
    Args:
        image_path (str): Pointer resolving to the .jpg targeting representation.
        backbone (nn.Module): Truncated ResNet component.
        device (torch.device): Host processing memory allocator.
        
    Returns:
        torch.Tensor: Frozen 1D Tensor shaped `(512,)`.
    """
    # Strict fallback transforms matching original dataset configurations logically bound
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Isolate image
    image = Image.open(image_path).convert("RGB")
    tensor = transform(image).unsqueeze(0).to(device)
    
    # Ensure memory overhead is frozen effectively bypassing Backpropagation mapping natively
    with torch.no_grad():
        features = backbone(tensor)
        
    return features.squeeze(0)  # Return tightly packed 1D Vector structure: `(512,)`

class HybridDataset(Dataset):
    """
    Memory intensive custom Dataset generator strictly precomputing 512 ResNet distributions
    mapped exactly adjacent to normalized tabular CSV math parameters generating `519` tensors.
    """
    def __init__(self, csv_file: str, class_config: str, device: torch.device):
        print("\nLoading Hybrid Memory Context (this sequence caches explicitly into RAM once)...")
        # Initialize the raw metadata
        self.df = pd.read_csv(csv_file)
        
        with open(class_config, "r") as f:
            self.class_mapping = json.load(f)

        # Truncate and configure CNN Extractor Head logic strictly using default weights
        backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        backbone.fc = nn.Identity()  # Route output cleanly to pool length directly bypassing classifier
        for param in backbone.parameters():
            param.requires_grad = False
        backbone = backbone.to(device)
        backbone.eval()
        
        # Enforce highly strict deterministic list bounding CSV arrays directly corresponding to indices  
        self.feature_columns = [
            "mean_r", "mean_g", "mean_b", 
            "color_variance", "edge_density", 
            "symmetry_score", "texture_entropy"
        ]
        
        # Extrapolate values dynamically and standardize explicitly.
        # This resolves outliers breaking downstream gradient layers mapping strictly to identical scales.
        # Note: In real production, parameters computed across train should apply identically 
        # against strictly isolated test structures.
        # Load feature normalization statistics
        with open("data/metadata/feature_stats.json") as f:
            feature_stats = json.load(f)

        for col in self.feature_columns:

            if col in ["mean_r", "mean_g", "mean_b"]:
                # RGB already bounded [0,255]
                self.df[col] = self.df[col] / 255.0
            else:
                mean_val = feature_stats[col]["mean"]
                std_val = feature_stats[col]["std"]

                if std_val > 0:
                    self.df[col] = (self.df[col] - mean_val) / std_val
                
        # Allocate caching arrays natively routing final structures sequentially.
        self.hybrid_tensors = []
        self.labels = []
        
        total_rows = len(self.df)
        
        for idx, row in self.df.iterrows():
            if idx % 50 == 0:
                print(f"  Precomputing Record [{idx}/{total_rows}]...")
                
            img_path = row["image_path"]
            label_str = row["label"]
            label_int = self.class_mapping[label_str]
            
            # Map Handcrafted 7-Dimension
            hand_vars = [row[col] for col in self.feature_columns]
            hand_tensor = torch.tensor(hand_vars, dtype=torch.float32).to(device)
            
            # Extract Core 512-Dimension 
            cnn_tensor = extract_cnn_features(img_path, backbone, device)
            
            # Generate Hybrid 519 Element (512 Deep Metrics + 7 Visual Bounds) Array Map
            hybrid_tensor = torch.cat([cnn_tensor, hand_tensor], dim=0)
            
            self.hybrid_tensors.append(hybrid_tensor)
            self.labels.append(label_int)
            
        print("Caching Logic Resolved.")

    def __len__(self):
        return len(self.hybrid_tensors)
        
    def __getitem__(self, idx):
        return self.hybrid_tensors[idx], self.labels[idx]

class HybridClassifier(nn.Module):
    """
    Custom 519-dimension mapping architecture classifying native artwork signatures.
    """
    def __init__(self):
        super(HybridClassifier, self).__init__()
        
        # Structuring core sequence block targeting bounds 519 -> 256 -> 8
        self.mlp = nn.Sequential(
            nn.Linear(512 + 7, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 8)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Computes logits directly mapped from hybrid embeddings natively."""
        return self.mlp(x)

def evaluate_hybrid_model(model: nn.Module, val_loader: DataLoader) -> tuple:
    """Standard evaluation sequence traversing precomputed indices returning losses / accuracy logs."""
    model.eval()
    criterion = nn.CrossEntropyLoss()
    val_loss = 0.0
    val_correct = 0
    val_total = 0
    
    with torch.no_grad():
        for batch_features, batch_labels in val_loader:
            batch_features = batch_features.to(next(model.parameters()).device)
            batch_labels = batch_labels.to(next(model.parameters()).device)

            outputs = model(batch_features)
            loss = criterion(outputs, batch_labels)
            
            val_loss += loss.item() * batch_features.size(0)
            
            _, predictions = torch.max(outputs, 1)
            val_correct += (predictions == batch_labels).sum().item()
            val_total += batch_labels.size(0)
            
    val_loss_avg = val_loss / val_total if val_total > 0 else 0.0
    val_acc_avg = (val_correct / val_total) * 100 if val_total > 0 else 0.0
    
    return val_loss_avg, val_acc_avg

def train_hybrid_model():
    """
    Handles initialization, memory caching, dimensionality validation, standard PyTorch 
    loss loops mapping 80/20 train intervals, and best-checkpoint constraints outputting natively.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Hybrid Sequence Bound To Device -> {device}")
    
    # 1. Load Precomputed Embeddings Map Array Native
    print("\n--- Phase 1: Embedding Allocations ---")
    dataset = HybridDataset(
        csv_file="data/features.csv",
        class_config="data/metadata/class_mapping.json",
        device=device
    )
    
    dataset_size = len(dataset)
    if dataset_size == 0:
        print("No valid CSV records extrapolated. Aborting.")
        return
        
    # Check dimensionality explicitly mapping logic expected strictly onto 519 array distributions.
    test_tensor, _ = dataset[0]
    expected_dim = 512 + 7
    assert test_tensor.shape[0] == expected_dim, f"Dimension Mismatch: Received {test_tensor.shape[0]}, Exepcted {expected_dim}"
    print(f"Structure Dimension Assertions Passed -> Vector Mapping Bounds [{expected_dim}]")

    # 2. Assign Native Shuffling / Validation Bounds
    print("\n--- Phase 2: Loading Subsets ---")
    train_size = int(0.8 * dataset_size)
    val_size = dataset_size - train_size
    
    # Ensures representations are thoroughly shuffled bypassing identical temporal ordering anomalies 
    generator = torch.Generator().manual_seed(42)
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=generator)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    # 3. Initialize Model Blocks
    model = HybridClassifier().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    print("\n--- Phase 3: Epoch Loop Computations ---")
    epochs = 10
    best_val_acc = 0.0
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        
        for batch_features, batch_labels in train_loader:
            batch_features = batch_features.to(device)
            batch_labels = batch_labels.to(device)

            optimizer.zero_grad()
            
            outputs = model(batch_features)
            loss = criterion(outputs, batch_labels)
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * batch_features.size(0)
            
        train_loss_avg = running_loss / train_size
        
        # Evaluate against validation context directly generating stats
        val_loss, val_acc = evaluate_hybrid_model(model, val_loader)
        
        print(f"Epoch {epoch + 1:02d}/{epochs} | "
              f"Train Loss: {train_loss_avg:.4f} | "
              f"Val Loss: {val_loss:.4f} | "
              f"Val Accuracy: {val_acc:.2f}%")
              
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            os.makedirs("models", exist_ok=True)
            torch.save({
                "model_state_dict": model.state_dict(),
                "best_val_accuracy": best_val_acc
            }, "models/hybrid_classifier.pth")
            print(f"  -> Checkpoint Output Extrapolated [Best Val: {best_val_acc:.2f}%]")

    print("\nHybrid Implementation Routine Concluded Natively.")

if __name__ == "__main__":
    train_hybrid_model()
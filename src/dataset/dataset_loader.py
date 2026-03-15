"""
Dataset loader module for Indian traditional painting classification.

This module provides the `PaintingDataset` PyTorch class for loading images from disk,
and the `create_dataloader` utility for creating train and validation splits.
"""
import os
import json
from PIL import Image
from typing import Tuple, Dict, Any, List

import torch
from torch.utils.data import Dataset, DataLoader, random_split, Subset
import torchvision.transforms as transforms

class PaintingDataset(Dataset):
    """
    PyTorch Dataset for loading traditional Indian paintings.
    
    Attributes:
        root_dir (str): Path to the root dataset directory containing class folders.
        transform (callable, optional): Optional transform to be applied on a sample.
        class_mapping (dict): Mapping from string class names to integer labels.
        image_paths (list): List of paths to the valid image files.
        labels (list): Integer labels corresponding to each image in `image_paths`.
    """
    
    def __init__(self, root_dir: str, mapping_file: str, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths: List[str] = []
        self.labels: List[int] = []
        
        # Load class mapping
        with open(mapping_file, 'r') as f:
            self.class_mapping: Dict[str, int] = json.load(f)
            
        valid_extensions = {".jpg", ".jpeg", ".png"}

        # Scan for images 
        # Ignoring hidden files natively by skipping files/folders starting with '.'
        for class_name in os.listdir(root_dir):
            if class_name.startswith('.'):
                continue
                
            class_path = os.path.join(root_dir, class_name)
            
            if not os.path.isdir(class_path):
                continue
                
            if class_name not in self.class_mapping:
                continue
                
            label = self.class_mapping[class_name]
                
            for file_name in os.listdir(class_path):
                if file_name.startswith('.'):
                    continue
                    
                ext = os.path.splitext(file_name)[1].lower()
                if ext in valid_extensions:
                    full_path = os.path.join(class_path, file_name)
                    self.image_paths.append(full_path)
                    self.labels.append(label)

        print(f"Loaded {len(self.image_paths)} images across {len(self.class_mapping)} classes")

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path = self.image_paths[idx]
        
        # Load image and explicitly convert to RGB
        image = Image.open(img_path).convert("RGB")
        
        if self.transform:
            image = self.transform(image)
            
        label = self.labels[idx]
        return image, label

def create_dataloader(
    root_dir: str, 
    mapping_file: str, 
    batch_size: int = 32, 
    val_split: float = 0.2, 
    num_workers: int = 4, 
    pin_memory: bool = True
) -> Tuple[DataLoader, DataLoader]:
    """
    Creates PyTorch DataLoaders for training and validation datasets.
    
    Args:
        root_dir (str): Path to the raw dataset folder.
        mapping_file (str): Path to the JSON file containing label translations.
        batch_size (int): The batch size to use for training/testing.
        val_split (float): Ratio of dataset to reserve for validation.
        num_workers (int): Number of subprocesses to use for data loading.
        pin_memory (bool): If True, allocates tensors into pinned memory.
        
    Returns:
        tuple: (train_loader, val_loader)
    """
    
    # 1. Base Transform for Validation Data (No augmentation)
    base_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 2. Train Transform with Augmentations
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 3. Initialize full dataset WITHOUT transforms initially so we can apply them separately
    full_dataset = PaintingDataset(root_dir=root_dir, mapping_file=mapping_file, transform=None)

    # 4. Create deterministic train/val splits
    torch.manual_seed(42)  # Set random seed for reproducibility
    dataset_size = len(full_dataset)
    val_size = int(val_split * dataset_size)
    train_size = dataset_size - val_size
    
    train_subset, val_subset = random_split(full_dataset, [train_size, val_size])
    
    # 5. Apply separate transformations to subsets via a custom wrapper
    class TransformWrapper(Dataset):
        def __init__(self, subset, transform):
            self.subset = subset
            self.transform = transform
            
        def __getitem__(self, idx):
            # Extract underlying raw image and label from the Subset -> Dataset graph
            image, label = self.subset[idx]
            if self.transform:
                image = self.transform(image)
            return image, label
            
        def __len__(self):
            return len(self.subset)

    train_dataset = TransformWrapper(train_subset, transform=train_transform)
    val_dataset = TransformWrapper(val_subset, transform=base_transform)

    # 6. Create DataLoaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers, 
        pin_memory=pin_memory
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers, 
        pin_memory=pin_memory
    )
    
    return train_loader, val_loader

"""
ResNet18-based transfer learning model for Indian painting style classification.

This module provides the `PaintingClassifier` which uses a pretrained 
ResNet18 backbone to classify images into traditional painting styles.
"""

import torch
import torch.nn as nn
import torchvision.models as models

class PaintingClassifier(nn.Module):
    """
    CNN classifier based on a ResNet18 backbone.
    
    Architecture:
    - Pretrained ResNet18 backbone
    - Dropout layer (p=0.5) for regularization
    - Linear classifier projecting to the target number of classes
    """
    
    def __init__(self, num_classes: int = 8):
        """
        Initializes the model architecture.
        
        Args:
            num_classes (int): The number of painting classes to output.
        """
        super(PaintingClassifier, self).__init__()
        
        # Load the pretrained ResNet18 model using modern weights
        self.backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        
        # Extract the input features from the original fully connected layer
        num_ftrs = self.backbone.fc.in_features
        
        # Replace the final layer with Dropout and a new Linear classifier
        self.backbone.fc = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(num_ftrs, num_classes)
        )
        
    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Runs the input through the backbone up to the layer before the classifier.
        Useful for combining CNN features with handcrafted features.
        
        Args:
            x (torch.Tensor): Input image tensor of shape (B, C, H, W).
            
        Returns:
            torch.Tensor: High-level semantic feature vector of shape (B, 512).
        """
        # Pass input through all layers except the final fully connected layer (fc)
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)

        x = self.backbone.avgpool(x)
        x = torch.flatten(x, 1)
        
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.
        
        Args:
            x (torch.Tensor): Input image tensor of shape (B, C, H, W).
            
        Returns:
            torch.Tensor: Raw logits for each class.
        """
        return self.backbone(x)

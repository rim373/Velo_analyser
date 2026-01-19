"""
Binary Classifier - ResNet50 for Intact vs Damaged
"""

import torch
import torch.nn as nn
from torchvision import models
from typing import Tuple


class BikeClassifier(nn.Module):
    """
    ResNet50-based binary classifier for bike defect detection
    """
    
    def __init__(
        self,
        num_classes: int = 2,
        pretrained: bool = True,
        dropout: float = 0.5
    ):
        """
        Args:
            num_classes: Number of output classes (default: 2 for intact/damaged)
            pretrained: Use ImageNet pretrained weights
            dropout: Dropout rate before final layer
        """
        super(BikeClassifier, self).__init__()
        
        # Load pretrained ResNet50
        self.backbone = models.resnet50(pretrained=pretrained)
        
        # Get number of input features to final layer
        num_features = self.backbone.fc.in_features
        
        # Replace final layer with custom classifier
        self.backbone.fc = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(num_features, num_classes)
        )
        
        self.num_classes = num_classes
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Input tensor [B, 3, H, W]
            
        Returns:
            Logits [B, num_classes]
        """
        return self.backbone(x)
    
    def predict(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Make prediction with probabilities
        
        Args:
            x: Input tensor [B, 3, H, W]
            
        Returns:
            Tuple of (predicted_class, probabilities)
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            probabilities = torch.softmax(logits, dim=1)
            predicted = torch.argmax(probabilities, dim=1)
        
        return predicted, probabilities
    
    def freeze_backbone(self):
        """Freeze backbone parameters (for transfer learning)"""
        for param in self.backbone.parameters():
            param.requires_grad = False
        
        # Unfreeze final layer
        for param in self.backbone.fc.parameters():
            param.requires_grad = True
    
    def unfreeze_backbone(self):
        """Unfreeze all parameters"""
        for param in self.backbone.parameters():
            param.requires_grad = True


class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance
    """
    
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        """
        Args:
            alpha: Weighting factor for class imbalance
            gamma: Focusing parameter
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute focal loss
        
        Args:
            inputs: Predictions [B, C]
            targets: Ground truth labels [B]
            
        Returns:
            Focal loss value
        """
        ce_loss = nn.CrossEntropyLoss(reduction='none')(inputs, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        return focal_loss.mean()


if __name__ == '__main__':
    # Test classifier
    print("Testing BikeClassifier...")
    
    # Create model
    model = BikeClassifier(num_classes=2, pretrained=True)
    print(f"✅ Model created")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test forward pass
    dummy_input = torch.randn(4, 3, 224, 224)
    output = model(dummy_input)
    print(f"✅ Forward pass: {dummy_input.shape} -> {output.shape}")
    
    # Test prediction
    predicted, probs = model.predict(dummy_input)
    print(f"✅ Predictions: {predicted}")
    print(f"✅ Probabilities shape: {probs.shape}")
    
    # Test focal loss
    criterion = FocalLoss()
    targets = torch.randint(0, 2, (4,))
    loss = criterion(output, targets)
    print(f"✅ Focal loss: {loss.item():.4f}")
    
    print("\n✅ All tests passed!")

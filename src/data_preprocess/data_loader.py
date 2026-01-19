"""
Data Loaders
PyTorch Dataset classes for binary classification and PatchCore
"""

import os
from pathlib import Path
from typing import Tuple, Optional, Callable, List
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np


class BikeClassificationDataset(Dataset):
    """
    Dataset for binary classification (intact vs damaged)
    """
    
    def __init__(
        self,
        intact_dir: str,
        damaged_dir: str,
        transform: Optional[Callable] = None,
        image_size: int = 224
    ):
        """
        Args:
            intact_dir: Directory containing intact bike images
            damaged_dir: Directory containing damaged bike images
            transform: Optional transform to apply
            image_size: Target image size
        """
        self.intact_dir = Path(intact_dir)
        self.damaged_dir = Path(damaged_dir)
        self.transform = transform
        self.image_size = image_size
        
        # Get all image paths
        self.intact_images = list(self.intact_dir.glob('*.jpg')) + \
                            list(self.intact_dir.glob('*.png'))
        self.damaged_images = list(self.damaged_dir.glob('*.jpg')) + \
                             list(self.damaged_dir.glob('*.png'))
        
        # Create labels (0 = intact, 1 = damaged)
        self.images = self.intact_images + self.damaged_images
        self.labels = [0] * len(self.intact_images) + [1] * len(self.damaged_images)
        
        # Default transform if none provided
        if self.transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])
        
        print(f"Dataset initialized:")
        print(f"  Intact images: {len(self.intact_images)}")
        print(f"  Damaged images: {len(self.damaged_images)}")
        print(f"  Total images: {len(self.images)}")
    
    def __len__(self) -> int:
        return len(self.images)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Get item by index
        
        Returns:
            Tuple of (image_tensor, label)
        """
        image_path = self.images[idx]
        label = self.labels[idx]
        
        # Load image
        image = Image.open(image_path).convert('RGB')
        
        # Apply transform
        if self.transform:
            image = self.transform(image)
        
        return image, label
    
    def get_class_weights(self) -> torch.Tensor:
        """
        Compute class weights for imbalanced dataset
        
        Returns:
            Tensor of class weights
        """
        class_counts = [
            len(self.intact_images),
            len(self.damaged_images)
        ]
        total = sum(class_counts)
        weights = [total / (2 * count) for count in class_counts]
        return torch.tensor(weights, dtype=torch.float32)


class PatchCoreDataset(Dataset):
    """
    Dataset for PatchCore training (intact images only)
    and testing (damaged images)
    """
    
    def __init__(
        self,
        image_dir: str,
        transform: Optional[Callable] = None,
        image_size: int = 256,
        is_training: bool = True
    ):
        """
        Args:
            image_dir: Directory containing images
            transform: Optional transform to apply
            image_size: Target image size
            is_training: True for training (intact only), False for testing
        """
        self.image_dir = Path(image_dir)
        self.transform = transform
        self.image_size = image_size
        self.is_training = is_training
        
        # Get all image paths
        self.images = list(self.image_dir.glob('*.jpg')) + \
                     list(self.image_dir.glob('*.png'))
        
        # Default transform if none provided
        if self.transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])
        
        print(f"PatchCore Dataset initialized:")
        print(f"  Mode: {'Training' if is_training else 'Testing'}")
        print(f"  Images: {len(self.images)}")
    
    def __len__(self) -> int:
        return len(self.images)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str]:
        """
        Get item by index
        
        Returns:
            Tuple of (image_tensor, image_path)
        """
        image_path = self.images[idx]
        
        # Load image
        image = Image.open(image_path).convert('RGB')
        
        # Apply transform
        if self.transform:
            image = self.transform(image)
        
        return image, str(image_path)


def create_classification_dataloaders(
    intact_dir: str,
    damaged_dir: str,
    batch_size: int = 32,
    num_workers: int = 4,
    train_split: float = 0.7,
    val_split: float = 0.15,
    test_split: float = 0.15,
    image_size: int = 224,
    train_transform: Optional[Callable] = None,
    val_transform: Optional[Callable] = None
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, validation, and test dataloaders for classification
    
    Args:
        intact_dir: Directory with intact images
        damaged_dir: Directory with damaged images
        batch_size: Batch size
        num_workers: Number of worker processes
        train_split: Training set proportion
        val_split: Validation set proportion
        test_split: Test set proportion
        image_size: Target image size
        train_transform: Transform for training set
        val_transform: Transform for val/test sets
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    from torch.utils.data import random_split
    
    # Create full dataset
    dataset = BikeClassificationDataset(
        intact_dir=intact_dir,
        damaged_dir=damaged_dir,
        image_size=image_size
    )
    
    # Calculate split sizes
    total_size = len(dataset)
    train_size = int(train_split * total_size)
    val_size = int(val_split * total_size)
    test_size = total_size - train_size - val_size
    
    # Split dataset
    train_dataset, val_dataset, test_dataset = random_split(
        dataset,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    # Apply specific transforms if provided
    if train_transform:
        train_dataset.dataset.transform = train_transform
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    print(f"\nDataLoader creation complete:")
    print(f"  Train samples: {len(train_dataset)}")
    print(f"  Val samples: {len(val_dataset)}")
    print(f"  Test samples: {len(test_dataset)}")
    
    return train_loader, val_loader, test_loader


def create_patchcore_dataloaders(
    intact_dir: str,
    damaged_dir: str,
    batch_size: int = 32,
    num_workers: int = 4,
    image_size: int = 256,
    val_split: float = 0.2
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create dataloaders for PatchCore
    
    Args:
        intact_dir: Directory with intact images (for training)
        damaged_dir: Directory with damaged images (for testing)
        batch_size: Batch size
        num_workers: Number of worker processes
        image_size: Target image size
        val_split: Validation split from intact images
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    from torch.utils.data import random_split
    
    # Training set (intact images only)
    train_dataset_full = PatchCoreDataset(
        image_dir=intact_dir,
        image_size=image_size,
        is_training=True
    )
    
    # Split into train and validation
    val_size = int(val_split * len(train_dataset_full))
    train_size = len(train_dataset_full) - val_size
    
    train_dataset, val_dataset = random_split(
        train_dataset_full,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    # Test set (damaged images)
    test_dataset = PatchCoreDataset(
        image_dir=damaged_dir,
        image_size=image_size,
        is_training=False
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False,  # Order doesn't matter for PatchCore
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,  # Process one at a time for testing
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    print(f"\nPatchCore DataLoader creation complete:")
    print(f"  Train samples (intact): {len(train_dataset)}")
    print(f"  Val samples (intact): {len(val_dataset)}")
    print(f"  Test samples (damaged): {len(test_dataset)}")
    
    return train_loader, val_loader, test_loader


if __name__ == '__main__':
    # Test dataloaders
    print("Testing Classification DataLoaders...")
    train_loader, val_loader, test_loader = create_classification_dataloaders(
        intact_dir='data/processed/intact',
        damaged_dir='data/processed/damaged',
        batch_size=8
    )
    
    # Test one batch
    images, labels = next(iter(train_loader))
    print(f"\nBatch shape: {images.shape}")
    print(f"Labels shape: {labels.shape}")
    print(f"Label distribution: {labels.sum().item()} damaged out of {len(labels)}")

"""
DRAEM Training Script
Train DRAEM model for bike defect detection and localization
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
import argparse
from tqdm import tqdm
import yaml

from src.models.draem import DRAEM
from src.data_preprocess.anomaly_generator import AnomalyGenerator


class DRAEMDataset(Dataset):
    """Dataset for DRAEM training with synthetic anomalies"""
    def __init__(self, intact_dir, anomaly_source_path=None, transform=None, image_size=256):
        """
        Args:
            intact_dir: Directory with normal/intact images only
            anomaly_source_path: Path to texture dataset for anomaly generation
            transform: Image transformations
            image_size: Target image size
        """
        self.intact_dir = Path(intact_dir)
        self.image_size = image_size

        # Get all intact images
        self.image_paths = []
        for ext in ['*.jpg', '*.png', '*.jpeg']:
            self.image_paths.extend(list(self.intact_dir.glob(f'**/{ext}')))

        print(f"[OK] Found {len(self.image_paths)} intact images for training")

        # Anomaly generator
        self.anomaly_generator = AnomalyGenerator(
            anomaly_source_path=anomaly_source_path,
            resize_shape=(image_size, image_size)
        )

        # Transforms
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Load intact image
        img_path = self.image_paths[idx]
        img = Image.open(img_path).convert('RGB')
        img = img.resize((self.image_size, self.image_size))
        img_np = np.array(img)

        # Generate synthetic anomaly
        aug_img, anomaly_mask = self.anomaly_generator.generate_anomaly(img_np)

        # Convert to PIL for transforms
        aug_img_pil = Image.fromarray(aug_img)

        # Apply transforms
        intact_tensor = self.transform(img)
        augmented_tensor = self.transform(aug_img_pil)

        # Mask to tensor
        mask_tensor = torch.from_numpy(anomaly_mask).unsqueeze(0).float()

        return intact_tensor, augmented_tensor, mask_tensor


class DRAEMLoss(nn.Module):
    """Combined loss for DRAEM training"""
    def __init__(self):
        super(DRAEMLoss, self).__init__()
        self.l2_loss = nn.MSELoss()
        self.focal_loss = FocalLoss()

    def forward(self, reconstruction, segmentation, target_img, target_mask):
        """
        Args:
            reconstruction: Reconstructed image
            segmentation: Predicted segmentation (2 channels: normal, anomaly)
            target_img: Target image (intact)
            target_mask: Ground truth anomaly mask

        Returns:
            total_loss, reconstruction_loss, segmentation_loss
        """
        # Reconstruction loss (L2)
        recon_loss = self.l2_loss(reconstruction, target_img)

        # SSIM loss
        ssim_loss = 1 - ssim(reconstruction, target_img)

        # Segmentation loss (Focal)
        # Convert mask to long tensor for cross entropy
        target_mask_long = target_mask.squeeze(1).long()
        seg_loss = self.focal_loss(segmentation, target_mask_long)

        # Combined loss
        total_loss = recon_loss + ssim_loss + seg_loss

        return total_loss, recon_loss, ssim_loss, seg_loss


class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance"""
    def __init__(self, alpha=0.25, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        """
        Args:
            inputs: Predictions (B, 2, H, W)
            targets: Ground truth (B, H, W) with values 0 or 1
        """
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        return focal_loss.mean()


def ssim(img1, img2, window_size=11, size_average=True):
    """
    Structural Similarity Index (SSIM)
    """
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    mu1 = F.avg_pool2d(img1, window_size, stride=1, padding=window_size // 2)
    mu2 = F.avg_pool2d(img2, window_size, stride=1, padding=window_size // 2)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.avg_pool2d(img1 * img1, window_size, stride=1, padding=window_size // 2) - mu1_sq
    sigma2_sq = F.avg_pool2d(img2 * img2, window_size, stride=1, padding=window_size // 2) - mu2_sq
    sigma12 = F.avg_pool2d(img1 * img2, window_size, stride=1, padding=window_size // 2) - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


def train_draem(config):
    """Train DRAEM model"""
    print("="*60)
    print("DRAEM TRAINING")
    print("="*60)

    # Device
    device = torch.device(config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
    print(f"Device: {device}")

    # Create dataset
    train_dataset = DRAEMDataset(
        intact_dir=config['data']['intact_dir'],
        anomaly_source_path=config['data'].get('anomaly_source_path'),
        image_size=config['data']['image_size']
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=config['data'].get('num_workers', 4),
        pin_memory=True
    )

    # Create model
    model = DRAEM(
        in_channels=3,
        out_channels_seg=2
    ).to(device)

    print(f"[OK] Model created")
    print(f"[OK] Training samples: {len(train_dataset)}")

    # Loss and optimizer
    criterion = DRAEMLoss()

    optimizer = torch.optim.Adam([
        {'params': model.reconstructive.parameters(), 'lr': config['training']['lr']},
        {'params': model.discriminative.parameters(), 'lr': config['training']['lr']}
    ])

    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=[
            int(config['training']['epochs'] * 0.8),
            int(config['training']['epochs'] * 0.9)
        ],
        gamma=0.2
    )

    # Training loop
    print("\n" + "="*60)
    print("PHASE 1: TRAINING")
    print("="*60)

    for epoch in range(config['training']['epochs']):
        model.train()

        epoch_loss = 0
        epoch_recon_loss = 0
        epoch_ssim_loss = 0
        epoch_seg_loss = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['training']['epochs']}")

        for batch_idx, (intact, augmented, mask) in enumerate(pbar):
            intact = intact.to(device)
            augmented = augmented.to(device)
            mask = mask.to(device)

            # Forward pass
            reconstruction, segmentation = model(augmented)

            # Compute loss
            total_loss, recon_loss, ssim_loss, seg_loss = criterion(
                reconstruction, segmentation, intact, mask
            )

            # Backward pass
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            # Track losses
            epoch_loss += total_loss.item()
            epoch_recon_loss += recon_loss.item()
            epoch_ssim_loss += ssim_loss.item()
            epoch_seg_loss += seg_loss.item()

            # Update progress bar
            pbar.set_postfix({
                'loss': total_loss.item(),
                'recon': recon_loss.item(),
                'seg': seg_loss.item()
            })

        # Epoch statistics
        n_batches = len(train_loader)
        print(f"\nEpoch {epoch+1} Summary:")
        print(f"  Total Loss: {epoch_loss/n_batches:.4f}")
        print(f"  Reconstruction Loss: {epoch_recon_loss/n_batches:.4f}")
        print(f"  SSIM Loss: {epoch_ssim_loss/n_batches:.4f}")
        print(f"  Segmentation Loss: {epoch_seg_loss/n_batches:.4f}")
        print(f"  LR: {optimizer.param_groups[0]['lr']:.6f}")

        # Step scheduler
        scheduler.step()

        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            checkpoint_dir = Path(config['checkpoint']['save_dir'])
            checkpoint_dir.mkdir(parents=True, exist_ok=True)

            checkpoint_path = checkpoint_dir / f'draem_epoch_{epoch+1}.pth'
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': epoch_loss / n_batches
            }, checkpoint_path)
            print(f"  [OK] Checkpoint saved: {checkpoint_path}")

    # Save final model
    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)

    checkpoint_dir = Path(config['checkpoint']['save_dir'])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    model_path = checkpoint_dir / 'draem_final.pth'
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config
    }, model_path)

    print(f"[OK] Final model saved: {model_path}")

    return model


def main():
    parser = argparse.ArgumentParser(description='Train DRAEM model')
    parser.add_argument('--config', type=str,
                       default='configs/draem_config.yaml',
                       help='Path to config file')
    parser.add_argument('--intact_dir', type=str,
                       help='Directory with intact images')
    parser.add_argument('--anomaly_source', type=str,
                       help='Directory with texture images for anomaly generation')
    parser.add_argument('--epochs', type=int,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int,
                       help='Batch size')
    parser.add_argument('--lr', type=float,
                       help='Learning rate')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use')

    args = parser.parse_args()

    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Override with command line args
    if args.intact_dir:
        config['data']['intact_dir'] = args.intact_dir
    if args.anomaly_source:
        config['data']['anomaly_source_path'] = args.anomaly_source
    if args.epochs:
        config['training']['epochs'] = args.epochs
    if args.batch_size:
        config['training']['batch_size'] = args.batch_size
    if args.lr:
        config['training']['lr'] = args.lr
    if args.device:
        config['device'] = args.device

    # Train
    model = train_draem(config)

    print("\n[OK] DRAEM training complete!")
    print("\nNext steps:")
    print("1. Test localization: python test_draem.py --image test.jpg")
    print("2. Evaluate model: python evaluate_draem.py")


if __name__ == '__main__':
    main()

"""
ANTI-OVERFITTING Training Script - Stage 1
Techniques: Mixup, Label Smoothing, Progressive Augmentation, Proper Regularization
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import yaml
from tqdm import tqdm
import argparse
from typing import Dict
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve
import seaborn as sns

from src.models.classifier import BikeClassifier, FocalLoss
from src.data_preprocess.data_loader import create_classification_dataloaders


def mixup_data(x, y, alpha=0.4):
    """Mixup augmentation to prevent overfitting"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(x.device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """Mixup loss calculation"""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


class LabelSmoothingCrossEntropy(nn.Module):
    """Label smoothing to prevent overconfidence"""
    def __init__(self, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing

    def forward(self, pred, target):
        confidence = 1.0 - self.smoothing
        n_classes = pred.size(-1)

        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (n_classes - 1))
            true_dist.scatter_(1, target.unsqueeze(1), confidence)

        return torch.mean(torch.sum(-true_dist * F.log_softmax(pred, dim=-1), dim=-1))


class OptimizedTrainer:
    """Anti-overfitting trainer with Mixup, Label Smoothing, and proper regularization"""

    def __init__(self, config: Dict, output_dir: Path):
        self.config = config
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Model with MODERATE dropout (not too high!)
        self.model = BikeClassifier(
            num_classes=2,
            pretrained=True,
            dropout=0.4  # BALANCED - was 0.7 (too high!)
        ).to(self.device)

        print(f"✅ Model on {self.device}")

        # Label Smoothing for main criterion (prevents overconfidence)
        self.criterion = LabelSmoothingCrossEntropy(smoothing=0.1)

        # Focal Loss for validation only
        self.focal_criterion = FocalLoss(alpha=0.25, gamma=2.0)
        print(f"✅ Label Smoothing (0.1) + Focal Loss (gamma=2.0)")

        # Optimizer with MODERATE weight decay
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=config['training']['learning_rate'],
            weight_decay=0.01  # BALANCED - was 0.05 (too high!)
        )

        self.scheduler = None
        self.best_val_f1 = 0.0
        self.patience_counter = 0
        self.mixup_alpha = 0.4  # Mixup strength

        # Tracking
        self.history = {
            'train_loss': [], 'val_loss': [],
            'train_acc': [], 'val_acc': [],
            'val_precision': [], 'val_recall': [], 'val_f1': []
        }
    
    def create_balanced_sampler(self, dataset):
        """Create weighted sampler to balance classes - MODERATE boost"""
        # Get labels
        labels = []
        for i in range(len(dataset)):
            _, label = dataset[i]
            labels.append(label)

        labels = np.array(labels)

        # Count classes
        class_counts = np.bincount(labels)
        print(f"\n📊 Training set distribution:")
        print(f"   Intact:  {class_counts[0]} images")
        print(f"   Damaged: {class_counts[1]} images")
        print(f"   Ratio:   {class_counts[0]/class_counts[1]:.1f}:1")

        # Calculate weights - MODERATE boost (not 3x!)
        class_weights = 1.0 / class_counts
        sample_weights = class_weights[labels]

        # MODERATE boost (1.5x instead of 3x)
        sample_weights[labels == 1] *= 1.5

        print(f"✅ Balanced sampler: Damaged samples boosted by 1.5x (moderate)")

        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True
        )

        return sampler
    
    def train_epoch(self, train_loader: DataLoader, use_mixup: bool = True) -> Dict:
        """Train one epoch with Mixup augmentation"""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(train_loader, desc=f"Epoch {len(self.history['train_loss'])+1}")

        for images, labels in pbar:
            images = images.to(self.device)
            labels = labels.to(self.device)

            # Apply Mixup with 50% probability
            if use_mixup and np.random.rand() < 0.5:
                mixed_images, labels_a, labels_b, lam = mixup_data(images, labels, self.mixup_alpha)
                outputs = self.model(mixed_images)
                loss = mixup_criterion(self.criterion, outputs, labels_a, labels_b, lam)
            else:
                # Normal forward pass
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

            # Backward
            self.optimizer.zero_grad()
            loss.backward()

            # Gradient clipping (moderate)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.optimizer.step()

            if self.scheduler:
                self.scheduler.step()

            # Stats
            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'acc': f"{100*correct/total:.1f}%"
            })

        return {
            'loss': total_loss / len(train_loader),
            'accuracy': 100 * correct / total
        }
    
    def validate(self, val_loader: DataLoader, threshold: float = 0.4, use_focal: bool = False) -> Dict:
        """Validate with adjustable threshold"""
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_labels = []
        all_probs = []

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(images)
                # Use same loss as training for fair comparison (label smoothing)
                # Only use focal loss for final evaluation
                if use_focal:
                    loss = self.focal_criterion(outputs, labels)
                else:
                    loss = self.criterion(outputs, labels)
                total_loss += loss.item()

                probs = torch.softmax(outputs, dim=1)

                # Use custom threshold for damaged class
                predicted = (probs[:, 1] > threshold).long()

                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs[:, 1].cpu().numpy())
        
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        
        # Metrics
        correct = (all_preds == all_labels).sum()
        total = len(all_labels)
        
        tp = ((all_preds == 1) & (all_labels == 1)).sum()
        fp = ((all_preds == 1) & (all_labels == 0)).sum()
        fn = ((all_preds == 0) & (all_labels == 1)).sum()
        
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        
        return {
            'loss': total_loss / len(val_loader),
            'accuracy': 100 * correct / total,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'predictions': all_preds,
            'labels': all_labels,
            'probs': all_probs,
            'tp': int(tp), 'fp': int(fp), 'fn': int(fn)
        }
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader):
        """Training loop with anti-overfitting techniques"""
        epochs = self.config['training']['epochs']
        patience = self.config['training']['early_stopping']['patience']

        # Setup scheduler - Cosine Annealing with Warm Restarts
        self.scheduler = CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=10,  # First restart after 10 epochs
            T_mult=2,  # Double the period after each restart
            eta_min=1e-6  # Minimum learning rate
        )

        print("\n" + "="*70)
        print("ANTI-OVERFITTING TRAINING - STAGE 1")
        print("="*70)
        print(f"Epochs: {epochs}")
        print(f"Early stopping patience: {patience}")
        print(f"Detection threshold: 0.4")
        print("\nAnti-Overfitting Techniques:")
        print(f"  • Mixup augmentation (alpha=0.4)")
        print(f"  • Label smoothing (0.1) - SAME loss for train & val")
        print(f"  • Dropout: 0.4 (balanced)")
        print(f"  • Weight decay: 0.01 (moderate)")
        print(f"  • Cosine annealing with warm restarts")
        print(f"  • Moderate class balancing (1.5x)")
        print("\nNote: Train & Val use SAME loss function (Label Smoothing)")
        print("      for fair comparison. Final evaluation uses Focal Loss.")
        print("="*70 + "\n")
        
        for epoch in range(epochs):
            # Train with Mixup
            train_metrics = self.train_epoch(train_loader, use_mixup=True)

            # Validate with threshold=0.4 using SAME loss as training (label smoothing)
            val_metrics = self.validate(val_loader, threshold=0.4, use_focal=False)
            
            # Store
            self.history['train_loss'].append(train_metrics['loss'])
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['train_acc'].append(train_metrics['accuracy'])
            self.history['val_acc'].append(val_metrics['accuracy'])
            self.history['val_precision'].append(val_metrics['precision'])
            self.history['val_recall'].append(val_metrics['recall'])
            self.history['val_f1'].append(val_metrics['f1_score'])
            
            # Print with detailed damaged detection stats
            print(f"\nEpoch {epoch+1}/{epochs}")
            print(f"  Train: Loss={train_metrics['loss']:.4f}, Acc={train_metrics['accuracy']:.2f}%")
            print(f"  Val:   Loss={val_metrics['loss']:.4f}, Acc={val_metrics['accuracy']:.2f}%")
            print(f"         P={val_metrics['precision']:.3f}, R={val_metrics['recall']:.3f}, F1={val_metrics['f1_score']:.3f}")
            print(f"  Damaged: TP={val_metrics['tp']}, FP={val_metrics['fp']}, FN={val_metrics['fn']}")

            # Check best - use BALANCED metric (F1)
            is_best = val_metrics['f1_score'] > self.best_val_f1
            if is_best:
                self.best_val_f1 = val_metrics['f1_score']
                self.patience_counter = 0
                self.save_checkpoint(epoch, 'best_model.pth')
                print(f"  ⭐ Best F1: {self.best_val_f1:.4f}")
            else:
                self.patience_counter += 1

            # Checkpoint every 10 epochs
            if (epoch + 1) % 10 == 0:
                self.save_checkpoint(epoch, f'checkpoint_epoch_{epoch+1}.pth')

            # Early stopping with better criteria
            if self.patience_counter >= patience:
                print(f"\n⚠️  Early stopping at epoch {epoch+1} (no F1 improvement for {patience} epochs)")
                break

            # Warning for overfitting - more realistic threshold
            train_val_gap = train_metrics['accuracy'] - val_metrics['accuracy']
            if train_val_gap > 15:  # 15% gap
                print(f"  ⚠️  WARNING: Possible overfitting (train-val gap: {train_val_gap:.1f}%)")
        
        print("\n" + "="*70)
        print(f"Best Val F1: {self.best_val_f1:.4f}")
        print("="*70)
    
    def save_checkpoint(self, epoch: int, filename: str):
        """Save checkpoint"""
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_f1': self.best_val_f1,
            'history': self.history
        }, self.output_dir / filename)
    
    def plot_results(self):
        """Plot all results with integer epochs on x-axis"""
        epochs = list(range(1, len(self.history['train_loss']) + 1))

        # 1. Loss
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, self.history['train_loss'], 'b-o', label='Train', linewidth=2, markersize=4)
        plt.plot(epochs, self.history['val_loss'], 'r-o', label='Val', linewidth=2, markersize=4)
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Loss', fontsize=12)
        plt.title('Loss Curves (Same Loss Function)', fontsize=14, fontweight='bold')
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        # Force integer x-axis
        plt.xticks(epochs)
        plt.savefig(self.output_dir / 'loss.png', dpi=300, bbox_inches='tight')
        plt.close()

        # 2. Accuracy
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, self.history['train_acc'], 'b-o', label='Train', linewidth=2, markersize=4)
        plt.plot(epochs, self.history['val_acc'], 'r-o', label='Val', linewidth=2, markersize=4)
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Accuracy (%)', fontsize=12)
        plt.title('Accuracy Curves', fontsize=14, fontweight='bold')
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        # Force integer x-axis
        plt.xticks(epochs)
        plt.savefig(self.output_dir / 'accuracy.png', dpi=300, bbox_inches='tight')
        plt.close()

        # 3. Metrics
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, self.history['val_precision'], 'g-o', label='Precision', linewidth=2, markersize=4)
        plt.plot(epochs, self.history['val_recall'], 'b-o', label='Recall', linewidth=2, markersize=4)
        plt.plot(epochs, self.history['val_f1'], 'r-o', label='F1', linewidth=2, markersize=4)
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Score', fontsize=12)
        plt.title('Validation Metrics', fontsize=14, fontweight='bold')
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.ylim([0, 1])
        # Force integer x-axis
        plt.xticks(epochs)
        plt.savefig(self.output_dir / 'metrics.png', dpi=300, bbox_inches='tight')
        plt.close()

        # 4. Confusion Matrix
        print("\nGenerating plots...")
        print("  ✅ loss.png (integer epochs)")
        print("  ✅ accuracy.png (integer epochs)")
        print("  ✅ metrics.png (integer epochs)")
    
    def plot_confusion_matrix(self, metrics: Dict, split: str):
        """Plot confusion matrix"""
        cm = confusion_matrix(metrics['labels'], metrics['predictions'])
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['Intact', 'Damaged'],
                   yticklabels=['Intact', 'Damaged'])
        plt.title(f'Confusion Matrix - {split.upper()}')
        plt.ylabel('True')
        plt.xlabel('Predicted')
        
        # Add percentages
        total = cm.sum()
        for i in range(2):
            for j in range(2):
                pct = cm[i, j] / total * 100
                plt.text(j+0.5, i+0.7, f'({pct:.1f}%)',
                        ha='center', va='center', color='red')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / f'confusion_matrix_{split}.png', dpi=300)
        plt.close()
        print(f"  ✅ confusion_matrix_{split}.png")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/stage1.yaml')
    parser.add_argument('--data_dir', type=str, default='data/processed')
    parser.add_argument('--output_dir', type=str, default='outputs/stage1_anti_overfit')
    parser.add_argument('--epochs', type=int, default=60)
    parser.add_argument('--batch_size', type=int, default=16)

    args = parser.parse_args()

    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)

    # Override with ANTI-OVERFITTING settings
    config['training']['epochs'] = args.epochs
    config['training']['batch_size'] = args.batch_size
    config['training']['learning_rate'] = 0.0001  # Moderate LR (not too low!)
    config['training']['early_stopping']['patience'] = 15  # Reasonable patience

    print("="*70)
    print("ANTI-OVERFITTING TRAINING - STAGE 1")
    print("="*70)
    print("Configuration:")
    print(f"  • Epochs: {args.epochs}")
    print(f"  • Batch size: {args.batch_size}")
    print(f"  • Learning rate: {config['training']['learning_rate']}")
    print(f"  • Early stopping patience: {config['training']['early_stopping']['patience']}")
    print("\nAnti-Overfitting Techniques:")
    print("  • Dropout: 0.4 (balanced, not too high)")
    print("  • Label Smoothing: 0.1")
    print("  • Mixup: alpha=0.4")
    print("  • Weight decay: 0.01 (moderate)")
    print("  • Balanced sampling: 1.5x damaged (moderate)")
    print("  • Detection threshold: 0.4")
    print("  • Gradient clip: 1.0")
    print("  • Cosine Annealing with Warm Restarts")
    print("="*70)
    
    # Create dataloaders
    train_loader, val_loader, test_loader = create_classification_dataloaders(
        intact_dir=str(Path(args.data_dir) / 'intact'),
        damaged_dir=str(Path(args.data_dir) / 'damaged'),
        batch_size=args.batch_size,
        num_workers=4,
        image_size=224,
        train_split=0.7,
        val_split=0.15,
        test_split=0.15
    )
    
    # Create trainer
    output_dir = Path(args.output_dir)
    trainer = OptimizedTrainer(config, output_dir)
    
    # IMPORTANT: Replace train_loader with balanced sampler
    train_dataset = train_loader.dataset
    balanced_sampler = trainer.create_balanced_sampler(train_dataset)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=balanced_sampler,
        num_workers=4,
        pin_memory=True
    )
    
    # Train
    trainer.train(train_loader, val_loader)
    
    # Plot
    trainer.plot_results()
    
    # Evaluate on ALL splits using Focal Loss (better for imbalanced evaluation)
    print("\n" + "="*70)
    print("FINAL EVALUATION ON ALL SPLITS (threshold=0.4, Focal Loss)")
    print("="*70)

    # Train set
    print("\n--- TRAINING SET ---")
    train_metrics = trainer.validate(train_loader, threshold=0.4, use_focal=True)
    print(f"Samples:   {len(train_loader.dataset)}")
    print(f"Accuracy:  {train_metrics['accuracy']:.2f}%")
    print(f"Precision: {train_metrics['precision']:.4f}")
    print(f"Recall:    {train_metrics['recall']:.4f}")
    print(f"F1-Score:  {train_metrics['f1_score']:.4f}")
    print(f"Damaged: TP={train_metrics['tp']}, FP={train_metrics['fp']}, FN={train_metrics['fn']}")
    trainer.plot_confusion_matrix(train_metrics, 'train')

    # Validation set
    print("\n--- VALIDATION SET ---")
    val_metrics = trainer.validate(val_loader, threshold=0.4, use_focal=True)
    print(f"Samples:   {len(val_loader.dataset)}")
    print(f"Accuracy:  {val_metrics['accuracy']:.2f}%")
    print(f"Precision: {val_metrics['precision']:.4f}")
    print(f"Recall:    {val_metrics['recall']:.4f}")
    print(f"F1-Score:  {val_metrics['f1_score']:.4f}")
    print(f"Damaged: TP={val_metrics['tp']}, FP={val_metrics['fp']}, FN={val_metrics['fn']}")
    trainer.plot_confusion_matrix(val_metrics, 'val')

    # Test set
    print("\n--- TEST SET ---")
    test_metrics = trainer.validate(test_loader, threshold=0.4, use_focal=True)
    print(f"Samples:   {len(test_loader.dataset)}")
    print(f"Accuracy:  {test_metrics['accuracy']:.2f}%")
    print(f"Precision: {test_metrics['precision']:.4f}")
    print(f"Recall:    {test_metrics['recall']:.4f}")
    print(f"F1-Score:  {test_metrics['f1_score']:.4f}")
    print(f"Damaged: TP={test_metrics['tp']}, FP={test_metrics['fp']}, FN={test_metrics['fn']}")
    trainer.plot_confusion_matrix(test_metrics, 'test')
    
    print("\n" + "="*70)
    print(f"All outputs saved to: {output_dir}/")
    print("="*70)


if __name__ == '__main__':
    main()
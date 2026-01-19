"""
Comprehensive Data Exploration for Bike Defect Detection Dataset
================================================================
This script explores and visualizes the dataset including:
- Class distribution
- Train/Val/Test splits
- Image statistics (size, channels, aspect ratios)
- Sample images from each class
- Augmentation demonstrations
- Annotation analysis (bounding boxes)
- Color distribution analysis
- Image quality metrics

All outputs are saved to data_exploration/output/
"""

import os
import sys
import json
import random
from pathlib import Path
from collections import Counter, defaultdict
from dotenv import load_dotenv

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec
import seaborn as sns
from PIL import Image, ImageDraw, ImageFont, ImageEnhance, ImageFilter
import cv2

# Load environment variables from .env file
load_dotenv(Path(__file__).parent.parent / ".env")

# Set style for all plots
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# Paths (from .env file)
BASE_DIR = Path(os.getenv("BASE_DIR", Path(__file__).parent.parent))
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
OUTPUT_DIR = BASE_DIR / "data_exploration" / "output"

# Create output directory
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("="*80)
print("BIKE DEFECT DETECTION - COMPREHENSIVE DATA EXPLORATION")
print("="*80)
print(f"\nData Directory: {DATA_DIR}")
print(f"Output Directory: {OUTPUT_DIR}")
print()


def load_annotations(split):
    """Load annotations JSON for a given split"""
    ann_path = RAW_DIR / split / "_annotations.json"
    if ann_path.exists():
        with open(ann_path, 'r') as f:
            return json.load(f)
    return []


def get_image_paths(split):
    """Get all image paths for a split"""
    images_dir = RAW_DIR / split / "images"
    if images_dir.exists():
        return list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png")) + list(images_dir.glob("*.jpeg"))
    return []


def load_split_metadata():
    """Load processed split metadata"""
    meta_path = PROCESSED_DIR / "split_metadata.json"
    if meta_path.exists():
        with open(meta_path, 'r') as f:
            return json.load(f)
    return None


def analyze_annotations(annotations):
    """Analyze annotations to extract statistics"""
    stats = {
        'total_images': len(annotations),
        'label_counts': Counter(),
        'images_per_class': defaultdict(list),
        'bbox_sizes': [],
        'bbox_areas': [],
        'annotations_per_image': []
    }

    for ann in annotations:
        image_name = ann.get('image', '')
        image_annotations = ann.get('annotations', [])
        stats['annotations_per_image'].append(len(image_annotations))

        labels_in_image = set()
        for a in image_annotations:
            label = a.get('label', 'unknown')
            stats['label_counts'][label] += 1
            labels_in_image.add(label)

            coords = a.get('coordinates', {})
            if coords:
                w = coords.get('width', 0)
                h = coords.get('height', 0)
                stats['bbox_sizes'].append((w, h))
                stats['bbox_areas'].append(w * h)

        for label in labels_in_image:
            stats['images_per_class'][label].append(image_name)

    return stats


def plot_class_distribution(save_path):
    """Plot class distribution across all splits"""
    print("[1/12] Analyzing class distribution...")

    metadata = load_split_metadata()

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle('Class Distribution Across Dataset Splits', fontsize=16, fontweight='bold')

    splits = ['train', 'valid', 'test']
    colors = ['#2ecc71', '#e74c3c']  # Green for intact, Red for damaged

    for idx, split in enumerate(splits):
        if metadata and split in metadata.get('splits', {}):
            split_data = metadata['splits'][split]
            classes = ['Intact', 'Damaged']
            counts = [split_data.get('intact', 0), split_data.get('damaged', 0)]

            bars = axes[idx].bar(classes, counts, color=colors, edgecolor='black', linewidth=1.5)
            axes[idx].set_title(f'{split.upper()} Set\n(Total: {sum(counts)} images)', fontsize=12, fontweight='bold')
            axes[idx].set_ylabel('Number of Images')

            # Add value labels on bars
            for bar, count in zip(bars, counts):
                height = bar.get_height()
                axes[idx].annotate(f'{count}\n({count/sum(counts)*100:.1f}%)',
                                  xy=(bar.get_x() + bar.get_width()/2, height),
                                  ha='center', va='bottom', fontsize=11, fontweight='bold')

            axes[idx].set_ylim(0, max(counts) * 1.2)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"   Saved: {save_path.name}")


def plot_overall_distribution(save_path):
    """Plot overall dataset distribution with pie charts"""
    print("[2/12] Creating overall distribution visualization...")

    metadata = load_split_metadata()

    fig = plt.figure(figsize=(16, 6))
    gs = GridSpec(1, 3, figure=fig)

    # Pie chart for class distribution
    ax1 = fig.add_subplot(gs[0, 0])
    if metadata:
        total = metadata.get('total', {})
        sizes = [total.get('intact', 0), total.get('damaged', 0)]
        labels = ['Intact\n(Normal)', 'Damaged\n(Defective)']
        colors = ['#27ae60', '#c0392b']
        explode = (0, 0.05)

        wedges, texts, autotexts = ax1.pie(sizes, explode=explode, labels=labels, colors=colors,
                                           autopct='%1.1f%%', shadow=True, startangle=90,
                                           textprops={'fontsize': 11})
        autotexts[0].set_fontweight('bold')
        autotexts[1].set_fontweight('bold')
        ax1.set_title('Class Distribution\n(All Data)', fontsize=14, fontweight='bold')

    # Pie chart for split distribution
    ax2 = fig.add_subplot(gs[0, 1])
    if metadata:
        splits = metadata.get('splits', {})
        sizes = [splits.get('train', {}).get('total', 0),
                splits.get('valid', {}).get('total', 0),
                splits.get('test', {}).get('total', 0)]
        labels = ['Train', 'Validation', 'Test']
        colors = ['#3498db', '#9b59b6', '#f39c12']

        wedges, texts, autotexts = ax2.pie(sizes, labels=labels, colors=colors,
                                           autopct='%1.1f%%', shadow=True, startangle=90,
                                           textprops={'fontsize': 11})
        for at in autotexts:
            at.set_fontweight('bold')
        ax2.set_title('Split Distribution\n(Train/Val/Test)', fontsize=14, fontweight='bold')

    # Summary statistics table
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.axis('off')

    if metadata:
        total = metadata.get('total', {})
        splits = metadata.get('splits', {})

        table_data = [
            ['Metric', 'Value'],
            ['Total Images', str(total.get('all', 0))],
            ['Intact Images', str(total.get('intact', 0))],
            ['Damaged Images', str(total.get('damaged', 0))],
            ['Class Imbalance Ratio', f"{total.get('intact', 0)/max(total.get('damaged', 1), 1):.2f}:1"],
            ['', ''],
            ['Train Set', str(splits.get('train', {}).get('total', 0))],
            ['Validation Set', str(splits.get('valid', {}).get('total', 0))],
            ['Test Set', str(splits.get('test', {}).get('total', 0))],
            ['', ''],
            ['Train %', f"{splits.get('train', {}).get('total', 0)/total.get('all', 1)*100:.1f}%"],
            ['Val %', f"{splits.get('valid', {}).get('total', 0)/total.get('all', 1)*100:.1f}%"],
            ['Test %', f"{splits.get('test', {}).get('total', 0)/total.get('all', 1)*100:.1f}%"],
        ]

        table = ax3.table(cellText=table_data, loc='center', cellLoc='center',
                         colWidths=[0.5, 0.4])
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.5)

        # Style header row
        for i in range(2):
            table[(0, i)].set_facecolor('#34495e')
            table[(0, i)].set_text_props(color='white', fontweight='bold')

        ax3.set_title('Dataset Summary', fontsize=14, fontweight='bold', pad=20)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"   Saved: {save_path.name}")


def plot_annotation_analysis(save_path):
    """Analyze and visualize annotations (labels, bounding boxes)"""
    print("[3/12] Analyzing annotations...")

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    all_labels = Counter()
    all_bbox_areas = []
    all_bbox_widths = []
    all_bbox_heights = []
    annotations_per_image_all = []

    for split in ['train', 'valid', 'test']:
        annotations = load_annotations(split)
        stats = analyze_annotations(annotations)
        all_labels.update(stats['label_counts'])
        all_bbox_areas.extend(stats['bbox_areas'])
        all_bbox_widths.extend([s[0] for s in stats['bbox_sizes']])
        all_bbox_heights.extend([s[1] for s in stats['bbox_sizes']])
        annotations_per_image_all.extend(stats['annotations_per_image'])

    # Plot 1: Label distribution
    ax1 = axes[0, 0]
    if all_labels:
        labels = list(all_labels.keys())
        counts = list(all_labels.values())
        colors = ['#e74c3c' if 'broken' in l.lower() else '#2ecc71' for l in labels]
        bars = ax1.barh(labels, counts, color=colors, edgecolor='black')
        ax1.set_xlabel('Count')
        ax1.set_title('Annotation Labels Distribution', fontsize=12, fontweight='bold')
        for bar, count in zip(bars, counts):
            ax1.text(bar.get_width() + 5, bar.get_y() + bar.get_height()/2,
                    f'{count}', va='center', fontsize=10)

    # Plot 2: Bounding box area distribution
    ax2 = axes[0, 1]
    if all_bbox_areas:
        ax2.hist(all_bbox_areas, bins=30, color='#3498db', edgecolor='black', alpha=0.7)
        ax2.axvline(np.mean(all_bbox_areas), color='red', linestyle='--',
                   label=f'Mean: {np.mean(all_bbox_areas):.0f}')
        ax2.axvline(np.median(all_bbox_areas), color='green', linestyle='--',
                   label=f'Median: {np.median(all_bbox_areas):.0f}')
        ax2.set_xlabel('Bounding Box Area (pixels)')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Bounding Box Area Distribution', fontsize=12, fontweight='bold')
        ax2.legend()

    # Plot 3: Bounding box dimensions scatter
    ax3 = axes[1, 0]
    if all_bbox_widths and all_bbox_heights:
        ax3.scatter(all_bbox_widths, all_bbox_heights, alpha=0.5, c='#9b59b6', edgecolor='white')
        ax3.set_xlabel('Width (pixels)')
        ax3.set_ylabel('Height (pixels)')
        ax3.set_title('Bounding Box Dimensions', fontsize=12, fontweight='bold')
        # Add diagonal line for square boxes
        max_dim = max(max(all_bbox_widths), max(all_bbox_heights))
        ax3.plot([0, max_dim], [0, max_dim], 'r--', alpha=0.5, label='Square (1:1)')
        ax3.legend()

    # Plot 4: Annotations per image
    ax4 = axes[1, 1]
    if annotations_per_image_all:
        unique, counts = np.unique(annotations_per_image_all, return_counts=True)
        ax4.bar(unique, counts, color='#f39c12', edgecolor='black')
        ax4.set_xlabel('Number of Annotations per Image')
        ax4.set_ylabel('Number of Images')
        ax4.set_title('Annotations per Image Distribution', fontsize=12, fontweight='bold')
        ax4.set_xticks(unique)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"   Saved: {save_path.name}")


def analyze_image_statistics(save_path):
    """Analyze image statistics (size, aspect ratio, etc.)"""
    print("[4/12] Analyzing image statistics...")

    widths = []
    heights = []
    aspect_ratios = []
    file_sizes = []
    channels_list = []

    for split in ['train', 'valid', 'test']:
        image_paths = get_image_paths(split)
        for img_path in image_paths[:100]:  # Sample for speed
            try:
                img = Image.open(img_path)
                w, h = img.size
                widths.append(w)
                heights.append(h)
                aspect_ratios.append(w / h)
                file_sizes.append(img_path.stat().st_size / 1024)  # KB
                channels_list.append(len(img.getbands()))
            except Exception as e:
                continue

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Image dimensions
    ax1 = axes[0, 0]
    ax1.scatter(widths, heights, alpha=0.5, c='#3498db', edgecolor='white', s=50)
    ax1.set_xlabel('Width (pixels)')
    ax1.set_ylabel('Height (pixels)')
    ax1.set_title('Image Dimensions Distribution', fontsize=12, fontweight='bold')
    if widths:
        ax1.axhline(np.mean(heights), color='red', linestyle='--', alpha=0.7,
                   label=f'Mean Height: {np.mean(heights):.0f}')
        ax1.axvline(np.mean(widths), color='green', linestyle='--', alpha=0.7,
                   label=f'Mean Width: {np.mean(widths):.0f}')
        ax1.legend()

    # Plot 2: Aspect ratio distribution
    ax2 = axes[0, 1]
    if aspect_ratios:
        ax2.hist(aspect_ratios, bins=20, color='#9b59b6', edgecolor='black', alpha=0.7)
        ax2.axvline(1.0, color='red', linestyle='--', label='Square (1:1)')
        ax2.axvline(np.mean(aspect_ratios), color='green', linestyle='--',
                   label=f'Mean: {np.mean(aspect_ratios):.2f}')
        ax2.set_xlabel('Aspect Ratio (Width/Height)')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Aspect Ratio Distribution', fontsize=12, fontweight='bold')
        ax2.legend()

    # Plot 3: File size distribution
    ax3 = axes[1, 0]
    if file_sizes:
        ax3.hist(file_sizes, bins=25, color='#e74c3c', edgecolor='black', alpha=0.7)
        ax3.axvline(np.mean(file_sizes), color='blue', linestyle='--',
                   label=f'Mean: {np.mean(file_sizes):.1f} KB')
        ax3.set_xlabel('File Size (KB)')
        ax3.set_ylabel('Frequency')
        ax3.set_title('File Size Distribution', fontsize=12, fontweight='bold')
        ax3.legend()

    # Plot 4: Statistics summary
    ax4 = axes[1, 1]
    ax4.axis('off')

    if widths:
        stats_text = f"""
IMAGE STATISTICS SUMMARY
========================

Dimensions:
  - Width:  Min={min(widths)}, Max={max(widths)}, Mean={np.mean(widths):.0f}
  - Height: Min={min(heights)}, Max={max(heights)}, Mean={np.mean(heights):.0f}

Aspect Ratios:
  - Min: {min(aspect_ratios):.2f}
  - Max: {max(aspect_ratios):.2f}
  - Mean: {np.mean(aspect_ratios):.2f}
  - Std: {np.std(aspect_ratios):.2f}

File Sizes:
  - Min: {min(file_sizes):.1f} KB
  - Max: {max(file_sizes):.1f} KB
  - Mean: {np.mean(file_sizes):.1f} KB
  - Total: {sum(file_sizes)/1024:.1f} MB

Channels:
  - Most common: {max(set(channels_list), key=channels_list.count)} channels (RGB)
  - Sample size: {len(widths)} images
"""
        ax4.text(0.1, 0.5, stats_text, transform=ax4.transAxes, fontsize=11,
                verticalalignment='center', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"   Saved: {save_path.name}")


def plot_sample_images(save_path):
    """Display sample images from each class"""
    print("[5/12] Creating sample images grid...")

    fig, axes = plt.subplots(3, 6, figsize=(18, 10))
    fig.suptitle('Sample Images from Dataset', fontsize=16, fontweight='bold')

    row_labels = ['TRAIN', 'VALID', 'TEST']

    for row_idx, split in enumerate(['train', 'valid', 'test']):
        annotations = load_annotations(split)

        # Separate intact and damaged
        intact_images = []
        damaged_images = []

        for ann in annotations:
            labels = [a['label'] for a in ann.get('annotations', [])]
            if any('broken' in l.lower() for l in labels):
                damaged_images.append(ann['image'])
            else:
                intact_images.append(ann['image'])

        # Sample images
        images_dir = RAW_DIR / split / "images"

        # Get 3 intact and 3 damaged samples
        intact_samples = random.sample(intact_images, min(3, len(intact_images))) if intact_images else []
        damaged_samples = random.sample(damaged_images, min(3, len(damaged_images))) if damaged_images else []

        samples = intact_samples + damaged_samples
        labels = ['Intact']*len(intact_samples) + ['Damaged']*len(damaged_samples)

        for col_idx in range(6):
            ax = axes[row_idx, col_idx]

            if col_idx < len(samples):
                img_path = images_dir / samples[col_idx]
                if img_path.exists():
                    img = Image.open(img_path)
                    ax.imshow(img)
                    label = labels[col_idx]
                    color = 'green' if label == 'Intact' else 'red'
                    ax.set_title(label, fontsize=10, fontweight='bold', color=color)

            ax.axis('off')

            # Add row label on first column
            if col_idx == 0:
                ax.text(-0.3, 0.5, row_labels[row_idx], transform=ax.transAxes,
                       fontsize=12, fontweight='bold', rotation=90, va='center')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"   Saved: {save_path.name}")


def demonstrate_augmentations(save_path):
    """Demonstrate data augmentation techniques"""
    print("[6/12] Demonstrating augmentation techniques...")

    # Find a sample image
    train_images = get_image_paths('train')
    if not train_images:
        print("   No training images found!")
        return

    sample_path = random.choice(train_images)
    original_img = Image.open(sample_path)
    original_np = np.array(original_img)

    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    fig.suptitle('Data Augmentation Techniques Demonstration', fontsize=16, fontweight='bold')

    augmentations = [
        ('Original', original_img),
        ('Horizontal Flip', original_img.transpose(Image.FLIP_LEFT_RIGHT)),
        ('Vertical Flip', original_img.transpose(Image.FLIP_TOP_BOTTOM)),
        ('Rotation (15 deg)', original_img.rotate(15, fillcolor=(128, 128, 128))),
        ('Rotation (-15 deg)', original_img.rotate(-15, fillcolor=(128, 128, 128))),
        ('Brightness +30%', ImageEnhance.Brightness(original_img).enhance(1.3)),
        ('Brightness -30%', ImageEnhance.Brightness(original_img).enhance(0.7)),
        ('Contrast +30%', ImageEnhance.Contrast(original_img).enhance(1.3)),
        ('Contrast -30%', ImageEnhance.Contrast(original_img).enhance(0.7)),
        ('Saturation +30%', ImageEnhance.Color(original_img).enhance(1.3)),
        ('Gaussian Blur', original_img.filter(ImageFilter.GaussianBlur(radius=2))),
        ('Random Crop', original_img.crop((50, 50, original_img.width-50, original_img.height-50)).resize(original_img.size)),
    ]

    for idx, (name, img) in enumerate(augmentations):
        row = idx // 4
        col = idx % 4
        axes[row, col].imshow(img)
        axes[row, col].set_title(name, fontsize=11, fontweight='bold')
        axes[row, col].axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"   Saved: {save_path.name}")


def demonstrate_preprocessing(save_path):
    """Demonstrate preprocessing steps"""
    print("[7/12] Demonstrating preprocessing steps...")

    train_images = get_image_paths('train')
    if not train_images:
        return

    sample_path = random.choice(train_images)
    original_img = Image.open(sample_path)
    original_np = np.array(original_img)

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    fig.suptitle('Preprocessing Pipeline Demonstration', fontsize=16, fontweight='bold')

    # Row 1: Stage 1 (Classification) preprocessing
    axes[0, 0].imshow(original_img)
    axes[0, 0].set_title(f'Original\n{original_img.size}', fontsize=10)
    axes[0, 0].axis('off')

    # Resize to 224x224
    resized_224 = original_img.resize((224, 224))
    axes[0, 1].imshow(resized_224)
    axes[0, 1].set_title('Resized (224x224)\nFor ResNet-50', fontsize=10)
    axes[0, 1].axis('off')

    # Convert to tensor and normalize
    img_np = np.array(resized_224).astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    normalized = (img_np - mean) / std

    # Show normalized (clip for visualization)
    normalized_vis = np.clip((normalized - normalized.min()) / (normalized.max() - normalized.min()), 0, 1)
    axes[0, 2].imshow(normalized_vis)
    axes[0, 2].set_title('Normalized\n(ImageNet stats)', fontsize=10)
    axes[0, 2].axis('off')

    # Show channel means
    axes[0, 3].bar(['R', 'G', 'B'], mean, color=['red', 'green', 'blue'], alpha=0.7)
    axes[0, 3].set_title('ImageNet Mean\nSubtracted', fontsize=10)
    axes[0, 3].set_ylim(0, 0.6)

    # Row 2: Stage 2 (DRAEM) preprocessing
    axes[1, 0].imshow(original_img)
    axes[1, 0].set_title(f'Original\n{original_img.size}', fontsize=10)
    axes[1, 0].axis('off')

    # Resize to 256x256
    resized_256 = original_img.resize((256, 256))
    axes[1, 1].imshow(resized_256)
    axes[1, 1].set_title('Resized (256x256)\nFor DRAEM', fontsize=10)
    axes[1, 1].axis('off')

    # Normalize for DRAEM
    img_np_256 = np.array(resized_256).astype(np.float32) / 255.0
    normalized_256 = (img_np_256 - mean) / std
    normalized_vis_256 = np.clip((normalized_256 - normalized_256.min()) / (normalized_256.max() - normalized_256.min()), 0, 1)
    axes[1, 2].imshow(normalized_vis_256)
    axes[1, 2].set_title('Normalized\n(Same stats)', fontsize=10)
    axes[1, 2].axis('off')

    # Show STD values
    axes[1, 3].bar(['R', 'G', 'B'], std, color=['red', 'green', 'blue'], alpha=0.7)
    axes[1, 3].set_title('ImageNet Std\nDivided', fontsize=10)
    axes[1, 3].set_ylim(0, 0.3)

    # Add row labels
    fig.text(0.02, 0.75, 'Stage 1\n(ResNet-50)', fontsize=12, fontweight='bold',
             rotation=90, va='center', ha='center')
    fig.text(0.02, 0.25, 'Stage 2\n(DRAEM)', fontsize=12, fontweight='bold',
             rotation=90, va='center', ha='center')

    plt.tight_layout(rect=[0.05, 0, 1, 1])
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"   Saved: {save_path.name}")


def analyze_color_distribution(save_path):
    """Analyze color distribution in images"""
    print("[8/12] Analyzing color distributions...")

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Color Distribution Analysis', fontsize=16, fontweight='bold')

    # Collect color statistics
    intact_colors = {'R': [], 'G': [], 'B': []}
    damaged_colors = {'R': [], 'G': [], 'B': []}

    for split in ['train', 'valid']:
        annotations = load_annotations(split)
        images_dir = RAW_DIR / split / "images"

        for ann in annotations[:50]:  # Sample for speed
            img_path = images_dir / ann['image']
            if not img_path.exists():
                continue

            try:
                img = Image.open(img_path)
                img_np = np.array(img)

                if img_np.ndim < 3:
                    continue

                labels = [a['label'] for a in ann.get('annotations', [])]
                is_damaged = any('broken' in l.lower() for l in labels)

                target = damaged_colors if is_damaged else intact_colors
                target['R'].extend(img_np[:, :, 0].flatten()[::100])
                target['G'].extend(img_np[:, :, 1].flatten()[::100])
                target['B'].extend(img_np[:, :, 2].flatten()[::100])
            except:
                continue

    # Plot histograms for intact images
    colors = ['red', 'green', 'blue']
    for idx, (channel, color) in enumerate(zip(['R', 'G', 'B'], colors)):
        if intact_colors[channel]:
            axes[0, idx].hist(intact_colors[channel], bins=50, color=color, alpha=0.7, density=True)
            axes[0, idx].set_title(f'Intact - {channel} Channel', fontsize=11, fontweight='bold')
            axes[0, idx].set_xlabel('Pixel Value')
            axes[0, idx].set_ylabel('Density')

    # Plot histograms for damaged images
    for idx, (channel, color) in enumerate(zip(['R', 'G', 'B'], colors)):
        if damaged_colors[channel]:
            axes[1, idx].hist(damaged_colors[channel], bins=50, color=color, alpha=0.7, density=True)
            axes[1, idx].set_title(f'Damaged - {channel} Channel', fontsize=11, fontweight='bold')
            axes[1, idx].set_xlabel('Pixel Value')
            axes[1, idx].set_ylabel('Density')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"   Saved: {save_path.name}")


def plot_images_with_bboxes(save_path):
    """Plot sample images with bounding box annotations"""
    print("[9/12] Creating bounding box visualizations...")

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    fig.suptitle('Sample Images with Bounding Box Annotations', fontsize=16, fontweight='bold')

    sample_count = 0
    for split in ['train', 'valid', 'test']:
        annotations = load_annotations(split)
        images_dir = RAW_DIR / split / "images"

        # Find images with annotations
        annotated = [ann for ann in annotations if ann.get('annotations')]
        random.shuffle(annotated)

        for ann in annotated[:4]:
            if sample_count >= 8:
                break

            img_path = images_dir / ann['image']
            if not img_path.exists():
                continue

            try:
                img = Image.open(img_path)

                row = sample_count // 4
                col = sample_count % 4
                ax = axes[row, col]

                ax.imshow(img)

                # Draw bounding boxes
                for annotation in ann.get('annotations', []):
                    coords = annotation.get('coordinates', {})
                    label = annotation.get('label', 'unknown')

                    x = coords.get('x', 0) - coords.get('width', 0) / 2
                    y = coords.get('y', 0) - coords.get('height', 0) / 2
                    w = coords.get('width', 0)
                    h = coords.get('height', 0)

                    color = 'red' if 'broken' in label.lower() else 'green'
                    rect = patches.Rectangle((x, y), w, h, linewidth=2,
                                            edgecolor=color, facecolor='none')
                    ax.add_patch(rect)
                    ax.text(x, y - 5, label, fontsize=8, color=color,
                           fontweight='bold', bbox=dict(boxstyle='round',
                           facecolor='white', alpha=0.7))

                ax.set_title(f'{split.upper()}: {ann["image"][:20]}...', fontsize=9)
                ax.axis('off')
                sample_count += 1

            except Exception as e:
                continue

    # Hide unused subplots
    for idx in range(sample_count, 8):
        row = idx // 4
        col = idx % 4
        axes[row, col].axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"   Saved: {save_path.name}")


def create_class_comparison(save_path):
    """Create side-by-side comparison of intact vs damaged"""
    print("[10/12] Creating class comparison visualization...")

    fig, axes = plt.subplots(2, 5, figsize=(18, 8))
    fig.suptitle('Intact vs Damaged: Visual Comparison', fontsize=16, fontweight='bold')

    # Collect samples
    intact_samples = []
    damaged_samples = []

    for split in ['train', 'valid', 'test']:
        annotations = load_annotations(split)
        images_dir = RAW_DIR / split / "images"

        for ann in annotations:
            labels = [a['label'] for a in ann.get('annotations', [])]
            img_path = images_dir / ann['image']

            if img_path.exists():
                if any('broken' in l.lower() for l in labels):
                    damaged_samples.append(img_path)
                else:
                    intact_samples.append(img_path)

    # Sample 5 from each
    random.shuffle(intact_samples)
    random.shuffle(damaged_samples)

    # Row 0: Intact images
    for idx, img_path in enumerate(intact_samples[:5]):
        try:
            img = Image.open(img_path)
            axes[0, idx].imshow(img)
            axes[0, idx].axis('off')
            if idx == 0:
                axes[0, idx].set_ylabel('INTACT', fontsize=14, fontweight='bold')
        except:
            pass

    # Row 1: Damaged images
    for idx, img_path in enumerate(damaged_samples[:5]):
        try:
            img = Image.open(img_path)
            axes[1, idx].imshow(img)
            axes[1, idx].axis('off')
            if idx == 0:
                axes[1, idx].set_ylabel('DAMAGED', fontsize=14, fontweight='bold')
        except:
            pass

    # Add row labels
    fig.text(0.02, 0.75, 'INTACT\n(Normal)', fontsize=14, fontweight='bold',
             color='green', rotation=90, va='center', ha='center')
    fig.text(0.02, 0.25, 'DAMAGED\n(Defective)', fontsize=14, fontweight='bold',
             color='red', rotation=90, va='center', ha='center')

    plt.tight_layout(rect=[0.05, 0, 1, 1])
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"   Saved: {save_path.name}")


def create_summary_dashboard(save_path):
    """Create a comprehensive summary dashboard"""
    print("[11/12] Creating summary dashboard...")

    metadata = load_split_metadata()

    fig = plt.figure(figsize=(20, 14))
    fig.suptitle('BIKE DEFECT DETECTION DATASET - EXPLORATION SUMMARY',
                fontsize=20, fontweight='bold', y=0.98)

    gs = GridSpec(3, 4, figure=fig, hspace=0.3, wspace=0.3)

    # 1. Class Distribution Bar Chart
    ax1 = fig.add_subplot(gs[0, 0:2])
    if metadata:
        splits = ['Train', 'Valid', 'Test']
        intact = [metadata['splits']['train']['intact'],
                 metadata['splits']['valid']['intact'],
                 metadata['splits']['test']['intact']]
        damaged = [metadata['splits']['train']['damaged'],
                  metadata['splits']['valid']['damaged'],
                  metadata['splits']['test']['damaged']]

        x = np.arange(len(splits))
        width = 0.35

        bars1 = ax1.bar(x - width/2, intact, width, label='Intact', color='#27ae60')
        bars2 = ax1.bar(x + width/2, damaged, width, label='Damaged', color='#c0392b')

        ax1.set_ylabel('Number of Images')
        ax1.set_title('Class Distribution by Split', fontsize=12, fontweight='bold')
        ax1.set_xticks(x)
        ax1.set_xticklabels(splits)
        ax1.legend()

        # Add value labels
        for bar in bars1:
            ax1.annotate(f'{int(bar.get_height())}', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                        ha='center', va='bottom', fontsize=9)
        for bar in bars2:
            ax1.annotate(f'{int(bar.get_height())}', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                        ha='center', va='bottom', fontsize=9)

    # 2. Overall Pie Chart
    ax2 = fig.add_subplot(gs[0, 2])
    if metadata:
        sizes = [metadata['total']['intact'], metadata['total']['damaged']]
        colors = ['#27ae60', '#c0392b']
        ax2.pie(sizes, labels=['Intact', 'Damaged'], colors=colors, autopct='%1.1f%%',
               shadow=True, startangle=90)
        ax2.set_title('Overall Class Balance', fontsize=12, fontweight='bold')

    # 3. Split Pie Chart
    ax3 = fig.add_subplot(gs[0, 3])
    if metadata:
        sizes = [metadata['splits']['train']['total'],
                metadata['splits']['valid']['total'],
                metadata['splits']['test']['total']]
        colors = ['#3498db', '#9b59b6', '#f39c12']
        ax3.pie(sizes, labels=['Train', 'Valid', 'Test'], colors=colors, autopct='%1.1f%%',
               shadow=True, startangle=90)
        ax3.set_title('Data Split Ratio', fontsize=12, fontweight='bold')

    # 4. Statistics Table
    ax4 = fig.add_subplot(gs[1, 0:2])
    ax4.axis('off')

    if metadata:
        table_data = [
            ['Metric', 'Train', 'Valid', 'Test', 'Total'],
            ['Intact', str(metadata['splits']['train']['intact']),
             str(metadata['splits']['valid']['intact']),
             str(metadata['splits']['test']['intact']),
             str(metadata['total']['intact'])],
            ['Damaged', str(metadata['splits']['train']['damaged']),
             str(metadata['splits']['valid']['damaged']),
             str(metadata['splits']['test']['damaged']),
             str(metadata['total']['damaged'])],
            ['Total', str(metadata['splits']['train']['total']),
             str(metadata['splits']['valid']['total']),
             str(metadata['splits']['test']['total']),
             str(metadata['total']['all'])],
            ['Damage %',
             f"{metadata['splits']['train']['damaged']/metadata['splits']['train']['total']*100:.1f}%",
             f"{metadata['splits']['valid']['damaged']/metadata['splits']['valid']['total']*100:.1f}%",
             f"{metadata['splits']['test']['damaged']/metadata['splits']['test']['total']*100:.1f}%",
             f"{metadata['total']['damaged']/metadata['total']['all']*100:.1f}%"],
        ]

        table = ax4.table(cellText=table_data, loc='center', cellLoc='center',
                         colWidths=[0.2]*5)
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.8)

        # Style header
        for i in range(5):
            table[(0, i)].set_facecolor('#34495e')
            table[(0, i)].set_text_props(color='white', fontweight='bold')

        ax4.set_title('Dataset Statistics Summary', fontsize=12, fontweight='bold', pad=20)

    # 5. Class Imbalance Bar
    ax5 = fig.add_subplot(gs[1, 2:4])
    if metadata:
        ratio = metadata['total']['intact'] / max(metadata['total']['damaged'], 1)
        ax5.barh(['Imbalance\nRatio'], [ratio], color='#e74c3c', height=0.4)
        ax5.set_xlabel('Intact : Damaged Ratio')
        ax5.set_title(f'Class Imbalance: {ratio:.1f}:1', fontsize=12, fontweight='bold')
        ax5.axvline(1, color='green', linestyle='--', label='Balanced (1:1)')
        ax5.legend()
        ax5.set_xlim(0, ratio * 1.2)

    # 6. Key Findings Text
    ax6 = fig.add_subplot(gs[2, :])
    ax6.axis('off')

    if metadata:
        findings = f"""
KEY FINDINGS & RECOMMENDATIONS
{'='*80}

DATASET OVERVIEW:
  - Total Images: {metadata['total']['all']} images across train/valid/test splits
  - Intact (Normal) Images: {metadata['total']['intact']} ({metadata['total']['intact']/metadata['total']['all']*100:.1f}%)
  - Damaged (Defective) Images: {metadata['total']['damaged']} ({metadata['total']['damaged']/metadata['total']['all']*100:.1f}%)

CLASS IMBALANCE:
  - The dataset is IMBALANCED with {metadata['total']['intact']/metadata['total']['damaged']:.1f}x more intact images than damaged
  - This requires handling via: weighted sampling, class weights in loss function, or data augmentation

SPLIT ANALYSIS:
  - Train: {metadata['splits']['train']['total']} images ({metadata['splits']['train']['total']/metadata['total']['all']*100:.1f}%) - Primary learning set
  - Valid: {metadata['splits']['valid']['total']} images ({metadata['splits']['valid']['total']/metadata['total']['all']*100:.1f}%) - Hyperparameter tuning
  - Test: {metadata['splits']['test']['total']} images ({metadata['splits']['test']['total']/metadata['total']['all']*100:.1f}%) - Final evaluation

RECOMMENDATIONS FOR TRAINING:
  1. Use weighted loss function with weight ~{metadata['total']['intact']/metadata['total']['damaged']:.1f}x for damaged class
  2. Apply data augmentation (flip, rotate, color jitter) to increase damaged samples
  3. Consider oversampling damaged class or undersampling intact class
  4. Monitor precision/recall for damaged class separately (not just accuracy)
  5. Use DRAEM for zero-annotation anomaly detection - eliminates annotation cost!
"""
        ax6.text(0.02, 0.95, findings, transform=ax6.transAxes, fontsize=10,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"   Saved: {save_path.name}")


def generate_text_report(save_path):
    """Generate a comprehensive text report"""
    print("[12/12] Generating text report...")

    metadata = load_split_metadata()

    report = []
    report.append("="*80)
    report.append("BIKE DEFECT DETECTION DATASET - EXPLORATION REPORT")
    report.append("="*80)
    report.append(f"\nGenerated: {Path(__file__).name}")
    report.append(f"Data Directory: {DATA_DIR}")
    report.append(f"Output Directory: {OUTPUT_DIR}")
    report.append("")

    # Dataset Overview
    report.append("-"*80)
    report.append("1. DATASET OVERVIEW")
    report.append("-"*80)

    if metadata:
        report.append(f"\nTotal Images: {metadata['total']['all']}")
        report.append(f"  - Intact (Normal): {metadata['total']['intact']} ({metadata['total']['intact']/metadata['total']['all']*100:.1f}%)")
        report.append(f"  - Damaged (Defective): {metadata['total']['damaged']} ({metadata['total']['damaged']/metadata['total']['all']*100:.1f}%)")
        report.append(f"\nClass Imbalance Ratio: {metadata['total']['intact']/metadata['total']['damaged']:.2f}:1 (Intact:Damaged)")

    # Split Analysis
    report.append("\n" + "-"*80)
    report.append("2. DATA SPLITS")
    report.append("-"*80)

    if metadata:
        for split in ['train', 'valid', 'test']:
            split_data = metadata['splits'][split]
            report.append(f"\n{split.upper()} SET:")
            report.append(f"  - Total: {split_data['total']} images")
            report.append(f"  - Intact: {split_data['intact']} ({split_data['intact']/split_data['total']*100:.1f}%)")
            report.append(f"  - Damaged: {split_data['damaged']} ({split_data['damaged']/split_data['total']*100:.1f}%)")
            report.append(f"  - % of Total Dataset: {split_data['total']/metadata['total']['all']*100:.1f}%")

    # Annotation Analysis
    report.append("\n" + "-"*80)
    report.append("3. ANNOTATION ANALYSIS")
    report.append("-"*80)

    all_labels = Counter()
    all_ann_counts = []

    for split in ['train', 'valid', 'test']:
        annotations = load_annotations(split)
        stats = analyze_annotations(annotations)
        all_labels.update(stats['label_counts'])
        all_ann_counts.extend(stats['annotations_per_image'])

    report.append("\nLabel Distribution:")
    for label, count in all_labels.most_common():
        report.append(f"  - {label}: {count}")

    if all_ann_counts:
        report.append(f"\nAnnotations per Image:")
        report.append(f"  - Min: {min(all_ann_counts)}")
        report.append(f"  - Max: {max(all_ann_counts)}")
        report.append(f"  - Mean: {np.mean(all_ann_counts):.2f}")

    # Preprocessing Info
    report.append("\n" + "-"*80)
    report.append("4. PREPROCESSING PIPELINE")
    report.append("-"*80)

    report.append("""
Stage 1 (ResNet-50 Classification):
  - Resize: 224x224 pixels
  - Normalization: ImageNet statistics
    * Mean: [0.485, 0.456, 0.406]
    * Std: [0.229, 0.224, 0.225]

Stage 2 (DRAEM Localization):
  - Resize: 256x256 pixels
  - Normalization: ImageNet statistics (same as Stage 1)
""")

    # Augmentation Info
    report.append("\n" + "-"*80)
    report.append("5. DATA AUGMENTATION")
    report.append("-"*80)

    report.append("""
Applied Augmentations (Training Only):
  - RandomHorizontalFlip (p=0.5)
  - RandomVerticalFlip (p=0.5)
  - RandomRotation (±15 degrees)
  - ColorJitter:
    * Brightness: ±20%
    * Contrast: ±20%
    * Saturation: ±20%
    * Hue: ±10%
  - RandomResizedCrop (scale=0.8-1.0)

For DRAEM (Synthetic Anomaly Generation):
  - Perlin Noise Masks
  - Texture Anomalies (40%)
  - Color Anomalies (30%)
  - Brightness Anomalies (30%)
""")

    # Recommendations
    report.append("\n" + "-"*80)
    report.append("6. RECOMMENDATIONS")
    report.append("-"*80)

    if metadata:
        imbalance = metadata['total']['intact'] / metadata['total']['damaged']
        report.append(f"""
Based on the exploration, here are recommendations:

1. CLASS IMBALANCE ({imbalance:.1f}:1):
   - Use weighted CrossEntropyLoss with weight ~{imbalance:.1f} for damaged class
   - Apply WeightedRandomSampler during training
   - Augment damaged images more aggressively

2. DATA AUGMENTATION:
   - Essential for increasing effective dataset size
   - Focus on damaged class augmentation
   - Use DRAEM's synthetic anomaly generation for Stage 2

3. EVALUATION METRICS:
   - Don't rely on accuracy alone (biased by majority class)
   - Monitor Precision, Recall, F1 for damaged class
   - Use AUROC for pixel-level localization

4. TRAIN/VAL/TEST SPLIT:
   - Current split: {metadata['splits']['train']['total']/metadata['total']['all']*100:.1f}% / {metadata['splits']['valid']['total']/metadata['total']['all']*100:.1f}% / {metadata['splits']['test']['total']/metadata['total']['all']*100:.1f}%
   - Standard recommendation: 70% / 15% / 15%
   - Consider stratified splitting to maintain class ratios
""")

    # Generated Files
    report.append("\n" + "-"*80)
    report.append("7. GENERATED VISUALIZATION FILES")
    report.append("-"*80)

    report.append("""
All visualizations saved to: data_exploration/output/

  1. class_distribution.png - Class distribution across splits
  2. overall_distribution.png - Pie charts and summary statistics
  3. annotation_analysis.png - Bounding box and label analysis
  4. image_statistics.png - Image dimensions and file sizes
  5. sample_images.png - Sample images from each split
  6. augmentation_demo.png - Data augmentation techniques
  7. preprocessing_demo.png - Preprocessing pipeline steps
  8. color_distribution.png - Color channel analysis
  9. bounding_boxes.png - Images with annotated bounding boxes
  10. class_comparison.png - Side-by-side intact vs damaged
  11. summary_dashboard.png - Comprehensive summary dashboard
  12. exploration_report.txt - This text report
""")

    report.append("\n" + "="*80)
    report.append("END OF REPORT")
    report.append("="*80)

    # Write report
    with open(save_path, 'w') as f:
        f.write('\n'.join(report))

    print(f"   Saved: {save_path.name}")


def main():
    """Main function to run all explorations"""
    print("\nStarting comprehensive data exploration...\n")

    # Generate all visualizations
    plot_class_distribution(OUTPUT_DIR / "01_class_distribution.png")
    plot_overall_distribution(OUTPUT_DIR / "02_overall_distribution.png")
    plot_annotation_analysis(OUTPUT_DIR / "03_annotation_analysis.png")
    analyze_image_statistics(OUTPUT_DIR / "04_image_statistics.png")
    plot_sample_images(OUTPUT_DIR / "05_sample_images.png")
    demonstrate_augmentations(OUTPUT_DIR / "06_augmentation_demo.png")
    demonstrate_preprocessing(OUTPUT_DIR / "07_preprocessing_demo.png")
    analyze_color_distribution(OUTPUT_DIR / "08_color_distribution.png")
    plot_images_with_bboxes(OUTPUT_DIR / "09_bounding_boxes.png")
    create_class_comparison(OUTPUT_DIR / "10_class_comparison.png")
    create_summary_dashboard(OUTPUT_DIR / "11_summary_dashboard.png")
    generate_text_report(OUTPUT_DIR / "12_exploration_report.txt")

    print("\n" + "="*80)
    print("DATA EXPLORATION COMPLETE!")
    print("="*80)
    print(f"\nAll outputs saved to: {OUTPUT_DIR}")
    print(f"\nGenerated Files:")
    for f in sorted(OUTPUT_DIR.glob("*")):
        size = f.stat().st_size / 1024
        print(f"  - {f.name} ({size:.1f} KB)")
    print()


if __name__ == "__main__":
    main()

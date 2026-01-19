"""
Script 2: Split Images into Intact and Damaged
Based on annotations, separate images into two categories
"""

import sys
from pathlib import Path
import json
import shutil
from typing import Dict, List
import argparse
from tqdm import tqdm

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))


def load_annotations(annotation_file: Path) -> List[Dict]:
    """Load COCO format annotations"""
    with open(annotation_file, 'r') as f:
        data = json.load(f)
    return data


def is_damaged(annotations: List[Dict]) -> bool:
    """
    Check if image contains damaged bike
    
    Args:
        annotations: List of annotation dictionaries
        
    Returns:
        True if any annotation has 'broken' or 'damaged' label
    """
    damage_keywords = ['broken', 'damaged', 'defect', 'crack']
    
    for ann in annotations:
        label = ann.get('label', '').lower()
        if any(keyword in label for keyword in damage_keywords):
            return True
    return False


def split_images(
    images_dir: Path,
    annotation_file: Path,
    output_intact: Path,
    output_damaged: Path,
    copy_files: bool = True
):
    """
    Split images based on annotations
    
    Args:
        images_dir: Directory containing images
        annotation_file: Path to annotation JSON file
        output_intact: Output directory for intact images
        output_damaged: Output directory for damaged images
        copy_files: If True, copy files; if False, move files
    """
    # Create output directories
    output_intact.mkdir(parents=True, exist_ok=True)
    output_damaged.mkdir(parents=True, exist_ok=True)
    
    # Load annotations
    print(f"\n📂 Loading annotations from: {annotation_file}")
    annotations = load_annotations(annotation_file)
    
    # Statistics
    intact_count = 0
    damaged_count = 0
    missing_count = 0
    
    # Process each image
    print(f"\n🔍 Processing {len(annotations)} images...")
    
    for item in tqdm(annotations, desc="Splitting images"):
        image_name = item['image']
        image_path = images_dir / image_name
        
        # Check if image exists
        if not image_path.exists():
            print(f"⚠️  Warning: Image not found: {image_name}")
            missing_count += 1
            continue
        
        # Determine if damaged
        item_annotations = item.get('annotations', [])
        damaged = is_damaged(item_annotations)
        
        # Copy or move to appropriate directory
        if damaged:
            dst_path = output_damaged / image_name
            damaged_count += 1
        else:
            dst_path = output_intact / image_name
            intact_count += 1
        
        # Copy or move file
        if copy_files:
            shutil.copy2(image_path, dst_path)
        else:
            shutil.move(str(image_path), str(dst_path))
    
    # Print summary
    print(f"\n{'='*60}")
    print("SPLIT SUMMARY")
    print(f"{'='*60}")
    print(f"✅ Intact images:   {intact_count:4d}")
    print(f"❌ Damaged images:  {damaged_count:4d}")
    print(f"⚠️  Missing images:  {missing_count:4d}")
    print(f"📊 Total processed: {intact_count + damaged_count:4d}")
    print(f"\nDamage ratio: {100 * damaged_count / (intact_count + damaged_count):.1f}%")
    
    return {
        'intact': intact_count,
        'damaged': damaged_count,
        'missing': missing_count,
        'total': intact_count + damaged_count
    }


def process_all_splits(data_root: Path, output_root: Path, copy_files: bool = True):
    """
    Process all train/valid/test splits
    
    Args:
        data_root: Root directory with train/valid/test folders
        output_root: Output root directory
        copy_files: If True, copy; if False, move
    """
    splits = ['train', 'valid', 'test']
    all_stats = {}
    
    for split in splits:
        split_dir = data_root / split
        
        # Check if split exists
        if not split_dir.exists():
            print(f"⚠️  Skipping {split} (directory not found)")
            continue
        
        print(f"\n{'='*60}")
        print(f"PROCESSING {split.upper()} SET")
        print(f"{'='*60}")
        
        # Find images directory
        images_dir = split_dir / 'images'
        if not images_dir.exists():
            images_dir = split_dir  # Images might be directly in split dir
        
        # Find annotation file
        annotation_file = None
        for ann_name in ['_annotations.json', '_annotations_createml.json', 'annotations.json']:
            ann_path = split_dir / ann_name
            if ann_path.exists():
                annotation_file = ann_path
                break
        
        if annotation_file is None:
            print(f"❌ No annotation file found in {split_dir}")
            continue
        
        # Output directories for this split
        output_intact = output_root / 'intact' / split
        output_damaged = output_root / 'damaged' / split
        
        # Split images
        stats = split_images(
            images_dir=images_dir,
            annotation_file=annotation_file,
            output_intact=output_intact,
            output_damaged=output_damaged,
            copy_files=copy_files
        )
        
        all_stats[split] = stats
    
    # Print overall summary
    print(f"\n{'='*60}")
    print("OVERALL SUMMARY")
    print(f"{'='*60}")
    
    total_intact = sum(s['intact'] for s in all_stats.values())
    total_damaged = sum(s['damaged'] for s in all_stats.values())
    total_all = total_intact + total_damaged
    
    for split, stats in all_stats.items():
        print(f"\n{split.upper()}:")
        print(f"  Intact:  {stats['intact']:4d}")
        print(f"  Damaged: {stats['damaged']:4d}")
        print(f"  Total:   {stats['total']:4d}")
    
    print(f"\nGRAND TOTAL:")
    print(f"  Intact:  {total_intact:4d} ({100*total_intact/total_all:.1f}%)")
    print(f"  Damaged: {total_damaged:4d} ({100*total_damaged/total_all:.1f}%)")
    print(f"  Total:   {total_all:4d}")
    
    # Save metadata
    metadata = {
        'splits': all_stats,
        'total': {
            'intact': total_intact,
            'damaged': total_damaged,
            'all': total_all
        }
    }
    
    metadata_file = output_root / 'split_metadata.json'
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n📄 Metadata saved to: {metadata_file}")
    
    return metadata


def main():
    parser = argparse.ArgumentParser(
        description='Split images into intact and damaged categories'
    )
    parser.add_argument(
        '--data_root',
        type=str,
        default='data/raw',
        help='Root directory with train/valid/test folders'
    )
    parser.add_argument(
        '--output_root',
        type=str,
        default='data/processed',
        help='Output root directory'
    )
    parser.add_argument(
        '--copy',
        action='store_true',
        help='Copy files instead of moving them'
    )
    parser.add_argument(
        '--single-split',
        type=str,
        choices=['train', 'valid', 'test'],
        help='Process only one split instead of all'
    )
    
    args = parser.parse_args()
    
    data_root = Path(args.data_root)
    output_root = Path(args.output_root)
    
    print("="*60)
    print("IMAGE SPLITTER - INTACT vs DAMAGED")
    print("="*60)
    print(f"Input:  {data_root}")
    print(f"Output: {output_root}")
    print(f"Mode:   {'COPY' if args.copy else 'MOVE'}")
    
    if args.single_split:
        # Process single split
        split = args.single_split
        split_dir = data_root / split
        images_dir = split_dir / 'images'
        
        # Find annotation file
        annotation_file = None
        for ann_name in ['_annotations.json', '_annotations_createml.json']:
            ann_path = split_dir / ann_name
            if ann_path.exists():
                annotation_file = ann_path
                break
        
        if annotation_file is None:
            print(f"❌ No annotation file found in {split_dir}")
            return
        
        output_intact = output_root / 'intact' / split
        output_damaged = output_root / 'damaged' / split
        
        split_images(
            images_dir=images_dir,
            annotation_file=annotation_file,
            output_intact=output_intact,
            output_damaged=output_damaged,
            copy_files=args.copy
        )
    else:
        # Process all splits
        process_all_splits(
            data_root=data_root,
            output_root=output_root,
            copy_files=args.copy
        )
    
    print(f"\n{'='*60}")
    print("SPLIT COMPLETE!")
    print(f"{'='*60}")
    print(f"\n📁 Output structure:")
    print(f"{output_root}/")
    print(f"├── intact/")
    print(f"│   ├── train/")
    print(f"│   ├── valid/")
    print(f"│   └── test/")
    print(f"├── damaged/")
    print(f"│   ├── train/")
    print(f"│   ├── valid/")
    print(f"│   └── test/")
    print(f"└── split_metadata.json")
    
    print(f"\n{'='*60}")
    print("NEXT STEPS:")
    print(f"{'='*60}")
    print("1. Check the output directories")
    print("2. Verify the split is correct")
    print("3. Start training: python train.py --stage 1")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()

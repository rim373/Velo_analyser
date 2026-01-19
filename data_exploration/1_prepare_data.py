"""
Script 1: Prepare Raw Data
Organize raw data from COCO format into processed structure
"""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.data_preprocess.preprocessor import BikeDataPreprocessor
import argparse
import json


def main():
    parser = argparse.ArgumentParser(description='Prepare raw bike defect data')
    parser.add_argument(
        '--data_root',
        type=str,
        default='data/raw',
        help='Root directory containing train/valid/test folders'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='data/processed',
        help='Output directory for processed data'
    )
    parser.add_argument(
        '--visualize',
        action='store_true',
        help='Visualize sample images'
    )
    
    args = parser.parse_args()
    
    print("="*60)
    print("STEP 1: DATA PREPARATION")
    print("="*60)
    
    # Initialize preprocessor
    preprocessor = BikeDataPreprocessor(args.data_root)
    
    # Get and print statistics
    print("\n📊 Dataset Statistics:")
    print("-"*60)
    stats = preprocessor.get_statistics()
    
    for split in ['train', 'valid', 'test']:
        if stats[split]['total'] > 0:
            print(f"\n{split.upper()}:")
            print(f"  ✓ Intact:  {stats[split]['intact']:3d} images")
            print(f"  ✗ Damaged: {stats[split]['damaged']:3d} images")
            print(f"  Total:     {stats[split]['total']:3d} images")
    
    # Calculate totals
    total_intact = sum(s['intact'] for s in stats.values())
    total_damaged = sum(s['damaged'] for s in stats.values())
    total_all = sum(s['total'] for s in stats.values())
    
    print(f"\nTOTAL ACROSS ALL SPLITS:")
    print(f"  ✓ Intact:  {total_intact:3d} images")
    print(f"  ✗ Damaged: {total_damaged:3d} images")
    print(f"  Total:     {total_all:3d} images")
    print(f"  Damage ratio: {100*total_damaged/total_all:.1f}%")
    
    # Create processed dataset
    print("\n" + "="*60)
    print("Creating processed dataset...")
    print("="*60)
    
    metadata = preprocessor.create_processed_dataset(args.output_dir)
    
    print("\n✅ Data preparation complete!")
    print(f"\n📁 Output structure:")
    print(f"  {args.output_dir}/")
    print(f"  ├── intact/     ({metadata['intact_count']} images)")
    print(f"  ├── damaged/    ({metadata['damaged_count']} images)")
    print(f"  └── metadata.json")
    
    # Visualize samples if requested
    if args.visualize:
        print("\n" + "="*60)
        print("Visualizing sample images...")
        print("="*60)
        
        try:
            # Visualize one intact and one damaged image from train set
            preprocessor.visualize_sample('train', index=0)
        except Exception as e:
            print(f"Visualization failed: {e}")
    
    print("\n" + "="*60)
    print("NEXT STEPS:")
    print("="*60)
    print("1. Run: python scripts/2_create_splits.py")
    print("2. Then run training: python train.py --stage 1")
    print("="*60)


if __name__ == '__main__':
    main()

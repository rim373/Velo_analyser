"""
Data Preprocessor
Handles loading and organizing raw data from COCO format annotations
"""

import json
import os
import shutil
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
from PIL import Image
from tqdm import tqdm


class BikeDataPreprocessor:
    """
    Preprocessor for bike defect dataset
    Converts COCO format annotations to organized structure
    """
    
    def __init__(self, data_root: str):
        """
        Args:
            data_root: Root directory containing train/valid/test folders
        """
        self.data_root = Path(data_root)
        self.train_dir = self.data_root / 'train'
        self.valid_dir = self.data_root / 'valid'
        self.test_dir = self.data_root / 'test'
        
        # Labels that indicate damage
        self.damage_labels = ['broken-bike', 'damaged-bike', 'defect']
        self.intact_labels = ['bike']
        
    def load_annotations(self, split: str) -> List[Dict]:
        """
        Load COCO format annotations
        
        Args:
            split: 'train', 'valid', or 'test'
            
        Returns:
            List of annotation dictionaries
        """
        split_dir = getattr(self, f'{split}_dir')
        annotation_file = split_dir / '_annotations.json'
        
        # Try different annotation file names
        if not annotation_file.exists():
            annotation_file = split_dir / '_annotations_createml.json'
        
        if not annotation_file.exists():
            raise FileNotFoundError(f"Annotation file not found in {split_dir}")
        
        with open(annotation_file, 'r') as f:
            annotations = json.load(f)
        
        return annotations
    
    def is_damaged(self, annotations: List[Dict]) -> bool:
        """
        Determine if image contains damage based on annotations
        
        Args:
            annotations: List of annotation dicts for an image
            
        Returns:
            True if image contains damage
        """
        for ann in annotations:
            label = ann.get('label', '').lower()
            if any(damage_label in label for damage_label in self.damage_labels):
                return True
        return False
    
    def get_image_info(self, split: str) -> List[Dict]:
        """
        Get organized image information with damage labels
        
        Args:
            split: 'train', 'valid', or 'test'
            
        Returns:
            List of dicts with image_path, is_damaged, annotations
        """
        annotations = self.load_annotations(split)
        split_dir = getattr(self, f'{split}_dir')
        images_dir = split_dir / 'images'
        
        image_info = []
        
        for item in tqdm(annotations, desc=f'Processing {split}'):
            image_name = item['image']
            image_path = images_dir / image_name
            
            if not image_path.exists():
                print(f"Warning: Image not found: {image_path}")
                continue
            
            item_annotations = item.get('annotations', [])
            is_damaged = self.is_damaged(item_annotations)
            
            image_info.append({
                'image_path': str(image_path),
                'image_name': image_name,
                'is_damaged': is_damaged,
                'annotations': item_annotations,
                'split': split
            })
        
        return image_info
    
    def create_processed_dataset(self, output_dir: str):
        """
        Create processed dataset organized by class
        
        Args:
            output_dir: Output directory for processed data
        """
        output_path = Path(output_dir)
        intact_dir = output_path / 'intact'
        damaged_dir = output_path / 'damaged'
        
        # Create directories
        intact_dir.mkdir(parents=True, exist_ok=True)
        damaged_dir.mkdir(parents=True, exist_ok=True)
        
        # Process all splits
        all_data = []
        for split in ['train', 'valid', 'test']:
            if not getattr(self, f'{split}_dir').exists():
                print(f"Warning: {split} directory not found, skipping...")
                continue
            
            data = self.get_image_info(split)
            all_data.extend(data)
        
        # Copy images to appropriate folders
        intact_count = 0
        damaged_count = 0
        
        for item in tqdm(all_data, desc='Copying images'):
            src_path = item['image_path']
            image_name = item['image_name']
            
            if item['is_damaged']:
                dst_path = damaged_dir / f"{item['split']}_{image_name}"
                shutil.copy2(src_path, dst_path)
                damaged_count += 1
            else:
                dst_path = intact_dir / f"{item['split']}_{image_name}"
                shutil.copy2(src_path, dst_path)
                intact_count += 1
        
        print(f"\nDataset processing complete!")
        print(f"Intact images: {intact_count}")
        print(f"Damaged images: {damaged_count}")
        print(f"Total images: {intact_count + damaged_count}")
        print(f"\nOutput directory: {output_path}")
        
        # Save metadata
        metadata = {
            'intact_count': intact_count,
            'damaged_count': damaged_count,
            'total_count': intact_count + damaged_count,
            'intact_dir': str(intact_dir),
            'damaged_dir': str(damaged_dir)
        }
        
        with open(output_path / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        return metadata
    
    def get_statistics(self) -> Dict:
        """
        Get dataset statistics
        
        Returns:
            Dictionary with dataset statistics
        """
        stats = {
            'train': {'intact': 0, 'damaged': 0, 'total': 0},
            'valid': {'intact': 0, 'damaged': 0, 'total': 0},
            'test': {'intact': 0, 'damaged': 0, 'total': 0}
        }
        
        for split in ['train', 'valid', 'test']:
            if not getattr(self, f'{split}_dir').exists():
                continue
            
            data = self.get_image_info(split)
            
            for item in data:
                if item['is_damaged']:
                    stats[split]['damaged'] += 1
                else:
                    stats[split]['intact'] += 1
                stats[split]['total'] += 1
        
        return stats
    
    def visualize_sample(self, split: str, index: int = 0):
        """
        Visualize a sample image with annotations
        
        Args:
            split: 'train', 'valid', or 'test'
            index: Index of image to visualize
        """
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches
        
        data = self.get_image_info(split)
        
        if index >= len(data):
            print(f"Index {index} out of range. Max index: {len(data)-1}")
            return
        
        item = data[index]
        
        # Load image
        img = Image.open(item['image_path'])
        
        # Create figure
        fig, ax = plt.subplots(1, figsize=(12, 8))
        ax.imshow(img)
        
        # Draw bounding boxes
        for ann in item['annotations']:
            coords = ann['coordinates']
            x, y = coords['x'], coords['y']
            w, h = coords['width'], coords['height']
            
            # Convert center coordinates to top-left if needed
            x_min = x - w/2
            y_min = y - h/2
            
            color = 'red' if 'broken' in ann['label'].lower() else 'green'
            rect = patches.Rectangle((x_min, y_min), w, h, 
                                     linewidth=2, edgecolor=color, 
                                     facecolor='none')
            ax.add_patch(rect)
            
            # Add label
            ax.text(x_min, y_min-5, ann['label'], 
                   bbox=dict(boxstyle='round', facecolor=color, alpha=0.5),
                   fontsize=10, color='white')
        
        title = f"{'DAMAGED' if item['is_damaged'] else 'INTACT'} - {item['image_name']}"
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.axis('off')
        
        plt.tight_layout()
        plt.show()


if __name__ == '__main__':
    # Example usage
    preprocessor = BikeDataPreprocessor('data/raw')
    
    # Get statistics
    stats = preprocessor.get_statistics()
    print("\nDataset Statistics:")
    print(json.dumps(stats, indent=2))
    
    # Create processed dataset
    metadata = preprocessor.create_processed_dataset('data/processed')

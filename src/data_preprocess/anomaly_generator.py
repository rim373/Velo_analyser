"""
Synthetic Anomaly Generation for DRAEM Training
Creates realistic defects by combining normal images with texture patches and Perlin noise
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import numpy as np
import torch
import cv2
from PIL import Image
import random

from src.utils.perlin import generate_perlin_noise_mask, generate_smooth_anomaly


class AnomalyGenerator:
    """
    Generate synthetic anomalies for training DRAEM
    """
    def __init__(self, anomaly_source_path=None, resize_shape=(256, 256)):
        """
        Args:
            anomaly_source_path: Path to texture dataset (DTD or similar)
            resize_shape: Target image size
        """
        self.resize_shape = resize_shape
        self.anomaly_source_path = anomaly_source_path

        # Load anomaly source images if provided
        if anomaly_source_path and Path(anomaly_source_path).exists():
            self.anomaly_source_images = self._load_anomaly_sources()
        else:
            self.anomaly_source_images = []
            print("[INFO] No anomaly source images provided, will use only Perlin noise")

    def _load_anomaly_sources(self):
        """Load texture images from anomaly source directory"""
        source_path = Path(self.anomaly_source_path)
        image_files = []

        for ext in ['*.jpg', '*.png', '*.jpeg']:
            image_files.extend(list(source_path.glob(f'**/{ext}')))

        print(f"[OK] Loaded {len(image_files)} anomaly source images")
        return image_files

    def generate_anomaly(self, img):
        """
        Generate synthetic anomaly on normal image

        Args:
            img: PIL Image or numpy array (H, W, 3) in range [0, 255]

        Returns:
            augmented_img: Image with synthetic anomaly
            anomaly_mask: Binary mask of anomaly region (H, W)
        """
        # Convert to numpy if PIL Image
        if isinstance(img, Image.Image):
            img = np.array(img)

        h, w = img.shape[:2]

        # Generate Perlin noise mask for anomaly region
        perlin_mask = generate_perlin_noise_mask(
            (h, w),
            min_perlin_scale=0,
            max_perlin_scale=6
        )

        # Random erosion/dilation to vary mask shape
        perlin_mask = self._augment_mask(perlin_mask)

        # Choose anomaly type
        anomaly_type = random.choice(['texture', 'noise', 'brightness'])

        if anomaly_type == 'texture' and len(self.anomaly_source_images) > 0:
            augmented_img = self._texture_anomaly(img, perlin_mask)
        elif anomaly_type == 'noise':
            augmented_img = self._noise_anomaly(img, perlin_mask)
        else:  # brightness
            augmented_img = self._brightness_anomaly(img, perlin_mask)

        # Ensure output is in correct range
        augmented_img = np.clip(augmented_img, 0, 255).astype(np.uint8)
        anomaly_mask = (perlin_mask > 0).astype(np.float32)

        return augmented_img, anomaly_mask

    def _texture_anomaly(self, img, mask):
        """Apply texture from anomaly source"""
        # Select random texture
        texture_path = random.choice(self.anomaly_source_images)
        texture = Image.open(texture_path).convert('RGB')
        texture = texture.resize((img.shape[1], img.shape[0]))
        texture = np.array(texture)

        # Generate smooth blending weight
        smooth_weight = generate_smooth_anomaly(
            (img.shape[0], img.shape[1]),
            min_perlin_scale=0,
            max_perlin_scale=4
        )

        # Blend texture with original image using mask
        mask_3ch = np.stack([mask] * 3, axis=2)
        weight_3ch = np.stack([smooth_weight] * 3, axis=2)

        augmented = img * (1 - mask_3ch) + texture * mask_3ch * weight_3ch + img * mask_3ch * (1 - weight_3ch)

        return augmented

    def _noise_anomaly(self, img, mask):
        """Apply random noise to masked region"""
        # Generate random noise
        noise = np.random.randint(0, 255, img.shape, dtype=np.uint8)

        # Smooth noise
        noise = cv2.GaussianBlur(noise, (5, 5), 0)

        # Generate blending weight
        smooth_weight = generate_smooth_anomaly(
            (img.shape[0], img.shape[1]),
            min_perlin_scale=1,
            max_perlin_scale=5
        )

        # Blend
        mask_3ch = np.stack([mask] * 3, axis=2)
        weight_3ch = np.stack([smooth_weight] * 3, axis=2)

        augmented = img * (1 - mask_3ch * weight_3ch) + noise * mask_3ch * weight_3ch

        return augmented

    def _brightness_anomaly(self, img, mask):
        """Apply brightness change to masked region"""
        # Random brightness factor
        brightness_factor = random.uniform(0.3, 2.0)

        # Apply to masked region
        mask_3ch = np.stack([mask] * 3, axis=2)

        augmented = img.copy().astype(np.float32)
        augmented = augmented * (1 - mask_3ch) + augmented * brightness_factor * mask_3ch

        return augmented

    def _augment_mask(self, mask):
        """Apply morphological operations to mask"""
        # Random kernel size
        kernel_size = random.choice([3, 5, 7])
        kernel = np.ones((kernel_size, kernel_size), np.uint8)

        # Random operation
        operation = random.choice(['erode', 'dilate', 'open', 'close', 'none'])

        mask_uint8 = (mask * 255).astype(np.uint8)

        if operation == 'erode':
            mask = cv2.erode(mask_uint8, kernel, iterations=1) / 255.0
        elif operation == 'dilate':
            mask = cv2.dilate(mask_uint8, kernel, iterations=1) / 255.0
        elif operation == 'open':
            mask = cv2.morphologyEx(mask_uint8, cv2.MORPH_OPEN, kernel) / 255.0
        elif operation == 'close':
            mask = cv2.morphologyEx(mask_uint8, cv2.MORPH_CLOSE, kernel) / 255.0
        else:
            mask = mask_uint8 / 255.0

        return mask


if __name__ == '__main__':
    # Test anomaly generation
    import matplotlib.pyplot as plt

    # Create dummy image
    img = np.random.randint(50, 200, (256, 256, 3), dtype=np.uint8)

    # Generate anomaly
    generator = AnomalyGenerator()

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    for i in range(3):
        aug_img, mask = generator.generate_anomaly(img)

        axes[0, i].imshow(aug_img)
        axes[0, i].set_title(f'Augmented Image {i+1}')
        axes[0, i].axis('off')

        axes[1, i].imshow(mask, cmap='gray')
        axes[1, i].set_title(f'Anomaly Mask {i+1}')
        axes[1, i].axis('off')

    plt.tight_layout()
    plt.savefig('anomaly_generation_examples.png', dpi=150)
    print("Saved anomaly_generation_examples.png")

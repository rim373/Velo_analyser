"""
🔥 DRAEM INFERENCE SCRIPT - WITH POST-PROCESSING
================================================
Bike defect detection with heatmap refinement and background masking
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from pathlib import Path
import matplotlib.pyplot as plt
from PIL import Image


# ============================================================================
# MODEL ARCHITECTURE (Same as training)
# ============================================================================

class ReconstructiveSubNetwork(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, 4, 2, 1), nn.BatchNorm2d(32), nn.ReLU(True),
            nn.Conv2d(32, 64, 4, 2, 1), nn.BatchNorm2d(64), nn.ReLU(True),
            nn.Conv2d(64, 128, 4, 2, 1), nn.BatchNorm2d(128), nn.ReLU(True),
            nn.Conv2d(128, 256, 4, 2, 1), nn.BatchNorm2d(256), nn.ReLU(True),
        )
        self.bottleneck = nn.Sequential(
            nn.Conv2d(256, 256, 3, 1, 1), nn.BatchNorm2d(256), nn.ReLU(True),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1), nn.BatchNorm2d(128), nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.BatchNorm2d(64), nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.BatchNorm2d(32), nn.ReLU(True),
            nn.ConvTranspose2d(32, out_channels, 4, 2, 1), nn.Tanh()
        )
    
    def forward(self, x):
        return self.decoder(self.bottleneck(self.encoder(x)))


class DiscriminativeSubNetwork(nn.Module):
    def __init__(self, in_channels=6, out_channels=2):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 64, 4, 2, 1), nn.BatchNorm2d(64), nn.LeakyReLU(0.2, True),
            nn.Conv2d(64, 128, 4, 2, 1), nn.BatchNorm2d(128), nn.LeakyReLU(0.2, True),
            nn.Conv2d(128, 256, 4, 2, 1), nn.BatchNorm2d(256), nn.LeakyReLU(0.2, True),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1), nn.BatchNorm2d(128), nn.LeakyReLU(0.2, True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.BatchNorm2d(64), nn.LeakyReLU(0.2, True),
            nn.ConvTranspose2d(64, out_channels, 4, 2, 1),
        )
    
    def forward(self, x):
        return self.decoder(self.encoder(x))


class DRAEM(nn.Module):
    def __init__(self):
        super().__init__()
        self.reconstructive = ReconstructiveSubNetwork()
        self.discriminative = DiscriminativeSubNetwork()
    
    def forward(self, x):
        reconstruction = self.reconstructive(x)
        segmentation = self.discriminative(torch.cat([x, reconstruction], dim=1))
        return reconstruction, segmentation


# ============================================================================
# POST-PROCESSING FUNCTIONS
# ============================================================================

def refine_heatmap(heatmap, threshold=0.6, kernel_size=5):
    """
    🔥 Refine heatmap to reduce noise and false positives
    
    Args:
        heatmap: numpy array [H, W], values in [0, 1]
        threshold: Remove values below this (default 0.6)
        kernel_size: Morphological operations kernel size
    
    Returns:
        Refined heatmap [H, W]
    """
    # 1. Aggressive thresholding
    heatmap = np.where(heatmap > threshold, heatmap, 0)
    
    # 2. Convert to uint8 for morphology
    heatmap_uint8 = (heatmap * 255).astype(np.uint8)
    
    # 3. Morphological operations to remove noise
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    heatmap_uint8 = cv2.morphologyEx(heatmap_uint8, cv2.MORPH_OPEN, kernel)
    heatmap_uint8 = cv2.morphologyEx(heatmap_uint8, cv2.MORPH_CLOSE, kernel)
    
    # 4. Gaussian smoothing
    heatmap_uint8 = cv2.GaussianBlur(heatmap_uint8, (7, 7), 0)
    
    # Convert back to [0, 1]
    return heatmap_uint8.astype(np.float32) / 255.0


def segment_bike_mask(image):
    """
    🔥 Segment bike from background to reduce false positives
    
    Args:
        image: RGB image [H, W, 3], uint8
    
    Returns:
        mask: Binary mask [H, W], 1=bike, 0=background
    """
    # Convert to HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    
    # Create mask for background (gray/brown ground)
    # Adjust these values based on your background
    lower_bg = np.array([0, 0, 30])   # Dark background
    upper_bg = np.array([180, 80, 180])  # Light background
    
    bg_mask = cv2.inRange(hsv, lower_bg, upper_bg)
    
    # Bike mask = NOT background
    bike_mask = 255 - bg_mask
    
    # Clean up with morphology
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    bike_mask = cv2.morphologyEx(bike_mask, cv2.MORPH_CLOSE, kernel)
    bike_mask = cv2.morphologyEx(bike_mask, cv2.MORPH_OPEN, kernel)
    
    # Convert to float [0, 1]
    return bike_mask.astype(np.float32) / 255.0


def apply_bike_mask(heatmap, bike_mask):
    """
    Apply bike mask to heatmap to ignore background
    
    Args:
        heatmap: Anomaly heatmap [H, W]
        bike_mask: Bike segmentation mask [H, W]
    
    Returns:
        Masked heatmap [H, W]
    """
    return heatmap * bike_mask


# ============================================================================
# INFERENCE CLASS
# ============================================================================

class DRAEMInference:
    """
    🔥 DRAEM Inference with post-processing
    """
    
    def __init__(self, model_path, device='cuda', image_size=256):
        """
        Args:
            model_path: Path to draem_best.pth
            device: 'cuda' or 'cpu'
            image_size: Input image size (default 256)
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.image_size = image_size
        
        # Normalization (ImageNet)
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(self.device)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(self.device)
        
        # Load model
        self.model = DRAEM().to(self.device)
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        print(f"✅ Model loaded from {model_path}")
        print(f"   Device: {self.device}")
    
    def preprocess(self, image):
        """
        Preprocess image for model input
        
        Args:
            image: RGB image [H, W, 3], uint8
        
        Returns:
            tensor: [1, 3, 256, 256], normalized
        """
        # Resize
        image = cv2.resize(image, (self.image_size, self.image_size))
        
        # To tensor [1, 3, H, W]
        tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        tensor = tensor.to(self.device)
        
        # Normalize
        tensor = (tensor - self.mean) / self.std
        
        return tensor
    
    def denormalize(self, tensor):
        """
        Denormalize tensor back to [0, 1]
        
        Args:
            tensor: [1, 3, H, W], normalized
        
        Returns:
            numpy array [H, W, 3], uint8
        """
        tensor = tensor * self.std + self.mean
        tensor = torch.clamp(tensor, 0, 1)
        image = tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
        return (image * 255).astype(np.uint8)
    
    def predict(self, image, use_bike_mask=True, refine_heatmap_flag=True, 
                threshold=0.6, kernel_size=5):
        """
        🔥 MAIN INFERENCE FUNCTION
        
        Args:
            image: RGB image [H, W, 3], uint8
            use_bike_mask: Apply bike segmentation mask
            refine_heatmap_flag: Apply post-processing to heatmap
            threshold: Heatmap threshold (default 0.6)
            kernel_size: Morphology kernel size
        
        Returns:
            dict with:
                - reconstruction: RGB image [H, W, 3]
                - heatmap: Anomaly heatmap [H, W]
                - heatmap_overlay: Heatmap overlaid on original
                - anomaly_score: Float in [0, 1]
                - is_damaged: Boolean
        """
        original_h, original_w = image.shape[:2]
        
        # Preprocess
        input_tensor = self.preprocess(image)
        
        # Inference
        with torch.no_grad():
            reconstruction, segmentation = self.model(input_tensor)
        
        # Get anomaly heatmap
        segmentation_probs = torch.softmax(segmentation, dim=1)
        heatmap = segmentation_probs[0, 1].cpu().numpy()  # [H, W], anomaly channel
        
        # Reconstruction
        reconstruction_img = self.denormalize(reconstruction)
        
        # Resize to original
        heatmap = cv2.resize(heatmap, (original_w, original_h))
        reconstruction_img = cv2.resize(reconstruction_img, (original_w, original_h))
        
        # 🔥 POST-PROCESSING
        
        # 1. Bike mask (optional)
        if use_bike_mask:
            bike_mask = segment_bike_mask(image)
            heatmap = apply_bike_mask(heatmap, bike_mask)
        
        # 2. Refine heatmap (optional)
        if refine_heatmap_flag:
            heatmap = refine_heatmap(heatmap, threshold=threshold, kernel_size=kernel_size)
        
        # Anomaly score
        anomaly_score = float(heatmap.mean())
        
        # Decision threshold
        is_damaged = anomaly_score > 0.3  # Adjust based on your data
        
        # Create overlay
        heatmap_colored = cv2.applyColorMap((heatmap * 255).astype(np.uint8), cv2.COLORMAP_JET)
        heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
        heatmap_overlay = cv2.addWeighted(image, 0.6, heatmap_colored, 0.4, 0)
        
        return {
            'reconstruction': reconstruction_img,
            'heatmap': heatmap,
            'heatmap_overlay': heatmap_overlay,
            'anomaly_score': anomaly_score,
            'is_damaged': is_damaged,
            'confidence': anomaly_score if is_damaged else (1 - anomaly_score)
        }
    
    def visualize(self, image, results, save_path=None):
        """
        Visualize results
        
        Args:
            image: Original RGB image
            results: Output from predict()
            save_path: Optional path to save figure
        """
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Original
        axes[0, 0].imshow(image)
        axes[0, 0].set_title(f"Original Image\\nStatus: {'DAMAGED' if results['is_damaged'] else 'INTACT'}", 
                            fontsize=12, fontweight='bold')
        axes[0, 0].axis('off')
        
        # Reconstruction
        axes[0, 1].imshow(results['reconstruction'])
        axes[0, 1].set_title('Reconstruction\\n(Anomaly Removed)', fontsize=12, fontweight='bold')
        axes[0, 1].axis('off')
        
        # Heatmap
        im = axes[1, 0].imshow(results['heatmap'], cmap='hot', vmin=0, vmax=1)
        axes[1, 0].set_title(f"Anomaly Heatmap\\nScore: {results['anomaly_score']:.2%}", 
                            fontsize=12, fontweight='bold')
        axes[1, 0].axis('off')
        plt.colorbar(im, ax=axes[1, 0], fraction=0.046, pad=0.04)
        
        # Overlay
        axes[1, 1].imshow(results['heatmap_overlay'])
        axes[1, 1].set_title(f"Defect Localization\\nConfidence: {results['confidence']:.2%}", 
                            fontsize=12, fontweight='bold')
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✅ Saved to {save_path}")
        
        plt.show()


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    import argparse

    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='DRAEM Inference for Bike Defect Detection')
    parser.add_argument('--model', type=str, default='checkpoints/draem/draem_best_v0.pth',
                        help='Path to DRAEM model checkpoint')
    parser.add_argument('--image', type=str, required=True,
                        help='Path to input image')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda or cpu)')
    parser.add_argument('--threshold', type=float, default=0.6,
                        help='Anomaly threshold')
    parser.add_argument('--output', type=str, default='draem_result.png',
                        help='Output path for visualization')
    parser.add_argument('--no-mask', action='store_true',
                        help='Disable bike masking')
    parser.add_argument('--no-refine', action='store_true',
                        help='Disable heatmap refinement')

    args = parser.parse_args()

    # Initialize inference
    inferencer = DRAEMInference(
        model_path=args.model,
        device=args.device,
        image_size=256
    )

    # Load test image
    image = cv2.imread(args.image)
    if image is None:
        print(f"Error: Could not load image from {args.image}")
        exit(1)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Predict with post-processing
    results = inferencer.predict(
        image,
        use_bike_mask=not args.no_mask,
        refine_heatmap_flag=not args.no_refine,
        threshold=args.threshold,
        kernel_size=5
    )

    # Print results
    print(f"\n{'='*50}")
    print(f"DRAEM INFERENCE RESULTS")
    print(f"{'='*50}")
    print(f"Anomaly Score: {results['anomaly_score']:.2%}")
    print(f"Status: {'DAMAGED' if results['is_damaged'] else 'INTACT'}")
    print(f"Confidence: {results['confidence']:.2%}")
    print(f"{'='*50}")

    # Visualize
    inferencer.visualize(image, results, save_path=args.output)
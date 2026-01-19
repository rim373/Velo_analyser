"""
PatchCore Implementation
State-of-the-art anomaly detection and localization
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from typing import List, Tuple, Dict, Optional
import numpy as np
from sklearn.random_projection import SparseRandomProjection
from scipy.ndimage import gaussian_filter
import faiss


class FeatureExtractor(nn.Module):
    """
    Extract multi-scale features from pretrained backbone
    """
    
    def __init__(
        self,
        backbone_name: str = 'wide_resnet50_2',
        layers: List[str] = ['layer2', 'layer3'],
        pretrained: bool = True
    ):
        """
        Args:
            backbone_name: Name of backbone network
            layers: List of layer names to extract features from
            pretrained: Use pretrained weights
        """
        super().__init__()
        
        self.layers = layers
        self.features = {}
        
        # Load backbone
        if backbone_name == 'resnet18':
            self.backbone = models.resnet18(pretrained=pretrained)
        elif backbone_name == 'resnet50':
            self.backbone = models.resnet50(pretrained=pretrained)
        elif backbone_name == 'wide_resnet50_2':
            self.backbone = models.wide_resnet50_2(pretrained=pretrained)
        else:
            raise ValueError(f"Unknown backbone: {backbone_name}")
        
        # Register hooks to extract intermediate features
        self._register_hooks()
        
        # Set to evaluation mode
        self.backbone.eval()
        
        # Freeze all parameters
        for param in self.backbone.parameters():
            param.requires_grad = False
    
    def _register_hooks(self):
        """Register forward hooks to extract features"""
        def get_activation(name):
            def hook(model, input, output):
                self.features[name] = output
            return hook
        
        for name, module in self.backbone.named_modules():
            if name in self.layers:
                module.register_forward_hook(get_activation(name))
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Extract features from input
        
        Args:
            x: Input tensor [B, C, H, W]
            
        Returns:
            Dictionary of {layer_name: features}
        """
        self.features = {}
        with torch.no_grad():
            _ = self.backbone(x)
        return self.features


class PatchCore:
    """
    PatchCore anomaly detection model
    """
    
    def __init__(
        self,
        backbone: str = 'wide_resnet50_2',
        layers: List[str] = ['layer2', 'layer3'],
        input_size: Tuple[int, int] = (256, 256),
        coreset_sampling_ratio: float = 0.01,
        num_neighbors: int = 9,
        device: str = 'cuda'
    ):
        """
        Args:
            backbone: Backbone network name
            layers: Layers to extract features from
            input_size: Input image size (H, W)
            coreset_sampling_ratio: Fraction of patches to keep
            num_neighbors: K for K-NN anomaly scoring
            device: Device to run on
        """
        self.backbone_name = backbone
        self.layers = layers
        self.input_size = input_size
        self.coreset_sampling_ratio = coreset_sampling_ratio
        self.num_neighbors = num_neighbors
        self.device = device
        
        # Feature extractor
        self.feature_extractor = FeatureExtractor(
            backbone_name=backbone,
            layers=layers,
            pretrained=True
        ).to(device)
        
        # Memory bank (will be filled during training)
        self.memory_bank = None
        self.index = None  # FAISS index for fast NN search
        
        # Threshold for anomaly detection
        self.threshold = None
    
    def extract_features(self, images: torch.Tensor) -> torch.Tensor:
        """
        Extract and aggregate patch features
        
        Args:
            images: Input images [B, 3, H, W]
            
        Returns:
            Aggregated patch features [B, N_patches, D]
        """
        # Extract multi-layer features
        features_dict = self.feature_extractor(images)

        # Aggregate features from multiple layers
        # First, upsample all features to the same spatial size
        target_size = None
        upsampled_features = []

        for layer_name in self.layers:
            layer_features = features_dict[layer_name]  # [B, C, H, W]

            # Use the largest spatial size as target
            if target_size is None:
                target_size = layer_features.shape[2:]

            # Upsample to target size if needed
            if layer_features.shape[2:] != target_size:
                layer_features = F.interpolate(
                    layer_features,
                    size=target_size,
                    mode='bilinear',
                    align_corners=False
                )

            upsampled_features.append(layer_features)

        # Concatenate along channel dimension
        # Result: [B, sum(C_i), H, W]
        combined_features = torch.cat(upsampled_features, dim=1)

        # Reshape to [B, H*W, C]
        B, C, H, W = combined_features.shape
        patch_features = combined_features.reshape(B, C, H * W)
        patch_features = patch_features.permute(0, 2, 1)  # [B, H*W, C]

        return patch_features
    
    def build_memory_bank(self, dataloader):
        """
        Build memory bank from training data (intact images)
        
        Args:
            dataloader: DataLoader with intact images
        """
        print("Building memory bank...")
        all_patch_features = []
        
        self.feature_extractor.eval()
        
        with torch.no_grad():
            for images, _ in dataloader:
                images = images.to(self.device)
                
                # Extract patch features
                patch_features = self.extract_features(images)  # [B, N, D]
                
                # Flatten batch dimension
                patch_features = patch_features.reshape(-1, patch_features.shape[-1])
                all_patch_features.append(patch_features.cpu().numpy())
        
        # Concatenate all features
        all_patch_features = np.concatenate(all_patch_features, axis=0)
        print(f"Total patches before coreset: {all_patch_features.shape[0]}")
        
        # Apply coreset subsampling
        self.memory_bank = self._coreset_sampling(
            all_patch_features,
            self.coreset_sampling_ratio
        )
        
        print(f"Memory bank size after coreset: {self.memory_bank.shape[0]}")
        
        # Build FAISS index for efficient nearest neighbor search
        self._build_faiss_index()
    
    def _coreset_sampling(
        self,
        features: np.ndarray,
        sampling_ratio: float
    ) -> np.ndarray:
        """
        Greedy coreset subsampling
        
        Args:
            features: Feature array [N, D]
            sampling_ratio: Fraction to keep
            
        Returns:
            Subsampled features
        """
        num_samples = int(len(features) * sampling_ratio)
        
        if num_samples >= len(features):
            return features
        
        # Simple random sampling (can be replaced with greedy k-center)
        indices = np.random.choice(len(features), num_samples, replace=False)
        return features[indices]
    
    def _build_faiss_index(self):
        """Build FAISS index for fast nearest neighbor search"""
        dimension = self.memory_bank.shape[1]
        
        # Use L2 distance
        self.index = faiss.IndexFlatL2(dimension)
        
        # Add features to index
        self.index.add(self.memory_bank.astype(np.float32))
        
        print(f"FAISS index built with {self.index.ntotal} vectors")
    
    def compute_anomaly_score(
        self,
        patch_features: torch.Tensor
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute anomaly scores for patches
        
        Args:
            patch_features: Patch features [N, D]
            
        Returns:
            Tuple of (anomaly_scores [N], spatial_shape)
        """
        patch_features_np = patch_features.cpu().numpy()
        
        # Search for K nearest neighbors
        distances, _ = self.index.search(
            patch_features_np.astype(np.float32),
            self.num_neighbors
        )
        
        # Anomaly score is the distance to nearest neighbor
        anomaly_scores = distances[:, 0]  # Distance to 1st NN
        
        # Can also use mean of K neighbors:
        # anomaly_scores = distances.mean(axis=1)
        
        return anomaly_scores
    
    def predict(
        self,
        image: torch.Tensor,
        return_heatmap: bool = True
    ) -> Dict:
        """
        Predict anomalies for an image
        
        Args:
            image: Input image [1, 3, H, W]
            return_heatmap: Whether to return anomaly heatmap
            
        Returns:
            Dictionary with prediction results
        """
        self.feature_extractor.eval()
        
        with torch.no_grad():
            # Extract features
            patch_features = self.extract_features(image)  # [1, N, D]
            patch_features = patch_features.squeeze(0)  # [N, D]
            
            # Compute anomaly scores
            anomaly_scores = self.compute_anomaly_score(patch_features)
            
            # Reshape to spatial dimensions
            spatial_size = int(np.sqrt(len(anomaly_scores)))
            anomaly_map = anomaly_scores.reshape(spatial_size, spatial_size)
            
            # Apply Gaussian smoothing
            anomaly_map = gaussian_filter(anomaly_map, sigma=4)
            
            # Resize to input size
            anomaly_map = torch.from_numpy(anomaly_map).unsqueeze(0).unsqueeze(0)
            anomaly_map = F.interpolate(
                anomaly_map,
                size=self.input_size,
                mode='bilinear',
                align_corners=False
            )
            anomaly_map = anomaly_map.squeeze().numpy()
            
            # Compute image-level score (max of all patches)
            image_score = float(anomaly_scores.max())
            
            # Binary prediction
            is_anomaly = image_score > self.threshold if self.threshold else None
            
            result = {
                'image_score': image_score,
                'is_anomaly': is_anomaly,
                'anomaly_map': anomaly_map if return_heatmap else None
            }
            
            return result
    
    def compute_threshold(self, val_loader):
        """
        Compute anomaly threshold from validation set (intact images)
        
        Args:
            val_loader: Validation dataloader with intact images
        """
        print("Computing anomaly threshold from validation set...")
        scores = []
        
        for images, _ in val_loader:
            images = images.to(self.device)
            
            with torch.no_grad():
                patch_features = self.extract_features(images)
                
                for i in range(len(images)):
                    pf = patch_features[i]
                    anomaly_scores = self.compute_anomaly_score(pf)
                    image_score = anomaly_scores.max()
                    scores.append(image_score)
        
        # Use 95th percentile as threshold
        self.threshold = float(np.percentile(scores, 95))
        print(f"Threshold set to: {self.threshold:.4f}")
        
        return self.threshold
    
    def save(self, path: str):
        """Save model"""
        torch.save({
            'memory_bank': self.memory_bank,
            'threshold': self.threshold,
            'config': {
                'backbone': self.backbone_name,
                'layers': self.layers,
                'input_size': self.input_size,
                'coreset_sampling_ratio': self.coreset_sampling_ratio,
                'num_neighbors': self.num_neighbors
            }
        }, path)
        print(f"Model saved to {path}")
    
    def load(self, path: str):
        """Load model"""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.memory_bank = checkpoint['memory_bank']
        self.threshold = checkpoint['threshold']
        self._build_faiss_index()
        print(f"Model loaded from {path}")


if __name__ == '__main__':
    # Test PatchCore
    print("Testing PatchCore implementation...")
    
    # Create model
    model = PatchCore(
        backbone='wide_resnet50_2',
        layers=['layer2', 'layer3'],
        coreset_sampling_ratio=0.01,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    # Test feature extraction
    dummy_input = torch.randn(2, 3, 256, 256).to(model.device)
    features = model.extract_features(dummy_input)
    print(f"Extracted features shape: {features.shape}")

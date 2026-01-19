"""
DRAEM (Discriminatively Trained Reconstruction Embedding for Surface Anomaly Detection)
Based on: https://github.com/vitjanz/draem

Two-branch architecture:
1. Reconstructive SubNetwork: Learns to reconstruct normal images
2. Discriminative SubNetwork: Learns to segment anomalies
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ReconstructiveSubNetwork(nn.Module):
    """
    Reconstructive branch: U-Net that reconstructs normal images
    """
    def __init__(self, in_channels=3, out_channels=3, base_width=128):
        super(ReconstructiveSubNetwork, self).__init__()
        self.encoder = EncoderReconstructive(in_channels, base_width)
        self.decoder = DecoderReconstructive(base_width, out_channels)

    def forward(self, x):
        b5 = self.encoder(x)
        output = self.decoder(b5)
        return output


class DiscriminativeSubNetwork(nn.Module):
    """
    Discriminative branch: Segments anomalies by comparing input and reconstruction
    Input: 6 channels (3 original + 3 reconstructed)
    Output: 2 channels (normal/anomaly segmentation)
    """
    def __init__(self, in_channels=6, out_channels=2, base_width=64):
        super(DiscriminativeSubNetwork, self).__init__()
        self.encoder = EncoderDiscriminative(in_channels, base_width)
        self.decoder = DecoderDiscriminative(base_width, out_channels)

    def forward(self, x):
        b1, b2, b3, b4, b5, b6 = self.encoder(x)
        output = self.decoder(b1, b2, b3, b4, b5, b6)
        return output


# ============================================================================
# Encoder/Decoder Components
# ============================================================================

class EncoderReconstructive(nn.Module):
    """Encoder for Reconstructive SubNetwork"""
    def __init__(self, in_channels, base_width):
        super(EncoderReconstructive, self).__init__()

        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels, base_width, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_width, base_width, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width),
            nn.ReLU(inplace=True)
        )
        self.mp1 = nn.MaxPool2d(2)

        self.block2 = nn.Sequential(
            nn.Conv2d(base_width, base_width * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_width * 2, base_width * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width * 2),
            nn.ReLU(inplace=True)
        )
        self.mp2 = nn.MaxPool2d(2)

        self.block3 = nn.Sequential(
            nn.Conv2d(base_width * 2, base_width * 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width * 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_width * 4, base_width * 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width * 4),
            nn.ReLU(inplace=True)
        )
        self.mp3 = nn.MaxPool2d(2)

        self.block4 = nn.Sequential(
            nn.Conv2d(base_width * 4, base_width * 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width * 8),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_width * 8, base_width * 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width * 8),
            nn.ReLU(inplace=True)
        )
        self.mp4 = nn.MaxPool2d(2)

        self.block5 = nn.Sequential(
            nn.Conv2d(base_width * 8, base_width * 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width * 8),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_width * 8, base_width * 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width * 8),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        b1 = self.block1(x)
        mp1 = self.mp1(b1)

        b2 = self.block2(mp1)
        mp2 = self.mp2(b2)

        b3 = self.block3(mp2)
        mp3 = self.mp3(b3)

        b4 = self.block4(mp3)
        mp4 = self.mp4(b4)

        b5 = self.block5(mp4)

        return b5


class DecoderReconstructive(nn.Module):
    """Decoder for Reconstructive SubNetwork"""
    def __init__(self, base_width, out_channels):
        super(DecoderReconstructive, self).__init__()

        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(base_width * 8, base_width * 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width * 8),
            nn.ReLU(inplace=True)
        )

        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(base_width * 8, base_width * 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width * 4),
            nn.ReLU(inplace=True)
        )

        self.up3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(base_width * 4, base_width * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width * 2),
            nn.ReLU(inplace=True)
        )

        self.up4 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(base_width * 2, base_width, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width),
            nn.ReLU(inplace=True)
        )

        self.final = nn.Conv2d(base_width, out_channels, kernel_size=1)

    def forward(self, b5):
        up1 = self.up1(b5)
        up2 = self.up2(up1)
        up3 = self.up3(up2)
        up4 = self.up4(up3)

        output = self.final(up4)
        return output


class EncoderDiscriminative(nn.Module):
    """Encoder for Discriminative SubNetwork with skip connections"""
    def __init__(self, in_channels, base_width):
        super(EncoderDiscriminative, self).__init__()

        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels, base_width, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_width, base_width, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width),
            nn.ReLU(inplace=True)
        )
        self.mp1 = nn.MaxPool2d(2)

        self.block2 = nn.Sequential(
            nn.Conv2d(base_width, base_width * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_width * 2, base_width * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width * 2),
            nn.ReLU(inplace=True)
        )
        self.mp2 = nn.MaxPool2d(2)

        self.block3 = nn.Sequential(
            nn.Conv2d(base_width * 2, base_width * 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width * 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_width * 4, base_width * 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width * 4),
            nn.ReLU(inplace=True)
        )
        self.mp3 = nn.MaxPool2d(2)

        self.block4 = nn.Sequential(
            nn.Conv2d(base_width * 4, base_width * 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width * 8),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_width * 8, base_width * 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width * 8),
            nn.ReLU(inplace=True)
        )
        self.mp4 = nn.MaxPool2d(2)

        self.block5 = nn.Sequential(
            nn.Conv2d(base_width * 8, base_width * 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width * 8),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_width * 8, base_width * 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width * 8),
            nn.ReLU(inplace=True)
        )
        self.mp5 = nn.MaxPool2d(2)

        self.block6 = nn.Sequential(
            nn.Conv2d(base_width * 8, base_width * 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width * 8),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_width * 8, base_width * 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width * 8),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        b1 = self.block1(x)
        mp1 = self.mp1(b1)

        b2 = self.block2(mp1)
        mp2 = self.mp2(b2)

        b3 = self.block3(mp2)
        mp3 = self.mp3(b3)

        b4 = self.block4(mp3)
        mp4 = self.mp4(b4)

        b5 = self.block5(mp4)
        mp5 = self.mp5(b5)

        b6 = self.block6(mp5)

        return b1, b2, b3, b4, b5, b6


class DecoderDiscriminative(nn.Module):
    """Decoder for Discriminative SubNetwork with skip connections"""
    def __init__(self, base_width, out_channels):
        super(DecoderDiscriminative, self).__init__()

        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(base_width * 8, base_width * 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width * 8),
            nn.ReLU(inplace=True)
        )

        self.db1 = nn.Sequential(
            nn.Conv2d(base_width * (8 + 8), base_width * 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width * 8),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_width * 8, base_width * 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width * 8),
            nn.ReLU(inplace=True)
        )

        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(base_width * 8, base_width * 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width * 8),
            nn.ReLU(inplace=True)
        )

        self.db2 = nn.Sequential(
            nn.Conv2d(base_width * (8 + 8), base_width * 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width * 8),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_width * 8, base_width * 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width * 8),
            nn.ReLU(inplace=True)
        )

        self.up3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(base_width * 8, base_width * 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width * 4),
            nn.ReLU(inplace=True)
        )

        self.db3 = nn.Sequential(
            nn.Conv2d(base_width * (4 + 4), base_width * 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width * 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_width * 4, base_width * 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width * 4),
            nn.ReLU(inplace=True)
        )

        self.up4 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(base_width * 4, base_width * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width * 2),
            nn.ReLU(inplace=True)
        )

        self.db4 = nn.Sequential(
            nn.Conv2d(base_width * (2 + 2), base_width * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_width * 2, base_width * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width * 2),
            nn.ReLU(inplace=True)
        )

        self.up5 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(base_width * 2, base_width, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width),
            nn.ReLU(inplace=True)
        )

        self.db5 = nn.Sequential(
            nn.Conv2d(base_width * 2, base_width, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_width, base_width, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width),
            nn.ReLU(inplace=True)
        )

        self.final = nn.Conv2d(base_width, out_channels, kernel_size=1)

    def forward(self, b1, b2, b3, b4, b5, b6):
        up1 = self.up1(b6)
        cat1 = torch.cat((up1, b5), dim=1)
        db1 = self.db1(cat1)

        up2 = self.up2(db1)
        cat2 = torch.cat((up2, b4), dim=1)
        db2 = self.db2(cat2)

        up3 = self.up3(db2)
        cat3 = torch.cat((up3, b3), dim=1)
        db3 = self.db3(cat3)

        up4 = self.up4(db3)
        cat4 = torch.cat((up4, b2), dim=1)
        db4 = self.db4(cat4)

        up5 = self.up5(db4)
        cat5 = torch.cat((up5, b1), dim=1)
        db5 = self.db5(cat5)

        output = self.final(db5)
        return output


# ============================================================================
# Combined DRAEM Model
# ============================================================================

class DRAEM(nn.Module):
    """
    Complete DRAEM model combining both subnetworks
    """
    def __init__(self, in_channels=3, out_channels_seg=2):
        super(DRAEM, self).__init__()

        self.reconstructive = ReconstructiveSubNetwork(
            in_channels=in_channels,
            out_channels=in_channels,
            base_width=128
        )

        self.discriminative = DiscriminativeSubNetwork(
            in_channels=in_channels * 2,  # Concatenate original and reconstructed
            out_channels=out_channels_seg,
            base_width=64
        )

    def forward(self, x):
        # Reconstructive branch
        reconstruction = self.reconstructive(x)

        # Discriminative branch
        # Concatenate original and reconstructed images
        combined = torch.cat([x, reconstruction], dim=1)
        segmentation = self.discriminative(combined)

        return reconstruction, segmentation


if __name__ == '__main__':
    # Test model
    model = DRAEM()
    x = torch.randn(2, 3, 256, 256)
    reconstruction, segmentation = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Reconstruction shape: {reconstruction.shape}")
    print(f"Segmentation shape: {segmentation.shape}")

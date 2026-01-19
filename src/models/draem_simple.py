"""
DRAEM - Simple Sequential Architecture
This matches the architecture used in your trained model
"""

import torch
import torch.nn as nn


class ReconstructiveSubNetwork(nn.Module):
    """Simpler reconstructive network with sequential layers"""
    def __init__(self, in_channels=3, out_channels=3, base_width=32):
        super().__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, base_width, 4, stride=2, padding=1),  # -> 128x128
            nn.BatchNorm2d(base_width),
            nn.ReLU(True),
            nn.Conv2d(base_width, base_width * 2, 4, stride=2, padding=1),  # -> 64x64
            nn.BatchNorm2d(base_width * 2),
            nn.ReLU(True),
            nn.Conv2d(base_width * 2, base_width * 4, 4, stride=2, padding=1),  # -> 32x32
            nn.BatchNorm2d(base_width * 4),
            nn.ReLU(True),
            nn.Conv2d(base_width * 4, base_width * 8, 4, stride=2, padding=1),  # -> 16x16
            nn.BatchNorm2d(base_width * 8),
            nn.ReLU(True),
        )

        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(base_width * 8, base_width * 8, 3, padding=1),
            nn.BatchNorm2d(base_width * 8),
            nn.ReLU(True),
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(base_width * 8, base_width * 4, 4, stride=2, padding=1),  # -> 32x32
            nn.BatchNorm2d(base_width * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(base_width * 4, base_width * 2, 4, stride=2, padding=1),  # -> 64x64
            nn.BatchNorm2d(base_width * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(base_width * 2, base_width, 4, stride=2, padding=1),  # -> 128x128
            nn.BatchNorm2d(base_width),
            nn.ReLU(True),
            nn.ConvTranspose2d(base_width, out_channels, 4, stride=2, padding=1),  # -> 256x256
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.bottleneck(x)
        x = self.decoder(x)
        return x


class DiscriminativeSubNetwork(nn.Module):
    """Simpler discriminative network with sequential layers"""
    def __init__(self, in_channels=6, out_channels=2, base_width=32):
        super().__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, base_width, 4, stride=2, padding=1),  # -> 128x128
            nn.BatchNorm2d(base_width),
            nn.ReLU(True),
            nn.Conv2d(base_width, base_width * 2, 4, stride=2, padding=1),  # -> 64x64
            nn.BatchNorm2d(base_width * 2),
            nn.ReLU(True),
            nn.Conv2d(base_width * 2, base_width * 4, 4, stride=2, padding=1),  # -> 32x32
            nn.BatchNorm2d(base_width * 4),
            nn.ReLU(True),
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(base_width * 4, base_width * 2, 4, stride=2, padding=1),  # -> 64x64
            nn.BatchNorm2d(base_width * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(base_width * 2, base_width, 4, stride=2, padding=1),  # -> 128x128
            nn.BatchNorm2d(base_width),
            nn.ReLU(True),
            nn.ConvTranspose2d(base_width, out_channels, 4, stride=2, padding=1),  # -> 256x256
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


class DRAEMSimple(nn.Module):
    """Complete DRAEM model with simple architecture"""
    def __init__(self, in_channels=3, out_channels_seg=2):
        super().__init__()
        self.reconstructive = ReconstructiveSubNetwork(
            in_channels=in_channels,
            out_channels=in_channels,
            base_width=32
        )
        self.discriminative = DiscriminativeSubNetwork(
            in_channels=in_channels * 2,
            out_channels=out_channels_seg,
            base_width=64
        )

    def forward(self, x):
        reconstruction = self.reconstructive(x)
        combined = torch.cat([x, reconstruction], dim=1)
        segmentation = self.discriminative(combined)
        return reconstruction, segmentation

"""
Perlin Noise Generator for Synthetic Anomaly Creation
Used in DRAEM to create realistic defect patterns during training
"""

import numpy as np
import torch


def rand_perlin_2d(shape, res, fade=lambda t: 6*t**5 - 15*t**4 + 10*t**3):
    """
    Generate 2D Perlin noise

    Args:
        shape: Output shape (height, width)
        res: Resolution of noise grid
        fade: Fade function for smooth interpolation

    Returns:
        Perlin noise array of shape (height, width)
    """
    delta = (res[0] / shape[0], res[1] / shape[1])
    d = (shape[0] // res[0], shape[1] // res[1])

    grid = np.mgrid[0:res[0]:delta[0], 0:res[1]:delta[1]].transpose(1, 2, 0) % 1

    # Gradients
    angles = 2 * np.pi * np.random.rand(res[0] + 1, res[1] + 1)
    gradients = np.dstack((np.cos(angles), np.sin(angles)))

    # Get grid coordinates
    g00 = gradients[0:-1, 0:-1].repeat(d[0], 0).repeat(d[1], 1)
    g10 = gradients[1:, 0:-1].repeat(d[0], 0).repeat(d[1], 1)
    g01 = gradients[0:-1, 1:].repeat(d[0], 0).repeat(d[1], 1)
    g11 = gradients[1:, 1:].repeat(d[0], 0).repeat(d[1], 1)

    # Ramps
    n00 = np.sum(np.dstack((grid[:, :, 0], grid[:, :, 1])) * g00, 2)
    n10 = np.sum(np.dstack((grid[:, :, 0] - 1, grid[:, :, 1])) * g10, 2)
    n01 = np.sum(np.dstack((grid[:, :, 0], grid[:, :, 1] - 1)) * g01, 2)
    n11 = np.sum(np.dstack((grid[:, :, 0] - 1, grid[:, :, 1] - 1)) * g11, 2)

    # Interpolation
    t = fade(grid)
    n0 = n00 * (1 - t[:, :, 0]) + t[:, :, 0] * n10
    n1 = n01 * (1 - t[:, :, 0]) + t[:, :, 0] * n11

    return np.sqrt(2) * ((1 - t[:, :, 1]) * n0 + t[:, :, 1] * n1)


def rand_perlin_2d_octaves(shape, res, octaves=1, persistence=0.5):
    """
    Generate 2D Perlin noise with multiple octaves for more detail

    Args:
        shape: Output shape (height, width)
        res: Base resolution
        octaves: Number of octaves to combine
        persistence: Amplitude factor for each octave

    Returns:
        Multi-octave Perlin noise
    """
    noise = np.zeros(shape)
    frequency = 1
    amplitude = 1

    for _ in range(octaves):
        noise += amplitude * rand_perlin_2d(
            shape,
            (frequency * res[0], frequency * res[1])
        )
        frequency *= 2
        amplitude *= persistence

    return noise


def generate_perlin_noise_mask(img_size, min_perlin_scale=0, max_perlin_scale=6):
    """
    Generate a binary anomaly mask using Perlin noise

    Args:
        img_size: Image size (height, width)
        min_perlin_scale: Minimum scale for Perlin noise
        max_perlin_scale: Maximum scale for Perlin noise

    Returns:
        Binary mask with anomaly region
    """
    # Random Perlin noise scale
    perlin_scale = 2 ** np.random.randint(min_perlin_scale, max_perlin_scale)

    # Generate Perlin noise
    perlin_noise = rand_perlin_2d_octaves(
        img_size,
        (perlin_scale, perlin_scale),
        octaves=3
    )

    # Threshold to create binary mask
    threshold = np.random.uniform(0.0, 0.5)
    mask = np.where(perlin_noise > threshold, 1.0, 0.0)

    return mask


def generate_smooth_anomaly(img_size, min_perlin_scale=0, max_perlin_scale=6):
    """
    Generate smooth anomaly pattern for realistic defects

    Args:
        img_size: Image size (height, width)
        min_perlin_scale: Minimum scale
        max_perlin_scale: Maximum scale

    Returns:
        Smooth anomaly pattern [0, 1]
    """
    perlin_scale = 2 ** np.random.randint(min_perlin_scale, max_perlin_scale)

    perlin_noise = rand_perlin_2d_octaves(
        img_size,
        (perlin_scale, perlin_scale),
        octaves=4,
        persistence=0.6
    )

    # Normalize to [0, 1]
    perlin_noise = (perlin_noise - perlin_noise.min()) / (perlin_noise.max() - perlin_noise.min() + 1e-8)

    return perlin_noise


if __name__ == '__main__':
    # Test Perlin noise generation
    import matplotlib.pyplot as plt

    # Generate noise
    noise = rand_perlin_2d_octaves((256, 256), (4, 4), octaves=4)
    mask = generate_perlin_noise_mask((256, 256))
    smooth = generate_smooth_anomaly((256, 256))

    # Visualize
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(noise, cmap='gray')
    axes[0].set_title('Perlin Noise (4 octaves)')
    axes[0].axis('off')

    axes[1].imshow(mask, cmap='gray')
    axes[1].set_title('Binary Mask')
    axes[1].axis('off')

    axes[2].imshow(smooth, cmap='gray')
    axes[2].set_title('Smooth Anomaly')
    axes[2].axis('off')

    plt.tight_layout()
    plt.savefig('perlin_noise_examples.png', dpi=150)
    print("Saved perlin_noise_examples.png")

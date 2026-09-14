"""
1. Global gamma collection 
2. Local gamma collection with 2 subimage sizes by dividing the image into 2x2 and 3x3. Find 
the most suitable gamma value for each subimage.  
"""

import cv2
from matplotlib import image
import numpy as np
import matplotlib.pyplot as plt

gamma_value = 0.0

def global_gamma_correction(image, gamma_value):
    """
    Perform global gamma correction on a grayscale image.
    Args:
        image (np.ndarray): Grayscale image (2D array).
        gamma_value (float): Gamma value for correction.

    """
    img_height = image.shape[0]
    img_width = image.shape[1]
    corrected_image = np.zeros((img_height, img_width), dtype=np.uint8)

    " s = c * r^gamma "
    for i in range(img_height):
        for j in range(img_width):
            r_norm = image[i, j] / 255.0
            s = (r_norm ** gamma_value) * 255.0
            corrected_image[i, j] = int(round(s))

    return corrected_image


def local_gamma_correction(
    image: np.ndarray, 
    grid_shape: tuple[int, int], 
    gamma_value: float | np.ndarray
) -> np.ndarray:
    """
    Applies gamma correction across an N x M grid of subimage blocks.
    
    Supports single scalar gamma values and block-specific 2D gamma arrays.
    Compatible with both grayscale (2D) and multi-channel color (3D) images.
    """
    img_height, img_width = image.shape[:2]
    grid_rows, grid_cols = grid_shape
    
    # Fast path: uniform scalar gamma using direct Lookup Table (LUT)
    if np.isscalar(gamma_value):
        lut = np.clip(np.round(((np.arange(256) / 255.0) ** float(gamma_value)) * 255.0), 0, 255).astype(np.uint8)
        return lut[image]
    
    gamma_grid = np.asarray(gamma_value, dtype=float)
    if gamma_grid.shape != (grid_rows, grid_cols):
        raise ValueError(f"gamma_value shape {gamma_grid.shape} must match grid_shape {grid_shape}")

    corrected_image = np.empty_like(image)
    
    block_h = img_height // grid_rows
    block_w = img_width // grid_cols

    for r in range(grid_rows):
        r_start = r * block_h
        r_end = img_height if r == grid_rows - 1 else (r + 1) * block_h
        
        for c in range(grid_cols):
            c_start = c * block_w
            c_end = img_width if c == grid_cols - 1 else (c + 1) * block_w
            
            # Precompute 256-entry LUT for this block's specific gamma
            lut = np.clip(np.round(((np.arange(256) / 255.0) ** gamma_grid[r, c]) * 255.0), 0, 255).astype(np.uint8)
            corrected_image[r_start:r_end, c_start:c_end] = lut[image[r_start:r_end, c_start:c_end]]

    return corrected_image
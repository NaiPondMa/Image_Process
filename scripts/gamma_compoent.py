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

import numpy as np

def local_gamma_correction(image, grid_shape) -> np.ndarray:
    """
    Divides the image into an N x N grid of subimages (e.g., (2,2) or (3,3)) 
    and automatically computes the optimal gamma value for each subimage block.
    """
    img_height, img_width = image.shape
    grid_rows, grid_cols = grid_shape
    corrected_image = np.zeros_like(image, dtype=np.uint8)
    
    # Calculate base dimensions for subimage blocks
    block_h = img_height // grid_rows
    block_w = img_width // grid_cols
    
    for r in range(grid_rows):
        for c in range(grid_cols):
            # Define block boundaries (handles uneven image division at borders)
            r_start, r_end = r * block_h, (r + 1) * block_h if r < grid_rows - 1 else img_height
            c_start, c_end = c * block_w, (c + 1) * block_w if c < grid_cols - 1 else img_width
            
            # Extract subimage block
            subimage = image[r_start:r_end, c_start:c_end]
            
            # Calculate normalized mean intensity of the subimage
            mean_intensity = np.mean(subimage) / 255.0
            mean_intensity = np.clip(mean_intensity, 1e-4, 1.0 - 1e-4)  # Avoid log(0) or log(1)
            
            # Find the dynamic gamma for this specific subimage (maps mean to mid-gray 0.5)
            gamma_suitable = np.log(0.5) / np.log(mean_intensity)
            
            # Apply gamma correction to the subimage
            sub_norm = subimage / 255.0
            sub_corrected = (sub_norm ** gamma_suitable) * 255.0
            
            corrected_image[r_start:r_end, c_start:c_end] = np.clip(np.round(sub_corrected), 0, 255).astype(np.uint8)
            
    return corrected_image
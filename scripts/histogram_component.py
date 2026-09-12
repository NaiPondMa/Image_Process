"""
1. Global histogram equalization 
2. Local histogram equalization with 3 neighborhood sizes: 3x3, 7x7, and 11x11. Find the 
values for 𝑘0, 𝑘1, and  𝑘2 that you think are the most suitable values.

"""
import cv2
import numpy as np
import matplotlib.pyplot as plt

def global_histogram_equalization(image):
    """
    Perform global histogram equalization on a grayscale image.
    input: 2D grayscale image
    output: 2D grayscale image after histogram equalization

    """
    img_height = image.shape[0]
    img_width = image.shape[1]
    histogram = np.zeros(256, dtype=np.int32)

    #calculate histogram
    for i in range(0,img_height):
        for j in range(0,img_width):
            pixel_value = image[i,j]
            histogram[pixel_value] += 1

    #calculate pdf to the image
    pdf_img = histogram / (img_height * img_width)

    #calculate cdf to the image
    cdf_img = np.zeros(256, float)
    cdf_img[0] = pdf_img[0]
    for i in range(1,256):
        cdf_img[i] = cdf_img[i-1] + pdf_img[i]

    cdf_eq = np.round(cdf_img * 255,0)

    imgEqualized = np.zeros((img_height, img_width), dtype=np.uint8)

    #for mapping input image to s
    for i in range(0,img_height):
        for j in range(0,img_width):
            r = image[i,j]
            s = cdf_eq[r]
            imgEqualized[i,j] = s

    return imgEqualized
def local_histogram_equalization(image, neighborhood_size):
    """
    Perform local histogram equalization on a grayscale image without built-in equalization functions.
    
    Args:
        image (np.ndarray): Grayscale image (2D array).
        neighborhood_size (int): Size of the local neighborhood (must be odd and >= 3).
    
    Returns:
        np.ndarray: Locally histogram-equalized image.
    """

    if neighborhood_size < 3 or neighborhood_size % 2 == 0:
        raise ValueError("neighborhood_size must be an odd integer >= 3.")
    
    height, width = image.shape
    pad_size = neighborhood_size // 2
    
    # Pad image to handle edge pixels cleanly without reducing output dimensions
    padded_image = np.pad(image, pad_size, mode='reflect')
    output = np.zeros_like(image, dtype=np.uint8)
    
    total_pixels = neighborhood_size * neighborhood_size

    # Slide the window over every pixel in the image
    for r in range(height):
        for c in range(width):
            # Extract the local spatial window
            window = padded_image[r:r + neighborhood_size, c:c + neighborhood_size]
            center_val = image[r, c]
            
            # Compute 256-bin local histogram and Cumulative Distribution Function (CDF)
            hist = np.bincount(window.ravel(), minlength=256)
            cdf = hist.cumsum()
            
            # Minimum non-zero value in CDF
            cdf_min = cdf[cdf > 0][0] if np.any(cdf > 0) else 0
            
            # Apply standard equalization formula: T(v) = round(((cdf(v) - cdf_min) / (N - cdf_min)) * 255)
            if total_pixels - cdf_min > 0:
                equalized_pixel = np.round(((cdf[center_val] - cdf_min) / (total_pixels - cdf_min)) * 255)
            else:
                equalized_pixel = center_val
                
            output[r, c] = np.clip(equalized_pixel, 0, 255)

    return output



    return cv2.equalizeHist(image)  # Placeholder - replace with actual local histogram equalization implementation

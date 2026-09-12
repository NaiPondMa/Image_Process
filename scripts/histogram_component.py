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
def local_histogram_equalization(image: np.ndarray, neighborhood_size: int, k0: float, k1: float, k2: float, E: float = 4.0) -> np.ndarray:
    """
    Local histogram statistics enhancement using mean and standard deviation thresholds.
    
    Args:
        image (np.ndarray): 2D grayscale input image.
        neighborhood_size (int): Neighborhood window dimension (3, 7, 11).
        k0 (float): Mean intensity multiplier (m_S <= k0 * m_G).
        k1 (float): Lower standard deviation multiplier (k1 * sigma_G <= sigma_S).
        k2 (float): Upper standard deviation multiplier (sigma_S <= k2 * sigma_G).
        E (float): Multiplier gain factor for dark pixels.
    """
    if neighborhood_size < 3 or neighborhood_size % 2 == 0:
        raise ValueError("neighborhood_size must be an odd integer >= 3.")
    
    height, width = image.shape
    pad_size = neighborhood_size // 2
    padded_image = np.pad(image, pad_size, mode='reflect')
    
    # Global statistics
    m_G = np.mean(image)
    sigma_G = np.std(image)
    
    output = image.astype(np.float64).copy()
    
    # Process local window around each pixel
    for r in range(height):
        for c in range(width):
            window = padded_image[r:r + neighborhood_size, c:c + neighborhood_size]
            m_S = np.mean(window)
            sigma_S = np.std(window)
            
            # Local enhancement criteria
            if (m_S <= k0 * m_G) and (k1 * sigma_G <= sigma_S <= k2 * sigma_G):
                output[r, c] = E * image[r, c]
                
    return np.clip(output, 0, 255).astype(np.uint8)

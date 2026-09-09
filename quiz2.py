"""
Write a program to enhance the given image so that the dark part on the right side of an image 
brings out more details using:  
1. Global histogram equalization 
2. Local histogram equalization with 3 neighborhood sizes: 3x3, 7x7, and 11x11. Find the 
    values for 𝑘0, 𝑘1, and  𝑘2 that you think are the most suitable values. 
3. Global gamma collection 
4. Local gamma collection with 2 subimage sizes by dividing the image into 2x2 and 3x3. Find 
the most suitable gamma value for each subimage.  
"""

import cv2
import matplotlib.pyplot as plt

from scripts.histogram_component import global_histogram_equalization, local_histogram_equalization
from scripts.gamma_compoent import global_gamma_collection, local_gamma_collection

if __name__ == "__main__":
    path = (r"D:\Github\Image_Process\images\Filament.jpg")
    image = cv2.imread(path)

    if image is None:
        raise FileNotFoundError("Image not found:" + path)

    # Global histogram equalization
    global_hist_eq_image = global_histogram_equalization(image)

    # Local histogram equalization with different neighborhood sizes
    local_hist_eq_image_3x3 = local_histogram_equalization(image, 3)
    local_hist_eq_image_7x7 = local_histogram_equalization(image, 7)
    local_hist_eq_image_11x11 = local_histogram_equalization(image, 11)

    # Global gamma collection
    global_gamma_image = global_gamma_collection(image, 1.5)  # Example gamma value

    # Local gamma collection with different subimage sizes
    local_gamma_image_2x2 = local_gamma_collection(image, (2, 2), 1.5)  # Example gamma value
    local_gamma_image_3x3 = local_gamma_collection(image, (3, 3), 1.5)  # Example gamma value

    plt.subplot(2, 4, 1)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title("Original Image")

    plt.subplot(2, 4, 2)
    plt.imshow(global_hist_eq_image, cmap='gray')
    plt.title("Global Histogram Equalization")

    plt.subplot(2, 4, 3)
    plt.imshow(local_hist_eq_image_3x3, cmap='gray')
    plt.title("Local Histogram Equalization (3x3)")

    plt.subplot(2, 4, 4)
    plt.imshow(local_hist_eq_image_7x7, cmap='gray')
    plt.title("Local Histogram Equalization (7x7)")

    plt.subplot(2, 4, 5)
    plt.imshow(local_hist_eq_image_11x11, cmap='gray')
    plt.title("Local Histogram Equalization (11x11)")

    plt.subplot(2, 4, 6)
    plt.imshow(global_gamma_image, cmap='gray')
    plt.title("Global Gamma Collection")

    plt.subplot(2, 4, 7)
    plt.imshow(local_gamma_image_2x2, cmap='gray')
    plt.title("Local Gamma Collection (2x2)")
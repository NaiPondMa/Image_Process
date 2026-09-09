"""
1. Global histogram equalization 
2. Local histogram equalization with 3 neighborhood sizes: 3x3, 7x7, and 11x11. Find the 
values for 𝑘0, 𝑘1, and  𝑘2 that you think are the most suitable values.

"""
import cv2
import numpy as np
import matplotlib.pyplot as plt

def global_histogram_equalization(image):
    return cv2.equalizeHist(image)
def local_histogram_equalization(image, neighborhood_size):
    return cv2.equalizeHist(image)  # Placeholder - replace with actual local histogram equalization implementation

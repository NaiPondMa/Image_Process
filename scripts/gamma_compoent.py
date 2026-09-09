"""
1. Global gamma collection 
2. Local gamma collection with 2 subimage sizes by dividing the image into 2x2 and 3x3. Find 
the most suitable gamma value for each subimage.  
"""

import cv2
from matplotlib import image
import numpy as np
import matplotlib.pyplot as plt

def global_gamma_collection(image, gamma_value):
    return cv2.pow(image / 255.0, gamma_value) * 255.0
def local_gamma_collection(image, subimage_size, gamma_value):
    return cv2.pow(image / 255.0, gamma_value) * 255.0  # Placeholder - replace with actual local gamma collection implementation
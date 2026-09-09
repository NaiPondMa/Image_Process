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
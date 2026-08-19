import cv2
import numpy as np
import matplotlib.pyplot as plt

def quantize_image(image, k):
    """
    Quantizes an 8-bit 2D list matrix to target_bits without built-in functions.
    
    :param image_matrix: 2D list of integer pixel values [0, 255]
    :param target_bits: Int from 1 to 8
    :return: 2D list of quantized pixel values
    """

    #check target bit
    if not (1 <= k <= 8):
        raise ValueError("target_bits must be between 1 and 8")

    shift

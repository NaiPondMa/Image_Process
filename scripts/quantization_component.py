import numpy as np
import cv2

def quantize_image(image, k):
    """
    Convert an image to grayscale and quantize it to k bits.
    
    :param image: A 2D grayscale or 3-channel BGR uint8 image
    :param k: Int from 1 to 8
    :return: 2D list of quantized pixel values
    """

    #conver to gray scale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #check target bit
    if not (1 <= k <= 8):
        raise ValueError("target_bits must be between 1 and 8")

    # Calculate the quantization step size
    L = 2 ** k
    bin_width = 256 // L
    scale = 255 / (L-1) if L > 1 else 0

    quantized_image = []
    for row in gray_image:
        quantized_row = []
        for pixel in row:
            # 1. Map pixel to discrete level index
            level = pixel // bin_width

            # 2. Map level index back to quantized pixel value
            quantized_pixel = round(level * scale)
            quantized_row.append(quantized_pixel)
        quantized_image.append(quantized_row)

    return quantized_image

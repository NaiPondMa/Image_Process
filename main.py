import cv2
import matplotlib.pyplot as plt

from scripts.sampling_component import manual_zoom_function, manual_shrinking_function
from scripts.quantization_component import quantize_image

if __name__ == "__main__":
    path = (r"D:\Github\Image_Process\images\kittens.jpg")
    image = cv2.imread(path)

    if image is None:
        raise FileNotFoundError("Image not found:" + path)

    zoomed_image = manual_zoom_function(image, 4.0)
    shrinked_image = manual_shrinking_function(image, 4.0)
    quantized_image_1bit = quantize_image(image, 1)
    quantized_image_3bit = quantize_image(image, 3)
    quantized_image_6bit = quantize_image(image, 6)

    plt.subplot(2, 3, 1)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title("Original Image")

    plt.subplot(2, 3, 2)
    plt.imshow(zoomed_image, cmap='gray')
    plt.title("Zoomed Image 4x")

    plt.subplot(2, 3, 3)
    plt.imshow(shrinked_image, cmap='gray')
    plt.title("Shrinked Image 1/4x")

    plt.subplot(2, 3, 4)
    plt.imshow(quantized_image_1bit, cmap='gray')
    plt.title("1-bit Quantized Image")

    plt.subplot(2, 3, 5)
    plt.imshow(quantized_image_3bit, cmap='gray')
    plt.title("3-bit Quantized Image")

    plt.subplot(2, 3, 6)
    plt.imshow(quantized_image_6bit, cmap='gray')
    plt.title("6-bit Quantized Image")

    plt.show()

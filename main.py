import cv2
import matplotlib.pyplot as plt

from sampling_component import manual_zoom_function
from sampling_component import manual_shrinking_function

if __name__ == "__main__":
    image = cv2.imread(r"D:\Github\Image_Process\images\cartoon_6a7499925e31c.jpg")

    if image is None:
        raise FileNotFoundError("Image not found: D:\Github\Image_Process\images\cartoon_6a7499925e31c.jpg")

    zoomed_image = manual_zoom_function(image, 2.0)
    shrinked_image = manual_shrinking_function(image, 4.0)

    plt.subplot(1, 3, 1)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title("Original Image")

    plt.subplot(1, 3, 2)
    plt.imshow(zoomed_image, cmap='gray')
    plt.title("Zoomed Image")

    plt.subplot(1, 3, 3)
    plt.imshow(shrinked_image, cmap='gray')
    plt.title("Shrinked Image")

    plt.show()

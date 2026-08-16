import cv2
import matplotlib.pyplot as plt

from function_zoom import manual_zoom_function

if __name__ == "__main__":
    image = cv2.imread(r"D:\Github\Image_Process\images\cartoon_6a7499925e31c.jpg")

    if image is None:
        raise FileNotFoundError("Image not found: D:\Github\Image_Process\images\cartoon_6a7499925e31c.jpg")

    zoomed_image = manual_zoom_function(image, 2.0)

    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title("Original Image")

    plt.subplot(1, 2, 2)
    plt.imshow(zoomed_image, cmap='gray')
    plt.title("Zoomed Image")

    plt.show()

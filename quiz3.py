"""
Write a short report to explain the processes of enhancing the given four images.

Write programs to remove noise and enhance the given images. Which filters and what
sizes are suitable for each of the noisy image?

Write programs to enhance the two blurry images. Which filters and what sizes are
suitable for each of the blurry images?
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

from scripts.filter_component import *
from scripts.sharpening_component import *


def load_rgb(path):
    img = cv2.imread(path, cv2.IMREAD_COLOR)        # OpenCV loads as BGR
    if img is None:                                  # cv2.imread fails silently
        raise FileNotFoundError(path)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def per_channel(func, rgb, *args, **kwargs):
    """Apply a grayscale filter to R, G and B separately (for noise removal)."""
    return np.stack([func(rgb[..., c], *args, **kwargs) for c in range(3)], axis=-1)


def on_luminance(func, rgb, *args, **kwargs):
    """Apply a grayscale enhancement to the brightness (Y) channel only.

    Sharpening R, G, B separately can create colour fringes at edges, so we
    convert to YCrCb, enhance Y, keep the colour (Cr, Cb) unchanged, and
    convert back to RGB.
    """
    ycrcb = cv2.cvtColor(rgb, cv2.COLOR_RGB2YCrCb)
    ycrcb[..., 0] = func(ycrcb[..., 0], *args, **kwargs)
    return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2RGB)


def sharpen_and_stretch(y, strength):
    """Combined Laplacian x smoothed Sobel (slides 153-159) + contrast stretch."""
    sharp = combined_sharpen(y, kernel_type="8pos", strength=strength, smooth_size=5)
    return contrast_stretch(sharp)


def show_grid(rows, title, save_as=None):
    """rows: list of rows, each a list of (image, caption). Images are RGB."""
    n_rows, n_cols = len(rows), max(len(r) for r in rows)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3.2 * n_rows),
                             squeeze=False)
    for r, row in enumerate(rows):
        for c in range(n_cols):
            ax = axes[r][c]
            ax.axis("off")
            if c < len(row):
                img, caption = row[c]
                if img.ndim == 2:
                    ax.imshow(img, cmap="gray", vmin=0, vmax=255)
                else:
                    ax.imshow(img)                   # RGB uint8, shown as-is
                ax.set_title(caption, fontsize=10)
    fig.suptitle(title)
    fig.tight_layout()
    if save_as:
        fig.savefig(save_as, dpi=120)


if __name__ == "__main__":
    folder = r"D:\Github\Image_Process\images\quiz3"
    cabin1 = load_rgb(rf"{folder}\noisy_cabin1_1.jpg")
    cabin2 = load_rgb(rf"{folder}\noisy_cabin1_2.jpg")
    stopsign = load_rgb(rf"{folder}\blury_stop_sign.jpg")
    moon = load_rgb(rf"{folder}\blurry_moon.jpg")

    # ---- Noise removal: each colour channel filtered separately -------------
    filter_cabin1 = per_channel(median_filter, cabin1, 3)                 # salt-and-pepper
    filter_cabin2 = per_channel(gaussian_filter, cabin2, ksize=5, sigma=1.0)  # Gaussian noise

    # ---- Sharpening: brightness channel only --------------------------------
    result_moon = on_luminance(sharpen_and_stretch, moon, strength=3)
    result_sign = on_luminance(sharpen_and_stretch, stopsign, strength=4)

    show_grid(
        [[(cabin1, "Noisy cabin (type 1)"), (cabin2, "Noisy cabin (type 2)"),
          (moon, "Blurry moon"), (stopsign, "Blurry stop sign")],
         [(filter_cabin1, "Median 3x3"), (filter_cabin2, "Gaussian 5x5, sigma=1"),
          (result_moon, "Laplacian x smoothed Sobel\n+ contrast stretch"),
          (result_sign, "Laplacian x smoothed Sobel\n+ contrast stretch")]],
        "Quiz 3 - final results")

    plt.show()
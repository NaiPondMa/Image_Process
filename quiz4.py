"""
Assignment #4: Frequency Domain Filtering and Periodic Noise Removal
Course: Digital Image Processing
Due Date: November 13, 2026

Tasks:
  1. Notch (Ideal) Filters:
     a. Low-pass filter with radius r in {10, 50, 100}
     b. High-pass filter with radius r in {10, 50, 100}
  2. Gaussian Filters:
     a. Low-pass filter with cutoff D0 in {10, 50, 100}
     b. High-pass filter with cutoff D0 in {10, 50, 100}
  3. Periodic Noise Removal:
     Remove periodic sinusoidal interference from:
       - Noisy_flower1_horizontal.jpg (horizontal noise stripes)
       - Noisy_flower1_vertical.jpg   (vertical noise stripes)
       - diag_noisy_animals.jpg       (diagonal periodic noise)
     using Gaussian Notch Reject Filters (GNRF) to eliminate ringing.
"""

import os
import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt

from scripts.notch_filter_component import (
    notch_lowpass_filter,
    notch_highpass_filter,
    ideal_notch_reject_filter,
    apply_frequency_filter,
    compute_spectrum,
)
from scripts.gaussian_filter_component import (
    gaussian_lowpass_filter,
    gaussian_highpass_filter,
    gaussian_notch_reject_filter,
    apply_gaussian_notch_reject,
)


def load_rgb(path: str) -> np.ndarray:
    """Loads an image as RGB uint8 (OpenCV loads as BGR)."""
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Image not found at: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def load_gray(path: str) -> np.ndarray:
    """Loads an image as grayscale uint8."""
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Image not found at: {path}")
    return img


def show_grid(rows, title: str, save_as: str = None):
    """
    Renders and optionally saves a grid of subplots.
    rows: list of rows, each a list of (image, caption).
    Images can be 2D grayscale uint8 or 3D RGB uint8.
    """
    n_rows = len(rows)
    n_cols = max(len(r) for r in rows)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.0 * n_cols, 3.2 * n_rows), squeeze=False)
    
    for r, row in enumerate(rows):
        for c in range(n_cols):
            ax = axes[r][c]
            ax.axis("off")
            if c < len(row):
                img, caption = row[c]
                if img.ndim == 2:
                    ax.imshow(img, cmap="gray", vmin=0, vmax=255)
                else:
                    ax.imshow(img)
                ax.set_title(caption, fontsize=10, fontweight="bold")
                
    fig.suptitle(title, fontsize=14, fontweight="bold")
    fig.tight_layout()
    if save_as:
        os.makedirs(os.path.dirname(os.path.abspath(save_as)), exist_ok=True)
        fig.savefig(save_as, dpi=130)
        print(f"[+] Saved visualization to: {save_as}")
    return fig


# =============================================================================
# Task 1 & 2: Frequency Domain Filtering (Notch vs Gaussian)
# =============================================================================
def run_task1_and_task2(image_path: str, output_prefix: str, save_dir: str = "images/result"):
    """
    Executes Tasks 1 & 2 on a single test image:
      - Notch (Ideal) Low-Pass & High-Pass with r in {10, 50, 100}
      - Gaussian Low-Pass & High-Pass with D0 in {10, 50, 100}
    """
    img_rgb = load_rgb(image_path)
    shape = img_rgb.shape[:2]
    radii = [10, 50, 100]
    
    # 1. Notch Filters (Assignment 1.a & 1.b)
    notch_lp_results = []
    notch_hp_results = []
    for r in radii:
        H_lp = notch_lowpass_filter(shape, r)
        H_hp = notch_highpass_filter(shape, r)
        res_lp = apply_frequency_filter(img_rgb, H_lp)
        res_hp = apply_frequency_filter(img_rgb, H_hp)
        notch_lp_results.append((res_lp, f"Notch LPF (r={r})"))
        notch_hp_results.append((res_hp, f"Notch HPF (r={r})"))
        
    # Grid 1: Notch Filters
    rows_notch = [
        [(img_rgb, "Original Input")] + notch_lp_results,
        [(img_rgb, "Original Input")] + notch_hp_results
    ]
    show_grid(
        rows_notch,
        f"Task 1: Notch (Ideal) Filters on {output_prefix.capitalize()}",
        save_as=os.path.join(save_dir, f"quiz4_{output_prefix}_task1_notch.png")
    )
    
    # 2. Gaussian Filters (Assignment 2.a & 2.b)
    gauss_lp_results = []
    gauss_hp_results = []
    for d0 in radii:
        H_lp = gaussian_lowpass_filter(shape, d0)
        H_hp = gaussian_highpass_filter(shape, d0)
        res_lp = apply_frequency_filter(img_rgb, H_lp)
        res_hp = apply_frequency_filter(img_rgb, H_hp)
        gauss_lp_results.append((res_lp, f"Gaussian LPF (D0={d0})"))
        gauss_hp_results.append((res_hp, f"Gaussian HPF (D0={d0})"))
        
    # Grid 2: Gaussian Filters
    rows_gauss = [
        [(img_rgb, "Original Input")] + gauss_lp_results,
        [(img_rgb, "Original Input")] + gauss_hp_results
    ]
    show_grid(
        rows_gauss,
        f"Task 2: Gaussian Filters on {output_prefix.capitalize()}",
        save_as=os.path.join(save_dir, f"quiz4_{output_prefix}_task2_gaussian.png")
    )


# =============================================================================
# Task 3: Periodic Noise Removal via Gaussian Notch Reject Filtering
# =============================================================================
def remove_periodic_noise(
    image_path: str,
    notches: list,
    d0: float = 6.0,
    filter_type: str = "gaussian",
    title: str = "Periodic Noise Removal",
    save_prefix: str = "periodic_noise",
    save_dir: str = "images/result"
):
    """
    Removes periodic interference spikes using Notch Reject Filtering.
    
    Why Gaussian Notch Reject Filter (GNRF) is most suitable:
      1. Ideal notch filters introduce sharp step discontinuities in frequency,
         causing severe ringing (Gibbs phenomenon) in the spatial domain.
      2. The Gaussian Notch Reject Filter has a smooth, exponentially decaying
         attenuation profile that zeros out the sinusoidal interference spikes
         while preserving all nearby structural frequencies without ringing.
    
    Parameters:
        image_path: Path to the noisy image.
        notches: List of (dx, dy) frequency offsets relative to center.
        d0: Notch bandwidth/radius parameter.
        filter_type: 'gaussian' (recommended) or 'ideal'.
    """
    noisy_img = load_gray(image_path)
    shape = noisy_img.shape
    H_img, W_img = shape
    
    # Compute 2D Fourier spectrum
    spectrum_noisy = compute_spectrum(noisy_img)
    
    # Construct Notch Reject Transfer Function
    if filter_type.lower() == "gaussian":
        H_filter = gaussian_notch_reject_filter(shape, notches, d0=d0)
    else:
        H_filter = ideal_notch_reject_filter(shape, notches, radius=d0)
        
    # Apply filter in frequency domain
    restored_img = apply_frequency_filter(noisy_img, H_filter)
    
    # Compute filtered spectrum
    spectrum_restored = compute_spectrum(restored_img)
    
    # Visual overlay of notch reject regions on the noisy spectrum
    cy, cx = H_img // 2, W_img // 2
    spec_norm = cv2.normalize(spectrum_noisy, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    spec_rgb = cv2.cvtColor(spec_norm, cv2.COLOR_GRAY2RGB)
    
    # Highlight notch locations with red circles on the spectrum
    for dx, dy in notches:
        cv2.circle(spec_rgb, (cx + dx, cy + dy), int(d0 + 2), (255, 40, 40), 2)
        cv2.circle(spec_rgb, (cx - dx, cy - dy), int(d0 + 2), (255, 40, 40), 2)
        
    # Render 4-panel diagnostic figure
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.5))
    axes[0].imshow(noisy_img, cmap="gray", vmin=0, vmax=255)
    axes[0].set_title(f"Noisy Input Image\n({title})", fontsize=11, fontweight="bold")
    axes[0].axis("off")
    
    axes[1].imshow(spec_rgb)
    axes[1].set_title(f"Noisy Spectrum\nNotches circled (D0={d0})", fontsize=11, fontweight="bold")
    axes[1].axis("off")
    
    axes[2].imshow(H_filter, cmap="gray", vmin=0, vmax=1)
    axes[2].set_title(f"GNRF Transfer Function H(u,v)\n{len(notches)} Notch Pairs", fontsize=11, fontweight="bold")
    axes[2].axis("off")
    
    axes[3].imshow(restored_img, cmap="gray", vmin=0, vmax=255)
    axes[3].set_title(f"Restored Output Image\n(Periodic Noise Removed)", fontsize=11, fontweight="bold")
    axes[3].axis("off")
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, f"quiz4_{save_prefix}_restored.png")
    os.makedirs(save_dir, exist_ok=True)
    fig.savefig(save_path, dpi=140)
    print(f"[+] Saved Task 3 restoration to: {save_path}")
    
    return restored_img, H_filter


def run_task3(quiz4_folder: str, save_dir: str = "images/result"):
    """
    Executes Task 3 for all three periodic noise images with precise parameters:
      1. Noisy_flower1_horizontal:
         Interference: Horizontal sinusoids -> Spikes on vertical axis.
         Notches: (dx=0, dy=±69) and 2nd harmonic (dx=0, dy=±138), D0=6.0.
      2. Noisy_flower1_vertical:
         Interference: Vertical sinusoids -> Spikes on horizontal axis.
         Notches: (dx=±96, dy=0) and 2nd harmonic (dx=±192, dy=0), D0=6.0.
      3. diag_noisy_animals:
         Interference: Diagonal sinusoidal pattern.
         Notches: (dx=±28, dy=±37), (dx=±28, dy=±139), (dx=±28, dy=∓65), D0=5.0.
    """
    print("\n" + "=" * 80)
    print("TASK 3: PERIODIC NOISE REMOVAL USING GAUSSIAN NOTCH REJECT FILTERS")
    print("=" * 80)
    
    # 1. Horizontal Flower Noise
    p_flower_h = glob.glob(os.path.join(quiz4_folder, "*flower1_horizontal*.jpg"))[0]
    notches_h = [(0, 69), (0, 138)]
    print(f"[*] Processing Horizontal Flower Noise: {os.path.basename(p_flower_h)}")
    print(f"    - Filter: Gaussian Notch Reject Filter (GNRF)")
    print(f"    - Notch Centers (dx, dy): {notches_h} + conjugate symmetric points")
    print(f"    - Cutoff radius D0 = 6.0")
    remove_periodic_noise(
        p_flower_h,
        notches=notches_h,
        d0=6.0,
        filter_type="gaussian",
        title="Horizontal Noise",
        save_prefix="flower_horizontal",
        save_dir=save_dir
    )
    
    # 2. Vertical Flower Noise
    p_flower_v = glob.glob(os.path.join(quiz4_folder, "*flower1_vertical*.jpg"))[0]
    notches_v = [(96, 0), (192, 0)]
    print(f"[*] Processing Vertical Flower Noise: {os.path.basename(p_flower_v)}")
    print(f"    - Filter: Gaussian Notch Reject Filter (GNRF)")
    print(f"    - Notch Centers (dx, dy): {notches_v} + conjugate symmetric points")
    print(f"    - Cutoff radius D0 = 6.0")
    remove_periodic_noise(
        p_flower_v,
        notches=notches_v,
        d0=6.0,
        filter_type="gaussian",
        title="Vertical Noise",
        save_prefix="flower_vertical",
        save_dir=save_dir
    )
    
    # 3. Diagonal Animals Noise
    p_animals_diag = glob.glob(os.path.join(quiz4_folder, "*diag_noisy_animals*.jpg"))[0]
    notches_diag = [(28, 37), (28, 139), (28, -65)]
    print(f"[*] Processing Diagonal Animals Noise: {os.path.basename(p_animals_diag)}")
    print(f"    - Filter: Gaussian Notch Reject Filter (GNRF)")
    print(f"    - Notch Centers (dx, dy): {notches_diag} + conjugate symmetric points")
    print(f"    - Cutoff radius D0 = 5.0")
    remove_periodic_noise(
        p_animals_diag,
        notches=notches_diag,
        d0=5.0,
        filter_type="gaussian",
        title="Diagonal Animals Noise",
        save_prefix="animals_diagonal",
        save_dir=save_dir
    )


# =============================================================================
# Main Program
# =============================================================================
if __name__ == "__main__":
    folder = r"images\quiz4"
    save_folder = r"images\result"
    os.makedirs(save_folder, exist_ok=True)
    
    # Locate clean test images for Task 1 and Task 2
    flower_clean = glob.glob(os.path.join(folder, "*flower1_6*.jpg"))[0]
    animals_clean = glob.glob(os.path.join(folder, "*animals_6*.jpg"))[0]
    
    print("=" * 80)
    print("ASSIGNMENT #4 - FREQUENCY DOMAIN ENHANCEMENT & NOISE REMOVAL")
    print("=" * 80)
    
    # Run Tasks 1 & 2 on clean images
    print("\n--- Running Task 1 & 2: Notch and Gaussian Filters on flower1.jpg ---")
    run_task1_and_task2(flower_clean, output_prefix="flower1", save_dir=save_folder)
    
    print("\n--- Running Task 1 & 2: Notch and Gaussian Filters on animals.jpg ---")
    run_task1_and_task2(animals_clean, output_prefix="animals", save_dir=save_folder)
    
    # Run Task 3: Periodic Noise Removal
    run_task3(quiz4_folder=folder, save_dir=save_folder)
    
    print("\n[+] All tasks finished successfully! Output images are saved in 'images/result/'.")

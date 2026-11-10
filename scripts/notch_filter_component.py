"""
Frequency Domain Notch & Ideal Filters (Chapter 4, Gonzalez & Woods)
====================================================================

Assignment #4 Component: Notch / Ideal Filters in 2D Discrete Fourier Domain

1. Ideal / Center-Notch Low-Pass Filter (Assignment 1.a):
       H(u, v) = { 1  if D(u, v) <= r  (closer to center than r)
                 { 0  if D(u, v) >  r  (further from center than r)

2. Ideal / Center-Notch High-Pass Filter (Assignment 1.b):
       H(u, v) = { 0  if D(u, v) <= r  (closer to center than r)
                 { 1  if D(u, v) >  r  (further from center than r)

3. Notch Reject Filter (Periodic Noise Removal for Task 3):
       Attenuates specific periodic noise frequencies at (u_k, v_k) and
       their conjugate symmetric coordinates (-u_k, -v_k).

4. Notch Pass Filter:
       Extracts the periodic interference pattern: H_NP(u, v) = 1 - H_NR(u, v)

All spatial filtering functions accept either a 2D grayscale uint8 array or
a 3D RGB array, and return a uint8 array clipped to [0, 255].
"""

import numpy as np
from typing import Tuple, List, Union, Optional

__all__ = [
    "create_centered_distance_grid",
    "notch_lowpass_filter",
    "notch_highpass_filter",
    "ideal_lowpass_filter",
    "ideal_highpass_filter",
    "ideal_notch_reject_filter",
    "ideal_notch_pass_filter",
    "apply_filter_2d",
    "apply_frequency_filter",
    "compute_spectrum",
]


def create_centered_distance_grid(shape: Tuple[int, int]) -> Tuple[np.ndarray, Tuple[int, int]]:
    """
    Computes Euclidean distance D(u, v) from the shifted Fourier center (M//2, N//2).
    
    Parameters:
        shape: (M, N) height and width of the image / frequency grid.
        
    Returns:
        D: 2D array of shape (M, N) containing distances to (M//2, N//2).
        center: (cy, cx) tuple of center coordinates.
    """
    M, N = shape
    cy, cx = M // 2, N // 2
    u = np.arange(M) - cy
    v = np.arange(N) - cx
    V, U = np.meshgrid(v, u)
    D = np.sqrt(U.astype(np.float64)**2 + V.astype(np.float64)**2)
    return D, (cy, cx)


def notch_lowpass_filter(shape: Tuple[int, int], radius: float) -> np.ndarray:
    """
    Notch / Ideal Low-Pass Filter transfer function H(u, v) (Assignment 1.a).
    
    H(u, v) = 1 if D(u, v) <= radius, 0 if D(u, v) > radius.
    
    Parameters:
        shape: (M, N) dimensions of the frequency grid.
        radius: Cutoff radius r (e.g. 10, 50, 100).
        
    Returns:
        H: Float64 transfer function of shape (M, N).
    """
    D, _ = create_centered_distance_grid(shape)
    H = np.where(D <= radius, 1.0, 0.0)
    return H.astype(np.float64)


# Alias as ideal lowpass filter
ideal_lowpass_filter = notch_lowpass_filter


def notch_highpass_filter(shape: Tuple[int, int], radius: float) -> np.ndarray:
    """
    Notch / Ideal High-Pass Filter transfer function H(u, v) (Assignment 1.b).
    
    H(u, v) = 0 if D(u, v) <= radius, 1 if D(u, v) > radius.
    
    Parameters:
        shape: (M, N) dimensions of the frequency grid.
        radius: Cutoff radius r (e.g. 10, 50, 100).
        
    Returns:
        H: Float64 transfer function of shape (M, N).
    """
    D, _ = create_centered_distance_grid(shape)
    H = np.where(D > radius, 1.0, 0.0)
    return H.astype(np.float64)


# Alias as ideal highpass filter
ideal_highpass_filter = notch_highpass_filter


def ideal_notch_reject_filter(
    shape: Tuple[int, int],
    notches: List[Tuple[int, int]],
    radius: float = 5.0
) -> np.ndarray:
    """
    Ideal Notch Reject Filter for removing periodic noise spikes (Task 3).
    
    Attenuates symmetric pairs of noise frequencies at (cx + dx, cy + dy)
    and (cx - dx, cy - dy).
    
    Parameters:
        shape: (M, N) dimensions of the image.
        notches: List of (dx, dy) coordinate offsets relative to center (M//2, N//2).
        radius: Radius of the cylindrical notch rejection zone around each spike.
        
    Returns:
        H: Float64 2D notch reject transfer function.
    """
    M, N = shape
    cy, cx = M // 2, N // 2
    H = np.ones((M, N), dtype=np.float64)
    
    y = np.arange(M)
    x = np.arange(N)
    X, Y = np.meshgrid(x, y)
    
    for dx, dy in notches:
        # Positive and negative symmetric notch centers
        u1, v1 = cy + dy, cx + dx
        u2, v2 = cy - dy, cx - dx
        
        d1 = np.sqrt((X - v1)**2 + (Y - u1)**2)
        d2 = np.sqrt((X - v2)**2 + (Y - u2)**2)
        
        # Zero out frequencies within radius of either notch center
        H[d1 <= radius] = 0.0
        H[d2 <= radius] = 0.0
        
    return H


def ideal_notch_pass_filter(
    shape: Tuple[int, int],
    notches: List[Tuple[int, int]],
    radius: float = 5.0
) -> np.ndarray:
    """
    Ideal Notch Pass Filter: extracts only the periodic noise components.
    H_NP(u, v) = 1.0 - H_NR(u, v).
    """
    return 1.0 - ideal_notch_reject_filter(shape, notches, radius)


def apply_filter_2d(gray_image: np.ndarray, H: np.ndarray) -> np.ndarray:
    """
    Applies a centered frequency transfer function H(u, v) to a 2D grayscale image.
    
    Steps:
      1. Compute 2D Fast Fourier Transform: F = FFT2(image)
      2. Shift DC to center: F_shifted = FFTShift(F)
      3. Multiply by transfer function: G_shifted = F_shifted * H
      4. Shift back and compute Inverse FFT: g = IFFT2(IFFTShift(G_shifted))
      5. Extract real part and clip to [0, 255] uint8.
    """
    if gray_image.ndim != 2:
        raise ValueError("apply_filter_2d requires a 2D grayscale image.")
    
    # 2D FFT & shift
    F = np.fft.fft2(gray_image.astype(np.float64))
    F_shift = np.fft.fftshift(F)
    
    # Filtering
    G_shift = F_shift * H
    
    # Inverse FFT & unshift
    G = np.fft.ifftshift(G_shift)
    g = np.fft.ifft2(G)
    
    # Take real part and clip
    g_real = np.real(g)
    return np.clip(g_real, 0, 255).astype(np.uint8)


def apply_frequency_filter(image: np.ndarray, H: np.ndarray) -> np.ndarray:
    """
    Applies frequency domain filter H to a 2D (grayscale) or 3D (RGB) image.
    If 3D, processes each color channel independently.
    """
    if image.ndim == 2:
        return apply_filter_2d(image, H)
    elif image.ndim == 3:
        channels = [apply_filter_2d(image[..., c], H) for c in range(image.shape[2])]
        return np.stack(channels, axis=-1)
    else:
        raise ValueError(f"Unsupported image shape: {image.shape}")


def compute_spectrum(gray_image: np.ndarray) -> np.ndarray:
    """
    Computes centered log magnitude spectrum for visualization: log(1 + |F(u, v)|).
    """
    F = np.fft.fft2(gray_image.astype(np.float64))
    F_shift = np.fft.fftshift(F)
    mag = np.log1p(np.abs(F_shift))
    return mag

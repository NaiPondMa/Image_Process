"""
Frequency Domain Gaussian Filters (Chapter 4, Gonzalez & Woods)
===============================================================

Assignment #4 Component: Gaussian Filters in 2D Discrete Fourier Domain

1. Gaussian Low-Pass Filter (GLPF, Assignment 2.a):
       H(u, v) = exp( -D^2(u, v) / (2 * D0^2) )
       Smooth transitions with zero ringing (unlike Ideal/Notch filters).

2. Gaussian High-Pass Filter (GHPF, Assignment 2.b):
       H(u, v) = 1 - exp( -D^2(u, v) / (2 * D0^2) )
       Sharpening and edge enhancement without ringing artifacts.

3. Gaussian Notch Reject Filter (GNRF, Periodic Noise Removal for Task 3):
       Smoothly suppresses periodic sinusoidal noise spikes at (u_k, v_k)
       and (-u_k, -v_k) without ringing:
       H_k(u, v) = ( 1 - exp( -D_k^2 / (2 * D0^2) ) ) * ( 1 - exp( -D_{-k}^2 / (2 * D0^2) ) )

4. Gaussian Notch Pass Filter (GNPF):
       H_GNP(u, v) = 1 - H_GNR(u, v)

All spatial filtering functions accept either a 2D grayscale uint8 array or
a 3D RGB array, and return a uint8 array clipped to [0, 255].
"""

import numpy as np
from typing import Tuple, List, Union, Optional
from scripts.notch_filter_component import (
    create_centered_distance_grid,
    apply_filter_2d,
    apply_frequency_filter,
    compute_spectrum,
)

__all__ = [
    "gaussian_lowpass_filter",
    "gaussian_highpass_filter",
    "gaussian_notch_reject_filter",
    "gaussian_notch_pass_filter",
    "apply_gaussian_lowpass",
    "apply_gaussian_highpass",
    "apply_gaussian_notch_reject",
]


def gaussian_lowpass_filter(shape: Tuple[int, int], cutoff: float) -> np.ndarray:
    """
    Gaussian Low-Pass Filter transfer function H(u, v) (Assignment 2.a).
    
    H(u, v) = exp( -D^2(u, v) / (2 * D0^2) )
    
    Parameters:
        shape: (M, N) dimensions of the frequency grid.
        cutoff: Cutoff frequency D0 (e.g. 10, 50, 100).
        
    Returns:
        H: Float64 2D Gaussian low-pass transfer function with values in [0, 1].
    """
    if cutoff <= 0:
        raise ValueError("Cutoff frequency D0 must be strictly positive.")
    D, _ = create_centered_distance_grid(shape)
    H = np.exp(-(D**2) / (2.0 * (cutoff**2)))
    return H.astype(np.float64)


def gaussian_highpass_filter(shape: Tuple[int, int], cutoff: float) -> np.ndarray:
    """
    Gaussian High-Pass Filter transfer function H(u, v) (Assignment 2.b).
    
    H(u, v) = 1 - exp( -D^2(u, v) / (2 * D0^2) )
    
    Parameters:
        shape: (M, N) dimensions of the frequency grid.
        cutoff: Cutoff frequency D0 (e.g. 10, 50, 100).
        
    Returns:
        H: Float64 2D Gaussian high-pass transfer function with values in [0, 1].
    """
    return (1.0 - gaussian_lowpass_filter(shape, cutoff)).astype(np.float64)


def gaussian_notch_reject_filter(
    shape: Tuple[int, int],
    notches: List[Tuple[int, int]],
    d0: float = 6.0
) -> np.ndarray:
    """
    Gaussian Notch Reject Filter for periodic noise removal (Task 3).
    
    Attenuates noise spikes at (cx + dx, cy + dy) and (cx - dx, cy - dy)
    using smooth Gaussian profiles, completely avoiding ringing artifacts.
    
    Parameters:
        shape: (M, N) dimensions of the image.
        notches: List of (dx, dy) coordinate offsets relative to center (M//2, N//2).
        d0: Notch radius parameter (controls notch width).
        
    Returns:
        H: Float64 2D Gaussian notch reject transfer function.
    """
    M, N = shape
    cy, cx = M // 2, N // 2
    H = np.ones((M, N), dtype=np.float64)
    
    y = np.arange(M)
    x = np.arange(N)
    X, Y = np.meshgrid(x, y)
    
    two_d0_sq = 2.0 * (d0**2)
    
    for dx, dy in notches:
        u1, v1 = cy + dy, cx + dx
        u2, v2 = cy - dy, cx - dx
        
        d1_sq = (X - v1)**2 + (Y - u1)**2
        d2_sq = (X - v2)**2 + (Y - u2)**2
        
        # Smooth product of two Gaussian notches
        h_pair = (1.0 - np.exp(-d1_sq / two_d0_sq)) * (1.0 - np.exp(-d2_sq / two_d0_sq))
        H *= h_pair
        
    return H


def gaussian_notch_pass_filter(
    shape: Tuple[int, int],
    notches: List[Tuple[int, int]],
    d0: float = 6.0
) -> np.ndarray:
    """
    Gaussian Notch Pass Filter: extracts periodic noise pattern without DC.
    H_GNP(u, v) = 1.0 - H_GNR(u, v).
    """
    return 1.0 - gaussian_notch_reject_filter(shape, notches, d0)


def apply_gaussian_lowpass(image: np.ndarray, cutoff: float) -> np.ndarray:
    """Applies Gaussian low-pass filter to a 2D or 3D image."""
    shape = image.shape[:2]
    H = gaussian_lowpass_filter(shape, cutoff)
    return apply_frequency_filter(image, H)


def apply_gaussian_highpass(image: np.ndarray, cutoff: float) -> np.ndarray:
    """Applies Gaussian high-pass filter to a 2D or 3D image."""
    shape = image.shape[:2]
    H = gaussian_highpass_filter(shape, cutoff)
    return apply_frequency_filter(image, H)


def apply_gaussian_notch_reject(
    image: np.ndarray,
    notches: List[Tuple[int, int]],
    d0: float = 6.0
) -> np.ndarray:
    """Applies Gaussian notch reject filter to a 2D or 3D image."""
    shape = image.shape[:2]
    H = gaussian_notch_reject_filter(shape, notches, d0)
    return apply_frequency_filter(image, H)

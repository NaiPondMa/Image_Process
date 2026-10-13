
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

def _windows(image, ksize):
    """All ksize x ksize neighbourhoods, shape (H, W, ksize, ksize)."""
    pad = ksize // 2
    padded = np.pad(image, pad, mode="reflect")
    return sliding_window_view(padded, (ksize, ksize))
"""
median filter 
"""

def median_filter(image, kernel_size=3):
    """
    Applies a median filter to a 2D grayscale image array using NumPy.
    
    Parameters:
        image (np.ndarray): 2D array representing the grayscale image.
        kernel_size (int): Size of the square neighborhood (must be odd).
    
    Returns:
        np.ndarray: Filtered image array.
    """
    pad = kernel_size // 2
    height, width = image.shape
    
    """ padding session """
    # Pad the image edges (e.g., repeating edge values or zero-padding)
    padded_image = np.pad(image, pad, mode='edge')
    output = np.zeros_like(image)
    
    """ having pixel re-arrange and find median and replace at the center """
    # Iterate over every pixel in the image
    for i in range(height):
        for j in range(width):
            # Extract the local neighborhood window
            window = padded_image[i:i + kernel_size, j:j + kernel_size]
            # Replace the central pixel with the median value
            output[i, j] = np.median(window)
            
    return output

"""
adaptive median filter
"""

def adaptive_median_filter(image, max_ksize=7):
    """Adaptive median filter.
 
    For each pixel, starting at a 3x3 window:
      Stage A: if z_min < z_med < z_max, the median is not an impulse -> go to B;
               otherwise enlarge the window (up to max_ksize), else output z_med.
      Stage B: if z_min < z_xy < z_max, the pixel itself is not an impulse ->
               keep it unchanged; otherwise output z_med.
 
    Only noisy pixels get replaced, so detail is preserved better than with a
    plain median filter, and it copes with denser noise.
    """
    img = np.asarray(image)
    if img.ndim == 3:
        return np.stack([adaptive_median_filter(img[..., c], max_ksize)
                         for c in range(img.shape[2])], axis=-1)
 
    f = img.astype(np.int32)
    out = f.copy()
    undecided = np.ones(f.shape, dtype=bool)
 
    for k in range(3, max_ksize + 1, 2):
        win = _windows(f, k)
        z_min = win.min(axis=(-2, -1))
        z_max = win.max(axis=(-2, -1))
        z_med = np.median(win, axis=(-2, -1))
 
        median_ok = (z_min < z_med) & (z_med < z_max)        # Stage A passes
        pixel_ok = (z_min < f) & (f < z_max)                  # Stage B test
 
        a = undecided & median_ok
        out[a & ~pixel_ok] = z_med[a & ~pixel_ok]             # impulse -> median
        # (a & pixel_ok) keeps the original value
        undecided &= ~a
 
        if k == max_ksize:                                    # window maxed out
            out[undecided] = z_med[undecided]
 
    return out.astype(img.dtype)

"""
gaussian filter

"""
def gaussian_kernel(ksize=5, sigma=1.0):
    ax = np.arange(ksize) - ksize // 2
    g = np.exp(-(ax ** 2) / (2 * sigma ** 2))
    k = np.outer(g, g)
    return k / k.sum()

def convolve2d(image, kernel):
    """Correlation with reflect padding (kernel is symmetric -> convolution)."""
    img = np.asarray(image, dtype=np.float64)
    kh, kw = kernel.shape
    padded = np.pad(img, ((kh // 2,) * 2, (kw // 2,) * 2), mode="reflect")
    out = np.zeros_like(img)
    for i in range(kh):
        for j in range(kw):
            out += kernel[i, j] * padded[i:i + img.shape[0], j:j + img.shape[1]]
    return out

def gaussian_filter(image, ksize=5, sigma=1.0):
    """Weighted smoothing; better than a box filter for Gaussian noise."""
    out = convolve2d(image, gaussian_kernel(ksize, sigma))
    return np.clip(out, 0, 255).astype(np.uint8)
 
 
def estimate_noise_variance(image):
    """Robust noise variance estimate (Immerkaer's method)."""
    img = np.asarray(image, dtype=np.float64)
    M = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], dtype=np.float64)
    r = convolve2d(img, M)[1:-1, 1:-1]
    sigma = np.sqrt(np.pi / 2) * np.abs(r).mean() / 6.0
    return sigma ** 2
 
 
def adaptive_local_noise_filter(image, ksize=7, noise_var=None):
    """Adaptive local noise reduction filter (Gonzalez & Woods, sec. 5.3).
 
        f_hat = g - (var_noise / var_local) * (g - mean_local)
 
    Flat areas (var_local ~ var_noise) -> replaced by the local mean (smoothed).
    Edges   (var_local >> var_noise)   -> left almost unchanged (kept sharp).
    """
    g = np.asarray(image, dtype=np.float64)
    if noise_var is None:
        noise_var = estimate_noise_variance(g)
    win = _windows(g, ksize)
    m_local = win.mean(axis=(-2, -1))
    v_local = win.var(axis=(-2, -1))
    ratio = np.minimum(noise_var / np.maximum(v_local, 1e-6), 1.0)  # ratio <= 1
    out = g - ratio * (g - m_local)
    return np.clip(out, 0, 255).astype(np.uint8)

"""
Sharpening Spatial Filters (Chapter 3, Gonzalez & Woods) — pure NumPy
=====================================================================

1. Laplacian (2nd derivative)                     slides 136-145
       g = f + c * lap(f),  c = -1 (negative centre) / +1 (positive centre)

2. Sobel gradient (1st derivative)                slides 146-152
       |grad f| ~ |Gx| + |Gy|                         (Eq. 3.49)

3. Combining spatial enhancement methods          slides 153-159, Fig. 3.46
       (a) f                  original
       (b) lap(f)             Laplacian -> fine detail
       (d) |grad f|           Sobel gradient -> strong edges
       (e) average 5x5 of (d) smoothed gradient
       (f) (b) x (e)          Laplacian masked by smoothed gradient
       (g) f + (f)            sharpened (strong at edges, weak in flat/noisy areas)
       (h) power-law, gamma   increase dynamic range

4. Gray-level helpers: power_law (gamma), contrast_stretch.

All functions take a grayscale uint8 image (2-D) and, unless stated
otherwise, return uint8 clipped to [0, 255].
"""

import numpy as np

__all__ = [
    "LAPLACIAN_KERNELS", "SOBEL_X", "SOBEL_Y",
    "filter2d", "laplacian", "scale_for_display", "scale_laplacian_for_display",
    "laplacian_sharpen", "laplacian_sharpen_composite_kernel",
    "sobel_gradient", "average_filter",
    "combined_sharpen", "power_law", "contrast_stretch",
]

# ---------------------------------------------------------------------------
# Kernels
# ---------------------------------------------------------------------------
LAPLACIAN_KERNELS = {
    # Fig. 3.39(a): 4-neighbour, negative centre
    "4neg": np.array([[0,  1, 0],
                      [1, -4, 1],
                      [0,  1, 0]], dtype=np.float64),
    # Fig. 3.39(b): 8-neighbour (diagonals), negative centre
    "8neg": np.array([[1,  1, 1],
                      [1, -8, 1],
                      [1,  1, 1]], dtype=np.float64),
    # Fig. 3.39(c): 4-neighbour, positive centre
    "4pos": np.array([[ 0, -1,  0],
                      [-1,  4, -1],
                      [ 0, -1,  0]], dtype=np.float64),
    # Fig. 3.39(d): 8-neighbour, positive centre
    "8pos": np.array([[-1, -1, -1],
                      [-1,  8, -1],
                      [-1, -1, -1]], dtype=np.float64),
}

# Sobel masks (Fig. 3.44)
SOBEL_X = np.array([[-1, -2, -1],
                    [ 0,  0,  0],
                    [ 1,  2,  1]], dtype=np.float64)
SOBEL_Y = np.array([[-1, 0, 1],
                    [-2, 0, 2],
                    [-1, 0, 1]], dtype=np.float64)


def _to_uint8(x):
    return np.clip(x, 0, 255).astype(np.uint8)


def _sign(kernel_type):
    """c in Eq. 3.40: subtract for negative centre, add for positive centre."""
    return -1.0 if kernel_type.endswith("neg") else 1.0


# ---------------------------------------------------------------------------
# Spatial filtering (correlation) with padding
# ---------------------------------------------------------------------------
def filter2d(image, kernel, padding="reflect"):
    """Apply a 2-D mask to a grayscale image. Returns float64."""
    image = np.asarray(image, dtype=np.float64)
    kh, kw = kernel.shape
    ph, pw = kh // 2, kw // 2
    padded = np.pad(image, ((ph, ph), (pw, pw)), mode=padding)
    out = np.zeros_like(image)
    for i in range(kh):
        for j in range(kw):
            w = kernel[i, j]
            if w != 0:
                out += w * padded[i:i + image.shape[0], j:j + image.shape[1]]
    return out


def scale_for_display(x):
    """Map any float image (may contain negatives) to 0-255 uint8 for viewing."""
    x = np.asarray(x, dtype=np.float64)
    x = x - x.min()
    if x.max() > 0:
        x = x * (255.0 / x.max())
    return x.astype(np.uint8)


scale_laplacian_for_display = scale_for_display      # old name still works


# ---------------------------------------------------------------------------
# 1. Laplacian
# ---------------------------------------------------------------------------
def laplacian(image, kernel_type="4neg", padding="reflect"):
    """Raw Laplacian (float64, contains negative values)."""
    if kernel_type not in LAPLACIAN_KERNELS:
        raise ValueError(f"kernel_type must be one of {list(LAPLACIAN_KERNELS)}")
    return filter2d(image, LAPLACIAN_KERNELS[kernel_type], padding)


def laplacian_sharpen(image, kernel_type="4neg", strength=1.0, padding="reflect"):
    """Eq. 3.40:  g = f + c * strength * lap(f)."""
    f = np.asarray(image, dtype=np.float64)
    return _to_uint8(f + _sign(kernel_type) * strength * laplacian(f, kernel_type, padding))


def laplacian_sharpen_composite_kernel(image, kernel_type="4neg", padding="reflect"):
    """Eq. 3.41: the same result in one pass with a single composite mask
    (Fig. 3.41), e.g. [0 -1 0; -1 5 -1; 0 -1 0] or [-1 -1 -1; -1 9 -1; -1 -1 -1]."""
    L = LAPLACIAN_KERNELS[kernel_type]
    identity = np.zeros_like(L)
    identity[1, 1] = 1.0
    return _to_uint8(filter2d(image, identity + _sign(kernel_type) * L, padding))


# ---------------------------------------------------------------------------
# 2. Sobel gradient and averaging filter
# ---------------------------------------------------------------------------
def sobel_gradient(image, padding="reflect"):
    """Gradient magnitude approximation |Gx| + |Gy| (Eq. 3.49). Returns float64."""
    gx = filter2d(image, SOBEL_X, padding)
    gy = filter2d(image, SOBEL_Y, padding)
    return np.abs(gx) + np.abs(gy)


def average_filter(image, ksize=5, padding="reflect"):
    """Box (averaging) filter of size ksize x ksize. Returns float64."""
    kernel = np.full((ksize, ksize), 1.0 / (ksize * ksize))
    return filter2d(image, kernel, padding)


# ---------------------------------------------------------------------------
# 3. Combining spatial enhancement methods (Fig. 3.46)
# ---------------------------------------------------------------------------
def combined_sharpen(image, kernel_type="8pos", strength=1.0, smooth_size=5,
                     gamma=None, return_steps=False):
    """Laplacian masked by the smoothed Sobel gradient, added to the original.

    Parameters
    ----------
    kernel_type : Laplacian mask; the slides use Fig. 3.39(d) = '8pos'.
    strength    : how much of the masked Laplacian to add (try 1-4).
    smooth_size : averaging filter size for the gradient (slides: 5).
    gamma       : if given, apply power-law s = 255*(r/255)**gamma at the end
                  (slides: gamma = 0.5, c = 1). None = skip.
    return_steps: also return a dict of the intermediate images (b)-(h),
                  scaled to uint8, for figures in the report.
    """
    f = np.asarray(image, dtype=np.float64)

    lap = laplacian(f, kernel_type)                      # (b) fine detail
    grad = sobel_gradient(f)                             # (d) strong edges
    grad_smooth = average_filter(grad, smooth_size)      # (e) smoothed gradient

    # Normalise the smoothed gradient to [0, 1] so it acts as a weight:
    # ~1 on strong edges, ~0 in flat (noisy) regions.
    weight = grad_smooth / max(grad_smooth.max(), 1e-12)
    product = lap * weight                               # (f) masked Laplacian

    g = _to_uint8(f + _sign(kernel_type) * strength * product)   # (g) sharpened
    out = power_law(g, gamma) if gamma is not None else g        # (h)

    if return_steps:
        steps = {
            "(b) Laplacian": scale_for_display(lap),
            "(c) Original + Laplacian": laplacian_sharpen(f, kernel_type, strength),
            "(d) Sobel gradient": scale_for_display(grad),
            f"(e) Gradient smoothed {smooth_size}x{smooth_size}": scale_for_display(grad_smooth),
            "(f) Laplacian x smoothed gradient": scale_for_display(product),
            "(g) Original + (f)": g,
        }
        if gamma is not None:
            steps[f"(h) Power-law, gamma={gamma}"] = out
        return out, steps
    return out


# ---------------------------------------------------------------------------
# 4. Gray-level transformations (for the final dynamic-range step)
# ---------------------------------------------------------------------------
def power_law(image, gamma=0.5, c=1.0):
    """s = c * r**gamma on intensities normalised to [0, 1] (Eq. 3.2-3).
    gamma < 1 brightens dark regions, gamma > 1 darkens bright regions."""
    r = np.asarray(image, dtype=np.float64) / 255.0
    return _to_uint8(255.0 * c * np.power(r, gamma))


def contrast_stretch(image, low_pct=1, high_pct=99):
    """Linear contrast stretch between two percentiles to the full 0-255 range
    (piecewise-linear contrast stretching, slides 32-35)."""
    f = np.asarray(image, dtype=np.float64)
    lo, hi = np.percentile(f, (low_pct, high_pct))
    return _to_uint8((f - lo) * 255.0 / max(hi - lo, 1e-12))
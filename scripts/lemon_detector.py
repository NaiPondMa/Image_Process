"""
Lemon Detection and Counting Pipeline Component
Term Project - Digital Image Processing

Unified processing pipeline for detecting and counting lemons in tree foliage images:
1. Color Space Conversion (RGB -> HSV / HSI)
2. Color Slicing / Binarization (Yellow & optional Unripe Green Hue isolation)
3. Morphological Cleaning (Opening to remove specks, Closing to seal holes)
4. Overlap Separation (Euclidean Distance Transform + Marker-Controlled Watershed)
5. Connected Component Analysis & Feature Filtering (Area, Circularity, Aspect Ratio)
6. Object Counting & Visualization
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any, List, Tuple, Optional


def rgb_to_hsv(rgb_image: np.ndarray) -> np.ndarray:
    """
    Converts an RGB image to the HSV color space.
    
    In OpenCV, H is mapped to [0, 179], S to [0, 255], and V to [0, 255].
    Hue isolates the chromatic purity (yellow vs. green leaves) independent
    of shadows and direct sun glare captured in Value (V).
    """
    return cv2.cvtColor(rgb_image, cv2.COLOR_RGB2HSV)


def color_slice(
    hsv_image: np.ndarray,
    h_range: Tuple[int, int] = (13, 35),
    s_min: int = 50,
    v_min: int = 50,
    include_green: bool = False,
    green_h_range: Tuple[int, int] = (40, 56),
    green_s_min: int = 120,
    green_v_min: int = 95
) -> np.ndarray:
    """
    Applies color slicing / thresholding in HSV color space.
    
    Parameters:
        hsv_image: Input HSV image (H: 0-179, S: 0-255, V: 0-255).
        h_range: (min_hue, max_hue) for ripe yellow lemons.
                 For images with sun-disturbed yellow leaves (lemon_tree3),
                 restricting h_max to 25-26 effectively suppresses yellow-green foliage.
        s_min: Minimum saturation to exclude grayish background / reflections.
        v_min: Minimum brightness value to exclude deep shadows.
        include_green: If True, also includes unripe green lemons.
        green_h_range: (min_hue, max_hue) for green lemon range.
        green_s_min: Stricter saturation threshold for green lemons.
        green_v_min: Stricter value threshold for green lemons.
        
    Returns:
        Binary mask (uint8, 0 or 255) containing segmented lemon regions.
    """
    H = hsv_image[:, :, 0]
    S = hsv_image[:, :, 1]
    V = hsv_image[:, :, 2]
    
    # Yellow ripe lemon mask
    mask_yellow = (
        (H >= h_range[0]) & (H <= h_range[1]) &
        (S >= s_min) &
        (V >= v_min)
    )
    
    if include_green:
        # Green unripe lemon mask (stricter S and V to avoid dark background leaves)
        mask_green = (
            (H >= green_h_range[0]) & (H <= green_h_range[1]) &
            (S >= green_s_min) &
            (V >= green_v_min)
        )
        combined = mask_yellow | mask_green
    else:
        combined = mask_yellow
        
    return (combined.astype(np.uint8)) * 255


def morphological_clean(
    binary_mask: np.ndarray,
    ksize: int = 5,
    open_iters: int = 2,
    close_iters: int = 2
) -> np.ndarray:
    """
    Cleans binary mask noise using morphological Opening and Closing.
    
    - Opening (Erosion followed by Dilation): Removes thin stems, leaf specks,
      and isolated background noise.
    - Closing (Dilation followed by Erosion): Fills internal holes caused by
      specular reflections or texture variation on lemon rinds.
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    
    # 1. Opening: Removes isolated false positives & thin connectors
    opened = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel, iterations=open_iters)
    
    # 2. Closing: Fills internal holes and bridges small cracks
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel, iterations=close_iters)
    
    return closed


def separate_overlapping_watershed(
    cleaned_mask: np.ndarray,
    bgr_image: np.ndarray,
    peak_ksize: int = 25,
    min_dist_val: float = 12.0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Separates overlapping and touching lemons using Euclidean Distance Transform
    and Marker-Controlled Watershed segmentation.
    
    Parameters:
        cleaned_mask: Binary mask after morphological cleaning.
        bgr_image: Original color image in BGR format (required by cv2.watershed).
        peak_ksize: Neighborhood kernel diameter for morphological peak dilation.
        min_dist_val: Minimum distance transform value to qualify as a lemon core.
        
    Returns:
        dist: Normalized float32 Euclidean distance transform map.
        markers: Watershed label matrix (markers > 1 are segmented objects).
        peaks: Binary mask of detected local maxima seeds.
    """
    # 1. Euclidean Distance Transform (L2 metric)
    dist = cv2.distanceTransform(cleaned_mask, cv2.DIST_L2, 5)
    
    # 2. Local Peak Detection via Morphological Dilation
    # A pixel is a local maximum if dist == dilate(dist) and dist >= min_dist_val
    peak_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (peak_ksize, peak_ksize))
    dist_dilated = cv2.dilate(dist, peak_kernel)
    peaks = (dist == dist_dilated) & (dist >= min_dist_val)
    
    # 3. Create Sure Foreground Markers
    # Dilate peaks slightly so each seed has a solid core
    peaks_dilated = cv2.dilate(peaks.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    num_markers, markers = cv2.connectedComponents(peaks_dilated)
    
    # 4. Create Sure Background and Unknown Boundary Region
    # Dilate mask to ensure background marker definitely covers non-lemon pixels
    sure_bg = cv2.dilate(cleaned_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)), iterations=2)
    unknown = cv2.subtract(sure_bg, peaks_dilated * 255)
    
    # 5. Format markers for cv2.watershed:
    # 0 = unknown boundary, 1 = sure background, 2..N = object seed markers
    markers = markers + 1
    markers[unknown == 255] = 0
    
    # 6. Apply Marker-Controlled Watershed
    ws_markers = cv2.watershed(bgr_image, markers)
    
    return dist, ws_markers, peaks


def filter_connected_components(
    ws_markers: np.ndarray,
    cleaned_mask: np.ndarray,
    hsv_image: np.ndarray,
    min_area: float = 300.0,
    max_area: Optional[float] = None,
    min_circularity: float = 0.20,
    max_aspect_ratio: float = 3.2,
    unripe_min_circularity: float = 0.45,
    unripe_max_aspect_ratio: float = 1.6
) -> Tuple[List[Dict[str, Any]], np.ndarray]:
    """
    Extracts features for each watershed region using Connected Component Analysis
    and filters out non-lemon artifacts based on area and shape compactness.
    
    Distinguishes ripe yellow lemons from green unripe fruits:
    - Ripe lemons: evaluated with min_circularity and max_aspect_ratio.
    - Unripe lemons (Mean Hue > 36): subject to stricter circularity and aspect ratio
      constraints to strictly eliminate flat, elongated green foliage.
    """
    valid_lemons = []
    filtered_mask = np.zeros_like(cleaned_mask, dtype=np.uint8)
    H = hsv_image[:, :, 0]
    
    unique_labels = [label for label in np.unique(ws_markers) if label > 1]
    
    for label in unique_labels:
        # Extract binary region belonging to this marker, bounded by cleaned mask
        region = ((ws_markers == label) & (cleaned_mask > 0)).astype(np.uint8)
        
        contours, _ = cv2.findContours(region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
            
        cnt = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(cnt)
        
        # Area filtering
        if area < min_area:
            continue
        if max_area is not None and area > max_area:
            continue
            
        # Perimeter and Circularity calculation
        perimeter = cv2.arcLength(cnt, True)
        if perimeter == 0:
            continue
        circularity = (4.0 * np.pi * area) / (perimeter * perimeter)
        
        # Bounding box & Aspect ratio
        x, y, w, h = cv2.boundingRect(cnt)
        aspect_ratio = max(w, h) / max(1.0, min(w, h))
        
        # Measure mean Hue to classify ripe vs unripe
        c_mask = np.zeros_like(cleaned_mask)
        cv2.drawContours(c_mask, [cnt], -1, 255, -1)
        mean_h = float(np.mean(H[c_mask == 255])) if np.any(c_mask == 255) else 20.0
        is_unripe = (mean_h > 36.0)
        
        # Shape filter depending on maturity state
        if is_unripe:
            if circularity < unripe_min_circularity or aspect_ratio > unripe_max_aspect_ratio:
                continue
        else:
            if circularity < min_circularity or aspect_ratio > max_aspect_ratio:
                continue
            
        # Centroid calculation via spatial moments
        M = cv2.moments(cnt)
        if M["m00"] > 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
        else:
            cx, cy = x + w // 2, y + h // 2
            
        lemon_id = len(valid_lemons) + 1
        valid_lemons.append({
            "id": lemon_id,
            "center": (cx, cy),
            "bbox": (x, y, w, h),
            "area": area,
            "perimeter": perimeter,
            "circularity": circularity,
            "aspect_ratio": aspect_ratio,
            "mean_hue": mean_h,
            "is_unripe": is_unripe,
            "contour": cnt
        })
        
        filtered_mask = cv2.bitwise_or(filtered_mask, region * 255)
        
    return valid_lemons, filtered_mask


class LemonDetector:
    """
    Unified Lemon Detection System implementing the end-to-end pipeline:
    RGB Input -> Gaussian Smoothing -> RGB to HSV -> Color Slicing ->
    Morphological Cleaning -> Overlap Separation (Watershed) ->
    Connected Component Feature Filtering -> Final Counting Annotation.
    """
    
    def __init__(self, default_config: Optional[Dict[str, Any]] = None):
        self.default_config = {
            "blur_ksize": 5,
            "h_range": (13, 35),
            "s_min": 50,
            "v_min": 50,
            "include_green": False,
            "green_h_range": (40, 56),
            "green_s_min": 120,
            "green_v_min": 95,
            "morph_ksize": 5,
            "open_iters": 2,
            "close_iters": 2,
            "peak_ksize": 25,
            "min_dist_val": 12.0,
            "min_area": 350.0,
            "max_area": None,
            "min_circularity": 0.20,
            "max_aspect_ratio": 3.2,
            "unripe_min_circularity": 0.45,
            "unripe_max_aspect_ratio": 1.6
        }
        if default_config:
            self.default_config.update(default_config)
            
    def process(self, image_input: Any, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Executes the unified detection algorithm on an input image.
        
        Parameters:
            image_input: File path (str) or RGB numpy array (H, W, 3).
            config: Optional parameter dictionary to tune specific thresholds.
            
        Returns:
            Dictionary containing final count, lemon metrics, and intermediate images.
        """
        # Resolve configuration
        cfg = self.default_config.copy()
        if config:
            cfg.update(config)
            
        # 1. Load image
        if isinstance(image_input, str):
            bgr = cv2.imread(image_input, cv2.IMREAD_COLOR)
            if bgr is None:
                raise FileNotFoundError(f"Failed to read image at '{image_input}'")
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        else:
            rgb = image_input.copy()
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            
        # 2. Preprocessing / Smoothing (Gaussian filter to reduce foliage speckles)
        blur_k = cfg.get("blur_ksize", 5)
        if blur_k > 1:
            blurred = cv2.GaussianBlur(rgb, (blur_k, blur_k), 0)
        else:
            blurred = rgb.copy()
            
        # 3. Color Space Conversion (RGB -> HSV)
        hsv = rgb_to_hsv(blurred)
        
        # 4. Color Slicing / Binarization
        raw_mask = color_slice(
            hsv,
            h_range=cfg.get("h_range", (13, 35)),
            s_min=cfg.get("s_min", 50),
            v_min=cfg.get("v_min", 50),
            include_green=cfg.get("include_green", False),
            green_h_range=cfg.get("green_h_range", (40, 56)),
            green_s_min=cfg.get("green_s_min", 120),
            green_v_min=cfg.get("green_v_min", 95)
        )
        
        # 5. Morphological Cleaning
        cleaned_mask = morphological_clean(
            raw_mask,
            ksize=cfg.get("morph_ksize", 5),
            open_iters=cfg.get("open_iters", 2),
            close_iters=cfg.get("close_iters", 2)
        )
        
        # 6. Overlap Separation (Distance Transform + Watershed)
        dist_map, ws_markers, peaks_mask = separate_overlapping_watershed(
            cleaned_mask,
            bgr,
            peak_ksize=cfg.get("peak_ksize", 25),
            min_dist_val=cfg.get("min_dist_val", 12.0)
        )
        
        # 7. Connected Component Analysis & Feature Filtering
        lemons, filtered_mask = filter_connected_components(
            ws_markers,
            cleaned_mask,
            hsv,
            min_area=cfg.get("min_area", 350.0),
            max_area=cfg.get("max_area", None),
            min_circularity=cfg.get("min_circularity", 0.20),
            max_aspect_ratio=cfg.get("max_aspect_ratio", 3.2),
            unripe_min_circularity=cfg.get("unripe_min_circularity", 0.45),
            unripe_max_aspect_ratio=cfg.get("unripe_max_aspect_ratio", 1.6)
        )
        
        # 8. Render Annotated Output Image
        annotated = rgb.copy()
        ripe_count = sum(1 for l in lemons if not l.get("is_unripe", False))
        unripe_count = sum(1 for l in lemons if l.get("is_unripe", False))
        
        for lemon in lemons:
            lid = lemon["id"]
            cx, cy = lemon["center"]
            x, y, w, h = lemon["bbox"]
            cnt = lemon["contour"]
            is_unripe = lemon.get("is_unripe", False)
            
            if is_unripe:
                # Cyan/Lime box & label for unripe fruit
                contour_color = (0, 255, 200)
                box_color = (0, 230, 255)
                label_text = f"#{lid} (Unripe)"
            else:
                # Green contour & Golden yellow box for ripe lemon
                contour_color = (30, 255, 30)
                box_color = (255, 215, 0)
                label_text = f"#{lid}"
                
            cv2.drawContours(annotated, [cnt], -1, contour_color, 2)
            cv2.rectangle(annotated, (x, y), (x + w, y + h), box_color, 2)
            cv2.circle(annotated, (cx, cy), 4, (255, 0, 0), -1)
            
            # Text shadow for maximum legibility
            cv2.putText(
                annotated, label_text, (x, max(15, y - 5)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3
            )
            cv2.putText(
                annotated, label_text, (x, max(15, y - 5)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1
            )
            
        # Draw header banner with total count
        if unripe_count > 0:
            banner_text = f"Lemons: {len(lemons)} ({ripe_count} Ripe + {unripe_count} Unripe)"
            banner_w = min(rgb.shape[1], 420)
        else:
            banner_text = f"Lemons Detected: {len(lemons)}"
            banner_w = min(rgb.shape[1], 330)
            
        cv2.rectangle(annotated, (0, 0), (banner_w, 36), (0, 0, 0), -1)
        cv2.putText(
            annotated, banner_text, (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2
        )
        
        return {
            "count": len(lemons),
            "ripe_count": ripe_count,
            "unripe_count": unripe_count,
            "lemons": lemons,
            "config_used": cfg,
            "images": {
                "original_rgb": rgb,
                "hsv_hue": hsv[:, :, 0],
                "raw_mask": raw_mask,
                "cleaned_mask": cleaned_mask,
                "dist_transform": dist_map,
                "peaks": peaks_mask,
                "watershed_markers": ws_markers,
                "filtered_mask": filtered_mask,
                "annotated": annotated
            }
        }


def plot_pipeline_stages(
    result: Dict[str, Any],
    title: str = "Lemon Detection Processing Pipeline",
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Renders a 2x3 comprehensive visual grid of all pipeline intermediate stages.
    """
    imgs = result["images"]
    count = result["count"]
    ripe_c = result.get("ripe_count", count)
    unripe_c = result.get("unripe_count", 0)
    
    if unripe_c > 0:
        hdr_count = f"{count} Lemons ({ripe_c} Ripe + {unripe_c} Unripe)"
    else:
        hdr_count = f"{count} Lemons"
        
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    fig.suptitle(f"{title} - Total: {hdr_count}", fontsize=15, fontweight="bold")
    
    # 1. Original RGB
    axes[0, 0].imshow(imgs["original_rgb"])
    axes[0, 0].set_title("(1) Input RGB Image")
    axes[0, 0].axis("off")
    
    # 2. Hue Channel
    axes[0, 1].imshow(imgs["hsv_hue"], cmap="hsv", vmin=0, vmax=179)
    axes[0, 1].set_title("(2) HSV Hue Channel")
    axes[0, 1].axis("off")
    
    # 3. Raw Color Slicing Mask
    axes[0, 2].imshow(imgs["raw_mask"], cmap="gray")
    axes[0, 2].set_title("(3) Color Slicing Mask")
    axes[0, 2].axis("off")
    
    # 4. Morphological Cleaning
    axes[1, 0].imshow(imgs["cleaned_mask"], cmap="gray")
    axes[1, 0].set_title("(4) Morphological Opening + Closing")
    axes[1, 0].axis("off")
    
    # 5. Distance Transform
    norm_dist = cv2.normalize(imgs["dist_transform"], None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    axes[1, 1].imshow(norm_dist, cmap="jet")
    axes[1, 1].set_title("(5) Distance Transform & Watershed")
    axes[1, 1].axis("off")
    
    # 6. Final Annotated Detection
    axes[1, 2].imshow(imgs["annotated"])
    axes[1, 2].set_title(f"(6) Final Output: {hdr_count}")
    axes[1, 2].axis("off")
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved pipeline visualization to '{save_path}'")
        
    return fig

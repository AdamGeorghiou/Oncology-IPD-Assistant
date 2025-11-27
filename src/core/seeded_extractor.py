# src/core/seeded_extractor.py
"""
Enhanced Seeded Curve Extraction

Click on a curve to extract it. Improvements:
1. Multi-pixel sampling (5x5 region) for robust color detection
2. Delta-E color distance instead of fixed HSV bounds
3. Region growing algorithm to follow connected pixels
4. Handles dashed/dotted lines better

Author: IPD Assistant
"""

import cv2
import numpy as np
import pandas as pd
from typing import Optional, Tuple, List
from scipy.ndimage import label
from scipy.interpolate import interp1d


# =============================================================================
# COLOR SAMPLING
# =============================================================================

def get_hsv_at_point(img_bgr: np.ndarray, x: int, y: int, sample_size: int = 1) -> Optional[np.ndarray]:
    """
    Returns the HSV color at the specific (x, y) coordinate.
    
    Args:
        img_bgr: BGR image
        x, y: Click coordinates
        sample_size: 1 for single pixel, 5 for 5x5 region (recommended)
    
    Returns:
        HSV array [H, S, V] or None if out of bounds
    """
    if img_bgr is None:
        return None
    
    h, w = img_bgr.shape[:2]
    
    if x < 0 or x >= w or y < 0 or y >= h:
        return None
    
    if sample_size == 1:
        # Single pixel (legacy behavior)
        pixel_bgr = img_bgr[y, x]
        pixel_bgr_reshaped = np.array([[pixel_bgr]], dtype=np.uint8)
        pixel_hsv = cv2.cvtColor(pixel_bgr_reshaped, cv2.COLOR_BGR2HSV)
        return pixel_hsv[0][0]
    else:
        # Multi-pixel sampling
        return sample_region_color(img_bgr, x, y, sample_size)


def sample_region_color(img_bgr: np.ndarray, x: int, y: int, size: int = 5) -> Optional[np.ndarray]:
    """
    Sample a region around (x, y) and return the dominant color.
    
    Uses median of saturated pixels to be robust against:
    - Antialiasing (blended edge pixels)
    - Noise
    - Accidentally clicking slightly off the line
    """
    h, w = img_bgr.shape[:2]
    half = size // 2
    
    # Define region bounds
    y1 = max(0, y - half)
    y2 = min(h, y + half + 1)
    x1 = max(0, x - half)
    x2 = min(w, x + half + 1)
    
    region_bgr = img_bgr[y1:y2, x1:x2]
    
    if region_bgr.size == 0:
        return None
    
    # Convert to HSV
    region_hsv = cv2.cvtColor(region_bgr, cv2.COLOR_BGR2HSV)
    pixels = region_hsv.reshape(-1, 3)
    
    # Filter for saturated pixels (actual line colors, not background)
    saturated_mask = pixels[:, 1] > 30
    saturated_pixels = pixels[saturated_mask]
    
    if len(saturated_pixels) == 0:
        # Fallback to all pixels if no saturated ones found
        saturated_pixels = pixels
    
    # Use median for robustness
    median_hsv = np.median(saturated_pixels, axis=0).astype(np.uint8)
    
    return median_hsv


def get_color_at_point_lab(img_bgr: np.ndarray, x: int, y: int, sample_size: int = 5) -> Optional[np.ndarray]:
    """
    Get color in LAB space for Delta-E calculations.
    """
    if img_bgr is None:
        return None
    
    h, w = img_bgr.shape[:2]
    half = sample_size // 2
    
    y1 = max(0, y - half)
    y2 = min(h, y + half + 1)
    x1 = max(0, x - half)
    x2 = min(w, x + half + 1)
    
    region_bgr = img_bgr[y1:y2, x1:x2]
    region_lab = cv2.cvtColor(region_bgr, cv2.COLOR_BGR2LAB)
    
    # Median color
    pixels = region_lab.reshape(-1, 3)
    median_lab = np.median(pixels, axis=0)
    
    return median_lab


# =============================================================================
# COLOR DISTANCE METRICS
# =============================================================================

def delta_e_cie76(lab1: np.ndarray, lab2: np.ndarray) -> float:
    """
    Calculate CIE76 Delta-E color difference.
    
    This is a perceptually uniform color distance metric.
    Lower = more similar. Generally:
    - < 1: Not perceptible
    - 1-2: Perceptible through close observation
    - 2-10: Perceptible at a glance
    - 11-49: Colors are more similar than opposite
    - 100: Colors are exact opposite
    """
    return np.sqrt(np.sum((lab1.astype(float) - lab2.astype(float)) ** 2))


def create_color_distance_mask(img_bgr: np.ndarray, target_lab: np.ndarray, 
                                threshold: float = 25.0) -> np.ndarray:
    """
    Create a mask of pixels within Delta-E threshold of target color.
    
    More accurate than HSV range for matching similar colors.
    """
    img_lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(float)
    
    # Calculate Delta-E for each pixel
    diff = img_lab - target_lab.astype(float)
    delta_e = np.sqrt(np.sum(diff ** 2, axis=2))
    
    # Create binary mask
    mask = (delta_e < threshold).astype(np.uint8) * 255
    
    return mask


# =============================================================================
# CURVE EXTRACTION
# =============================================================================

def extract_single_color_arm(img_bgr: np.ndarray, target_hsv: np.ndarray, 
                              max_time_x: float, max_surv_y: float = 100.0,
                              use_delta_e: bool = True) -> Optional[pd.DataFrame]:
    """
    Extract a curve based on a specific target color.
    
    Args:
        img_bgr: BGR image
        target_hsv: Target HSV color [H, S, V]
        max_time_x: X-axis maximum value
        max_surv_y: Y-axis maximum value
        use_delta_e: If True, use LAB/Delta-E matching (more accurate)
    
    Returns:
        DataFrame with 'time' and 'survival' columns
    """
    height, width = img_bgr.shape[:2]
    
    if use_delta_e:
        # Convert target HSV to LAB for Delta-E matching
        target_hsv_img = np.array([[target_hsv]], dtype=np.uint8)
        target_bgr = cv2.cvtColor(target_hsv_img, cv2.COLOR_HSV2BGR)
        target_lab = cv2.cvtColor(target_bgr, cv2.COLOR_BGR2LAB)[0][0]
        
        # Adaptive threshold based on saturation
        # Low saturation colors need tighter threshold
        sat = target_hsv[1]
        threshold = 20 + (sat / 255) * 15  # Range: 20-35
        
        mask = create_color_distance_mask(img_bgr, target_lab, threshold)
    else:
        # Fallback to HSV range matching
        img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        h, s, v = target_hsv
        
        # Adaptive tolerance
        h_tol = 12 if s > 100 else 18
        s_tol = 50 if s > 100 else 70
        v_tol = 50
        
        lower = np.array([max(0, h - h_tol), max(20, s - s_tol), max(30, v - v_tol)])
        upper = np.array([min(180, h + h_tol), min(255, s + s_tol), min(255, v + v_tol)])
        
        mask = cv2.inRange(img_hsv, lower, upper)
    
    # Morphological cleanup
    kernel_open = np.ones((2, 2), np.uint8)
    kernel_close = np.ones((3, 3), np.uint8)
    
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close)
    
    # Extract curve points
    points = []
    
    for x in range(width):
        col = mask[:, x]
        ys = np.where(col > 0)[0]
        
        if len(ys) > 0:
            # Use median for robustness
            median_y = np.median(ys)
            
            time_val = x * (max_time_x / width)
            surv_val = (height - median_y) * (max_surv_y / height)
            
            points.append({'time': time_val, 'survival': surv_val})
    
    if len(points) < 15:
        return None
    
    df = pd.DataFrame(points)
    
    # Filter unreasonable values
    df = df[(df['survival'] >= -5) & (df['survival'] <= 105)]
    
    # Smooth
    window = max(3, len(df) // 50)
    df['survival'] = df['survival'].rolling(window, center=True, min_periods=2).median()
    df = df.dropna()
    
    if len(df) < 10:
        return None
    
    return df[['time', 'survival']].reset_index(drop=True)


def extract_with_region_growing(img_bgr: np.ndarray, seed_x: int, seed_y: int,
                                 max_time_x: float, max_surv_y: float = 100.0,
                                 tolerance: float = 25.0) -> Optional[pd.DataFrame]:
    """
    Extract curve using region growing from seed point.
    
    Better for following connected lines, handles gaps in dashed lines.
    """
    height, width = img_bgr.shape[:2]
    
    # Get seed color in LAB
    seed_lab = get_color_at_point_lab(img_bgr, seed_x, seed_y, sample_size=5)
    if seed_lab is None:
        return None
    
    # Create initial color mask
    mask = create_color_distance_mask(img_bgr, seed_lab, tolerance)
    
    # Find connected components
    labeled, num_features = label(mask)
    
    if num_features == 0:
        return None
    
    # Find which component contains the seed point
    seed_label = labeled[seed_y, seed_x]
    
    if seed_label == 0:
        # Seed point not in any component - expand search
        # Look for nearest component
        for radius in range(1, 20):
            for dy in range(-radius, radius + 1):
                for dx in range(-radius, radius + 1):
                    ny, nx = seed_y + dy, seed_x + dx
                    if 0 <= ny < height and 0 <= nx < width:
                        if labeled[ny, nx] > 0:
                            seed_label = labeled[ny, nx]
                            break
                if seed_label > 0:
                    break
            if seed_label > 0:
                break
    
    if seed_label == 0:
        return None
    
    # Create mask of just this component
    component_mask = (labeled == seed_label).astype(np.uint8) * 255
    
    # Dilate slightly to connect nearby segments (for dashed lines)
    kernel = np.ones((3, 5), np.uint8)  # Horizontal bias for KM curves
    component_mask = cv2.dilate(component_mask, kernel, iterations=1)
    
    # Extract curve from this component
    points = []
    
    for x in range(width):
        col = component_mask[:, x]
        ys = np.where(col > 0)[0]
        
        if len(ys) > 0:
            median_y = np.median(ys)
            time_val = x * (max_time_x / width)
            surv_val = (height - median_y) * (max_surv_y / height)
            points.append({'time': time_val, 'survival': surv_val})
    
    if len(points) < 15:
        return None
    
    df = pd.DataFrame(points)
    df = df[(df['survival'] >= -5) & (df['survival'] <= 105)]
    
    window = max(3, len(df) // 50)
    df['survival'] = df['survival'].rolling(window, center=True, min_periods=2).median()
    df = df.dropna()
    
    return df[['time', 'survival']].reset_index(drop=True) if len(df) >= 10 else None


# =============================================================================
# VISUALIZATION HELPERS
# =============================================================================

def visualize_color_match(img_bgr: np.ndarray, target_hsv: np.ndarray) -> np.ndarray:
    """
    Create a visualization showing which pixels match the target color.
    
    Useful for debugging/preview before extraction.
    """
    # Convert to LAB for Delta-E
    target_hsv_img = np.array([[target_hsv]], dtype=np.uint8)
    target_bgr = cv2.cvtColor(target_hsv_img, cv2.COLOR_HSV2BGR)
    target_lab = cv2.cvtColor(target_bgr, cv2.COLOR_BGR2LAB)[0][0]
    
    mask = create_color_distance_mask(img_bgr, target_lab, threshold=25)
    
    # Create overlay
    overlay = img_bgr.copy()
    overlay[mask > 0] = [0, 255, 0]  # Highlight matches in green
    
    # Blend with original
    result = cv2.addWeighted(img_bgr, 0.7, overlay, 0.3, 0)
    
    return result


def get_color_info(hsv: np.ndarray) -> dict:
    """
    Get human-readable information about a color.
    """
    h, s, v = hsv
    
    # Approximate color name based on hue
    if s < 30:
        color_name = "Gray/White"
    elif h < 10 or h > 170:
        color_name = "Red"
    elif h < 25:
        color_name = "Orange"
    elif h < 35:
        color_name = "Yellow"
    elif h < 80:
        color_name = "Green"
    elif h < 130:
        color_name = "Blue"
    elif h < 160:
        color_name = "Purple"
    else:
        color_name = "Pink"
    
    return {
        'hsv': (int(h), int(s), int(v)),
        'color_name': color_name,
        'saturation_level': 'High' if s > 150 else 'Medium' if s > 80 else 'Low',
        'brightness_level': 'Bright' if v > 200 else 'Medium' if v > 100 else 'Dark'
    }


# =============================================================================
# TEST
# =============================================================================

if __name__ == "__main__":
    print("Enhanced Seeded Extractor Module")
    print("Functions available:")
    print("  - get_hsv_at_point(img, x, y, sample_size=5)")
    print("  - extract_single_color_arm(img, hsv, max_x, max_y)")
    print("  - extract_with_region_growing(img, seed_x, seed_y, max_x, max_y)")
    print("  - visualize_color_match(img, hsv)")
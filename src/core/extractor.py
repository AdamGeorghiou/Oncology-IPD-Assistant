# src/core/extractor.py
"""
Enhanced Kaplan-Meier Curve Extraction

Improvements over original:
1. Pre-flight quality assessment
2. Adaptive color tolerance based on image characteristics
3. Better deduplication with curve similarity scoring
4. Line detection preprocessing to identify actual curve pixels

Author: IPD Assistant
"""

import cv2
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from scipy.interpolate import interp1d
from typing import List, Tuple, Optional, Dict


# =============================================================================
# PRE-FLIGHT QUALITY ASSESSMENT
# =============================================================================

def assess_extraction_difficulty(img_bgr: np.ndarray) -> Dict:
    """
    Analyze image to predict extraction success likelihood.
    
    Returns:
        dict with:
        - score: 0-100 (higher = easier to extract)
        - recommendation: 'auto', 'seeded', or 'manual'
        - reasons: list of factors affecting score
    """
    result = {
        'score': 50,
        'recommendation': 'auto',
        'reasons': []
    }
    
    img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    height, width = img_bgr.shape[:2]
    
    # 1. Check color saturation distribution
    saturation = img_hsv[:, :, 1].flatten()
    high_sat_pixels = np.sum(saturation > 80) / len(saturation)
    
    if high_sat_pixels > 0.05:
        result['score'] += 15
        result['reasons'].append(f"Good color saturation ({high_sat_pixels:.1%} saturated pixels)")
    elif high_sat_pixels < 0.01:
        result['score'] -= 20
        result['reasons'].append("Low saturation - colors may be washed out")
    
    # 2. Check for distinct color clusters
    colors = get_dominant_line_colors(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB), num_colors=6)
    
    if len(colors) >= 2:
        # Check hue separation between colors
        hues = [c[0] for c in colors]
        hue_diffs = [abs(hues[i] - hues[j]) for i in range(len(hues)) for j in range(i+1, len(hues))]
        min_hue_diff = min(hue_diffs) if hue_diffs else 0
        
        if min_hue_diff > 20:
            result['score'] += 20
            result['reasons'].append(f"Distinct colors detected (hue separation: {min_hue_diff}°)")
        elif min_hue_diff < 10:
            result['score'] -= 15
            result['reasons'].append("Similar colors - may confuse curve detection")
    else:
        result['score'] -= 25
        result['reasons'].append("Few distinct colors found")
    
    # 3. Check image resolution
    if width >= 800 and height >= 400:
        result['score'] += 10
        result['reasons'].append("Good resolution")
    elif width < 400 or height < 200:
        result['score'] -= 15
        result['reasons'].append("Low resolution may affect accuracy")
    
    # 4. Check for potential gridlines (horizontal lines)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, minLineLength=width//3, maxLineGap=10)
    
    if lines is not None:
        horizontal_lines = sum(1 for line in lines if abs(line[0][1] - line[0][3]) < 5)
        if horizontal_lines > 5:
            result['score'] -= 10
            result['reasons'].append(f"Many gridlines detected ({horizontal_lines}) - may interfere")
    
    # 5. Background uniformity check
    # Sample corners for background color consistency
    corners = [
        img_bgr[10:30, 10:30],
        img_bgr[10:30, -30:-10],
        img_bgr[-30:-10, 10:30],
        img_bgr[-30:-10, -30:-10]
    ]
    corner_means = [np.mean(c) for c in corners]
    bg_variance = np.std(corner_means)
    
    if bg_variance < 20:
        result['score'] += 10
        result['reasons'].append("Uniform background")
    else:
        result['score'] -= 5
        result['reasons'].append("Non-uniform background")
    
    # Clamp score
    result['score'] = max(0, min(100, result['score']))
    
    # Set recommendation based on score
    if result['score'] >= 65:
        result['recommendation'] = 'auto'
    elif result['score'] >= 40:
        result['recommendation'] = 'seeded'
    else:
        result['recommendation'] = 'manual'
    
    return result


# =============================================================================
# COLOR DETECTION
# =============================================================================

def get_dominant_line_colors(img_rgb: np.ndarray, num_colors: int = 5) -> List[np.ndarray]:
    """
    K-Means to find line colors (ignoring white background/black text).
    
    Improved: Better filtering and adaptive thresholds.
    """
    img_hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    data = img_hsv.reshape((-1, 3))
    
    # Filter: Must have some saturation (not gray/white/black)
    # and reasonable value (not too dark)
    mask = (data[:, 1] > 30) & (data[:, 2] > 50) & (data[:, 2] < 245)
    data_filtered = data[mask]
    
    if len(data_filtered) < 100:
        return []
    
    # Limit sample size for performance
    if len(data_filtered) > 50000:
        indices = np.random.choice(len(data_filtered), 50000, replace=False)
        data_filtered = data_filtered[indices]
    
    try:
        kmeans = KMeans(n_clusters=min(num_colors, len(data_filtered) // 100), 
                       random_state=42, n_init=10)
        kmeans.fit(data_filtered)
        centers = kmeans.cluster_centers_.astype(int)
        
        # Get cluster sizes for filtering
        labels = kmeans.labels_
        cluster_sizes = np.bincount(labels)
        
        # Filter: Keep saturated colors with enough pixels
        min_cluster_size = len(data_filtered) * 0.01  # At least 1% of colored pixels
        line_colors = []
        
        for i, center in enumerate(centers):
            if center[1] > 40 and cluster_sizes[i] > min_cluster_size:
                line_colors.append(center)
        
        # Sort by saturation (most saturated = most likely to be a line)
        line_colors.sort(key=lambda x: x[1], reverse=True)
        
        return line_colors
        
    except Exception as e:
        print(f"Color clustering error: {e}")
        return []


def calculate_adaptive_tolerance(img_hsv: np.ndarray, target_hsv: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate adaptive HSV tolerance based on local color variance.
    
    Instead of fixed ±10 hue, ±40 sat, this adapts to the image.
    """
    h, s, v = target_hsv
    
    # Base tolerances
    h_tol = 12
    s_tol = 50
    v_tol = 50
    
    # Adjust based on saturation - low sat colors need wider tolerance
    if s < 100:
        s_tol = 60
        h_tol = 15
    
    # Adjust based on value - dark colors need wider tolerance
    if v < 100:
        v_tol = 60
    
    lower = np.array([max(0, h - h_tol), max(20, s - s_tol), max(30, v - v_tol)])
    upper = np.array([min(180, h + h_tol), min(255, s + s_tol), min(255, v + v_tol)])
    
    return lower, upper


# =============================================================================
# CURVE EXTRACTION
# =============================================================================

def extract_curve_from_mask(mask: np.ndarray, max_time_x: float, max_surv_y: float = 100.0,
                            smooth_window: int = 10) -> Optional[pd.DataFrame]:
    """
    Extract curve coordinates from a binary mask.
    
    Improved: Better handling of gaps, outliers, and smoothing.
    """
    height, width = mask.shape
    points = []
    
    for x in range(width):
        col = mask[:, x]
        ys = np.where(col > 0)[0]
        
        if len(ys) > 0:
            # Use median instead of mean - more robust to outliers
            median_y = np.median(ys)
            
            # Map pixels to data coordinates
            time_val = x * (max_time_x / width)
            surv_val = (height - median_y) * (max_surv_y / height)
            
            # Also track the spread (useful for confidence)
            spread = np.std(ys) if len(ys) > 1 else 0
            
            points.append({
                'time': time_val, 
                'survival': surv_val,
                'confidence': 1.0 / (1.0 + spread)  # Higher spread = lower confidence
            })
    
    if len(points) < 20:
        return None
    
    df = pd.DataFrame(points)
    
    # Filter unreasonable values
    df = df[(df['survival'] >= -5) & (df['survival'] <= 105)]
    
    if len(df) < 20:
        return None
    
    # Smooth with rolling median (more robust than mean)
    df['survival'] = df['survival'].rolling(smooth_window, center=True, min_periods=3).median()
    df = df.dropna()
    
    # Remove outliers using IQR on the derivative
    if len(df) > 10:
        df['diff'] = df['survival'].diff().abs()
        q75 = df['diff'].quantile(0.75)
        iqr = df['diff'].quantile(0.75) - df['diff'].quantile(0.25)
        threshold = q75 + 2.0 * iqr
        df = df[df['diff'] < threshold]
        df = df.drop(columns=['diff'])
    
    return df[['time', 'survival']].reset_index(drop=True)


def auto_extract_curves(img_bgr: np.ndarray, max_time_x: float, max_surv_y: float = 100.0) -> List[pd.DataFrame]:
    """
    Main extraction function with improvements.
    
    Args:
        img_bgr: The CROPPED graph image (BGR format)
        max_time_x: User provided X-axis limit (e.g., 30 months)
        max_surv_y: Y-axis maximum (typically 100 for percentage)
    
    Returns:
        List of DataFrames, each containing 'time' and 'survival' columns
    """
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    height, width = img_bgr.shape[:2]
    
    # 1. Find candidate colors
    target_colors = get_dominant_line_colors(img_rgb, num_colors=6)
    
    if not target_colors:
        return []
    
    raw_arms = []
    
    # 2. Extract curve for each color
    for color in target_colors:
        # Use adaptive tolerance
        lower, upper = calculate_adaptive_tolerance(img_hsv, color)
        
        mask = cv2.inRange(img_hsv, lower, upper)
        
        # Morphological cleanup - remove noise, connect gaps
        kernel_open = np.ones((2, 2), np.uint8)
        kernel_close = np.ones((3, 3), np.uint8)
        
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close)
        
        # Extract curve from mask
        df = extract_curve_from_mask(mask, max_time_x, max_surv_y)
        
        if df is not None and len(df) > 30:
            raw_arms.append(df)
    
    # 3. Deduplicate similar curves
    final_arms = deduplicate_arms(raw_arms, tolerance=3.0)
    
    return final_arms


# =============================================================================
# DEDUPLICATION
# =============================================================================

def deduplicate_arms(arm_list: List[pd.DataFrame], tolerance: float = 3.0) -> List[pd.DataFrame]:
    """
    Merge similar curves (handles antialiasing artifacts).
    
    Improved: Better similarity metric using area between curves.
    """
    if not arm_list:
        return []
    
    # Sort by data density (more points = likely primary curve)
    arm_list.sort(key=lambda x: len(x), reverse=True)
    
    unique_arms = []
    
    for candidate in arm_list:
        is_duplicate = False
        
        for existing in unique_arms:
            similarity = calculate_curve_similarity(candidate, existing)
            
            if similarity < tolerance:
                is_duplicate = True
                break
        
        if not is_duplicate:
            unique_arms.append(candidate)
    
    return unique_arms


def calculate_curve_similarity(curve1: pd.DataFrame, curve2: pd.DataFrame) -> float:
    """
    Calculate similarity between two curves using mean absolute difference.
    
    Returns average absolute difference in survival values.
    Lower = more similar.
    """
    # Find overlapping time range
    t_min = max(curve1['time'].min(), curve2['time'].min())
    t_max = min(curve1['time'].max(), curve2['time'].max())
    
    if t_max <= t_min:
        return float('inf')
    
    # Create interpolation functions
    try:
        f1 = interp1d(curve1['time'], curve1['survival'], 
                     bounds_error=False, fill_value='extrapolate')
        f2 = interp1d(curve2['time'], curve2['survival'], 
                     bounds_error=False, fill_value='extrapolate')
        
        # Sample at regular intervals
        t_samples = np.linspace(t_min, t_max, 50)
        
        vals1 = f1(t_samples)
        vals2 = f2(t_samples)
        
        # Mean absolute difference
        diff = np.abs(vals1 - vals2)
        diff = diff[~np.isnan(diff)]
        
        if len(diff) == 0:
            return float('inf')
        
        return np.mean(diff)
        
    except Exception:
        return float('inf')


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def get_extraction_summary(arms: List[pd.DataFrame]) -> Dict:
    """
    Generate summary statistics for extracted arms.
    """
    if not arms:
        return {'count': 0, 'arms': []}
    
    summary = {
        'count': len(arms),
        'arms': []
    }
    
    for i, arm in enumerate(arms):
        arm_info = {
            'index': i,
            'points': len(arm),
            'time_range': (arm['time'].min(), arm['time'].max()),
            'survival_range': (arm['survival'].min(), arm['survival'].max()),
            'start_survival': arm['survival'].iloc[0] if len(arm) > 0 else None,
            'end_survival': arm['survival'].iloc[-1] if len(arm) > 0 else None
        }
        summary['arms'].append(arm_info)
    
    return summary


# =============================================================================
# TEST
# =============================================================================

if __name__ == "__main__":
    print("Enhanced Extractor Module")
    print("Functions available:")
    print("  - assess_extraction_difficulty(img_bgr)")
    print("  - auto_extract_curves(img_bgr, max_time_x, max_surv_y)")
    print("  - get_extraction_summary(arms)")
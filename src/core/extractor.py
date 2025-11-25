# src/core/extractor.py
import cv2
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from scipy.interpolate import interp1d

def get_dominant_line_colors(img_rgb, num_colors=5):
    """K-Means to find line colors (ignoring white background/black text)"""
    # Convert to HSV for better color segmentation
    img_hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    data = img_hsv.reshape((-1, 3))
    
    # Filter: S>40 (Color), V>40 (Not too dark)
    mask = (data[:, 1] > 40) & (data[:, 2] > 40)
    data_filtered = data[mask]
    
    if len(data_filtered) < 100: return []

    kmeans = KMeans(n_clusters=num_colors, random_state=42, n_init=10)
    kmeans.fit(data_filtered)
    centers = kmeans.cluster_centers_.astype(int)
    
    # Filter clusters to keep only saturated ones
    line_colors = [c for c in centers if c[1] > 50]
    # Sort by Hue
    line_colors.sort(key=lambda x: x[0])
    return line_colors

def deduplicate_arms(arm_list, tolerance=2.5):
    """Merges similar curves (antialiasing fix)"""
    if not arm_list: return []
    
    # Sort by length (longest is primary)
    arm_list.sort(key=lambda x: len(x), reverse=True)
    unique_arms = []
    
    for candidate in arm_list:
        is_duplicate = False
        for existing in unique_arms:
            t_min = max(candidate['time'].min(), existing['time'].min())
            t_max = min(candidate['time'].max(), existing['time'].max())
            
            if t_max <= t_min: continue 
            
            f_exist = interp1d(existing['time'], existing['survival'], bounds_error=False)
            exist_vals = f_exist(candidate['time'])
            
            diff = np.abs(candidate['survival'] - exist_vals)
            diff = diff[~np.isnan(diff)]
            
            if len(diff) > 0 and np.mean(diff) < tolerance:
                is_duplicate = True
                break
        
        if not is_duplicate:
            unique_arms.append(candidate)
            
    return unique_arms

def auto_extract_curves(img_bgr, max_time_x, max_surv_y=100.0):
    """
    Main extraction function.
    img_bgr: The CROPPED graph image.
    max_time_x: User provided X-axis limit (e.g., 30 months).
    """
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    height, width = img_bgr.shape[:2]

    # 1. Find Colors
    target_colors = get_dominant_line_colors(img_rgb)
    
    raw_arms = []
    
    # 2. Extract for each color
    for color in target_colors:
        h, s, v = color
        lower = np.array([max(0, h-10), max(40, s-40), max(40, v-40)])
        upper = np.array([min(180, h+10), 255, 255])
        
        mask = cv2.inRange(img_hsv, lower, upper)
        # Clean noise
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((2,2), np.uint8))
        
        points = []
        for x in range(width):
            col = mask[:, x]
            ys = np.where(col > 0)[0]
            if len(ys) > 0:
                avg_y = np.mean(ys)
                # Map pixels to data
                time_val = x * (max_time_x / width)
                surv_val = (height - avg_y) * (max_surv_y / height)
                points.append({'time': time_val, 'survival': surv_val})
        
        if len(points) > 50:
            df = pd.DataFrame(points)
            # Smooth
            df = df[(df['survival'] >= 0) & (df['survival'] <= 105)]
            df['survival'] = df['survival'].rolling(10, center=True).mean()
            df = df.dropna()
            raw_arms.append(df)
            
    # 3. Deduplicate
    final_arms = deduplicate_arms(raw_arms)
    
    return final_arms
# src/core/seeded_extractor.py
import cv2
import numpy as np
import pandas as pd

def get_hsv_at_point(img_bgr, x, y):
    """
    Returns the HSV color at the specific (x, y) coordinate.
    """
    if img_bgr is None: return None
    
    # Check bounds
    h, w = img_bgr.shape[:2]
    if x < 0 or x >= w or y < 0 or y >= h:
        return None
        
    # Convert single pixel to HSV
    # We do this efficiently by grabbing the pixel first
    pixel_bgr = img_bgr[y, x]
    pixel_bgr_reshaped = np.array([[pixel_bgr]], dtype=np.uint8)
    pixel_hsv = cv2.cvtColor(pixel_bgr_reshaped, cv2.COLOR_BGR2HSV)
    
    return pixel_hsv[0][0]

def extract_single_color_arm(img_bgr, target_hsv, max_time_x, max_surv_y=100.0):
    """
    Extracts a curve based on a specific target HSV color.
    """
    img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    height, width = img_bgr.shape[:2]
    
    h, s, v = target_hsv
    
    # Dynamic Tolerance
    lower = np.array([max(0, h-10), max(20, s-50), max(20, v-50)])
    upper = np.array([min(180, h+10), 255, 255])
    
    mask = cv2.inRange(img_hsv, lower, upper)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((2,2), np.uint8))
    
    points = []
    for x in range(width):
        col = mask[:, x]
        ys = np.where(col > 0)[0]
        if len(ys) > 0:
            avg_y = np.mean(ys)
            time_val = x * (max_time_x / width)
            surv_val = (height - avg_y) * (max_surv_y / height)
            points.append({'time': time_val, 'survival': surv_val})
            
    if len(points) > 10:
        df = pd.DataFrame(points)
        df = df[(df['survival'] >= 0) & (df['survival'] <= 105)]
        df['survival'] = df['survival'].rolling(5, center=True).mean()
        return df.dropna()
        
    return None
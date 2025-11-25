# src/core/vision.py
import cv2
import numpy as np
from typing import Tuple, Optional

def detect_axes_and_crop(image_path: str) -> Tuple[Optional[np.ndarray], dict]:
    """
    Analyzes an image to find the graph area (X and Y axes).
    Returns the cropped image (graph only) and the axis coordinates.
    """
    img = cv2.imread(image_path)
    if img is None:
        return None, {"status": "failed", "msg": "Image not found"}
        
    height, width = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 1. Morphological cleanup
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    kernel = np.ones((3,3), np.uint8)
    edges_dilated = cv2.dilate(edges, kernel, iterations=1)
    
    # 2. Hough Transform
    lines = cv2.HoughLinesP(
        edges_dilated, 
        1, 
        np.pi/180, 
        100, 
        minLineLength=width//4, 
        maxLineGap=20
    )
    
    best_x_axis = None
    best_y_axis = None
    max_y_coord = 0
    min_x_coord = width 
    
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            
            # Filter borders
            if x1 < 10 or x2 > width-10 or y1 < 10 or y2 > height-10:
                continue

            angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
            
            # X-Axis Logic (Horizontal, lowest)
            if (abs(angle) < 5 or abs(angle - 180) < 5) and ((y1+y2)/2 > max_y_coord):
                max_y_coord = (y1+y2)/2
                best_x_axis = (x1, y1, x2, y2)
            
            # Y-Axis Logic (Vertical, left-most)
            if (abs(angle - 90) < 5 or abs(angle + 90) < 5) and ((x1+x2)/2 < min_x_coord):
                min_x_coord = (x1+x2)/2
                best_y_axis = (x1, y1, x2, y2)

    if not best_x_axis or not best_y_axis:
        return img, {"status": "failed", "msg": "Could not detect axes"}

    # 3. Calculate Origin and Crop
    origin_x = best_y_axis[0]
    origin_y = best_x_axis[1]
    
    # FIX: Ensure we grab the TOP of the Y-axis and RIGHT of X-axis regardless of line direction
    y_start = min(best_y_axis[1], best_y_axis[3])
    x_end = max(best_x_axis[0], best_x_axis[2])
    
    # Ensure bounds
    crop_x1 = max(0, origin_x)
    crop_x2 = min(width, x_end)
    crop_y1 = max(0, y_start)
    crop_y2 = min(height, origin_y)
    
    # Validation check to prevent empty crop
    if crop_x2 <= crop_x1 or crop_y2 <= crop_y1:
        return img, {"status": "failed", "msg": "Invalid crop dimensions detected."}
    
    cropped_img = img[crop_y1:crop_y2, crop_x1:crop_x2]
    
    # Draw detection on original for debug
    debug_img = img.copy()
    cv2.line(debug_img, (best_x_axis[0], best_x_axis[1]), (best_x_axis[2], best_x_axis[3]), (0, 255, 0), 3)
    cv2.line(debug_img, (best_y_axis[0], best_y_axis[1]), (best_y_axis[2], best_y_axis[3]), (0, 0, 255), 3)
    cv2.circle(debug_img, (origin_x, origin_y), 10, (0, 255, 255), -1)

    metadata = {
        "status": "success",
        "origin": (origin_x, origin_y),
        "x_len_px": x_end - origin_x,
        "y_len_px": origin_y - y_start,
        "crop_coords": (crop_x1, crop_y1, crop_x2, crop_y2),
        "debug_image": debug_img
    }
    
    return cropped_img, metadata
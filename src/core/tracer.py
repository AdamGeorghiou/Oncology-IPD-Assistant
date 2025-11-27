# src/core/tracer.py
"""
Manual Curve Tracer - Simple & Clean

Converts freehand drawings from streamlit_drawable_canvas into curve data.
"""

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from typing import Optional


def process_manual_drawing(canvas_result, x_len_px: int, y_len_px: int) -> Optional[pd.DataFrame]:
    """
    Convert FreeDraw path from Streamlit Canvas into a DataFrame.
    
    Args:
        canvas_result: Result from st_canvas
        x_len_px: Width of canvas in pixels
        y_len_px: Height of canvas in pixels
        
    Returns:
        DataFrame with 'rel_time' (0-1) and 'rel_surv' (0-1) columns, or None if failed
    """
    if canvas_result is None or canvas_result.json_data is None:
        return None
    
    objects = canvas_result.json_data.get("objects", [])
    if not objects:
        return None
    
    # Extract all points from drawn paths
    all_points = []
    
    for obj in objects:
        if obj.get("type") == "path":
            path = obj.get("path", [])
            for command in path:
                if len(command) >= 3:
                    # M = move, L = line - both have x,y at positions 1,2
                    if command[0] in ['M', 'L']:
                        all_points.append((command[1], command[2]))
                    elif command[0] == 'Q' and len(command) >= 5:
                        # Quadratic curve - endpoint at positions 3,4
                        all_points.append((command[3], command[4]))
    
    if len(all_points) < 3:
        return None
    
    # Convert to numpy arrays
    pts = np.array(all_points)
    xs_px = pts[:, 0]
    ys_px = pts[:, 1]
    
    # Sort by X coordinate
    sorted_idx = np.argsort(xs_px)
    xs_px = xs_px[sorted_idx]
    ys_px = ys_px[sorted_idx]
    
    # Remove duplicate X values
    _, unique_idx = np.unique(xs_px, return_index=True)
    unique_idx = np.sort(unique_idx)
    xs_clean = xs_px[unique_idx]
    ys_clean = ys_px[unique_idx]
    
    if len(xs_clean) < 3:
        return None
    
    try:
        # Create evenly spaced X values for interpolation
        x_new = np.linspace(xs_clean.min(), xs_clean.max(), num=200)
        
        # Linear interpolation
        f = interp1d(xs_clean, ys_clean, kind='linear', bounds_error=False)
        y_new = f(x_new)
        
        # Normalize to 0-1 range
        rel_x = x_new / x_len_px
        rel_y = (y_len_px - y_new) / y_len_px  # Invert Y (canvas Y=0 is top)
        
        # Clamp to valid range
        rel_x = np.clip(rel_x, 0, 1)
        rel_y = np.clip(rel_y, 0, 1)
        
        df = pd.DataFrame({
            'rel_time': rel_x,
            'rel_surv': rel_y
        })
        
        return df.dropna()
        
    except Exception as e:
        print(f"Tracer error: {e}")
        return None
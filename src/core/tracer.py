# src/core/tracer.py
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

def process_manual_drawing(canvas_result, x_len_px, y_len_px):
    """
    Converts the FreeDraw path from Streamlit Canvas into a DataFrame.
    """
    if canvas_result.json_data is None: return None
    
    objects = canvas_result.json_data["objects"]
    if not objects: return None
    
    # Get all points from the drawn paths
    all_points = []
    for obj in objects:
        if obj["type"] == "path":
            # Path is a list of [command, x, y]
            # We extract just the x,y coordinates
            for command in obj["path"]:
                if len(command) >= 3: # 'M', x, y or 'L', x, y
                    all_points.append((command[1], command[2]))
    
    if len(all_points) < 2: return None
    
    # Convert to numpy
    pts = np.array(all_points)
    xs_px = pts[:, 0]
    ys_px = pts[:, 1]
    
    # Sort by X
    sorted_indices = np.argsort(xs_px)
    xs_px = xs_px[sorted_indices]
    ys_px = ys_px[sorted_indices]
    
    # Interpolate
    try:
        # Remove duplicates in X to allow interpolation
        _, unique_indices = np.unique(xs_px, return_index=True)
        xs_clean = xs_px[unique_indices]
        ys_clean = ys_px[unique_indices]
        
        if len(xs_clean) < 2: return None

        # Create dense X range
        x_new_px = np.arange(min(xs_clean), max(xs_clean), 1.0)
        f = interp1d(xs_clean, ys_clean, kind='linear', bounds_error=False)
        y_new_px = f(x_new_px)
        
        # Normalize (0 to 1)
        # Note: Canvas Y is 0 at top, same as image logic
        rel_x = x_new_px / x_len_px
        rel_y = (y_len_px - y_new_px) / y_len_px # Invert Y for plot
        
        df = pd.DataFrame({'rel_time': rel_x, 'rel_surv': rel_y})
        return df
        
    except Exception as e:
        print(f"Tracing Error: {e}")
        return None
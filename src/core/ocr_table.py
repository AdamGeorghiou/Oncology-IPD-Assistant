# src/core/ocr_table.py
import easyocr
import cv2
import numpy as np
import pandas as pd
import re

# Initialize reader once to save time (it loads models)
# using gpu=False for compatibility, set True if you have CUDA
reader = easyocr.Reader(['en'], gpu=False, verbose=False)

def auto_read_risk_table(image_path, num_arms, num_timepoints):
    """
    Scans the bottom 25% of the image for the risk table.
    Returns a dataframe of shape (num_arms, num_timepoints) with best guesses.
    """
    img = cv2.imread(image_path)
    if img is None: return None

    h, w = img.shape[:2]
    
    # Heuristic: Table is usually in the bottom 20-25%
    crop_y = int(h * 0.75)
    roi = img[crop_y:h, 0:w]
    
    # Run OCR
    try:
        results = reader.readtext(roi)
    except Exception as e:
        print(f"OCR Error: {e}")
        return None

    # Extract numbers and their centroids (y-coordinates)
    found_numbers = []
    for (bbox, text, prob) in results:
        # Remove non-numeric chars (keep spaces for merged numbers)
        clean_text = re.sub(r"[^0-9\s]", "", text).strip()
        
        if clean_text:
            # Handle merged OCR issues (e.g., "355 201" read as one block)
            # Split by space if present
            parts = clean_text.split()
            
            # Get Y-centroid of the text box to group by row
            (tl, tr, br, bl) = bbox
            y_center = (tl[1] + bl[1]) / 2
            x_center = (tl[0] + tr[0]) / 2
            
            for part in parts:
                if part.isdigit():
                    # We add a slight x_offset for parts split from the same box
                    found_numbers.append({
                        'val': int(part), 
                        'y': y_center, 
                        'x': x_center + parts.index(part)*10 
                    })

    if not found_numbers:
        return None

    # CLUSTERING BY ROW (Y-Coordinate)
    # Simple logic: sort by Y, split where difference > threshold
    found_numbers.sort(key=lambda k: k['y'])
    
    rows = []
    current_row = [found_numbers[0]]
    
    for i in range(1, len(found_numbers)):
        # If y difference is small (< 20px), it's the same row
        if abs(found_numbers[i]['y'] - current_row[-1]['y']) < 20:
            current_row.append(found_numbers[i])
        else:
            rows.append(current_row)
            current_row = [found_numbers[i]]
    rows.append(current_row)
    
    # We only want the rows that look like data (length close to num_timepoints)
    # Sort rows by length (descending)
    rows.sort(key=len, reverse=True)
    
    # Take the top 'num_arms' rows
    best_rows = rows[:num_arms]
    
    # Format into DataFrame
    data = []
    for row in best_rows:
        # Sort by X to get time order
        row.sort(key=lambda k: k['x'])
        # Extract values
        vals = [item['val'] for item in row]
        
        # Pad or truncate to match num_timepoints
        if len(vals) < num_timepoints:
            vals += [0] * (num_timepoints - len(vals))
        else:
            vals = vals[:num_timepoints]
        data.append(vals)
        
    # If we found fewer rows than arms, pad with zeros
    while len(data) < num_arms:
        data.append([0] * num_timepoints)

    return data
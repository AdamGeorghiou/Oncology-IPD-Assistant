# src/core/ocr_table.py
"""
Multi-Strategy At-Risk Table Extraction

Three approaches for extracting numbers from at-risk tables:
1. Enhanced EasyOCR - Preprocessing + column segmentation
2. Gemini Vision LLM - Most robust, handles difficult cases
3. Manual fallback - User enters data directly

Author: IPD Assistant
"""

import cv2
import numpy as np
import re
import os
from typing import List, Optional, Tuple

# Lazy load EasyOCR (heavy import)
_reader = None

def _get_ocr_reader():
    """Lazy initialization of EasyOCR reader."""
    global _reader
    if _reader is None:
        import easyocr
        _reader = easyocr.Reader(['en'], gpu=False, verbose=False)
    return _reader


# =============================================================================
# METHOD 1: ENHANCED EASYOCR WITH PREPROCESSING
# =============================================================================

def preprocess_for_ocr(img_bgr: np.ndarray, scale: int = 3) -> Tuple[np.ndarray, int]:
    """
    Aggressive preprocessing to improve OCR accuracy on small text.
    
    Steps:
    1. Upscale image (small text becomes readable)
    2. Convert to grayscale
    3. Apply CLAHE for contrast enhancement
    4. Binarize with Otsu's threshold
    5. Optional morphological cleanup
    
    Returns:
        Tuple of (processed_image, scale_factor)
    """
    # 1. Upscale for better OCR accuracy
    img_large = cv2.resize(
        img_bgr, None, 
        fx=scale, fy=scale, 
        interpolation=cv2.INTER_CUBIC
    )
    
    # 2. Convert to grayscale
    gray = cv2.cvtColor(img_large, cv2.COLOR_BGR2GRAY)
    
    # 3. CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    
    # 4. Binarize - black text on white background
    _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # 5. Light morphological opening to reduce noise
    kernel = np.ones((2, 2), np.uint8)
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    
    return cleaned, scale


def extract_table_enhanced_ocr(
    img_crop_bgr: np.ndarray, 
    num_arms: int, 
    num_timepoints: int,
    use_column_segmentation: bool = True
) -> Optional[List[List[int]]]:
    """
    Enhanced EasyOCR extraction with preprocessing and optional column segmentation.
    
    Args:
        img_crop_bgr: BGR image of the cropped at-risk table
        num_arms: Number of treatment arms (rows)
        num_timepoints: Number of time columns
        use_column_segmentation: If True, segment by columns first
        
    Returns:
        2D list of integers [arm][timepoint] or None if failed
    """
    if img_crop_bgr is None or img_crop_bgr.size == 0:
        return None
    
    reader = _get_ocr_reader()
    
    # Preprocess
    processed, scale = preprocess_for_ocr(img_crop_bgr, scale=3)
    
    # Convert back to BGR for EasyOCR (it expects color or grayscale)
    processed_bgr = cv2.cvtColor(processed, cv2.COLOR_GRAY2BGR)
    
    if use_column_segmentation:
        return _extract_by_columns(processed_bgr, num_arms, num_timepoints, reader, scale)
    else:
        return _extract_by_clustering(processed_bgr, num_arms, num_timepoints, reader)


def _extract_by_columns(
    img_bgr: np.ndarray, 
    num_arms: int, 
    num_timepoints: int,
    reader,
    scale: int
) -> Optional[List[List[int]]]:
    """
    Column-based segmentation: divide image into grid cells and OCR each.
    More reliable for dense tables with known structure.
    """
    h, w = img_bgr.shape[:2]
    col_width = w // num_timepoints
    row_height = h // num_arms
    
    # Add padding to prevent edge clipping
    pad = int(5 * scale)
    
    results = []
    
    for row_idx in range(num_arms):
        row_values = []
        y1 = max(0, row_idx * row_height - pad)
        y2 = min(h, (row_idx + 1) * row_height + pad)
        
        for col_idx in range(num_timepoints):
            x1 = max(0, col_idx * col_width - pad)
            x2 = min(w, (col_idx + 1) * col_width + pad)
            
            cell = img_bgr[y1:y2, x1:x2]
            
            # OCR single cell
            try:
                cell_results = reader.readtext(cell, allowlist='0123456789,.')
                
                # Extract best number from cell
                best_val = 0
                best_conf = 0
                
                for (bbox, text, prob) in cell_results:
                    clean = re.sub(r'[^0-9]', '', text)
                    if clean and prob > best_conf:
                        val = int(clean)
                        if val < 1_000_000:  # Sanity check
                            best_val = val
                            best_conf = prob
                
                row_values.append(best_val)
                
            except Exception:
                row_values.append(0)
        
        results.append(row_values)
    
    return results


def _extract_by_clustering(
    img_bgr: np.ndarray, 
    num_arms: int, 
    num_timepoints: int,
    reader
) -> Optional[List[List[int]]]:
    """
    Original clustering approach with improved preprocessing.
    Groups detected numbers by Y-coordinate into rows.
    """
    try:
        results = reader.readtext(img_bgr, allowlist='0123456789,.')
    except Exception as e:
        print(f"OCR Error: {e}")
        return None
    
    # Extract raw numbers with coordinates
    raw_items = []
    for (bbox, text, prob) in results:
        clean_text = re.sub(r'[^0-9]', '', text)
        if clean_text:
            val = int(clean_text)
            if val < 1_000_000:
                (tl, tr, br, bl) = bbox
                y_center = (tl[1] + bl[1]) / 2
                x_center = (tl[0] + tr[0]) / 2
                raw_items.append({'val': val, 'y': y_center, 'x': x_center, 'conf': prob})
    
    if not raw_items:
        return None
    
    # Sort by Y (top to bottom)
    raw_items.sort(key=lambda k: k['y'])
    
    # Cluster into rows with adaptive threshold
    h = img_bgr.shape[0]
    row_threshold = h / (num_arms * 2)  # Adaptive based on expected rows
    
    rows = []
    current_row = [raw_items[0]]
    
    for i in range(1, len(raw_items)):
        if abs(raw_items[i]['y'] - current_row[-1]['y']) > row_threshold:
            rows.append(current_row)
            current_row = [raw_items[i]]
        else:
            current_row.append(raw_items[i])
    rows.append(current_row)
    
    # Take top num_arms rows
    valid_rows = rows[:num_arms]
    
    final_data = []
    for row_items in valid_rows:
        # Sort left to right
        row_items.sort(key=lambda k: k['x'])
        vals = [item['val'] for item in row_items]
        
        # Pad or truncate
        if len(vals) < num_timepoints:
            vals += [0] * (num_timepoints - len(vals))
        else:
            vals = vals[:num_timepoints]
        
        final_data.append(vals)
    
    # Pad missing rows
    while len(final_data) < num_arms:
        final_data.append([0] * num_timepoints)
    
    return final_data


# =============================================================================
# METHOD 2: GEMINI VISION LLM EXTRACTION
# =============================================================================

def extract_table_with_vision_llm(
    img_crop_bgr: np.ndarray,
    num_arms: int,
    num_timepoints: int,
    arm_names: Optional[List[str]] = None,
    time_points: Optional[List[float]] = None
) -> Optional[List[List[int]]]:
    """
    Use Gemini Vision to extract at-risk table numbers.
    
    This is the most robust method for difficult cases where OCR fails.
    Cost: ~$0.0003 per extraction with Gemini Flash.
    
    Args:
        img_crop_bgr: BGR image of the cropped at-risk table
        num_arms: Number of treatment arms (rows)
        num_timepoints: Number of time columns
        arm_names: Optional list of arm names for context
        time_points: Optional list of time values for context
        
    Returns:
        2D list of integers [arm][timepoint] or None if failed
    """
    if img_crop_bgr is None or img_crop_bgr.size == 0:
        return None
    
    import base64
    import json
    
    try:
        import google.generativeai as genai
    except ImportError:
        print("google-generativeai not installed. Run: pip install google-generativeai")
        return None
    
    # Configure API
    api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("No Gemini API key found. Set GOOGLE_API_KEY or GEMINI_API_KEY environment variable.")
        return None
    
    genai.configure(api_key=api_key)
    
    # Encode image to base64
    _, buffer = cv2.imencode('.png', img_crop_bgr)
    img_b64 = base64.b64encode(buffer).decode('utf-8')
    
    # Build context-aware prompt
    arm_context = ""
    if arm_names:
        arm_context = f"The arms are named (top to bottom): {', '.join(arm_names)}. "
    
    time_context = ""
    if time_points:
        time_context = f"Time points: {time_points}. "
    
    prompt = f"""Extract the numbers from this at-risk table image.

This is a "Number at Risk" table from a clinical trial survival curve.
{arm_context}{time_context}

There are exactly {num_arms} rows (treatment arms) and {num_timepoints} columns (time points).
Read the numbers from LEFT to RIGHT, TOP to BOTTOM.

IMPORTANT:
- Return ONLY a valid JSON array of arrays
- Each inner array represents one row (treatment arm)
- Numbers should decrease or stay the same over time (patients drop out)
- If a number is unclear, make your best estimate based on the pattern
- Do NOT include any text, explanation, or markdown - ONLY the JSON array

Example format for 2 arms, 5 timepoints:
[[373, 366, 350, 320, 280], [184, 171, 160, 145, 130]]

Your response (JSON only):"""

    try:
        # Use Gemini Flash for cost efficiency
        model = genai.GenerativeModel('gemini-2.0-flash')
        
        # Create image part
        image_part = {
            "mime_type": "image/png",
            "data": img_b64
        }
        
        response = model.generate_content([prompt, image_part])
        
        # Parse response
        response_text = response.text.strip()
        
        # Clean up response (remove markdown code blocks if present)
        if response_text.startswith('```'):
            # Remove ```json and ``` markers
            response_text = re.sub(r'^```(?:json)?\s*', '', response_text)
            response_text = re.sub(r'\s*```$', '', response_text)
        
        # Parse JSON
        data = json.loads(response_text)
        
        # Validate structure
        if not isinstance(data, list) or len(data) != num_arms:
            print(f"Invalid response structure: expected {num_arms} rows, got {len(data) if isinstance(data, list) else 'non-list'}")
            return None
        
        # Ensure all values are integers and rows have correct length
        result = []
        for row in data:
            if not isinstance(row, list):
                return None
            
            int_row = []
            for val in row[:num_timepoints]:
                try:
                    int_row.append(int(val))
                except (ValueError, TypeError):
                    int_row.append(0)
            
            # Pad if needed
            while len(int_row) < num_timepoints:
                int_row.append(0)
            
            result.append(int_row)
        
        return result
        
    except json.JSONDecodeError as e:
        print(f"Failed to parse Gemini response as JSON: {e}")
        print(f"Response was: {response_text[:500]}")
        return None
    except Exception as e:
        print(f"Gemini Vision error: {e}")
        return None


# =============================================================================
# LEGACY COMPATIBILITY FUNCTION
# =============================================================================

def extract_table_from_crop(
    img_crop_bgr: np.ndarray, 
    num_arms: int, 
    num_timepoints: int
) -> Optional[List[List[int]]]:
    """
    Legacy function for backwards compatibility.
    Uses enhanced OCR method by default.
    """
    return extract_table_enhanced_ocr(
        img_crop_bgr, 
        num_arms, 
        num_timepoints,
        use_column_segmentation=True
    )


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def validate_at_risk_data(data: List[List[int]]) -> dict:
    """
    Validate at-risk table data for common issues.
    
    Returns dict with:
        - valid: bool
        - warnings: list of warning messages
        - errors: list of error messages
    """
    result = {
        'valid': True,
        'warnings': [],
        'errors': []
    }
    
    if not data:
        result['valid'] = False
        result['errors'].append("No data provided")
        return result
    
    for i, row in enumerate(data):
        arm_name = f"Arm {i + 1}"
        
        # Check for all zeros
        if all(v == 0 for v in row):
            result['warnings'].append(f"{arm_name}: All values are zero - likely OCR failure")
        
        # Check for increasing values (should decrease over time)
        for j in range(1, len(row)):
            if row[j] > row[j-1] and row[j-1] > 0:
                result['warnings'].append(
                    f"{arm_name}: Value increases at timepoint {j} ({row[j-1]} → {row[j]})"
                )
        
        # Check for reasonable starting value
        if row[0] < 10:
            result['warnings'].append(
                f"{arm_name}: Very low starting count ({row[0]}) - verify first value"
            )
    
    return result


def compare_extractions(
    ocr_result: Optional[List[List[int]]], 
    llm_result: Optional[List[List[int]]]
) -> dict:
    """
    Compare OCR and LLM extraction results to help user choose.
    
    Returns comparison metrics and recommendation.
    """
    comparison = {
        'ocr_available': ocr_result is not None,
        'llm_available': llm_result is not None,
        'recommendation': None,
        'differences': []
    }
    
    if not comparison['ocr_available'] and not comparison['llm_available']:
        comparison['recommendation'] = 'manual'
        return comparison
    
    if not comparison['ocr_available']:
        comparison['recommendation'] = 'llm'
        return comparison
    
    if not comparison['llm_available']:
        comparison['recommendation'] = 'ocr'
        return comparison
    
    # Both available - compare
    ocr_validation = validate_at_risk_data(ocr_result)
    llm_validation = validate_at_risk_data(llm_result)
    
    # Count issues
    ocr_issues = len(ocr_validation['warnings']) + len(ocr_validation['errors'])
    llm_issues = len(llm_validation['warnings']) + len(llm_validation['errors'])
    
    # Find specific differences
    for i in range(min(len(ocr_result), len(llm_result))):
        for j in range(min(len(ocr_result[i]), len(llm_result[i]))):
            if ocr_result[i][j] != llm_result[i][j]:
                comparison['differences'].append({
                    'arm': i,
                    'timepoint': j,
                    'ocr_value': ocr_result[i][j],
                    'llm_value': llm_result[i][j]
                })
    
    # Recommend based on validation
    if llm_issues < ocr_issues:
        comparison['recommendation'] = 'llm'
    elif ocr_issues < llm_issues:
        comparison['recommendation'] = 'ocr'
    else:
        # Equal issues - prefer LLM as it's generally more reliable for this task
        comparison['recommendation'] = 'llm' if comparison['differences'] else 'either'
    
    return comparison


# =============================================================================
# TEST / DEBUG
# =============================================================================

if __name__ == "__main__":
    # Simple test
    print("OCR Table Module loaded successfully")
    print("Available methods:")
    print("  1. extract_table_enhanced_ocr() - Improved EasyOCR with preprocessing")
    print("  2. extract_table_with_vision_llm() - Gemini Vision extraction")
    print("  3. extract_table_from_crop() - Legacy compatibility wrapper")
# src/core/llm_extractor.py
"""
Vision LLM Curve Extraction - v2 (Improved Accuracy)

Use Gemini Vision to extract Kaplan-Meier curves when traditional
color-based methods fail. 

v2 Improvements:
- Two-stage extraction: First analyze, then extract
- Y-axis scale detection (0-1 vs 0-100)
- Endpoint verification
- More sampling points in critical regions
- Cross-validation prompts

Cost: ~$0.002-0.004 per extraction (still very cheap)

Author: IPD Assistant
"""

import cv2
import numpy as np
import pandas as pd
import json
import re
import os
import base64
from typing import List, Optional, Dict, Tuple


# =============================================================================
# MAIN EXTRACTION FUNCTION (v2 - Two Stage)
# =============================================================================

def extract_curves_with_vision_llm(
    img_bgr: np.ndarray,
    max_time_x: float,
    max_surv_y: float = 100.0,
    num_arms: Optional[int] = None,
    arm_names: Optional[List[str]] = None,
    include_confidence_intervals: bool = False
) -> Dict:
    """
    Extract Kaplan-Meier curves using Gemini Vision (v2 - improved accuracy).
    
    Uses two-stage approach:
    1. First analyze the image to understand scale and structure
    2. Then extract precise coordinates with validation
    
    Args:
        img_bgr: Cropped graph image (BGR format)
        max_time_x: Maximum value on X-axis (e.g., 30 months)
        max_surv_y: Maximum value on Y-axis (typically 100%)
        num_arms: Expected number of treatment arms (optional, auto-detect if None)
        arm_names: Names of arms if known (optional)
        include_confidence_intervals: Whether to try extracting CI bounds
        
    Returns:
        Dict with:
        - 'status': 'success' or 'error'
        - 'arms': List of DataFrames with 'time' and 'survival' columns
        - 'metadata': Additional info from extraction
        - 'error': Error message if failed
    """
    result = {
        'status': 'error',
        'arms': [],
        'metadata': {},
        'error': None
    }
    
    if img_bgr is None or img_bgr.size == 0:
        result['error'] = "Invalid image provided"
        return result
    
    try:
        import google.generativeai as genai
    except ImportError:
        result['error'] = "google-generativeai not installed. Run: pip install google-generativeai"
        return result
    
    # Configure API
    api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        result['error'] = "No Gemini API key found. Set GOOGLE_API_KEY environment variable."
        return result
    
    genai.configure(api_key=api_key)
    
    # Encode image
    _, buffer = cv2.imencode('.png', img_bgr)
    img_b64 = base64.b64encode(buffer).decode('utf-8')
    
    image_part = {
        "mime_type": "image/png",
        "data": img_b64
    }
    
    try:
        model = genai.GenerativeModel('gemini-2.0-flash')
        
        # =====================================================================
        # STAGE 1: Analyze the image structure
        # =====================================================================
        analysis = _analyze_image_structure(model, image_part, max_time_x, max_surv_y)
        
        if analysis['status'] != 'success':
            result['error'] = f"Analysis failed: {analysis.get('error', 'Unknown')}"
            return result
        
        # =====================================================================
        # STAGE 2: Extract precise coordinates
        # =====================================================================
        extraction = _extract_precise_coordinates(
            model, image_part, analysis, 
            max_time_x, num_arms, arm_names
        )
        
        if extraction['status'] == 'success':
            result['status'] = 'success'
            result['arms'] = extraction['arms']
            result['metadata'] = {
                **analysis.get('metadata', {}),
                **extraction.get('metadata', {})
            }
        else:
            result['error'] = extraction.get('error', 'Extraction failed')
            
    except Exception as e:
        result['error'] = f"Gemini API error: {str(e)}"
    
    return result


def _analyze_image_structure(model, image_part, max_time_x: float, max_surv_y: float) -> Dict:
    """
    Stage 1: Analyze the image to understand its structure before extraction.
    """
    
    analysis_prompt = f"""Analyze this Kaplan-Meier survival curve image carefully.

I need you to identify:

1. Y-AXIS SCALE: Is the Y-axis labeled as:
   - Percentage (0-100 or 0% to 100%)
   - Probability/Proportion (0.0 to 1.0)
   - Something else?

2. Y-AXIS RANGE: What are the actual min and max values shown on the Y-axis?

3. X-AXIS RANGE: What is the maximum time value shown? (User expects ~{max_time_x})

4. NUMBER OF CURVES: How many distinct survival curves are visible?

5. CURVE COLORS: What color is each curve?

6. KEY LANDMARKS - For EACH curve, tell me:
   - Starting survival value (at time 0)
   - Survival value at the MIDDLE of the time range
   - ENDING survival value (at the rightmost point of the curve)
   - The approximate time where the curve ends

7. CURVE SEPARATION: Do the curves cross each other, or does one stay above the other?

Return your analysis as JSON:
{{
    "y_axis_scale": "percentage" or "probability",
    "y_axis_min": <number>,
    "y_axis_max": <number>,
    "x_axis_max": <number>,
    "num_curves": <number>,
    "curves": [
        {{
            "color": "blue/green/red/etc",
            "start_value": <number in the ACTUAL scale shown>,
            "middle_value": <number>,
            "middle_time": <number>,
            "end_value": <number in the ACTUAL scale shown>,
            "end_time": <number>
        }}
    ],
    "curves_cross": true/false,
    "which_curve_on_top": "description of which curve has better survival"
}}

IMPORTANT: Report the values EXACTLY as you read them from the axis scale. 
If Y-axis shows 0.0 to 1.0, report values like 0.45, not 45.
If Y-axis shows 0 to 100, report values like 45, not 0.45.

JSON response:"""

    try:
        response = model.generate_content([analysis_prompt, image_part])
        response_text = response.text.strip()
        
        # Clean markdown
        cleaned = response_text
        if cleaned.startswith('```'):
            cleaned = re.sub(r'^```(?:json)?\s*', '', cleaned)
            cleaned = re.sub(r'\s*```$', '', cleaned)
        
        data = json.loads(cleaned)
        
        return {
            'status': 'success',
            'metadata': data
        }
        
    except Exception as e:
        return {
            'status': 'error',
            'error': str(e)
        }


def _extract_precise_coordinates(
    model, image_part, analysis: Dict,
    max_time_x: float, num_arms: Optional[int], arm_names: Optional[List[str]]
) -> Dict:
    """
    Stage 2: Extract precise coordinates using information from analysis.
    """
    
    result = {
        'status': 'error',
        'arms': [],
        'metadata': {},
        'error': None
    }
    
    meta = analysis.get('metadata', {})
    
    # Determine scale conversion
    y_scale = meta.get('y_axis_scale', 'percentage')
    y_max = meta.get('y_axis_max', 100)
    x_max = meta.get('x_axis_max', max_time_x)
    
    # Build context from analysis
    curves_info = meta.get('curves', [])
    num_curves = meta.get('num_curves', num_arms or 2)
    
    landmarks_text = ""
    for i, curve in enumerate(curves_info):
        landmarks_text += f"""
Curve {i+1} ({curve.get('color', 'unknown')}):
  - Starts at: {curve.get('start_value')} at time 0
  - Middle: ~{curve.get('middle_value')} at time {curve.get('middle_time')}
  - Ends at: {curve.get('end_value')} at time {curve.get('end_time')}
"""

    arms_hint = ""
    if arm_names:
        arms_hint = f"The arms should be named: {', '.join(arm_names)}. Match colors to names based on the legend."

    # Determine if we need to convert scale
    scale_note = ""
    if y_scale == 'probability' or y_max <= 1.0:
        scale_note = """
CRITICAL: The Y-axis uses a 0.0-1.0 scale (probability).
Convert ALL values to percentage by multiplying by 100.
Example: 0.45 on the graph → report as 45 in your response."""
    
    extraction_prompt = f"""Now extract precise coordinates for each survival curve.

Based on my analysis, here's what I found:
{landmarks_text}

{scale_note}

{arms_hint}

TIME POINTS TO SAMPLE:
Extract survival values at these specific times (or nearest visible point):
0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34

For each time point, read the survival value directly from where the curve intersects that time.

VERIFICATION CHECKLIST:
□ First value should be ~100 (or close) for both curves
□ Values should generally DECREASE over time (survival curves go down)
□ The curve that's visually HIGHER should have HIGHER survival values
□ End values should match what I identified above (converted to percentage if needed)

Return JSON:
{{
    "arms": [
        {{
            "name": "Name of arm 1",
            "color": "color",
            "points": [
                {{"time": 0, "survival": 100}},
                {{"time": 1, "survival": 99}},
                ... continue for all time points where curve exists ...
            ],
            "verified_end_value": <final survival % at end of curve>
        }},
        {{
            "name": "Name of arm 2", 
            "color": "color",
            "points": [...],
            "verified_end_value": <final survival %>
        }}
    ],
    "validation": {{
        "all_curves_start_near_100": true/false,
        "all_curves_decrease": true/false,
        "relative_positions_correct": true/false
    }}
}}

ALL SURVIVAL VALUES MUST BE IN PERCENTAGE (0-100 scale).
If you read 0.5 from a 0-1 axis, report it as 50.

JSON response:"""

    try:
        response = model.generate_content([extraction_prompt, image_part])
        response_text = response.text.strip()
        
        # Clean markdown
        cleaned = response_text
        if cleaned.startswith('```'):
            cleaned = re.sub(r'^```(?:json)?\s*', '', cleaned)
            cleaned = re.sub(r'\s*```$', '', cleaned)
        
        data = json.loads(cleaned)
        
        # Process arms
        arms = []
        for arm_data in data.get('arms', []):
            points = arm_data.get('points', [])
            
            if len(points) < 5:
                continue
            
            times = []
            survivals = []
            
            for pt in points:
                t = pt.get('time', pt.get('t', pt.get('x')))
                s = pt.get('survival', pt.get('s', pt.get('y')))
                
                if t is not None and s is not None:
                    t = float(t)
                    s = float(s)
                    
                    # Validate and fix scale if needed
                    if s <= 1.5 and s >= 0:  # Likely still in 0-1 scale
                        s = s * 100
                    
                    # Clamp to valid range
                    s = max(0, min(100, s))
                    
                    if 0 <= t <= max_time_x * 1.2:
                        times.append(t)
                        survivals.append(s)
            
            if len(times) >= 5:
                df = pd.DataFrame({
                    'time': times,
                    'survival': survivals
                })
                
                df = df.sort_values('time').reset_index(drop=True)
                df = df.drop_duplicates(subset=['time'], keep='first')
                
                arm_name = arm_data.get('name', f'Arm {len(arms) + 1}')
                df.attrs['name'] = arm_name
                df.attrs['color'] = arm_data.get('color', 'unknown')
                df.attrs['verified_end'] = arm_data.get('verified_end_value')
                
                arms.append(df)
        
        if not arms:
            result['error'] = "No valid curves extracted"
            return result
        
        result['status'] = 'success'
        result['arms'] = arms
        result['metadata'] = {
            'validation': data.get('validation', {}),
            'y_scale_detected': y_scale,
            'scale_converted': y_scale == 'probability'
        }
        
        return result
        
    except json.JSONDecodeError as e:
        result['error'] = f"JSON parse error: {str(e)}"
        return result
    except Exception as e:
        result['error'] = f"Extraction error: {str(e)}"
        return result


# =============================================================================
# SINGLE-STAGE EXTRACTION (Fallback/Simple)
# =============================================================================

def extract_curves_simple(
    img_bgr: np.ndarray,
    max_time_x: float,
    max_surv_y: float = 100.0,
    num_arms: int = 2
) -> Dict:
    """
    Simpler single-stage extraction as fallback.
    """
    result = {
        'status': 'error',
        'arms': [],
        'metadata': {},
        'error': None
    }
    
    try:
        import google.generativeai as genai
    except ImportError:
        result['error'] = "google-generativeai not installed"
        return result
    
    api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        result['error'] = "No API key"
        return result
    
    genai.configure(api_key=api_key)
    
    _, buffer = cv2.imencode('.png', img_bgr)
    img_b64 = base64.b64encode(buffer).decode('utf-8')
    
    prompt = f"""Extract the Kaplan-Meier survival curves from this image.

CRITICAL INSTRUCTIONS:
1. Check the Y-axis scale. If it shows 0.0-1.0, multiply all values by 100 to get percentages.
2. There are {num_arms} curves to extract.
3. Read values at time points: 0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34
4. The X-axis maximum is approximately {max_time_x}.

For each curve, carefully trace from left to right and note the survival percentage.

Return ONLY this JSON structure:
{{
    "y_axis_is_probability": true/false,
    "arms": [
        {{
            "name": "arm name",
            "color": "color",
            "points": [{{"time": 0, "survival": <percentage 0-100>}}, ...]
        }}
    ]
}}

All survival values MUST be percentages (0-100), not decimals.

JSON:"""

    try:
        model = genai.GenerativeModel('gemini-2.0-flash')
        
        image_part = {"mime_type": "image/png", "data": img_b64}
        response = model.generate_content([prompt, image_part])
        response_text = response.text.strip()
        
        cleaned = response_text
        if cleaned.startswith('```'):
            cleaned = re.sub(r'^```(?:json)?\s*', '', cleaned)
            cleaned = re.sub(r'\s*```$', '', cleaned)
        
        data = json.loads(cleaned)
        
        was_probability = data.get('y_axis_is_probability', False)
        
        arms = []
        for arm_data in data.get('arms', []):
            points = arm_data.get('points', [])
            if len(points) < 3:
                continue
            
            times = []
            survivals = []
            
            for pt in points:
                t = float(pt.get('time', 0))
                s = float(pt.get('survival', 0))
                
                # Auto-correct if values look like probabilities
                if s <= 1.0 and s > 0:
                    s = s * 100
                
                times.append(t)
                survivals.append(max(0, min(100, s)))
            
            df = pd.DataFrame({'time': times, 'survival': survivals})
            df = df.sort_values('time').reset_index(drop=True)
            df.attrs['name'] = arm_data.get('name', f'Arm {len(arms)+1}')
            df.attrs['color'] = arm_data.get('color', 'unknown')
            arms.append(df)
        
        if arms:
            result['status'] = 'success'
            result['arms'] = arms
            result['metadata'] = {'scale_was_probability': was_probability}
        else:
            result['error'] = "No curves extracted"
            
    except Exception as e:
        result['error'] = str(e)
    
    return result


# =============================================================================
# QUICK EXTRACTION API
# =============================================================================

def quick_extract(img_bgr: np.ndarray, max_time: float, max_survival: float = 100.0) -> List[pd.DataFrame]:
    """
    Simple one-liner extraction for quick use.
    """
    result = extract_curves_with_vision_llm(img_bgr, max_time, max_survival)
    
    if result['status'] == 'success':
        return result['arms']
    else:
        print(f"Extraction failed: {result.get('error', 'Unknown error')}")
        return []


# =============================================================================
# VALIDATION HELPERS
# =============================================================================

def validate_extraction_against_image(
    img_bgr: np.ndarray,
    extracted_arms: List[pd.DataFrame],
    max_time_x: float
) -> Dict:
    """
    Ask LLM to validate if the extraction looks correct.
    """
    try:
        import google.generativeai as genai
    except ImportError:
        return {'status': 'error', 'error': 'google-generativeai not installed'}
    
    api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        return {'status': 'error', 'error': 'No API key'}
    
    genai.configure(api_key=api_key)
    
    _, buffer = cv2.imencode('.png', img_bgr)
    img_b64 = base64.b64encode(buffer).decode('utf-8')
    
    # Summarize what was extracted
    extraction_summary = []
    for i, df in enumerate(extracted_arms):
        name = df.attrs.get('name', f'Arm {i+1}')
        start = df['survival'].iloc[0]
        end = df['survival'].iloc[-1]
        end_time = df['time'].iloc[-1]
        extraction_summary.append(f"{name}: starts at {start:.1f}%, ends at {end:.1f}% (time {end_time:.0f})")
    
    prompt = f"""I extracted these survival curves from the image:

{chr(10).join(extraction_summary)}

Please verify by looking at the image:
1. Do the END values match what you see at the rightmost point of each curve?
2. Are the curves correctly identified (colors match)?
3. Is the relative ordering correct (which curve is higher)?

Reply with JSON:
{{
    "end_values_correct": true/false,
    "colors_correct": true/false,
    "ordering_correct": true/false,
    "actual_end_values": {{"curve1": <value>, "curve2": <value>}},
    "issues": ["list of any problems found"]
}}

JSON:"""

    try:
        model = genai.GenerativeModel('gemini-2.0-flash')
        image_part = {"mime_type": "image/png", "data": img_b64}
        response = model.generate_content([prompt, image_part])
        
        cleaned = response.text.strip()
        if cleaned.startswith('```'):
            cleaned = re.sub(r'^```(?:json)?\s*', '', cleaned)
            cleaned = re.sub(r'\s*```$', '', cleaned)
        
        data = json.loads(cleaned)
        return {'status': 'success', 'validation': data}
        
    except Exception as e:
        return {'status': 'error', 'error': str(e)}


# =============================================================================
# TEST
# =============================================================================

if __name__ == "__main__":
    print("LLM Curve Extractor Module v2")
    print("=" * 50)
    print("Improvements in v2:")
    print("  - Two-stage extraction (analyze then extract)")
    print("  - Y-axis scale detection (0-1 vs 0-100)")
    print("  - Endpoint verification")
    print("  - Better sampling across time range")
    print()
    print("Functions:")
    print("  - extract_curves_with_vision_llm(img, max_time, max_surv)")
    print("  - extract_curves_simple(img, max_time, max_surv) - fallback")
    print("  - validate_extraction_against_image(img, arms, max_time)")
    print("  - quick_extract(img, max_time)")
# src/core/reconstructor.py
"""
IPD Reconstruction using the Guyot Algorithm

Reconstructs Individual Patient Data from digitised Kaplan-Meier curves
and at-risk tables using the IPDfromKM R package.
"""

import pandas as pd
import numpy as np
import rpy2.robjects as ro
from rpy2.robjects import pandas2ri
from rpy2.robjects.packages import importr
from rpy2.robjects.conversion import localconverter

# Load R Package once
try:
    IPD_PKG = importr('IPDfromKM')
except:
    print("Warning: IPDfromKM R package not found. Install it in R first.")
    IPD_PKG = None


def reconstruct_survival(curve_df, time_points, risk_counts):
    """
    Reconstructs IPD for a single arm.
    
    Automatically trims at-risk data to match the curve's time range.
    Each arm is handled independently - if one curve is shorter, only
    that arm's at-risk data is trimmed.
    
    Args:
        curve_df: DataFrame with 'time' and 'survival' columns
        time_points: List of time points from at-risk table
        risk_counts: List of patient counts at each time point
        
    Returns:
        DataFrame with reconstructed IPD (time, status, arm columns)
    """
    if IPD_PKG is None:
        raise RuntimeError("IPDfromKM R package not available")
    
    # Validate inputs
    if len(time_points) != len(risk_counts):
        raise ValueError(f"Mismatch: {len(time_points)} time points vs {len(risk_counts)} risk counts")
    
    # Get curve time range
    curve_min_time = curve_df['time'].min()
    curve_max_time = curve_df['time'].max()
    
    print(f"  Curve range: {curve_min_time:.1f} - {curve_max_time:.1f} months")
    
    # --- AUTO-TRIM: Only use at-risk data within THIS curve's range ---
    time_points = np.array(time_points)
    risk_counts = np.array(risk_counts)
    
    # Find valid time points for THIS arm
    valid_mask = time_points <= curve_max_time
    time_points_trimmed = time_points[valid_mask]
    risk_counts_trimmed = risk_counts[valid_mask]
    
    if len(time_points_trimmed) < len(time_points):
        n_removed = len(time_points) - len(time_points_trimmed)
        print(f"  Trimmed {n_removed} at-risk points (curve ends at {curve_max_time:.1f})")
    
    print(f"  Using {len(time_points_trimmed)} at-risk points (0 to {time_points_trimmed[-1]:.0f})")
    
    # Ensure we have enough data points
    if len(time_points_trimmed) < 3:
        raise ValueError("Not enough at-risk time points within curve range")
    
    # --- PREPARE DATA FOR R ---
    curve_data = curve_df[['time', 'survival']].copy()
    
    # Add starting point if curve doesn't start at 0
    if curve_data['time'].min() > 0.5:
        start_row = pd.DataFrame({'time': [0.0], 'survival': [100.0]})
        curve_data = pd.concat([start_row, curve_data], ignore_index=True)
    
    # Ensure survival is in percentage (0-100), not proportion (0-1)
    if curve_data['survival'].max() <= 1.5:
        curve_data['survival'] = curve_data['survival'] * 100
    
    # Sort by time
    curve_data = curve_data.sort_values('time').reset_index(drop=True)
    
    # Convert to R objects
    with localconverter(ro.default_converter + pandas2ri.converter):
        r_curve = ro.conversion.py2rpy(curve_data)
    
    r_trisk = ro.FloatVector(time_points_trimmed.tolist())
    r_nrisk = ro.FloatVector(risk_counts_trimmed.tolist())
    
    # --- CALL R: preprocess() ---
    try:
        prep = IPD_PKG.preprocess(dat=r_curve, trisk=r_trisk, nrisk=r_nrisk, maxy=100)
    except Exception as e:
        raise RuntimeError(f"R preprocess() failed: {str(e)}")
    
    # --- CALL R: getIPD() ---
    try:
        ipd_res = IPD_PKG.getIPD(prep=prep, armID=1, tot_events=ro.NULL)
    except Exception as e:
        raise RuntimeError(f"R getIPD() failed: {str(e)}")
    
    # Convert back to Python
    with localconverter(ro.default_converter + pandas2ri.converter):
        r_ipd = ipd_res.rx2('IPD')
        ipd_df = ro.conversion.rpy2py(r_ipd)
    
    if ipd_df.shape[1] == 3:
        ipd_df.columns = ['time', 'status', 'arm']
    
    print(f"  ✓ Reconstructed {len(ipd_df)} patients")
    
    return ipd_df


def calculate_km_from_ipd(ipd_df):
    """
    Calculates the KM curve from the reconstructed IPD for plotting/validation.
    
    Args:
        ipd_df: DataFrame with 'time' and 'status' columns
        
    Returns:
        DataFrame with 'time' and 'survival' columns
    """
    from lifelines import KaplanMeierFitter
    
    kmf = KaplanMeierFitter()
    kmf.fit(ipd_df['time'], ipd_df['status'])
    
    # Get the curve
    timeline = kmf.survival_function_.index.values
    probs = kmf.survival_function_['KM_estimate'].values * 100  # Convert to percentage
    
    return pd.DataFrame({'time': timeline, 'survival': probs})
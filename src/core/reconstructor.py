# src/core/reconstructor.py
"""
IPD Reconstruction using the Guyot Algorithm

Reconstructs Individual Patient Data from digitised Kaplan-Meier curves
and at-risk tables using the IPDfromKM R package.
"""

import pandas as pd
import numpy as np

# rpy2 imports
import rpy2.robjects as ro
from rpy2.robjects import pandas2ri
from rpy2.robjects.packages import importr
from rpy2.robjects.conversion import localconverter

# Global reference - loaded lazily
_IPD_PKG = None


def _get_ipd_package():
    """Lazily load the IPDfromKM R package."""
    global _IPD_PKG
    if _IPD_PKG is None:
        with localconverter(ro.default_converter + pandas2ri.converter):
            _IPD_PKG = importr('IPDfromKM')
        print("✓ IPDfromKM R package loaded successfully")
    return _IPD_PKG


def _extract_ipd_from_result(ipd_res):
    """Extract the IPD dataframe from R result, handling different rpy2 versions."""
    
    # Debug: print what we got
    print(f"  Result type: {type(ipd_res)}")
    
    # Try different access methods
    
    # Method 1: rx2 (older rpy2)
    if hasattr(ipd_res, 'rx2'):
        try:
            return ipd_res.rx2('IPD')
        except:
            pass
    
    # Method 2: It's a NamedList - find IPD by name
    if hasattr(ipd_res, 'names') and callable(ipd_res.names):
        names = ipd_res.names()
        if names is not None:
            names_list = list(names)
            print(f"  Result names: {names_list}")
            if 'IPD' in names_list:
                idx = names_list.index('IPD')
                return ipd_res[idx]
    
    # Method 3: Direct index access - IPD is usually first element
    try:
        return ipd_res[0]
    except:
        pass
    
    # Method 4: iterate and find dataframe-like object
    for i, item in enumerate(ipd_res):
        print(f"  Item {i}: {type(item)}")
        # Return first item that looks like a dataframe
        if hasattr(item, 'shape') or 'DataFrame' in str(type(item)):
            return item
    
    raise RuntimeError(f"Could not extract IPD from result. Type: {type(ipd_res)}")


def reconstruct_survival(curve_df, time_points, risk_counts):
    """
    Reconstructs IPD for a single arm.
    """
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
    
    valid_mask = time_points <= curve_max_time
    time_points_trimmed = time_points[valid_mask]
    risk_counts_trimmed = risk_counts[valid_mask]
    
    if len(time_points_trimmed) < len(time_points):
        n_removed = len(time_points) - len(time_points_trimmed)
        print(f"  Trimmed {n_removed} at-risk points (curve ends at {curve_max_time:.1f})")
    
    print(f"  Using {len(time_points_trimmed)} at-risk points (0 to {time_points_trimmed[-1]:.0f})")
    
    if len(time_points_trimmed) < 3:
        raise ValueError("Not enough at-risk time points within curve range")
    
    # --- PREPARE DATA ---
    curve_data = curve_df[['time', 'survival']].copy()
    
    if curve_data['time'].min() > 0.5:
        start_row = pd.DataFrame({'time': [0.0], 'survival': [100.0]})
        curve_data = pd.concat([start_row, curve_data], ignore_index=True)
    
    if curve_data['survival'].max() <= 1.5:
        curve_data['survival'] = curve_data['survival'] * 100
    
    curve_data = curve_data.sort_values('time').reset_index(drop=True)
    
    # --- CALL R FUNCTIONS ---
    with localconverter(ro.default_converter + pandas2ri.converter):
        ipd_pkg = _get_ipd_package()
        
        r_curve = ro.conversion.py2rpy(curve_data)
        r_trisk = ro.FloatVector(time_points_trimmed.tolist())
        r_nrisk = ro.FloatVector(risk_counts_trimmed.tolist())
        
        try:
            prep = ipd_pkg.preprocess(dat=r_curve, trisk=r_trisk, nrisk=r_nrisk, maxy=100)
        except Exception as e:
            raise RuntimeError(f"R preprocess() failed: {str(e)}")
        
        try:
            ipd_res = ipd_pkg.getIPD(prep=prep, armID=1, tot_events=ro.NULL)
        except Exception as e:
            raise RuntimeError(f"R getIPD() failed: {str(e)}")
        
        # Extract IPD using helper function
        r_ipd = _extract_ipd_from_result(ipd_res)
        ipd_df = ro.conversion.rpy2py(r_ipd)
    
    if ipd_df.shape[1] == 3:
        ipd_df.columns = ['time', 'status', 'arm']
    
    print(f"  ✓ Reconstructed {len(ipd_df)} patients")
    
    return ipd_df


def calculate_km_from_ipd(ipd_df):
    """Calculates the KM curve from the reconstructed IPD."""
    from lifelines import KaplanMeierFitter
    
    kmf = KaplanMeierFitter()
    kmf.fit(ipd_df['time'], ipd_df['status'])
    
    timeline = kmf.survival_function_.index.values
    probs = kmf.survival_function_['KM_estimate'].values * 100
    
    return pd.DataFrame({'time': timeline, 'survival': probs})
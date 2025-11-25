# src/core/reconstructor.py
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

def reconstruct_survival(curve_df, time_points, risk_counts):
    """
    Reconstructs IPD for a single arm.
    Returns: Reconstructed KM Curve (DataFrame) and IPD (DataFrame)
    """
    # 1. Preprocess
    # Convert curve to format R expects [time, survival]
    curve_data = curve_df[['time', 'survival']].copy()
    
    # Ensure data types
    with localconverter(ro.default_converter + pandas2ri.converter):
        r_curve = ro.conversion.py2rpy(curve_data)
    
    r_trisk = ro.FloatVector(time_points)
    r_nrisk = ro.FloatVector(risk_counts)
    
    # Call R: preprocess()
    prep = IPD_PKG.preprocess(dat=r_curve, trisk=r_trisk, nrisk=r_nrisk, maxy=100)
    
    # 2. Extract IPD
    # Call R: getIPD()
    # We pass armID=1 just as a dummy placeholder
    ipd_res = IPD_PKG.getIPD(prep=prep, armID=1, tot_events=ro.NULL)
    
    # Convert back to Python
    with localconverter(ro.default_converter + pandas2ri.converter):
        r_ipd = ipd_res.rx2('IPD')
        ipd_df = ro.conversion.rpy2py(r_ipd)

    if ipd_df.shape[1] == 3:
        ipd_df.columns = ['time', 'status', 'arm']
        
    return ipd_df

def calculate_km_from_ipd(ipd_df):
    """Calculates the KM curve from the reconstructed IPD for plotting"""
    from lifelines import KaplanMeierFitter
    kmf = KaplanMeierFitter()
    kmf.fit(ipd_df['time'], ipd_df['status'])
    
    # Get the curve
    timeline = kmf.survival_function_.index.values
    probs = kmf.survival_function_['KM_estimate'].values * 100
    
    return pd.DataFrame({'time': timeline, 'survival': probs})
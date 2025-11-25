# src/core/analysis.py
import pandas as pd
import numpy as np
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.statistics import logrank_test

def run_clinical_analysis(arm_data: dict):
    """
    Runs CoxPH and LogRank test on the reconstructed IPD.
    arm_data: Dictionary { 'Arm Name': {'ipd': df}, ... }
    """
    # 1. STRICT VALIDATION: Enforce exactly 2 arms for pairwise comparison
    if len(arm_data) != 2:
        return {
            "status": "error", 
            "msg": f"Clinical analysis currently supports exactly 2 arms. You provided {len(arm_data)}."
        }
    
    # 2. Combine Data
    combined_ipd = []
    arm_names = list(arm_data.keys())
    
    # Map: First Arm -> 0 (Reference), Second Arm -> 1 (Treatment/Experiment)
    for i, (name, data) in enumerate(arm_data.items()):
        df = data['ipd'].copy()
        df['group'] = i # 0 or 1
        df['group_name'] = name
        combined_ipd.append(df)
        
    full_df = pd.concat(combined_ipd)
    
    # 3. Cox Proportional Hazards
    cph = CoxPHFitter()
    try:
        cph.fit(full_df, duration_col='time', event_col='status', formula='group')
        
        # SAFE ACCESS: Use the summary table to guarantee 'exp' (HR) scale
        # summary.loc['group'] gives the row for our variable
        summary = cph.summary.loc['group']
        
        hr = summary['exp(coef)']
        lower = summary['exp(coef) lower 95%']
        upper = summary['exp(coef) upper 95%']
        p_val = summary['p']
        
        stats = {
            "hr": hr,
            "ci_lower": lower,
            "ci_upper": upper,
            "p_value": p_val
        }
        
    except Exception as e:
        return {"status": "error", "msg": f"Cox Model Failed: {str(e)}"}
    
    # 4. Median Survival (Using KaplanMeierFitter)
    medians = {}
    try:
        for name, data in arm_data.items():
            df = data['ipd']
            kmf = KaplanMeierFitter()
            kmf.fit(df['time'], df['status'])
            medians[name] = kmf.median_survival_time_
    except Exception as e:
        return {"status": "error", "msg": f"Median Calculation Failed: {str(e)}"}
        
    return {
        "status": "success",
        "stats": stats,
        "medians": medians,
        "comparison": f"{arm_names[1]} (Group 1) vs {arm_names[0]} (Group 0)"
    }
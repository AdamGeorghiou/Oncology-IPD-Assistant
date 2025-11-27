# src/core/analysis.py
"""
Clinical Analysis Module

Performs statistical analysis on reconstructed IPD:
- Cox Proportional Hazards (HR, CI, p-value)
- Log-Rank Test
- Median Survival Times
"""

import pandas as pd
import numpy as np
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.statistics import logrank_test


def run_clinical_analysis(arm_data: dict, reference_arm: str = None):
    """
    Runs CoxPH and LogRank test on the reconstructed IPD.
    
    Args:
        arm_data: Dictionary { 'Arm Name': {'ipd': df}, ... }
        reference_arm: Name of the reference/control arm. If None, uses first arm.
        
    Returns:
        Dictionary with status, stats, medians, comparison, n_per_arm
    """
    
    # 1. STRICT VALIDATION: Enforce exactly 2 arms for pairwise comparison
    if len(arm_data) != 2:
        return {
            "status": "error",
            "msg": f"Clinical analysis currently supports exactly 2 arms. You provided {len(arm_data)}."
        }
    
    # 2. Determine arm order based on reference selection
    arm_names = list(arm_data.keys())
    
    if reference_arm and reference_arm in arm_names:
        # Put reference arm first (group 0)
        comparator_arm = [a for a in arm_names if a != reference_arm][0]
        arm_order = [reference_arm, comparator_arm]
    else:
        # Default: first arm is reference
        arm_order = arm_names
    
    reference = arm_order[0]
    comparator = arm_order[1]
    
    # 3. Combine Data
    combined_ipd = []
    
    # Reference -> group 0, Comparator -> group 1
    # HR = hazard(group 1) / hazard(group 0) = hazard(comparator) / hazard(reference)
    for i, name in enumerate(arm_order):
        data = arm_data[name]
        df = data['ipd'].copy()
        df['group'] = i  # 0 = reference, 1 = comparator
        df['group_name'] = name
        combined_ipd.append(df)
    
    full_df = pd.concat(combined_ipd)
    
    # 4. Patient counts per arm
    n_per_arm = {name: len(arm_data[name]['ipd']) for name in arm_order}
    
    # 5. Cox Proportional Hazards
    cph = CoxPHFitter()
    try:
        cph.fit(full_df, duration_col='time', event_col='status', formula='group')
        
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
    
    # 6. Log-Rank Test
    try:
        T1 = full_df[full_df['group'] == 0]['time']
        E1 = full_df[full_df['group'] == 0]['status']
        T2 = full_df[full_df['group'] == 1]['time']
        E2 = full_df[full_df['group'] == 1]['status']
        
        logrank = logrank_test(T1, T2, event_observed_A=E1, event_observed_B=E2)
        stats['logrank_p'] = logrank.p_value
    except Exception as e:
        stats['logrank_p'] = None
    
    # 7. Median Survival
    medians = {}
    try:
        for name in arm_order:
            data = arm_data[name]
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
        "n_per_arm": n_per_arm,
        "comparison": f"{comparator} vs {reference}",
        "reference_arm": reference,
        "comparator_arm": comparator
    }
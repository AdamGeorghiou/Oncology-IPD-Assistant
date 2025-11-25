# src/core/project_io.py
import json
import pandas as pd
import os
import datetime
import numpy as np

PROJECTS_DIR = "data/projects"
os.makedirs(PROJECTS_DIR, exist_ok=True)

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer): return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        return super(NpEncoder, self).default(obj)

def _safe_name(name: str) -> str:
    return "".join([c for c in name if c.isalnum() or c in (" ", "-", "_")]).strip()

def _clean_crop_metadata(crop_metadata):
    if not crop_metadata: return None
    return {k: v for k, v in crop_metadata.items() if k != "debug_image"}

def save_project(session_state, meta_info):
    """
    Saves the finalized trial state + Enriched Metadata.
    
    meta_info: dict provided by UI containing:
      - name, disease, setting, phase, endpoint
      - hr, ci_lower, ci_upper, p_value, comparison, medians
    
    This function augments meta_info with:
      - n_per_arm (derived from risk_df column 0)
      - arms (list of names)
    """
    if not meta_info.get('name'): return False, "Trial Name is required."
    
    # --- 1. ENRICH METADATA ---
    # Derive patient counts (N) from the At-Risk table (Time 0)
    n_per_arm = {}
    arms_list = []
    
    try:
        if session_state.get('risk_df') is not None:
            # Assuming first column is T=0
            first_col = session_state.risk_df.columns[0]
            # Convert to dict {ArmName: Count}
            n_series = session_state.risk_df[first_col]
            n_per_arm = n_series.to_dict()
            arms_list = list(n_per_arm.keys())
    except Exception:
        pass # Fallback if risk table is missing/malformed

    # Update meta_info with these derived stats
    meta_info['n_per_arm'] = n_per_arm
    meta_info['arms'] = arms_list

    # --- 2. PREPARE RAW INPUTS ---
    extracted_arms_data = []
    if 'extracted_arms' in session_state:
        for arm in session_state.extracted_arms:
            extracted_arms_data.append({
                "id": arm.get('id'),
                "name": arm['name'],
                "color": arm['color'],
                "source": arm['source'],
                "data": arm['data'].to_dict(orient='records')
            })
            
    risk_data = None
    if session_state.get('risk_df') is not None:
        risk_data = session_state.risk_df.to_dict()

    # --- 3. PREPARE FROZEN RESULTS (No R needed on load) ---
    frozen_results = {}
    if session_state.get('reconstruction_results'):
        for arm_name, res in session_state['reconstruction_results'].items():
            frozen_results[arm_name] = {
                "ipd": res['ipd'].to_dict(orient='records'),
                "mae": res['mae'],
                "color": res.get('color', '#000000')
            }

    # --- 4. BUILD PAYLOAD ---
    payload = {
        "schema_version": "2.1", # Bumped version for richer meta
        "saved_at": str(datetime.datetime.now()),
        "meta": meta_info,       # The rich context for LLM
        "frozen_results": frozen_results, 
        "inputs": {
            "last_step": int(session_state.get("step", 1)),
            "image_path": session_state.get('current_image_path'),
            "crop_metadata": _clean_crop_metadata(session_state.get('crop_metadata')),
            "risk_times": session_state.get('risk_times'),
            "risk_data": risk_data,
            "arms_data": extracted_arms_data
        }
    }
    
    safe_name = _safe_name(meta_info['name'])
    filepath = os.path.join(PROJECTS_DIR, f"{safe_name}.json")
    
    try:
        with open(filepath, 'w') as f:
            json.dump(payload, f, cls=NpEncoder, indent=4)
        return True, f"Trial saved to Library: {safe_name}.json"
    except Exception as e:
        return False, str(e)

def load_project_names():
    if not os.path.exists(PROJECTS_DIR): return []
    return sorted([f.replace('.json', '') for f in os.listdir(PROJECTS_DIR) if f.endswith('.json')])

def load_project_data(project_name):
    filepath = os.path.join(PROJECTS_DIR, f"{project_name}.json")
    if not os.path.exists(filepath): return None
    
    with open(filepath, 'r') as f:
        data = json.load(f)
        
    # Hydrate Frozen Results (Pure Python KM calculation)
    from src.core.reconstructor import calculate_km_from_ipd
    
    hydrated_results = {}
    for arm_name, res in data.get('frozen_results', {}).items():
        ipd_df = pd.DataFrame(res['ipd'])
        km_curve = calculate_km_from_ipd(ipd_df)
        
        hydrated_results[arm_name] = {
            "ipd": ipd_df,
            "recon_curve": km_curve,
            "mae": res['mae'],
            "color": res.get('color', 'blue'),
            "original_curve": None 
        }

    return {
        "meta": data.get('meta', {}),
        "reconstruction_results": hydrated_results,
        "inputs": data.get('inputs', {})
    }
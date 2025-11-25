# src/core/table_manager.py
import pandas as pd
import numpy as np

def create_empty_risk_table(arm_names, time_points):
    """
    Creates an empty dataframe for the At-Risk table.
    Rows = Arms, Columns = Time Points
    """
    # Create column names as strings for display
    columns = [str(t) for t in time_points]
    
    # Initialize with 0 or empty strings
    df = pd.DataFrame(index=arm_names, columns=columns)
    df = df.fillna(0) # Fill with 0s to make it easier to type numbers
    return df

def parse_risk_table(edited_df):
    """
    Converts the edited dataframe back into the format needed for the backend.
    Returns: List of lists (one per arm) containing patient counts.
    """
    # Ensure all data is numeric
    try:
        clean_df = edited_df.astype(int)
        return clean_df
    except ValueError:
        return None
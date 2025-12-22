import re
import numpy as np
import pandas as pd

def parse_area_column(area_str):
    """Parse area column - handles multiple values"""
    if pd.isna(area_str):
        return None
    area_str = str(area_str).strip()
    values = re.findall(r'\d+(?:\.\d+)?', area_str)
    if len(values) == 0:
        return None
    areas = [float(v) for v in values]
    return np.mean(areas)

def extract_voltage_features(voltage_str):
    """Extract voltage features from string"""
    if pd.isna(voltage_str):
        return None, None, 0, None
    voltage_str = str(voltage_str).strip()
    voltages = re.findall(r'(\d+)(?:/|kV)?', voltage_str)
    voltages = [int(v) for v in voltages if int(v) >= 10]
    if len(voltages) == 0:
        return None, None, 0, None
    num_levels = len(voltages)
    primary_voltage = max(voltages)
    secondary_voltage = min(voltages) if num_levels > 1 else None
    voltage_ratio = primary_voltage / secondary_voltage if secondary_voltage and secondary_voltage > 0 else None
    return primary_voltage, secondary_voltage, num_levels, voltage_ratio

# Feature order for model input (CRITICAL!):
FEATURE_NAMES = ["Area_m2", "Primary_Voltage_kV", "Secondary_Voltage_kV", 
                 "Voltage_Levels", "Voltage_Ratio", "Has_Secondary", "Has_Voltage_Ratio"]

def prepare_features(area_m2, voltage_str):
    """Prepare features for model input"""
    # Extract voltage features
    primary_v, secondary_v, num_levels, v_ratio = extract_voltage_features(voltage_str)
    
    # Create feature vector
    features = {
        "Area_m2": area_m2,
        "Primary_Voltage_kV": primary_v if primary_v else 0,
        "Secondary_Voltage_kV": secondary_v if secondary_v else 0,
        "Voltage_Levels": num_levels,
        "Voltage_Ratio": v_ratio if v_ratio else 0,
        "Has_Secondary": 1 if secondary_v else 0,
        "Has_Voltage_Ratio": 1 if v_ratio else 0
    }
    
    # Return as array in correct order
    return np.array([features[f] for f in FEATURE_NAMES]).reshape(1, -1)

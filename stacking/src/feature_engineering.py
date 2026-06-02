import pandas as pd
import numpy as np

def engineer_features(df):
    """
    Enhanced feature engineering with clinical domain knowledge.
    Implements ABCDE rule features and other domain-specific transformations.
    
    Args:
        df (pd.DataFrame): Input dataframe containing raw metadata.
        
    Returns:
        pd.DataFrame: Dataframe with added engineered features.
    """
    # Avoid modifying the original dataframe
    df = df.copy()
    
    # Ensure numerical columns are float
    num_cols = [
        'age_approx', 'clin_size_long_diam_mm', 'tbp_lv_minorAxisMM', 
        'tbp_lv_areaMM2', 'tbp_lv_perimeterMM', 'tbp_lv_deltaB', 
        'tbp_lv_radial_color_std_max', 'tbp_lv_color_std_mean', 
        'tbp_lv_norm_color', 'tbp_lv_B', 'tbp_lv_H', 'tbp_lv_A'
    ]
    for col in num_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # --- 1. AGE FEATURES ---
    if 'age_approx' in df.columns:
        # Binning
        df['age_group'] = pd.cut(df['age_approx'], bins=[0, 30, 50, 70, 100],
                                 labels=['young', 'middle', 'senior', 'elderly'])
        # Risk flag (>50 is higher risk)
        df['age_risk'] = (df['age_approx'] > 50).astype(int)
        # Non-linear effect
        df['age_squared'] = df['age_approx'] ** 2
    
    # --- 2. SIZE FEATURES (Diameter) ---
    # Fill clinical size with machine size if missing
    if 'clin_size_long_diam_mm' in df.columns and 'tbp_lv_minorAxisMM' in df.columns:
        df['lesion_size_mm'] = df['clin_size_long_diam_mm'].fillna(df['tbp_lv_minorAxisMM'])
    elif 'tbp_lv_minorAxisMM' in df.columns:
        df['lesion_size_mm'] = df['tbp_lv_minorAxisMM']
    else:
        df['lesion_size_mm'] = np.nan
        
    if 'lesion_size_mm' in df.columns:
        # Size categories
        df['size_category'] = pd.cut(df['lesion_size_mm'], bins=[-1, 6, 10, 20, 1000],
                                     labels=['small', 'medium', 'large', 'very_large'])
        # ABCD rule: Diameter > 6mm is suspicious
        df['large_lesion'] = (df['lesion_size_mm'] > 6).astype(int)
        df['size_squared'] = df['lesion_size_mm'] ** 2
        df['log_size'] = np.log1p(df['lesion_size_mm'])

    # --- 3. SHAPE FEATURES (Border) ---
    if 'tbp_lv_areaMM2' in df.columns and 'tbp_lv_perimeterMM' in df.columns:
        # Shape Regularity: Area / Perimeter^2
        # Circle has max regularity. Irregular shapes have lower values.
        df['shape_regularity'] = df['tbp_lv_areaMM2'] / (df['tbp_lv_perimeterMM']**2 + 1e-6)
        
        # Compactness (Isoperimetric quotient): 4*pi*Area / Perimeter^2
        # 1 for circle, <1 for others
        df['compactness'] = (4 * np.pi * df['tbp_lv_areaMM2']) / (df['tbp_lv_perimeterMM']**2 + 1e-6)
        
        df['log_area'] = np.log1p(df['tbp_lv_areaMM2'])
        df['log_perimeter'] = np.log1p(df['tbp_lv_perimeterMM'])
        
        df['area_to_perimeter'] = df['tbp_lv_areaMM2'] / (df['tbp_lv_perimeterMM'] + 1e-6)

    if 'tbp_lv_minorAxisMM' in df.columns and 'tbp_lv_areaMM2' in df.columns:
        # Eccentricity approximation
        df['eccentricity'] = df['tbp_lv_minorAxisMM'] / (df['tbp_lv_areaMM2']**0.5 + 1e-6)

    # --- 4. COLOR FEATURES (Color) ---
    if all(c in df.columns for c in ['tbp_lv_deltaB', 'tbp_lv_radial_color_std_max', 'tbp_lv_color_std_mean']):
        # Color Variance: Magnitude of color variation vectors
        df['color_variance'] = np.sqrt(
            df['tbp_lv_deltaB']**2 + df['tbp_lv_radial_color_std_max']**2 +
            df['tbp_lv_color_std_mean']**2
        )
        
    if all(c in df.columns for c in ['tbp_lv_deltaB', 'tbp_lv_radial_color_std_max']):
        # Contrast
        df['color_contrast'] = df['tbp_lv_deltaB'] * df['tbp_lv_radial_color_std_max']

    if 'tbp_lv_norm_color' in df.columns:
        df['color_uniformity'] = 1 / (df['tbp_lv_norm_color'] + 1e-6)
        
    if 'tbp_lv_B' in df.columns and 'tbp_lv_H' in df.columns:
        # Darkness score (Blue/Hue ratio approximation)
        df['darkness_score'] = df['tbp_lv_B'] / (df['tbp_lv_H'] + 1e-6)
        df['h_to_b_ratio'] = df['tbp_lv_H'] / (df['tbp_lv_B'] + 1e-6)

    if 'tbp_lv_A' in df.columns and 'tbp_lv_B' in df.columns:
        df['a_to_b_ratio'] = df['tbp_lv_A'] / (df['tbp_lv_B'] + 1e-6)

    # --- 5. ANATOMICAL FEATURES ---
    if 'anatom_site_general' in df.columns:
        high_risk_sites = ['torso', 'upper extremity', 'posterior torso', 'anterior torso', 'head/neck']
        df['high_risk_site'] = df['anatom_site_general'].isin(high_risk_sites).astype(int)
        
        site_risk_map = {
            'head/neck': 4, 'torso': 3, 'posterior torso': 3, 'anterior torso': 3,
            'upper extremity': 2, 'lower extremity': 2,
            'palms/soles': 1, 'oral/genital': 1
        }
        df['site_risk_score'] = df['anatom_site_general'].map(site_risk_map).fillna(0)

    # --- 6. INTERACTION FEATURES (Evolution/Risk Combinations) ---
    # Combine Age (Evolution) with other factors
    if 'age_approx' in df.columns:
        if 'lesion_size_mm' in df.columns:
            df['age_size_risk'] = df['age_approx'] * df['lesion_size_mm']
        if 'site_risk_score' in df.columns:
            df['age_site_risk'] = df['age_approx'] * df['site_risk_score']
        if 'color_variance' in df.columns:
            df['age_color_risk'] = df['age_approx'] * df['color_variance']

    if 'color_variance' in df.columns and 'lesion_size_mm' in df.columns:
        df['color_size_risk'] = df['color_variance'] * df['lesion_size_mm']
        
    if 'site_risk_score' in df.columns and 'lesion_size_mm' in df.columns:
        df['site_size_risk'] = df['site_risk_score'] * df['lesion_size_mm']

    # --- 7. ASYMMETRY SCORE (Composite) ---
    # Combine Color Norm, Radial Color Std, and Shape Irregularity
    if all(c in df.columns for c in ['tbp_lv_norm_color', 'tbp_lv_radial_color_std_max', 'shape_regularity']):
        df['asymmetry_score'] = (
            df['tbp_lv_norm_color'] + df['tbp_lv_radial_color_std_max'] +
            (1 / (df['shape_regularity'] + 1e-6))
        ) / 3

    return df

NEW_FEATURES = [
    'age_group', 'age_risk', 'age_squared',
    'lesion_size_mm', 'size_category', 'large_lesion', 'size_squared', 'log_size',
    'shape_regularity', 'compactness', 'log_area', 'log_perimeter', 'area_to_perimeter', 'eccentricity',
    'color_variance', 'color_contrast', 'color_uniformity', 'darkness_score', 'h_to_b_ratio', 'a_to_b_ratio',
    'high_risk_site', 'site_risk_score',
    'age_size_risk', 'age_site_risk', 'age_color_risk', 'color_size_risk', 'site_size_risk',
    'asymmetry_score'
]

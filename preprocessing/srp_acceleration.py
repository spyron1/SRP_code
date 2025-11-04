"""
SRP 3D Acceleration Calculator Module

Calculate cannonball model SRP acceleration in 3D (x, y, z components).
Uses shadow factor, dynamic CR, area-to-mass ratio, and solar flux scaling.

Author: Divyanshu Panday
Date: October 2025
"""

import os
import numpy as np
import pandas as pd


# Physical constant
P0_N_M2 = 4.56e-6  # Solar radiation pressure at 1 AU (N/m^2)


def add_cannonball_srp_acceleration(
    df: pd.DataFrame,
    shadow_col: str = "shadow_factor",
    cr_col: str = "CR_eff",
    a_over_m_col: str = "A_OVER_M",
    au2_over_r2_col: str = "AU2_over_R2",
    ux_col: str = "sun_to_sat_ux",
    uy_col: str = "sun_to_sat_uy",
    uz_col: str = "sun_to_sat_uz",
    out_ax: str = "srp_ax_mps2",
    out_ay: str = "srp_ay_mps2",
    out_az: str = "srp_az_mps2",
    out_mag: str = "srp_a_mag_mps2",
    use_negative_sign: bool = False,
) -> pd.DataFrame:
    """
    Add cannonball model SRP acceleration components (x, y, z) to DataFrame.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with shadow, CR, A/M, AU²/R², and unit vectors
    shadow_col : str
        Shadow factor column (0.0 = umbra, 1.0 = full sunlight)
    cr_col : str
        Radiation pressure coefficient column (dynamic or static)
    a_over_m_col : str
        Area-to-mass ratio column (m²/kg)
    au2_over_r2_col : str
        Solar flux scaling factor (AU/R)² column
    ux_col, uy_col, uz_col : str
        Sun-to-satellite unit vector component columns
    out_ax, out_ay, out_az : str
        Output column names for acceleration components (m/s²)
    out_mag : str
        Output column name for acceleration magnitude (m/s²)
    use_negative_sign : bool
        If True, use negative sign (Sun-to-sat direction is away from Sun)
        
    Returns
    -------
    pd.DataFrame
        DataFrame with added SRP acceleration columns
        
    Notes
    -----
    Formula: a_srp = ν × P₀ × CR × (A/m) × (AU/R)² × û
    where:
    - ν = shadow_factor (eclipse condition)
    - P₀ = 4.56e-6 N/m² (solar radiation pressure at 1 AU)
    - CR = reflection coefficient
    - A/m = area-to-mass ratio
    - (AU/R)² = solar flux scaling
    - û = Sun-to-satellite unit vector
    """
    required = [shadow_col, cr_col, a_over_m_col, au2_over_r2_col, 
                ux_col, uy_col, uz_col]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")

    out = df.copy()
    
    # Calculate scalar magnitude: ν × P₀ × CR × (A/m) × (AU/R)²
    mag = (
        out[shadow_col].to_numpy(float)
        * P0_N_M2
        * out[cr_col].to_numpy(float)
        * out[a_over_m_col].to_numpy(float)
        * out[au2_over_r2_col].to_numpy(float)
    )

    # Apply sign (default: positive for Sun-to-sat direction)
    sign = -1.0 if use_negative_sign else 1.0
    
    # Get unit vector components
    ux = out[ux_col].to_numpy(float)
    uy = out[uy_col].to_numpy(float)
    uz = out[uz_col].to_numpy(float)

    # Calculate acceleration components
    ax = sign * mag * ux
    ay = sign * mag * uy
    az = sign * mag * uz

    # Add to DataFrame
    out[out_ax] = ax
    out[out_ay] = ay
    out[out_az] = az
    out[out_mag] = np.sqrt(ax*ax + ay*ay + az*az)
    
    # # Print statistics
    # valid_mag = out[out_mag].dropna()
    # if len(valid_mag) > 0:
    #     print(f"\n✓ SRP acceleration calculated:")
    #     print(f"  Mean magnitude: {valid_mag.mean():.3e} m/s²")
    #     print(f"  Max magnitude:  {valid_mag.max():.3e} m/s²")
    #     print(f"  Min magnitude:  {valid_mag.min():.3e} m/s²")
    
    return out


if __name__ == "__main__":
    """Test SRP acceleration calculator."""
    print("\n" + "="*60)
    print("Testing SRP Acceleration Calculator")
    print("="*60)
    
    # Import required modules
    try:
        from .data_selector import DataSelector
        from .position_calculator import add_skyfield_position_columns
        from .shadow_calculator import add_shadow_factor_column
        from .srp_features import SRPFeatureCalculator
    except ImportError:
        from data_selector import DataSelector
        from position_calculator import add_skyfield_position_columns
        from shadow_calculator import add_shadow_factor_column
        from srp_features import SRPFeatureCalculator
    
    # Test with real satellite data
    norad_id = 39634  # Sentinel-1A

    print(f"\nProcessing NORAD ID: {norad_id}")
    
    # Step 1: Load data
    selector = DataSelector(norad_id)
    df = selector.prepare_dataset()
    print(f"✓ Loaded {len(df)} rows")
    
    # Step 2: Add positions
    df = add_skyfield_position_columns(df)
    print(f"✓ Added position columns")
    
    # Step 3: Add shadow factor
    df = add_shadow_factor_column(df)
    
    # Step 4: Add SRP features
    calc = SRPFeatureCalculator()
    df = calc.add_all_features(df)
    
    # Step 5: Add SRP acceleration (3D components)
    df = add_cannonball_srp_acceleration(df, cr_col='CR_eff')
    
    # Save preprocessed data
    output_folder = "preprocessed_data"
    os.makedirs(output_folder, exist_ok=True)
    
    output_file = os.path.join(output_folder, f"NORAD_{norad_id}_preprocessed.xlsx")
    print(f"\nSaving preprocessed data to: {output_file}")
    df.to_excel(output_file, index=False)
    print(f"✓ Saved {len(df)} rows to {output_file}")
    
    # Show sample results
    print("\n--- Sample Output ---")
    cols_to_show = ['EPOCH', 'shadow_factor', 'CR_eff', 
                    'srp_ax_mps2', 'srp_ay_mps2', 'srp_az_mps2', 'srp_a_mag_mps2']
    print(df[cols_to_show].head(10))
    
    # Show eclipse vs sunlit comparison
    print("\n--- Eclipse vs Sunlit Comparison ---")
    sunlit = df[df['shadow_factor'] == 1.0]['srp_a_mag_mps2']
    umbra = df[df['shadow_factor'] == 0.0]['srp_a_mag_mps2']
    
    if len(sunlit) > 0:
        print(f"Sunlit mean SRP: {sunlit.mean():.3e} m/s²")
    if len(umbra) > 0:
        print(f"Umbra mean SRP:  {umbra.mean():.3e} m/s²")

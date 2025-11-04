"""
Position Calculator Module

Compute satellite and Sun positions using Skyfield.

Author: Divyanshu Panday
Date: October 2025
"""

import numpy as np
import pandas as pd
from skyfield.api import load, EarthSatellite


def add_skyfield_position_columns(df, line1_col='TLE_LINE1', line2_col='TLE_LINE2', epoch_col='EPOCH'):
    """
    Add satellite position/velocity and Sun position columns using Skyfield.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with TLE and epoch columns
    line1_col : str
        TLE line 1 column name
    line2_col : str
        TLE line 2 column name
    epoch_col : str
        Epoch column name
        
    Returns
    -------
    pd.DataFrame
        DataFrame with added columns:
        - sat_x_km, sat_y_km, sat_z_km (satellite position)
        - sat_vx_kmps, sat_vy_kmps, sat_vz_kmps (satellite velocity)
        - sun_x_km, sun_y_km, sun_z_km (Sun position - Earth to Sun vector)
    """
    print("\n" + "="*60)
    print("Computing Positions (Skyfield)")
    print("="*60)
    
    df_out = df.copy()
    ts = load.timescale()
    eph = load('de421.bsp')
    sun = eph['sun']
    earth = eph['earth']
    
    print("✓ Loaded ephemeris: de421.bsp")

    # Prepare output arrays
    n = len(df_out)
    sat_x = np.full(n, np.nan)
    sat_y = np.full(n, np.nan)
    sat_z = np.full(n, np.nan)
    sat_vx = np.full(n, np.nan)
    sat_vy = np.full(n, np.nan)
    sat_vz = np.full(n, np.nan)
    sun_x = np.full(n, np.nan)
    sun_y = np.full(n, np.nan)
    sun_z = np.full(n, np.nan)
    
    success = 0
    failed = 0

    for i, row in df_out.iterrows():
        l1 = row[line1_col]
        l2 = row[line2_col]
        epoch = pd.to_datetime(row[epoch_col], utc=True, errors='coerce')
        
        if pd.isna(epoch) or not isinstance(l1, str) or not isinstance(l2, str):
            failed += 1
            continue
            
        try:
            sat = EarthSatellite(l1, l2, 'SAT', ts)
            t = ts.from_datetime(epoch.to_pydatetime())
            
            # Satellite geocentric position and velocity
            geocentric = sat.at(t)
            pos = geocentric.position.km
            vel = geocentric.velocity.km_per_s
            sat_x[i], sat_y[i], sat_z[i] = pos
            sat_vx[i], sat_vy[i], sat_vz[i] = vel
            
            # Sun position (Earth-to-Sun vector)
            sun_vec = earth.at(t).observe(sun).position.km
            sun_x[i], sun_y[i], sun_z[i] = sun_vec
            
            success += 1
        except Exception:
            failed += 1
            continue

    df_out['sat_x_km'] = sat_x
    df_out['sat_y_km'] = sat_y
    df_out['sat_z_km'] = sat_z
    df_out['sat_vx_kmps'] = sat_vx
    df_out['sat_vy_kmps'] = sat_vy
    df_out['sat_vz_kmps'] = sat_vz
    df_out['sun_x_km'] = sun_x
    df_out['sun_y_km'] = sun_y
    df_out['sun_z_km'] = sun_z
    
    print(f"\n✓ Position calculation complete:")
    print(f"  Success: {success}/{n} rows")
    print(f"  Failed: {failed}/{n} rows")
    
    return df_out


if __name__ == "__main__":
    """Test position calculator with data_selector output."""
    import sys
    sys.path.insert(0, '.')
    
    try:
        from data_selector import DataSelector
    except ImportError:
        from preprocessing.data_selector import DataSelector
    
    print("\n" + "="*60)
    print("Testing Position Calculator")
    print("="*60)
    
    # Use data from data_selector
    norad_id = 41240
    selector = DataSelector(norad_id)
    df = selector.prepare_dataset()
    
    # Add positions
    df_with_pos = add_skyfield_position_columns(df)
    
    print("\n--- Sample Output ---")
    print(df_with_pos[['EPOCH', 'sat_x_km', 'sat_y_km', 'sat_z_km', 'sun_x_km']].head(3))



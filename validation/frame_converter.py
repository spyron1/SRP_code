# -*- coding: utf-8 -*-
"""
Frame Converter for Validation

Converts satellite position/velocity between ICRF and TEME frames.
- ICRF (GCRS): Used in preprocessing (Skyfield default output)
- TEME: Required for SGP4 propagation baseline comparison

This module extracts position and velocity vectors from preprocessing DataFrame
and provides them in both ICRF and TEME frames for validation purposes.

Author: Divyanshu Panday
Date: October 2025
"""

import numpy as np
import pandas as pd
from skyfield.api import load, EarthSatellite
from skyfield.sgp4lib import TEME


class FrameConverter:
    """
    Handles frame conversions between ICRF (GCRS) and TEME for validation
    """
    
    def __init__(self):
        """Initialize timescale and ephemeris"""
        self.ts = load.timescale()
        # Load JPL ephemeris for Sun position (GCRS frame)
        self.eph = load('de421.bsp')
        self.earth = self.eph['earth']
        self.sun = self.eph['sun']
        
    def extract_state_vectors_from_df(self, df, time_index=0):
        """
        Extract position and velocity from preprocessing DataFrame
        
        Parameters:
        -----------
        df : DataFrame
            Preprocessing output with columns:
            - EPOCH: datetime
            - sat_x_km, sat_y_km, sat_z_km (ICRF frame from Skyfield)
            - sat_vx_kmps, sat_vy_kmps, sat_vz_kmps (ICRF frame from Skyfield)
            - TLE_LINE1, TLE_LINE2 (for TEME extraction)
            
        time_index : int
            Which row to extract (default: 0 = first TLE)
            
        Returns:
        --------
        dict : {
            'epoch': skyfield Time object,
            'epoch_utc': str (ISO format),
            'icrf': {
                'position_km': np.array([x, y, z]),
                'velocity_km_s': np.array([vx, vy, vz])
            },
            'teme': {
                'position_km': np.array([x, y, z]),
                'velocity_km_s': np.array([vx, vy, vz])
            },
            'sun_gcrs_km': np.array([x, y, z])  # Sun position in GCRS/ICRF
        }
        """
        print(f"\n{'='*80}")
        print(f" Frame Converter: Extracting State Vectors (Row {time_index}) ".center(80))
        print(f"{'='*80}")
        
        # Get the row
        row = df.iloc[time_index]
        
        # Extract epoch
        epoch_datetime = pd.to_datetime(row['EPOCH'], utc=True)
        epoch_time = self.ts.from_datetime(epoch_datetime.to_pydatetime())
        epoch_utc_str = epoch_datetime.strftime("%Y-%m-%dT%H:%M:%S.%fZ")
        
        print(f"\nEpoch (UTC): {epoch_utc_str}")
        
        # --- ICRF Frame (already in preprocessing DataFrame) ---
        pos_icrf_km = np.array([
            row['sat_x_km'],
            row['sat_y_km'],
            row['sat_z_km']
        ])
        
        vel_icrf_km_s = np.array([
            row['sat_vx_kmps'],  # Note: column name is 'kmps' not 'km_s'
            row['sat_vy_kmps'],
            row['sat_vz_kmps']
        ])
        
        print(f"\n--- ICRF (GCRS) Frame [from Skyfield] ---")
        print(f"Position (km):  [{pos_icrf_km[0]:12.6f}, {pos_icrf_km[1]:12.6f}, {pos_icrf_km[2]:12.6f}]")
        print(f"Velocity (km/s): [{vel_icrf_km_s[0]:11.8f}, {vel_icrf_km_s[1]:11.8f}, {vel_icrf_km_s[2]:11.8f}]")
        
        # --- TEME Frame (from TLE using Skyfield) ---
        tle_line1 = row['TLE_LINE1']
        tle_line2 = row['TLE_LINE2']
        
        sat = EarthSatellite(tle_line1, tle_line2, "SAT", self.ts)
        state = sat.at(epoch_time)
        
        # Extract TEME position and velocity
        pos_teme_km = state.frame_xyz(TEME).km
        vel_teme_km_s = state.frame_xyz_and_velocity(TEME)[1].km_per_s
        
        print(f"\n--- TEME Frame [for SGP4 compatibility] ---")
        print(f"Position (km):  [{pos_teme_km[0]:12.6f}, {pos_teme_km[1]:12.6f}, {pos_teme_km[2]:12.6f}]")
        print(f"Velocity (km/s): [{vel_teme_km_s[0]:11.8f}, {vel_teme_km_s[1]:11.8f}, {vel_teme_km_s[2]:11.8f}]")
        
        # --- Sun Position (GCRS/ICRF frame) ---
        sun_gcrs_km = self.earth.at(epoch_time).observe(self.sun).apparent().position.km
        
        print(f"\n--- Sun Vector (GCRS/ICRF) ---")
        print(f"Earth→Sun (km): [{sun_gcrs_km[0]:14.6f}, {sun_gcrs_km[1]:14.6f}, {sun_gcrs_km[2]:14.6f}]")
        
        # --- Frame Comparison ---
        print(f"\n--- Frame Comparison (ICRF vs TEME) ---")
        pos_diff = pos_icrf_km - pos_teme_km
        vel_diff = vel_icrf_km_s - vel_teme_km_s
        pos_diff_mag = np.linalg.norm(pos_diff)
        vel_diff_mag = np.linalg.norm(vel_diff)
        
        print(f"Position difference (km): [{pos_diff[0]:10.6f}, {pos_diff[1]:10.6f}, {pos_diff[2]:10.6f}]")
        print(f"Position diff magnitude:  {pos_diff_mag:10.6f} km")
        print(f"Velocity difference (km/s): [{vel_diff[0]:10.8f}, {vel_diff[1]:10.8f}, {vel_diff[2]:10.8f}]")
        print(f"Velocity diff magnitude:  {vel_diff_mag:10.8f} km/s")
        
        print(f"\n{'='*80}")
        print(f"✅ State vectors extracted successfully!")
        print(f"{'='*80}\n")
        
        return {
            'epoch': epoch_time,
            'epoch_utc': epoch_utc_str,
            'icrf': {
                'position_km': pos_icrf_km,
                'velocity_km_s': vel_icrf_km_s
            },
            'teme': {
                'position_km': pos_teme_km,
                'velocity_km_s': vel_teme_km_s
            },
            'sun_gcrs_km': sun_gcrs_km
        }
    
    def extract_all_state_vectors(self, df):
        """
        Extract ALL position/velocity vectors from DataFrame (for 24hr, 2days, 7days)
        
        Parameters:
        -----------
        df : DataFrame
            Full preprocessing output
            
        Returns:
        --------
        list of dict : List of state vectors for each time step
        """
        print(f"\n{'='*80}")
        print(f" Extracting ALL State Vectors from DataFrame ".center(80))
        print(f"{'='*80}")
        print(f"\nTotal TLEs: {len(df)}")
        
        all_states = []
        
        for idx in range(len(df)):
            state = self.extract_state_vectors_from_df(df, time_index=idx)
            all_states.append(state)
            
            # Print summary for each (compact format)
            pos = state['icrf']['position_km']
            print(f"[{idx+1:2d}/{len(df)}] {state['epoch_utc'][:19]} → ICRF: [{pos[0]:10.3f}, {pos[1]:10.3f}, {pos[2]:10.3f}] km")
        
        print(f"\n{'='*80}")
        print(f"✅ Extracted {len(all_states)} state vectors")
        print(f"{'='*80}\n")
        
        return all_states


def main():
    """
    Example usage: Extract state vectors from preprocessing output
    """
    import sys
    import os
    
    # Add parent directory to path
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    from validation.preprocessing_runner import ValidationPreprocessor
    
    print("="*80)
    print(" Frame Converter Test ".center(80))
    print("="*80)
    
    # Step 1: Run preprocessing to get DataFrame
    preprocessor = ValidationPreprocessor()
    norad_id = 39634  # Example satellite
    time_range = '24hr'  # Options: '24hr', '2days', '7days'
    
    print(f"\n[1/2] Running preprocessing for NORAD {norad_id} ({time_range})...")
    cannonball_df = preprocessor.run_preprocessing(norad_id, time_range=time_range)
    
    if cannonball_df is None:
        print("❌ Preprocessing failed!")
        return
    
    # Get full DataFrame with all features
    full_df = preprocessor.last_preprocessed_df
    
    # Step 2: Extract state vectors
    print(f"\n[2/2] Extracting state vectors...")
    converter = FrameConverter()
    
    # Extract first state vector (initial conditions)
    print("\n" + "="*80)
    print(" INITIAL CONDITIONS (First TLE) ".center(80))
    print("="*80)
    initial_state = converter.extract_state_vectors_from_df(full_df, time_index=0)
    
    # Extract all state vectors
    print("\n" + "="*80)
    print(" ALL STATE VECTORS ".center(80))
    print("="*80)
    all_states = converter.extract_all_state_vectors(full_df)
    
    # Summary
    print("\n" + "="*80)
    print(" SUMMARY ".center(80))
    print("="*80)
    print(f"\nTotal state vectors extracted: {len(all_states)}")
    print(f"Time span: {all_states[0]['epoch_utc'][:19]} → {all_states[-1]['epoch_utc'][:19]}")
    
    print(f"\n--- Initial Conditions (ICRF) ---")
    pos_i = initial_state['icrf']['position_km']
    vel_i = initial_state['icrf']['velocity_km_s']
    print(f"Position (km):  [{pos_i[0]:12.6f}, {pos_i[1]:12.6f}, {pos_i[2]:12.6f}]")
    print(f"Velocity (km/s): [{vel_i[0]:11.8f}, {vel_i[1]:11.8f}, {vel_i[2]:11.8f}]")
    
    print(f"\n--- Initial Conditions (TEME) ---")
    pos_t = initial_state['teme']['position_km']
    vel_t = initial_state['teme']['velocity_km_s']
    print(f"Position (km):  [{pos_t[0]:12.6f}, {pos_t[1]:12.6f}, {pos_t[2]:12.6f}]")
    print(f"Velocity (km/s): [{vel_t[0]:11.8f}, {vel_t[1]:11.8f}, {vel_t[2]:11.8f}]")
    
    print("="*80)
    

if __name__ == '__main__':
    main()

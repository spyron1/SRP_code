# -*- coding: utf-8 -*-
"""
Quick Test: Frame Converter

Tests the frame converter to ensure ICRF and TEME extraction works correctly.
This is a simple test to verify the validation infrastructure is set up properly.

Run this first before running the full validation workflow.

Author: Divyanshu Panday
Date: October 2025
"""

import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from validation.preprocessing_runner import ValidationPreprocessor
from validation.frame_converter import FrameConverter


def test_frame_converter():
    """
    Test frame converter with 24hr TLE data
    """
    print("=" * 80)
    print(" Frame Converter Test ".center(80))
    print("=" * 80)
    
    # Step 1: Run preprocessing
    print("\n[1/2] Running preprocessing...")
    preprocessor = ValidationPreprocessor()
    
    # Use NORAD 43476 as example
    norad_id = 43476
    time_range = '24hr'
    
    cannonball_df = preprocessor.run_preprocessing(norad_id, time_range=time_range)
    
    if cannonball_df is None:
        print("❌ Preprocessing failed!")
        return False
    
    # Get full DataFrame
    full_df = preprocessor.last_preprocessed_df
    print(f"✅ Preprocessing complete! {len(full_df)} TLEs fetched")
    
    # Step 2: Test frame converter
    print("\n[2/2] Testing frame converter...")
    converter = FrameConverter()
    
    # Extract first state
    print("\n--- Extracting Initial State (First TLE) ---")
    state = converter.extract_state_vectors_from_df(full_df, time_index=0)
    
    # Verify outputs
    print("\n" + "=" * 80)
    print(" Verification ".center(80))
    print("=" * 80)
    
    # Check ICRF
    icrf_pos = state['icrf']['position_km']
    icrf_vel = state['icrf']['velocity_km_s']
    print(f"\n✅ ICRF Position: {icrf_pos}")
    print(f"✅ ICRF Velocity: {icrf_vel}")
    
    # Check TEME
    teme_pos = state['teme']['position_km']
    teme_vel = state['teme']['velocity_km_s']
    print(f"\n✅ TEME Position: {teme_pos}")
    print(f"✅ TEME Velocity: {teme_vel}")
    
    # Check Sun vector
    sun_vec = state['sun_gcrs_km']
    print(f"\n✅ Sun Vector (GCRS): {sun_vec}")
    
    # Verify magnitude differences (should be small)
    import numpy as np
    
    icrf_mag = np.linalg.norm(icrf_pos)
    teme_mag = np.linalg.norm(teme_pos)
    diff_km = abs(icrf_mag - teme_mag)
    
    print(f"\n--- Frame Comparison ---")
    print(f"ICRF Position Magnitude: {icrf_mag:.3f} km")
    print(f"TEME Position Magnitude: {teme_mag:.3f} km")
    print(f"Difference: {diff_km:.6f} km")
    
    if diff_km < 1.0:
        print("✅ Frame difference is small (< 1 km) - Good!")
    else:
        print("⚠️ Frame difference is large - Check implementation")
    
    print("\n" + "=" * 80)
    print(" Test Complete! ".center(80))
    print("=" * 80)
    
    return True


if __name__ == '__main__':
    success = test_frame_converter()
    
    if success:
        print("\n✅ Frame converter test passed!")
        print("\nNext step: Run validation_workflow.py")
    else:
        print("\n❌ Frame converter test failed!")

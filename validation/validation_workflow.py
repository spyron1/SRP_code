# -*- coding: utf-8 -*-
"""
Complete Validation Workflow

Implements the SRP Validation Strategy:
1. Fetch TLE data (24hr, 2days, or 7days)
2. Run preprocessing (Cannonball + ML SRP)
3. Propagate using 3 methods:
   - Method A: SGP4 Baseline
   - Method B: Custom Propagator + Cannonball SRP
   - Method C: Custom Propagator + ML SRP
4. Compare to ground truth TLE
5. Calculate errors and determine winner

Usage:
------
python validation_workflow.py

Author: Divyanshu Panday
Date: October 2025
"""

import sys
import os
import numpy as np
import pandas as pd
import joblib
from datetime import timedelta

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from validation.preprocessing_runner import ValidationPreprocessor
from validation.custom_propagator import (
    propagate_from_dataframe,
    sgp4_baseline_propagation,
    compare_to_ground_truth
)
from validation.frame_converter import FrameConverter


class ValidationWorkflow:
    """
    Complete validation workflow for SRP model comparison
    """
    
    def __init__(self, username=None, password=None):
        """
        Initialize validation workflow
        
        Parameters:
        -----------
        username : str, optional
            Space-Track username
        password : str, optional
            Space-Track password
        """
        self.preprocessor = ValidationPreprocessor(username, password)
        self.frame_converter = FrameConverter()
        
    def cannonball_srp_function(self, time, r_sat_km, r_sun_km):
        """
        Cannonball SRP acceleration function for custom propagator
        
        Parameters:
        -----------
        time : skyfield Time
            Current time
        r_sat_km : array [3]
            Satellite position in ICRF (km)
        r_sun_km : array [3]
            Sun position in ICRF (km)
            
        Returns:
        --------
        array [3] : SRP acceleration in km/s²
        """
        # Get satellite parameters (from last preprocessing run)
        if not hasattr(self, 'satellite_params'):
            raise ValueError("Run preprocessing first!")
        
        # Constants
        P0 = 4.56e-6  # N/m² (solar pressure at 1 AU)
        AU = 1.496e11  # m
        
        # Get satellite parameters
        A_over_m = self.satellite_params['A_over_m']  # m²/kg
        Cr = self.satellite_params['Cr_eff']  # dynamic Cr
        shadow_factor = self.satellite_params['shadow_factor']
        
        # Sun-to-satellite vector
        r_sun_to_sat = r_sat_km - r_sun_km
        r_sun_sat_norm = np.linalg.norm(r_sun_to_sat)
        
        # Unit vector
        s_hat = r_sun_to_sat / r_sun_sat_norm
        
        # Distance factor (AU / r)²
        r_sun_sat_m = r_sun_sat_norm * 1000  # Convert to meters
        distance_factor = (AU / r_sun_sat_m) ** 2
        
        # SRP acceleration (convert N/m² to km/s²)
        # P0 * (A/m) * Cr * (AU/r)² * shadow * s_hat
        # N/m² → km/s²: 1 N/m² = 1e-6 km/s² / (m²/kg)
        a_srp_km_s2 = shadow_factor * P0 * A_over_m * Cr * distance_factor * s_hat * 1e-6
        
        return a_srp_km_s2
    
    def ml_srp_function(self, time, r_sat_km, r_sun_km):
        """
        ML-predicted SRP acceleration function for custom propagator
        
        Parameters:
        -----------
        time : skyfield Time
            Current time
        r_sat_km : array [3]
            Satellite position in ICRF (km)
        r_sun_km : array [3]
            Sun position in ICRF (km)
            
        Returns:
        --------
        array [3] : SRP acceleration in km/s²
        """
        # TODO: Implement ML prediction
        # For now, return cannonball as placeholder
        return self.cannonball_srp_function(time, r_sat_km, r_sun_km)
    
    def run_validation(self, norad_id, time_range='24hr', C_d=2.2):
        """
        Run complete validation workflow
        
        Parameters:
        -----------
        norad_id : int
            NORAD catalog ID
        time_range : str
            '24hr', '2days', or '7days'
        C_d : float
            Drag coefficient (default: 2.2)
            
        Returns:
        --------
        dict : Validation results with errors for each method
        """
        print("=" * 80)
        print(f" SRP VALIDATION WORKFLOW ".center(80))
        print("=" * 80)
        print(f"\nNORAD ID: {norad_id}")
        print(f"Time Range: {time_range}")
        print(f"Drag Coefficient: {C_d}")
        
        # Parse time range
        duration_map = {
            '24hr': 24.0,
            '2days': 48.0,
            '7days': 168.0
        }
        duration_hours = duration_map[time_range]
        
        # =================================================================
        # STEP 1: RUN PREPROCESSING (Get TLE data + Cannonball SRP)
        # =================================================================
        print("\n" + "=" * 80)
        print(" STEP 1: PREPROCESSING ".center(80))
        print("=" * 80)
        
        cannonball_df = self.preprocessor.run_preprocessing(norad_id, time_range=time_range)
        
        if cannonball_df is None:
            print("❌ Preprocessing failed!")
            return None
        
        # Get full DataFrame with all features
        full_df = self.preprocessor.last_preprocessed_df
        
        # Store satellite parameters for SRP functions
        row0 = full_df.iloc[0]
        self.satellite_params = {
            'A_over_m': row0['A_OVER_M'],
            'Cr_eff': row0['CR_eff'],
            'shadow_factor': row0['shadow_factor']
        }
        
        # Get initial epoch
        initial_epoch = pd.to_datetime(row0['EPOCH'], utc=True)
        target_epoch = initial_epoch + timedelta(hours=duration_hours)
        
        print(f"\nValidation Timeline:")
        print(f"  Initial Epoch (Day 1): {initial_epoch.strftime('%Y-%m-%d %H:%M:%S UTC')}")
        print(f"  Target Epoch:          {target_epoch.strftime('%Y-%m-%d %H:%M:%S UTC')}")
        
        # =================================================================
        # STEP 2: METHOD A - SGP4 BASELINE
        # =================================================================
        print("\n" + "=" * 80)
        print(" STEP 2: METHOD A - SGP4 BASELINE ".center(80))
        print("=" * 80)
        
        sgp4_result = sgp4_baseline_propagation(full_df, duration_hours=duration_hours)
        
        # =================================================================
        # STEP 3: METHOD B - CUSTOM PROPAGATOR + CANNONBALL SRP
        # =================================================================
        print("\n" + "=" * 80)
        print(" STEP 3: METHOD B - CUSTOM PROPAGATOR + CANNONBALL SRP ".center(80))
        print("=" * 80)
        
        cannonball_result = propagate_from_dataframe(
            full_df,
            srp_accel_func=self.cannonball_srp_function,
            duration_hours=duration_hours,
            C_d=C_d
        )
        
        # =================================================================
        # STEP 4: METHOD C - CUSTOM PROPAGATOR + ML SRP
        # =================================================================
        print("\n" + "=" * 80)
        print(" STEP 4: METHOD C - CUSTOM PROPAGATOR + ML SRP ".center(80))
        print("=" * 80)
        
        ml_result = propagate_from_dataframe(
            full_df,
            srp_accel_func=self.ml_srp_function,
            duration_hours=duration_hours,
            C_d=C_d
        )
        
        # =================================================================
        # STEP 5: COMPARE TO GROUND TRUTH
        # =================================================================
        print("\n" + "=" * 80)
        print(" STEP 5: COMPARISON TO GROUND TRUTH ".center(80))
        print("=" * 80)
        
        # Use the LAST TLE as ground truth
        ground_truth_df = full_df.copy()
        
        print("\n--- Method A: SGP4 Baseline ---")
        error_sgp4 = compare_to_ground_truth(
            sgp4_result['positions_km'],
            ground_truth_df,
            target_epoch
        )
        
        print("\n--- Method B: Cannonball SRP ---")
        error_cannonball = compare_to_ground_truth(
            cannonball_result['positions_km'],
            ground_truth_df,
            target_epoch
        )
        
        print("\n--- Method C: ML SRP ---")
        error_ml = compare_to_ground_truth(
            ml_result['positions_km'],
            ground_truth_df,
            target_epoch
        )
        
        # =================================================================
        # STEP 6: RESULTS SUMMARY
        # =================================================================
        print("\n" + "=" * 80)
        print(" VALIDATION RESULTS ".center(80))
        print("=" * 80)
        
        results = {
            'norad_id': norad_id,
            'time_range': time_range,
            'duration_hours': duration_hours,
            'initial_epoch': initial_epoch,
            'target_epoch': target_epoch,
            'errors': {
                'sgp4_km': error_sgp4['error_km'],
                'cannonball_km': error_cannonball['error_km'],
                'ml_km': error_ml['error_km']
            },
            'propagations': {
                'sgp4': sgp4_result,
                'cannonball': cannonball_result,
                'ml': ml_result
            }
        }
        
        # Print summary table
        print(f"\n{'Method':<30} {'Error (km)':<15}")
        print("-" * 45)
        print(f"{'A. SGP4 Baseline':<30} {error_sgp4['error_km']:>10.3f}")
        print(f"{'B. Cannonball SRP':<30} {error_cannonball['error_km']:>10.3f}")
        print(f"{'C. ML SRP':<30} {error_ml['error_km']:>10.3f}")
        
        # Determine winner
        print("\n" + "=" * 80)
        if error_ml['error_km'] < error_cannonball['error_km']:
            improvement = (error_cannonball['error_km'] - error_ml['error_km']) / error_cannonball['error_km'] * 100
            print(f"✅ ML SRP is BETTER than Cannonball!")
            print(f"   Improvement: {improvement:.1f}%")
        else:
            degradation = (error_ml['error_km'] - error_cannonball['error_km']) / error_cannonball['error_km'] * 100
            print(f"⚠️ Cannonball is more accurate than ML")
            print(f"   ML degradation: {degradation:.1f}%")
        
        print("=" * 80)
        
        return results


def main():
    """
    Run validation workflow
    """
    # Initialize workflow
    workflow = ValidationWorkflow()
    
    # Run validation
    results = workflow.run_validation(
        norad_id=43476,  # Example satellite
        time_range='24hr',  # Options: '24hr', '2days', '7days'
        C_d=2.2  # Drag coefficient
    )
    
    if results:
        print("\n✅ Validation complete!")
        print(f"Results saved in: {results}")


if __name__ == '__main__':
    main()

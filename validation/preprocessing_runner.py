"""
Preprocessing Runner for Validation

Runs complete preprocessing pipeline on validation TLE data:
1. Fetch TLE data (24hr, 2days, or 7days)
2. Data selection (select required columns)
3. Position calculation (satellite + Sun positions)
4. Shadow calculation (eclipse detection)
5. SRP features (geometric features)
6. Physics-based SRP acceleration (Cannonball model)

Returns only: ax, ay, az (Cannonball SRP)
"""

import sys
import os
import pandas as pd
import joblib
from dotenv import load_dotenv

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from validation.tle_collector import ValidationTLECollector
from preprocessing.data_selector import DataSelector
from preprocessing.position_calculator import add_skyfield_position_columns
from preprocessing.shadow_calculator import add_shadow_factor_column
from preprocessing.srp_features import SRPFeatureCalculator
from preprocessing.srp_acceleration import add_cannonball_srp_acceleration
from preprocessing.satellite_constants import get_params


class ValidationPreprocessor:
    """
    Runs full preprocessing pipeline on validation data
    Fetches TLE → Preprocessing → Returns ax, ay, az
    """
    
    def __init__(self, username=None, password=None):
        """
        Initialize preprocessor with Space-Track credentials
        
        Parameters:
        -----------
        username : str, optional
            Space-Track username (will load from .env if not provided)
        password : str, optional
            Space-Track password (will load from .env if not provided)
        """
        # Load credentials
        if username is None or password is None:
            load_dotenv()
            username = os.getenv("SPACETRACK_USERNAME")
            password = os.getenv("SPACETRACK_PASSWORD")
        
        # Initialize TLE collector
        self.tle_collector = ValidationTLECollector(username, password)
        
    def run_preprocessing(self, norad_id, time_range='7days'):
        """
        Run complete preprocessing pipeline
        
        Parameters:
        -----------
        norad_id : int
            NORAD catalog ID (must be in satellite_constants.py)
        time_range : str
            Time range for TLE data: '24hr', '2days', or '7days' (default: '24hr')
            
        Returns:
        --------
        DataFrame : Only EPOCH, ax, ay, az columns (Cannonball SRP acceleration)
        """
        print("=" * 80)
        print(f" Validation Preprocessing: NORAD {norad_id} ({time_range}) ".center(80))
        print("=" * 80)
        
        # Step 1: Fetch TLE data
        print(f"\n[1/5] Fetching TLE data ({time_range})...")
        if time_range == '12hr':
            # Reuse 24hr collector then filter to last 12 hours
            df_full = self.tle_collector.collect_tle_last_24hrs(norad_id)
            if df_full is not None and 'EPOCH' in df_full.columns:
                df_full['EPOCH_DT'] = pd.to_datetime(df_full['EPOCH'], utc=True)
                latest_time = df_full['EPOCH_DT'].max()
                twelve_hours_ago = latest_time - pd.Timedelta(hours=12)
                df = df_full[df_full['EPOCH_DT'] >= twelve_hours_ago].copy()
                if len(df) == 0:
                    print("⚠️ No TLEs found within last 12 hours after filtering; falling back to all fetched (24hr)")
                    df = df_full.copy()
                df = df.drop(columns=['EPOCH_DT'])
                print(f"Filtered to {len(df)} TLEs within last 12 hours")
            else:
                df = df_full
        elif time_range == '24hr':
            df = self.tle_collector.collect_tle_last_24hrs(norad_id)
        elif time_range == '2days':
            df = self.tle_collector.collect_tle_last_2days(norad_id)
        elif time_range == '7days':
            df = self.tle_collector.collect_tle_last_7days(norad_id)
        else:
            raise ValueError(f"Invalid time_range: {time_range}. Use '12hr', '24hr', '2days', or '7days'")
        
        if df is None:
            return None
        
        # Step 2: Data Selection (same as preprocessing pipeline)
        print(f"\n[2/5] Selecting TLE columns...")
        selector = DataSelector(norad_id)
        df = selector.select_tle_columns(df)
        satellite_params = get_params(norad_id)
        df = selector.add_satellite_metadata(
            df,
            mass_kg=satellite_params['mass_kg'],
            surface_area_m2=satellite_params['surface_area_m2'],
            orbit_alt_km=satellite_params['orbit_alt_km'],
            cr=satellite_params['cr']
        )
        
        # Step 3: Position Calculation
        print(f"\n[3/5] Calculating positions...")
        df = add_skyfield_position_columns(df)
        
        # Step 4: Shadow Calculation
        print(f"\n[4/5] Computing shadow factors...")
        df = add_shadow_factor_column(df)
        
        # Step 5: SRP Features + Acceleration
        print(f"\n[5/5] Adding SRP features + Cannonball acceleration...")
        calc = SRPFeatureCalculator()
        df = calc.add_all_features(df)
        df = add_cannonball_srp_acceleration(df, cr_col='CR_eff')
        
        # Save full DataFrame for ML prediction (internal use)
        self.last_preprocessed_df = df.copy()
        
        # Return ONLY EPOCH and SRP acceleration (srp_ax_mps2, srp_ay_mps2, srp_az_mps2)
        srp_results = df[['EPOCH', 'srp_ax_mps2', 'srp_ay_mps2', 'srp_az_mps2']].copy()
        
        print(f"\n" + "=" * 80)
        print(f"✅ Preprocessing Complete! Total Records: {len(srp_results)}")
        print("=" * 80)
        
        # Display first 3 rows
        print(f"\nSRP Acceleration Results:")
        print(srp_results.to_string(index=False))
        print()
        
        return srp_results
    
    def run_ml_prediction(self, model_folder='../model/trained_models'):
        """
        Run ML prediction using preprocessed features from last run_preprocessing call
        
        Parameters:
        -----------
        model_folder : str
            Path to folder containing trained models. Filenames can be
            flexible (e.g. lgb_srp_ax_mps2.pkl, rf_srp_ax_mps2.pkl). Loader
            will auto-detect any .pkl file containing the target substring
            ("srp_ax_mps2", "srp_ay_mps2", "srp_az_mps2"). If multiple
            matches exist for a target, the most recently modified file is
            used.
            
        Returns:
        --------
        DataFrame : EPOCH, ml_ax_mps2, ml_ay_mps2, ml_az_mps2 (ML predictions)
        """
        # Check if preprocessing was run
        if not hasattr(self, 'last_preprocessed_df'):
            raise ValueError("Run run_preprocessing() first to generate features!")
        
        print("=" * 80)
        print(" ML Prediction ".center(80))
        print("=" * 80)
        
        # Use the saved DataFrame with all features
        df = self.last_preprocessed_df.copy()

        # --- PREPARE CATEGORICAL FEATURES FOR PREDICTION ---
        # Convert NORAD_CAT_ID to category dtype to match training
        df['NORAD_CAT_ID'] = df['NORAD_CAT_ID'].astype('category')
        print(f"✓ Converted NORAD_CAT_ID to category type for prediction.")
        # ----------------------------------------------------
        
        # ------------------------------------------------------------------
        # Dynamic model discovery: find latest .pkl per target component
        # ------------------------------------------------------------------
        print(f"\nLoading ML models dynamically from {model_folder}...")
        if not os.path.isdir(model_folder):
            raise FileNotFoundError(f"Model folder not found: {model_folder}")

        target_keys = {
            'srp_ax_mps2': None,
            'srp_ay_mps2': None,
            'srp_az_mps2': None,
        }

        # Scan directory for candidate files
        for fname in os.listdir(model_folder):
            if not fname.lower().endswith('.pkl'):
                continue
            fpath = os.path.join(model_folder, fname)
            for t in target_keys.keys():
                if t in fname:
                    # If multiple, choose most recently modified
                    current = target_keys[t]
                    if (current is None) or (os.path.getmtime(fpath) > os.path.getmtime(current)):
                        target_keys[t] = fpath

        missing = [t for t, path in target_keys.items() if path is None]
        if missing:
            raise FileNotFoundError(
                "Missing model files for targets: " + ", ".join(missing) +
                f". Ensure model filenames contain these substrings and are placed in {model_folder}."
            )

        # Load models
        model_ax = joblib.load(target_keys['srp_ax_mps2'])
        model_ay = joblib.load(target_keys['srp_ay_mps2'])
        model_az = joblib.load(target_keys['srp_az_mps2'])
        print("✅ Models loaded:")
        for k, v in target_keys.items():
            print(f"   {k} -> {os.path.basename(v)}")
        
        # Select features for ML (same as training)
        feature_cols = [
            "NORAD_CAT_ID",
            'A_OVER_M',
            'CR_eff',
            'AU2_over_R2',
            'shadow_factor',
            'sun_to_sat_ux',
            'sun_to_sat_uy',
            'sun_to_sat_uz',
            # New dynamic & temporal features
            'sun_velocity_angle_cos',
            'sun_position_angle_cos',
            'day_of_year_sin',
            'day_of_year_cos',
            'hour_of_day_sin',
            'hour_of_day_cos'
        ]
        
        print(f"\nExtracting features: {feature_cols}")
        X = df[feature_cols]
        print("\n--- Feature Matrix X (Input to Model) ---")
        print(X.to_string())
        print("-----------------------------------------\n")

        # Predict
        print("Predicting SRP acceleration...")
        df['ml_ax_mps2'] = model_ax.predict(X)
        df['ml_ay_mps2'] = model_ay.predict(X)
        df['ml_az_mps2'] = model_az.predict(X)
        
        # --- PHYSICS-INFORMED POST-PROCESSING ---
        # Enforce zero acceleration when in shadow (shadow_factor == 0)
        print("\nApplying physics constraint: Forcing SRP to zero in shadow...")
        shadow_indices = df[df['shadow_factor'] == 0].index
        count = len(shadow_indices)
        if count > 0:
            df.loc[shadow_indices, ['ml_ax_mps2', 'ml_ay_mps2', 'ml_az_mps2']] = 0.0
            print(f"✓ Set {count} predictions to zero based on shadow factor.")
        else:
            print("✓ No data points in full shadow.")
        # -----------------------------------------
        
        # Return only ML results
        ml_results = df[['EPOCH', 'ml_ax_mps2', 'ml_ay_mps2', 'ml_az_mps2']].copy()
        
        print(f"\n" + "=" * 80)
        print(f"✅ ML Prediction Complete! Total Records: {len(ml_results)}")
        print("=" * 80)
        
        # Display results
        print(f"\nML Acceleration Results:")
        print(ml_results.to_string(index=False))
        print()
        
        return ml_results


def main():
    """
    Example usage - Complete workflow
    """
    # Initialize preprocessor
    preprocessor = ValidationPreprocessor()
    
    # Example: NORAD 59386 with 24hr data
    print("=" * 80)
    print(" Complete Validation Workflow ".center(80))
    print("=" * 80)
    print()
    
    # Step 1: Run preprocessing (Cannonball SRP)
    cannonball_df = preprocessor.run_preprocessing(norad_id=43476

, time_range='7days')
    
    if cannonball_df is not None:
        print(f"\n✅ Cannonball SRP shape: {cannonball_df.shape}")
        
        # Step 2: Run ML prediction (uses features from Step 1)
        ml_df = preprocessor.run_ml_prediction(model_folder='../model/trained_models')
        
        print(f"\n✅ ML SRP shape: {ml_df.shape}")
        
        # Compare results
        print("\n" + "=" * 80)
        print(" Comparison: Cannonball vs ML ".center(80))
        print("=" * 80)
        print(f"\nCannonball:")
        print(cannonball_df.to_string(index=False))
        print(f"\nML Prediction :")
        print(ml_df.to_string(index=False))


if __name__ == '__main__':
    main()

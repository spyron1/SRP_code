"""
SRP Comparison

Compare Physics-based (Cannonball) SRP vs ML-predicted SRP.

Metrics:
- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- Residuals (Physics - ML)
"""

import sys
import os
import pandas as pd
import numpy as np
import joblib

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class SRPComparator:
    """
    Compare Physics-based vs ML-predicted SRP accelerations
    """
    
    def __init__(self, model_folder='trained_models'):
        """
        Initialize comparator with trained ML models
        
        Parameters:
        -----------
        model_folder : str
            Folder containing trained .pkl models (default: 'trained_models')
        """
        self.model_folder = model_folder
        
        # Load trained models
        self.model_ax = joblib.load(os.path.join(model_folder, 'rf_srp_ax_mps2.pkl'))
        self.model_ay = joblib.load(os.path.join(model_folder, 'rf_srp_ay_mps2.pkl'))
        self.model_az = joblib.load(os.path.join(model_folder, 'rf_srp_az_mps2.pkl'))
        
        print(f"✅ Loaded ML models from {model_folder}")
        
    def predict_ml_srp(self, df):
        """
        Predict SRP acceleration using ML models
        
        Parameters:
        -----------
        df : DataFrame
            Preprocessed validation data with features
            
        Returns:
        --------
        DataFrame : Original df with ML predictions added
        """
        # Define feature columns (same as training)
        feature_cols = [
            'sun_sat_distance_km', 'zenith_angle_deg', 'shadow_factor',
            'cos_sun_angle', 'CR_eff', 'SRP_force_N',
            'BSTAR', 'ECCENTRICITY', 'INCLINATION', 'MEAN_MOTION'
        ]
        
        # Check if all features exist
        missing_features = [col for col in feature_cols if col not in df.columns]
        if missing_features:
            raise ValueError(f"Missing features: {missing_features}")
            
        X = df[feature_cols]
        
        # Predict
        print("Predicting ML-based SRP acceleration...")
        df['ax_ml'] = self.model_ax.predict(X)
        df['ay_ml'] = self.model_ay.predict(X)
        df['az_ml'] = self.model_az.predict(X)
        
        print("✅ ML predictions added (ax_ml, ay_ml, az_ml)")
        return df
        
    def compare_results(self, df, output_folder='validation_data'):
        """
        Compare Physics vs ML SRP and calculate metrics
        
        Parameters:
        -----------
        df : DataFrame
            Data with both Physics (ax, ay, az) and ML (ax_ml, ay_ml, az_ml)
        output_folder : str
            Output folder for results (default: 'validation_data')
            
        Returns:
        --------
        dict : Comparison metrics
        """
        # Calculate residuals
        df['residual_ax'] = df['ax'] - df['ax_ml']
        df['residual_ay'] = df['ay'] - df['ay_ml']
        df['residual_az'] = df['az'] - df['az_ml']
        
        # Calculate magnitude
        df['physics_magnitude'] = np.sqrt(df['ax']**2 + df['ay']**2 + df['az']**2)
        df['ml_magnitude'] = np.sqrt(df['ax_ml']**2 + df['ay_ml']**2 + df['az_ml']**2)
        df['residual_magnitude'] = df['physics_magnitude'] - df['ml_magnitude']
        
        # Calculate metrics
        metrics = {}
        
        for axis in ['ax', 'ay', 'az']:
            mae = np.mean(np.abs(df[f'residual_{axis}']))
            rmse = np.sqrt(np.mean(df[f'residual_{axis}']**2))
            
            metrics[axis] = {
                'MAE (m/s²)': mae,
                'RMSE (m/s²)': rmse
            }
            
        # Magnitude metrics
        metrics['magnitude'] = {
            'MAE (m/s²)': np.mean(np.abs(df['residual_magnitude'])),
            'RMSE (m/s²)': np.sqrt(np.mean(df['residual_magnitude']**2))
        }
        
        # Save results
        os.makedirs(output_folder, exist_ok=True)
        output_path = os.path.join(output_folder, 'srp_comparison_results.xlsx')
        
        df.to_excel(output_path, index=False)
        print(f"\n✅ Saved comparison results to: {output_path}")
        
        # Print metrics
        print("\n" + "=" * 60)
        print("SRP Comparison Metrics (Physics vs ML)")
        print("=" * 60)
        
        for axis, vals in metrics.items():
            print(f"\n{axis.upper()}:")
            print(f"  MAE:  {vals['MAE (m/s²)']:.2e} m/s²")
            print(f"  RMSE: {vals['RMSE (m/s²)']:.2e} m/s²")
            
        return metrics


def main():
    """
    Example usage
    """
    # Load preprocessed validation data
    INPUT_FILE = 'validation_data/NORAD_58888_physics_srp.xlsx'
    
    df = pd.read_excel(INPUT_FILE)
    print(f"Loaded {len(df)} validation records")
    
    # Initialize comparator
    comparator = SRPComparator()
    
    # Predict ML SRP
    df = comparator.predict_ml_srp(df)
    
    # Compare results
    metrics = comparator.compare_results(df)
    
    # Display sample comparison
    print("\n" + "=" * 60)
    print("Sample Comparison (First 5 Records)")
    print("=" * 60)
    
    comparison_cols = [
        'EPOCH', 'shadow_factor',
        'ax', 'ax_ml', 'residual_ax',
        'ay', 'ay_ml', 'residual_ay',
        'az', 'az_ml', 'residual_az'
    ]
    
    print(df[comparison_cols].head())


if __name__ == '__main__':
    main()

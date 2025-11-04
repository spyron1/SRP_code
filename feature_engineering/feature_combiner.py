"""
Feature Combiner Module

Combines multiple processed satellite Excel files into a single training dataset.
Selects only required columns for ML model training.

Author: Divyanshu Panday
Date: October 2025
"""

import os
import pandas as pd
import numpy as np
import warnings
from sklearn.preprocessing import LabelEncoder
import joblib

warnings.filterwarnings("ignore")


# Required columns for ML model training
REQUIRED_COLS = [
    # Identifiers & Time
    "EPOCH",
    "NORAD_CAT_ID",
    
    # Core physical features
    "A_OVER_M", 
    "CR_eff",
    "AU2_over_R2", 
    "shadow_factor",
    
    # Sun direction vectors
    "sun_to_sat_ux", 
    "sun_to_sat_uy", 
    "sun_to_sat_uz",
    
    # New dynamic & temporal features
    'sun_velocity_angle_cos',
    'sun_position_angle_cos',
    'day_of_year_sin',
    'day_of_year_cos',
    'hour_of_day_sin',
    'hour_of_day_cos',

    # Target variables
    "srp_ax_mps2", 
    "srp_ay_mps2", 
    "srp_az_mps2"
]


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove duplicate rows based on features and targets.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
        
    Returns
    -------
    pd.DataFrame
        Dataframe with duplicates removed
    """
    print(f"\n{'='*60}")
    print("Checking for Duplicates")
    print(f"{'='*60}")
    print(f"Initial rows: {len(df)}")
    
    # Check duplicates on features + targets
    feature_cols = ['NORAD_CAT_ID','NORAD_CAT_ID_original'
        'A_OVER_M', 'CR_eff', 'AU2_over_R2', 'shadow_factor',
        'sun_to_sat_ux', 'sun_to_sat_uy', 'sun_to_sat_uz'
    ]
    target_cols = ['srp_ax_mps2', 'srp_ay_mps2', 'srp_az_mps2']
    
    dup_cols = [c for c in feature_cols + target_cols if c in df.columns]
    
    if len(dup_cols) > 0:
        df_clean = df.drop_duplicates(subset=dup_cols, keep='first')
        removed = len(df) - len(df_clean)
        
        if removed > 0:
            print(f"⚠️  Found {removed} duplicate rows ({removed/len(df)*100:.2f}%)")
            print(f"✓ Removed duplicates (kept first occurrence)")
        else:
            print("✓ No duplicates found")
        
        print(f"Final rows: {len(df_clean)}")
        print(f"{'='*60}\n")
        
        return df_clean
    
    return df


def combine_training_data(
    data_folder: str,
    output_file: str = None,
    required_cols: list = None,
    encode_norad: bool = True
) -> pd.DataFrame:
    """
    Combine multiple Excel files into a single training dataset.
    
    Parameters
    ----------
    data_folder : str
        Path to folder containing processed Excel files
    output_file : str, optional
        Path to save combined data (if None, doesn't save)
    required_cols : list, optional
        List of columns to keep (default: REQUIRED_COLS)
    encode_norad : bool
        Whether to label encode NORAD_CAT_ID (default: True)
        
    Returns
    -------
    pd.DataFrame
        Combined training data with selected columns
    """
    if required_cols is None:
        required_cols = REQUIRED_COLS
    
    print(f"\n{'='*60}")
    print("Combining Training Data")
    print(f"{'='*60}")
    print(f"Data folder: {data_folder}")
    
    # Find all Excel files
    excel_files = [
        os.path.join(data_folder, f) 
        for f in os.listdir(data_folder) 
        if f.endswith('.xlsx')
    ]
    
    if not excel_files:
        raise ValueError(f"No Excel files found in {data_folder}")
    
    print(f"Found {len(excel_files)} Excel files")
    
    # Read and combine all files
    print("Reading files...")
    dfs = []
    for f in excel_files:
        df = pd.read_excel(f)
        filename = os.path.basename(f)
        print(f"  - {filename}: {len(df)} rows")
        dfs.append(df)
    
    print("\nCombining dataframes...")
    combined_df = pd.concat(dfs, ignore_index=True)
    print(f"Total rows before filtering: {len(combined_df)}")
    
    # Keep only required columns (if present)
    available_cols = [col for col in required_cols if col in combined_df.columns]
    missing_cols = [col for col in required_cols if col not in combined_df.columns]
    
    if missing_cols:
        print(f"\nWarning: Missing columns: {missing_cols}")
    
    combined_df = combined_df[available_cols]
    print(f"Selected {len(available_cols)} columns")
    
    # Remove duplicates
    combined_df = remove_duplicates(combined_df)
    
    # Label encode NORAD_CAT_ID
    if encode_norad and "NORAD_CAT_ID" in combined_df.columns:
        print("\nLabel encoding NORAD_CAT_ID...")
        le = LabelEncoder()
        
        # Keep original NORAD_CAT_ID and add encoded version
        combined_df["NORAD_CAT_ID_original"] = combined_df["NORAD_CAT_ID"]
        combined_df["NORAD_CAT_ID"] = le.fit_transform(
            combined_df["NORAD_CAT_ID"].astype(str)
        )
        
        # Save the fitted encoder for use in validation
        encoder_dir = "encoders"
        os.makedirs(encoder_dir, exist_ok=True)
        encoder_path = os.path.join(encoder_dir, "norad_id_encoder.pkl")
        joblib.dump(le, encoder_path)
        print(f"✓ Saved LabelEncoder to {encoder_path}")
        
        # Reorder columns to put original ID first
        cols = combined_df.columns.tolist()
        cols.remove("NORAD_CAT_ID_original")
        cols.insert(0, "NORAD_CAT_ID_original")
        combined_df = combined_df[cols]
        
        print(f"Unique satellites: {combined_df['NORAD_CAT_ID'].nunique()}")
        print("\nLabel encoding mapping:")
        for orig, label in zip(le.classes_, le.transform(le.classes_)):
            print(f"  {orig} -> {label}")
    
    # Save if output file specified
    if output_file:
        print(f"\nSaving to: {output_file}")
        combined_df.to_excel(output_file, index=False)
        print("✓ Saved successfully")
    
    print(f"\n{'='*60}")
    print(f"Final dataset shape: {combined_df.shape}")
    print(f"{'='*60}\n")
    
    return combined_df


if __name__ == "__main__":
    """Test feature combiner with comprehensive validation."""
    
    # Read from preprocessing/preprocessed_data folder
    data_folder = "../preprocessing/preprocessed_data"
    output_file = "training_data_combined.xlsx"
    
    try:
        df = combine_training_data(
            data_folder=data_folder,
            output_file=output_file,
            encode_norad=True
        )
        
        print("="*70)
        print(" FINAL DATASET SUMMARY ".center(70, "="))
        print("="*70)
        
        print("\n--- Dataset Info ---")
        print(df.info())
        
        print("\n--- First 5 Rows ---")
        print(df.head())
        
        print("\n--- Statistical Summary ---")
        print(df.describe())
        
        if "NORAD_CAT_ID" in df.columns:
            print("\n--- NORAD_CAT_ID Distribution ---")
            print(df["NORAD_CAT_ID"].value_counts())
        
        if "EPOCH" in df.columns:
            print("\n--- Temporal Coverage ---")
            print(f"Start: {df['EPOCH'].min()}")
            print(f"End:   {df['EPOCH'].max()}")
            print(f"Range: {pd.to_datetime(df['EPOCH'].max()) - pd.to_datetime(df['EPOCH'].min())}")
        
        print("\n" + "="*70)
        print("✓ Training data ready!")
        print("="*70 + "\n")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

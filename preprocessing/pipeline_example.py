"""
Complete Preprocessing Pipeline

Automatically processes all Excel files from data_acquisition/raw_data folder
and saves preprocessed data to preprocessed_data folder.

Author: Divyanshu Panday
Date: October 2025
"""

import os
import glob
from data_selector import DataSelector
from position_calculator import add_skyfield_position_columns
from shadow_calculator import add_shadow_factor_column
from srp_features import SRPFeatureCalculator
from srp_acceleration import add_cannonball_srp_acceleration


def process_single_file(input_file: str, output_dir: str = "preprocessed_data") -> None:
    """
    Process a single Excel file through the complete pipeline.
    
    Parameters
    ----------
    input_file : str
        Path to input Excel file
    output_dir : str
        Output directory for preprocessed data
    """
    filename = os.path.basename(input_file)
    print(f"\n{'='*70}")
    print(f" Processing: {filename} ".center(70, "="))
    print(f"{'='*70}")
    
    # Extract NORAD ID from filename (format: NORAD_xxxxx.xlsx)
    try:
        norad_id = int(filename.split('_')[1].split('.')[0])
    except:
        print(f"⚠️  Skipping {filename}: Could not extract NORAD ID")
        return
    
    # Step 1: Load data with satellite parameters
    print("\n[1/5] Loading data with satellite parameters...")
    selector = DataSelector(norad_id)
    df = selector.prepare_dataset()
    print(f"✓ Loaded {len(df)} rows")
    
    # Step 2: Add satellite & Sun positions
    print("\n[2/5] Calculating positions...")
    df = add_skyfield_position_columns(df)
    print(f"✓ Added position columns")
    
    # Step 3: Add shadow factor
    print("\n[3/5] Computing shadow factors...")
    df = add_shadow_factor_column(df)
    
    # Step 4: Add SRP features
    print("\n[4/5] Adding SRP features...")
    calc = SRPFeatureCalculator()
    df = calc.add_all_features(df)
    
    # Step 5: Add SRP acceleration
    print("\n[5/5] Calculating SRP acceleration...")
    df = add_cannonball_srp_acceleration(df, cr_col='CR_eff')
    
    # Save preprocessed data
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"NORAD_{norad_id}_preprocessed.xlsx")
    
    print(f"\n💾 Saving to: {output_file}")
    df.to_excel(output_file, index=False)
    print(f"✓ Saved {len(df)} rows, {df.shape[1]} columns")
    
    print(f"\n{'='*70}")
    print(f" ✓ Completed: {filename} ".center(70, "="))
    print(f"{'='*70}\n")


def process_all_files(raw_data_dir: str = "../data_acquisition/raw_data",
                     output_dir: str = "preprocessed_data") -> None:
    """
    Process all Excel files from raw_data directory.
    
    Parameters
    ----------
    raw_data_dir : str
        Directory containing raw Excel files
    output_dir : str
        Output directory for preprocessed data
    """
    print("\n" + "="*70)
    print(" SRP PREPROCESSING PIPELINE ".center(70, "="))
    print("="*70)
    
    # Find all Excel files
    pattern = os.path.join(raw_data_dir, "*.xlsx")
    excel_files = glob.glob(pattern)
    
    if not excel_files:
        print(f"\n⚠️  No Excel files found in: {raw_data_dir}")
        return
    
    print(f"\nFound {len(excel_files)} Excel file(s) to process:")
    for f in excel_files:
        print(f"  - {os.path.basename(f)}")
    
    # Process each file
    for i, input_file in enumerate(excel_files, 1):
        print(f"\n\n{'#'*70}")
        print(f" File {i}/{len(excel_files)} ".center(70, "#"))
        print(f"{'#'*70}")
        
        try:
            process_single_file(input_file, output_dir)
        except Exception as e:
            print(f"\n❌ Error processing {os.path.basename(input_file)}: {e}")
            continue
    
    # Summary
    print("\n" + "="*10)
    print(" ALL FILES PROCESSED ".center(10, "="))
    print("="*10)
    print(f"\nProcessed: {len(excel_files)} files")
    print(f"Output directory: {output_dir}")
    print("\n✓ Ready for feature engineering!")
    print("="*10 + "\n")


if __name__ == "__main__":
    """Run preprocessing pipeline on all raw data files."""
    
    # Process all files from data_acquisition/raw_data
    process_all_files(
        raw_data_dir="../data_acquisition/raw_data",
        output_dir="preprocessed_data"
    )



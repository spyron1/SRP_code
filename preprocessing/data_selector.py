"""
Data Selector Module

Select TLE columns and add satellite metadata.

Author: Divyanshu Panday
Date: October 2025
"""

import os
import pandas as pd

try:
    from .satellite_constants import get_params
except ImportError:
    from satellite_constants import get_params


class DataSelector:
    """Select TLE columns and add satellite metadata."""
    
    # TLE columns to select
    DEFAULT_TLE_COLUMNS = [
        "NORAD_CAT_ID",
        "EPOCH",
        "INCLINATION",
        "RA_OF_ASC_NODE",
        "ECCENTRICITY",
        "ARG_OF_PERICENTER",
        "MEAN_ANOMALY",
        "MEAN_MOTION",
        "TLE_LINE1",
        "TLE_LINE2"
    ]
    
    def __init__(self, norad_id: int):
        """
        Initialize with NORAD ID.
        
        Parameters
        ----------
        norad_id : int
            NORAD Catalog ID
        """
        self.norad_id = norad_id
        self.params = get_params(norad_id)
    
    def select_tle_columns(self, df: pd.DataFrame, columns: list = None) -> pd.DataFrame:
        """
        Select TLE columns from raw data.
        
        Parameters
        ----------
        df : pd.DataFrame
            Raw DataFrame
        columns : list, optional
            Custom columns (uses DEFAULT_TLE_COLUMNS if None)
            
        Returns
        -------
        pd.DataFrame
            Selected columns
        """
        if columns is None:
            columns = self.DEFAULT_TLE_COLUMNS
        
        # Check available columns
        available_cols = [col for col in columns if col in df.columns]
        missing_cols = [col for col in columns if col not in df.columns]
        
        if missing_cols:
            print(f"⚠️  Missing columns: {missing_cols}")
        
        df_selected = df[available_cols].copy()
        
        print(f"✓ Selected {len(available_cols)} columns, {len(df_selected)} rows")
        
        return df_selected

    
    def add_satellite_metadata(self,
                              df: pd.DataFrame,
                              mass_kg: float,
                              surface_area_m2: float,
                              orbit_alt_km: float,
                              cr: float) -> pd.DataFrame:
        """
        Add satellite physical properties to DataFrame.
        
        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame
        mass_kg : float
            Satellite mass (kg)
        surface_area_m2 : float
            Cross-sectional area (m²)
        orbit_alt_km : float
            Orbital altitude (km)
        cr : float
            Reflectivity coefficient
            
        Returns
        -------
        pd.DataFrame
            DataFrame with added metadata columns
        """
        df_out = df.copy()
        
        # Add metadata
        df_out["ORBIT_ALT_KM"] = orbit_alt_km
        df_out["CR"] = cr
        df_out["A_OVER_M"] = surface_area_m2 / mass_kg
        
        print(f"\n✓ Added satellite metadata:")
        print(f"  Altitude: {orbit_alt_km:.1f} km")
        print(f"  Area: {surface_area_m2:.1f} m²")
        print(f"  Mass: {mass_kg:.1f} kg")
        print(f"  A/m: {surface_area_m2/mass_kg:.4f} m²/kg")
        print(f"  C_R: {cr:.2f}")
        
        return df_out
    
    def prepare_dataset(self, input_file: str = None) -> pd.DataFrame:
        """
        Load raw data, select columns, and add metadata from constants.
        
        Parameters
        ----------
        input_file : str, optional
            Path to raw Excel file. If None, uses NORAD_{norad_id}.xlsx from raw_data/
            
        Returns
        -------
        pd.DataFrame
            DataFrame with selected TLE columns + metadata columns
            (ORBIT_ALT_KM, CR, A_OVER_M)
        """
        print("\n" + "="*60)
        print(f"Preparing Dataset for NORAD ID: {self.norad_id}")
        print("="*60)
        
        # Get input file
        if input_file is None:
            input_file = f"../data_acquisition/raw_data/NORAD_{self.norad_id}.xlsx"
        
        # Load raw data
        if not os.path.exists(input_file):
            raise FileNotFoundError(f"Input file not found: {input_file}")
        
        print(f"\nLoading: {input_file}")
        df_raw = pd.read_excel(input_file)
        print(f"✓ Loaded {len(df_raw)} rows")
        
        # Select columns
        df_selected = self.select_tle_columns(df_raw)
        
        # Add metadata from constants
        df_prepared = self.add_satellite_metadata(
            df_selected,
            mass_kg=self.params['mass_kg'],
            surface_area_m2=self.params['surface_area_m2'],
            orbit_alt_km=self.params['orbit_alt_km'],
            cr=self.params['cr']
        )
        
        print(f"\n✓ Dataset ready! Shape: {df_prepared.shape}")
        
        return df_prepared


if __name__ == "__main__":
    """Test data selector."""
    
    # Test with NORAD ID
    norad_id = 41240
    
    print(f"\nTesting DataSelector with NORAD ID: {norad_id}")
    print("="*60)
    
    selector = DataSelector(norad_id)
    df = selector.prepare_dataset()
    
    print("\n--- Sample Data ---")
    print(df[['NORAD_CAT_ID', 'EPOCH', 'ORBIT_ALT_KM', 'CR', 'A_OVER_M']].head(3))


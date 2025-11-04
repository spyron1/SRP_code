"""
SRP (Solar Radiation Pressure) Features Module

Calculate SRP-related geometric and physical features:
- Sun-to-satellite unit vectors
- AU/R² scaling factor
- Beta angle (orbital plane to Sun angle)
- Dynamic C_R coefficient

Author: Divyanshu Panday
Date: October 2025
"""

from typing import Optional

import numpy as np
import pandas as pd


class SRPFeatureCalculator:
    """Calculate SRP-related features for orbit propagation."""
    
    # Physical constants
    AU_KM = 149597870.7  # Astronomical Unit in kilometers
    
    def __init__(self):
        """Initialize SRP feature calculator."""
        pass
    
    def add_au_over_r_column(self,
                            df: pd.DataFrame,
                            sat_x_col: str = "sat_x_km",
                            sat_y_col: str = "sat_y_km",
                            sat_z_col: str = "sat_z_km",
                            sun_x_col: str = "sun_x_km",
                            sun_y_col: str = "sun_y_km",
                            sun_z_col: str = "sun_z_km") -> pd.DataFrame:
        """
        Add AU²/R² scaling factor column (solar flux scaling).
        
        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame with satellite and Sun positions
        sat_x_col, sat_y_col, sat_z_col : str
            Column names for satellite position
        sun_x_col, sun_y_col, sun_z_col : str
            Column names for Sun position
            
        Returns
        -------
        pd.DataFrame
            DataFrame with added columns:
            - AU2_over_R2: (AU/r)² scaling factor
            - r_sun_sat_km: Distance from Sun to satellite (km)
        """
        df_out = df.copy()
        
        AU2_over_R2_list = []
        r_sun_sat_list = []
        
        for _, row in df_out.iterrows():
            sat_pos = np.array([row[sat_x_col], row[sat_y_col], row[sat_z_col]])
            sun_pos = np.array([row[sun_x_col], row[sun_y_col], row[sun_z_col]])
            
            # Handle NaN values
            if np.isnan(sat_pos).any() or np.isnan(sun_pos).any():
                AU2_over_R2_list.append(np.nan)
                r_sun_sat_list.append(np.nan)
                continue
            
            # Sun-to-satellite vector
            sun_to_sat_vec = sat_pos - sun_pos
            r_sun_sat = np.linalg.norm(sun_to_sat_vec)
            r_sun_sat_list.append(r_sun_sat)
            
            # Calculate (AU/R)²
            if r_sun_sat != 0:
                au2_over_r2 = (self.AU_KM / r_sun_sat) ** 2
                AU2_over_R2_list.append(au2_over_r2)
            else:
                AU2_over_R2_list.append(np.nan)
        
        df_out['AU2_over_R2'] = AU2_over_R2_list
        df_out['r_sun_sat_km'] = r_sun_sat_list
        
        print(f"✓ Added AU²/R² scaling factor")
        print(f"  Mean AU²/R²: {np.nanmean(AU2_over_R2_list):.6e}")
        
        return df_out
    
    def add_unit_vector_column(self,
                              df: pd.DataFrame,
                              sat_x_col: str = "sat_x_km",
                              sat_y_col: str = "sat_y_km",
                              sat_z_col: str = "sat_z_km",
                              sun_x_col: str = "sun_x_km",
                              sun_y_col: str = "sun_y_km",
                              sun_z_col: str = "sun_z_km") -> pd.DataFrame:
        """
        Add Sun-to-satellite unit vector components.
        
        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame with satellite and Sun positions
        sat_x_col, sat_y_col, sat_z_col : str
            Column names for satellite position
        sun_x_col, sun_y_col, sun_z_col : str
            Column names for Sun position
            
        Returns
        -------
        pd.DataFrame
            DataFrame with added columns:
            - sun_to_sat_ux, sun_to_sat_uy, sun_to_sat_uz: Unit vector components
        """
        df_out = df.copy()
        
        ux_list, uy_list, uz_list = [], [], []
        
        for _, row in df_out.iterrows():
            sat_pos = np.array([row[sat_x_col], row[sat_y_col], row[sat_z_col]])
            sun_pos = np.array([row[sun_x_col], row[sun_y_col], row[sun_z_col]])
            
            # Handle NaN values
            if np.isnan(sat_pos).any() or np.isnan(sun_pos).any():
                ux_list.append(np.nan)
                uy_list.append(np.nan)
                uz_list.append(np.nan)
                continue
            
            # Sun-to-satellite vector
            sun_to_sat_vec = sat_pos - sun_pos
            r_sun_sat_norm = np.linalg.norm(sun_to_sat_vec)
            
            # Unit vector components
            if r_sun_sat_norm != 0:
                ux_list.append(sun_to_sat_vec[0] / r_sun_sat_norm)
                uy_list.append(sun_to_sat_vec[1] / r_sun_sat_norm)
                uz_list.append(sun_to_sat_vec[2] / r_sun_sat_norm)
            else:
                ux_list.append(np.nan)
                uy_list.append(np.nan)
                uz_list.append(np.nan)
        
        df_out['sun_to_sat_ux'] = ux_list
        df_out['sun_to_sat_uy'] = uy_list
        df_out['sun_to_sat_uz'] = uz_list
        
        print(f"✓ Added Sun-to-satellite unit vectors")
        
        return df_out
    
    def add_beta_angle(self,
                      df: pd.DataFrame,
                      rx_col: str = 'sat_x_km',
                      ry_col: str = 'sat_y_km',
                      rz_col: str = 'sat_z_km',
                      vx_col: str = 'sat_vx_kmps',
                      vy_col: str = 'sat_vy_kmps',
                      vz_col: str = 'sat_vz_kmps',
                      sun_rx_col: str = 'sun_x_km',
                      sun_ry_col: str = 'sun_y_km',
                      sun_rz_col: str = 'sun_z_km',
                      out_col: str = 'beta_angle_rad') -> pd.DataFrame:
        """
        Add beta angle (angle between orbital plane and Sun vector).
        
        Beta angle = arcsin(n̂ · ŝ) where n̂ is orbit normal, ŝ is Sun direction
        
        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame with satellite position, velocity, and Sun position
        rx_col, ry_col, rz_col : str
            Column names for satellite position
        vx_col, vy_col, vz_col : str
            Column names for satellite velocity
        sun_rx_col, sun_ry_col, sun_rz_col : str
            Column names for Sun position
        out_col : str
            Output column name for beta angle (radians)
            
        Returns
        -------
        pd.DataFrame
            DataFrame with added beta_angle_rad column
        """
        df_out = df.copy()
        needed_cols = [rx_col, ry_col, rz_col, vx_col, vy_col, vz_col, 
                      sun_rx_col, sun_ry_col, sun_rz_col]
        
        # Return with NaNs if required columns are missing
        if not all(c in df_out.columns for c in needed_cols):
            df_out[out_col] = np.nan
            return df_out
        
        # Vectorized calculation
        r = df_out[[rx_col, ry_col, rz_col]].to_numpy(float)
        v = df_out[[vx_col, vy_col, vz_col]].to_numpy(float)
        s = df_out[[sun_rx_col, sun_ry_col, sun_rz_col]].to_numpy(float)
        
        # Calculate orbit normal vector (h = r × v)
        h = np.cross(r, v)
        h_norm = np.linalg.norm(h, axis=1)
        with np.errstate(invalid='ignore', divide='ignore'):
            n_hat = h / h_norm[:, None]
        
        # Calculate unit vector to the Sun
        s_norm = np.linalg.norm(s, axis=1)
        with np.errstate(invalid='ignore', divide='ignore'):
            s_hat = s / s_norm[:, None]
        
        # Calculate dot product and arcsin to get beta angle
        dot = np.sum(n_hat * s_hat, axis=1)
        df_out[out_col] = np.arcsin(np.clip(dot, -1.0, 1.0))
        
        # Display statistics
        beta_deg = np.rad2deg(df_out[out_col].dropna())
        if len(beta_deg) > 0:
            print(f"✓ Added beta angle")
            print(f"  Mean β: {np.mean(beta_deg):.2f}°")
            print(f"  Range:  [{np.min(beta_deg):.2f}°, {np.max(beta_deg):.2f}°]")
        
        return df_out
    
    def add_dynamic_cr(self,
                      df: pd.DataFrame,
                      beta_col: str = 'beta_angle_rad',
                      cr_base_col: str = 'CR',
                      out_col: str = 'CR_eff',
                      k1: float = 0.03,
                      min_cr: float = 1.0,
                      max_cr: float = 2.0) -> pd.DataFrame:
        """
        Add dynamic C_R coefficient based on beta angle.
        
        Applies linear adjustment: CR_eff = CR * (1 + k1 * beta)
        
        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame with beta angle and baseline C_R
        beta_col : str
            Column name for beta angle (radians)
        cr_base_col : str
            Column name for baseline C_R value
        out_col : str
            Output column name for effective C_R
        k1 : float
            Linear adjustment coefficient (default: 0.03)
        min_cr : float
            Minimum CR value clipping (default: 1.0)
        max_cr : float
            Maximum CR value clipping (default: 2.0)
            
        Returns
        -------
        pd.DataFrame
            DataFrame with added CR_eff column
        """
        df_out = df.copy()
        
        if cr_base_col not in df_out.columns:
            raise KeyError(f"Missing base CR column '{cr_base_col}'")
        
        # Vectorized calculation
        base_cr = df_out[cr_base_col].to_numpy(float)
        beta = df_out[beta_col].to_numpy(float) if beta_col in df_out.columns else np.zeros(len(df_out))
        
        # Apply linear scaling and clamp the result
        cr_eff = base_cr * (1.0 + k1 * beta)
        df_out[out_col] = np.clip(cr_eff, min_cr, max_cr)
        
        valid_cr = df_out[out_col].dropna()
        if len(valid_cr) > 0:
            print(f"✓ Added dynamic C_R (k1={k1})")
            print(f"  Mean CR_eff: {np.mean(valid_cr):.3f}")
            print(f"  Range: [{np.min(valid_cr):.3f}, {np.max(valid_cr):.3f}]")
        
        return df_out
    
    def add_dynamic_and_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Adds dynamic, orientation-based, and cyclical time-based features.
        
        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with satellite state vectors and sun vectors.
            
        Returns
        -------
        pd.DataFrame
            DataFrame with new dynamic and temporal features added.
        """
        print("\n" + "="*60)
        print("Adding Dynamic and Temporal Features")
        print("="*60)
        
        df_out = df.copy()
        
        # Ensure EPOCH is datetime for temporal feature extraction
        df_out['EPOCH'] = pd.to_datetime(df_out['EPOCH'])

        # --- 1. Time-Based Features (Cyclical) ---
        day_of_year = df_out['EPOCH'].dt.dayofyear
        hour_of_day = df_out['EPOCH'].dt.hour + (df_out['EPOCH'].dt.minute / 60.0)
        
        df_out['day_of_year_sin'] = np.sin(2 * np.pi * day_of_year / 365.25)
        df_out['day_of_year_cos'] = np.cos(2 * np.pi * day_of_year / 365.25)
        df_out['hour_of_day_sin'] = np.sin(2 * np.pi * hour_of_day / 24.0)
        df_out['hour_of_day_cos'] = np.cos(2 * np.pi * hour_of_day / 24.0)
        print("✓ Added cyclical time-based features (day/hour sin/cos)")

        # --- 2. Angle-Based Features (Vectorized) ---
        # Required vectors
        pos_vecs = df_out[['sat_x_km', 'sat_y_km', 'sat_z_km']].values
        vel_vecs = df_out[['sat_vx_kmps', 'sat_vy_kmps', 'sat_vz_kmps']].values
        sun_unit_vecs = df_out[['sun_to_sat_ux', 'sun_to_sat_uy', 'sun_to_sat_uz']].values

        # Normalize position vectors to get r_hat
        pos_norm = np.linalg.norm(pos_vecs, axis=1, keepdims=True)
        pos_unit_vecs = pos_vecs / pos_norm
        
        # Dot product for sun-position angle
        df_out['sun_position_angle_cos'] = np.sum(pos_unit_vecs * sun_unit_vecs, axis=1)
        print("✓ Added sun-position angle feature")

        # Normalize velocity vectors to get v_hat
        vel_norm = np.linalg.norm(vel_vecs, axis=1, keepdims=True)
        vel_unit_vecs = vel_vecs / vel_norm
        
        # Dot product for sun-velocity angle
        df_out['sun_velocity_angle_cos'] = np.sum(vel_unit_vecs * sun_unit_vecs, axis=1)
        print("✓ Added sun-velocity angle feature (most important)")
        
        return df_out

    def add_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add all SRP features in one call.
        
        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame with satellite/Sun position and velocity
            
        Returns
        -------
        pd.DataFrame
            DataFrame with all SRP features added
        """
        print("\n" + "="*60)
        print("Adding SRP Features")
        print("="*60)
        
        df_out = df.copy()
        
        # 1. AU²/R² scaling
        df_out = self.add_au_over_r_column(df_out)
        
        # 2. Unit vectors
        df_out = self.add_unit_vector_column(df_out)
        
        # 3. Beta angle (if velocity columns exist)
        if 'sat_vx_kmps' in df_out.columns:
            df_out = self.add_beta_angle(df_out)
            
            # 4. Dynamic C_R (if CR and beta exist)
            if 'CR' in df_out.columns and 'beta_angle_rad' in df_out.columns:
                df_out = self.add_dynamic_cr(df_out)

        # 5. Add new dynamic and temporal features
        df_out = self.add_dynamic_and_temporal_features(df_out)
        
        print("\n✓ All SRP features added successfully!")
        
        return df_out


if __name__ == "__main__":
    """Test SRP feature calculator with complete pipeline."""
    print("\n" + "="*60)
    print("Testing SRPFeatureCalculator - Complete Pipeline")
    print("="*60)
    
    # Import required modules
    try:
        from .data_selector import DataSelector
        from .position_calculator import add_skyfield_position_columns
        from .shadow_calculator import add_shadow_factor_column
    except ImportError:
        from data_selector import DataSelector
        from position_calculator import add_skyfield_position_columns
        from shadow_calculator import add_shadow_factor_column
    
    # Test with real satellite data
    norad_id = 41240  # RISAT
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
    
    # Show sample results
    print("\n--- Sample Output ---")
    print(df[['EPOCH', 'shadow_factor', 'beta_angle_rad', 'CR_eff','AU2_over_R2']].head(10))

    #other updated feature unit vector columns
    print(df[['sun_to_sat_ux', 'sun_to_sat_uy', 'sun_to_sat_uz']].head(10))

    # Show new dynamic features
    print("\n--- New Dynamic Features ---")
    new_cols = [
        'sun_velocity_angle_cos', 
        'sun_position_angle_cos',
        'day_of_year_sin', 
        'day_of_year_cos', 
        'hour_of_day_sin', 
        'hour_of_day_cos'
    ]
    print(df[new_cols].head(10))

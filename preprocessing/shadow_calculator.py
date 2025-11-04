"""
Shadow/Eclipse Calculator Module

Cone-based shadow model for computing eclipse conditions (umbra/penumbra).
Calculates illumination fraction for satellites based on geometric shadow analysis.

Author: Divyanshu Panday
Date: October 2025
"""

import numpy as np
import pandas as pd


def angle_between(v1: np.ndarray, v2: np.ndarray) -> float:
    """Returns the angle in radians between two vectors (safe for degenerates)."""
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 == 0.0 or n2 == 0.0:
        return 0.0
    cosang = np.dot(v1, v2) / (n1 * n2)
    return float(np.arccos(np.clip(cosang, -1.0, 1.0)))


# ...existing code...
def compute_shadow_factor(
    sat_pos_km: np.ndarray,
    sun_pos_km: np.ndarray,
    earth_radius_km: float = 6378.137,
    sun_radius_km: float = 696340.0,
    *,
    use_oblate: bool = True, 
    earth_equatorial_radius_km: float = 6378.137,
    earth_polar_radius_km: float = 6356.7523142,
    atmos_expansion_km: float = 0.0,
    return_region: bool = False,
    limb_darkening: bool = False,
    limb_u: float = 0.6,
    limb_grid: int = 120
) -> float | tuple[float, str]:
    """
    """
    sat_to_earth = -sat_pos_km
    sat_to_sun = sun_pos_km - sat_pos_km
    r_sc = np.linalg.norm(sat_pos_km)
    d_sun_sc = np.linalg.norm(sat_to_sun)
    if r_sc == 0.0 or d_sun_sc == 0.0:
        return (1.0, 'full') if return_region else 1.0

    # Effective Earth radius
    if use_oblate:
        x, y, z = sat_pos_km
        rho = np.hypot(x, y)
        if rho == 0.0:
            geoc_lat = np.pi/2 if z >= 0 else -np.pi/2
        else:
            geoc_lat = np.arctan2(z, rho)
        a = earth_equatorial_radius_km
        b = earth_polar_radius_km
        sinφ = np.sin(geoc_lat)
        cosφ = np.cos(geoc_lat)
        Reff = (a * b) / np.sqrt((a * sinφ)**2 + (b * cosφ)**2)
    else:
        Reff = earth_radius_km
    Reff += atmos_expansion_km

    alpha_earth = np.arcsin(np.clip(Reff / r_sc, -1.0, 1.0))
    alpha_sun = np.arcsin(np.clip(sun_radius_km / d_sun_sc, -1.0, 1.0))
    theta = angle_between(sat_to_earth, sat_to_sun)

    earth_larger = alpha_earth > alpha_sun

    # Regions (geometry only)
    if earth_larger and theta <= (alpha_earth - alpha_sun):
        # Umbra
        result = (0.0, 'umbra') if return_region else 0.0
        return result
    if (not earth_larger) and theta <= (alpha_sun - alpha_earth):
        # Antumbra (annular)
        if not limb_darkening:
            f = 1.0 - (alpha_earth / alpha_sun)**2
        else:
            r2 = (alpha_earth / alpha_sun)**2
            r = alpha_earth / alpha_sun
            # Blocked flux for inner disk with linear limb darkening analytic:
            # F_block = π[(1-u) r^2 + (2u/3)(1 - (1 - r^2)^{3/2})]
            # Total flux = π(1 - u/3)
            F_block = np.pi * ((1 - limb_u) * r2 + (2 * limb_u / 3.0) * (1 - (1 - r2)**1.5))
            F_tot = np.pi * (1 - limb_u / 3.0)
            f = 1.0 - F_block / F_tot
        f = float(np.clip(f, 0.0, 1.0))
        return (f, 'antumbra') if return_region else f
    if theta >= (alpha_earth + alpha_sun):
        # Full illumination
        result = (1.0, 'full') if return_region else 1.0
        return result

    # Penumbra
    if not limb_darkening:
        # Uniform disk analytic overlap
        R_big = max(alpha_earth, alpha_sun)
        r_small = min(alpha_earth, alpha_sun)
        d = theta
        if d <= 1e-15:
            overlap_area = np.pi * r_small**2
        else:
            arg1 = np.clip((d**2 + r_small**2 - R_big**2) / (2 * d * r_small), -1.0, 1.0)
            arg2 = np.clip((d**2 + R_big**2 - r_small**2) / (2 * d * R_big), -1.0, 1.0)
            part1 = r_small**2 * np.arccos(arg1)
            part2 = R_big**2 * np.arccos(arg2)
            part3 = 0.5 * np.sqrt(
                max(0.0,
                    (-d + r_small + R_big) *
                    (d + r_small - R_big) *
                    (d - r_small + R_big) *
                    (d + r_small + R_big))
            )
            overlap_area = part1 + part2 - part3
        sun_area = np.pi * alpha_sun**2
        f = 1.0 - overlap_area / sun_area
        f = float(np.clip(f, 0.0, 1.0))
        return (f, 'penumbra') if return_region else f
    else:
        # Limb darkening numeric integration over Sun disk
        # Normalize geometry: Sun radius = 1
        R_sun_norm = 1.0
        R_earth_norm = alpha_earth / alpha_sun
        d_norm = theta / alpha_sun  # center separation in Sun radii
        # Place Sun at (0,0), Earth at (d_norm,0)
        N = max(20, int(limb_grid))
        xs = np.linspace(-1.0, 1.0, N)
        dx = xs[1] - xs[0]
        area_cell = dx * dx
        blocked_flux = 0.0
        total_flux = 0.0
        R2_earth = R_earth_norm**2
        for x in xs:
            # y span reduction by circle limit
            for y in xs:
                if x*x + y*y <= 1.0:
                    r_sq = x*x + y*y
                    # mu = sqrt(1 - r^2)
                    mu = np.sqrt(max(0.0, 1.0 - r_sq))
                    I = (1 - limb_u + limb_u * mu)
                    total_flux += I * area_cell
                    # Occulted?
                    dx_e = x - d_norm
                    if dx_e*dx_e + y*y <= R2_earth:
                        blocked_flux += I * area_cell
        # Normalize: total flux should approximate π(1 - u/3)
        if total_flux <= 0:
            f = 1.0
        else:
            f = 1.0 - blocked_flux / total_flux
        f = float(np.clip(f, 0.0, 1.0))
        return (f, 'penumbra') if return_region else f


def add_shadow_factor_column(
    df: pd.DataFrame,
    sat_x_col: str = "sat_x_km",
    sat_y_col: str = "sat_y_km",
    sat_z_col: str = "sat_z_km",
    sun_x_col: str = "sun_x_km",
    sun_y_col: str = "sun_y_km",
    sun_z_col: str = "sun_z_km",
    **kwargs
) -> pd.DataFrame:
    """
    Add shadow_factor (and optionally shadow_region) columns.
    Extra kwargs go to compute_shadow_factor (e.g. return_region=True, use_oblate=True,
    atmos_expansion_km=50, limb_darkening=True, limb_u=0.6, limb_grid=140).
    """
    df_out = df.copy()
    want_region = kwargs.get("return_region", False)
    factors = []
    regions = []
    for _, row in df_out.iterrows():
        sat_pos = np.array([row[sat_x_col], row[sat_y_col], row[sat_z_col]])
        sun_pos = np.array([row[sun_x_col], row[sun_y_col], row[sun_z_col]])
        if np.isnan(sat_pos).any() or np.isnan(sun_pos).any():
            factors.append(np.nan)
            if want_region:
                regions.append(None)
            continue
        val = compute_shadow_factor(sat_pos, sun_pos, **kwargs)
        if want_region:
            f, reg = val
            factors.append(f)
            regions.append(reg)
        else:
            factors.append(val)
    df_out["shadow_factor"] = factors
    if want_region:
        df_out["shadow_region"] = regions
    return df_out


if __name__ == "__main__":
    """Test shadow calculator."""
    print("\n" + "="*60)
    print("Testing Shadow Calculator")
    print("="*60)
    
    # Import required modules
    try:
        from .data_selector import DataSelector
        from .position_calculator import add_skyfield_position_columns
    except ImportError:
        from data_selector import DataSelector
        from position_calculator import add_skyfield_position_columns
    
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
    
    # Show sample results
    print("\n--- Sample Output ---")
    df_filtered = df[(df['shadow_factor'] >= 0.1) & (df['shadow_factor'] < 0.9)].head(10)
    print(df_filtered[['EPOCH', 'shadow_factor']])
    



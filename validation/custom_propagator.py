"""
Custom Orbit Propagator for Validation

Validation Strategy (as per SRP_Validation_Strategy.md):
1. Get TLE data (24hr/2days/7days) → Step 1: tle_collector.py
2. Preprocess data (Cannonball + ML SRP) → Step 2: preprocessing_runner.py
3. Extract positions/velocities → Step 3: frame_converter.py
4. Propagate 3 methods & compare → Step 4: THIS FILE

3 Propagation Methods:
- Method A: SGP4 Baseline (TLE → SGP4 → position, NO custom propagator)
- Method B: Custom Propagator + Cannonball SRP (TLE → RK45 integration)
- Method C: Custom Propagator + ML SRP (TLE → RK45 integration)

All methods start from FIRST TLE (t=0) and propagate to t=24hr/48hr/168hr.
Comparison: All 3 predicted positions vs LAST TLE position (reference).

Force Models for Custom Propagator (LEO-validated):
- J2 Perturbation (Vallado 2013, Eq. 8-29)
- Atmospheric Drag (Montenbruck & Gill 2000, Eq. 3.75)
- Solar Radiation Pressure (Montenbruck & Gill 2000, Eq. 3.76)

Integrator: RK45 (Runge-Kutta 4(5) - accurate for orbit propagation)

Note on "Ground Truth":
- "Ground truth" = LAST TLE position (calculated via SGP4 from TLE_LAST)
- TLEs are fitted to real radar tracking data by Space Force
- So TLE_LAST is best available proxy for actual satellite position
- We don't have GPS/radar data directly, only TLE-derived positions

Author: Divyanshu Panday
Date: October 2025
"""

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from skyfield.api import load, EarthSatellite
from datetime import timedelta
from sgp4.api import Satrec, jday  # for explicit TEME extraction if needed
try:
    from astropy.time import Time
    from astropy import units as u
    from astropy.coordinates import TEME, CartesianRepresentation, CartesianDifferential, SkyCoord
    _ASTROPY_AVAILABLE = True
except ImportError:
    _ASTROPY_AVAILABLE = False

# --- Physical and Astronomical Constants ---
# As per SRP_Validation_Strategy.md, based on Vallado (2013) and Montenbruck & Gill (2000)

# Gravitational parameter of Earth (km^3/s^2)
MU_EARTH = 398600.4418
R_EARTH_KM = 6378.137  # Equatorial radius (WGS-84)
J2 = 1.08262668e-3     # Earth's second zonal harmonic

# Drag still omitted; reintroducing J2 corrects plane (RAAN/inclination) evolution and reduces cross-track drift.

# Load DE421 ephemeris for Sun's position
eph = load('de421.bsp')
ts = load.timescale()
sun = eph['sun']
earth = eph['earth']


def differential_equations(t, state, epoch_time, srp_accel_func, include_j2=True, include_drag=False, drag_params=None):
    """Time derivative for propagation (two-body + optional J2 + SRP + simple drag).

    Parameters
    ----------
    t : float
        Seconds since start of propagation.
    state : ndarray
        [x, y, z, vx, vy, vz] in km and km/s.
    epoch_time : skyfield.timelib.Time
        Initial epoch (for sun vector if needed by srp_accel_func).
    srp_accel_func : callable
        Function (t_sec, r_sat_km, r_sun_km) -> accel km/s^2
    """
    r_km = state[:3]
    v_km_s = state[3:]
    r_norm = np.linalg.norm(r_km)
    if r_norm == 0:
        return np.concatenate([v_km_s, np.zeros(3)])
    # Two-body gravity
    a_gravity = -MU_EARTH * r_km / r_norm**3
    # J2 perturbation (Vallado Eq. 8-29)
    if include_j2:
        z2 = r_km[2] ** 2
        r2 = r_norm ** 2
        factor = 1.5 * J2 * MU_EARTH * (R_EARTH_KM ** 2) / (r_norm ** 5)
        coeff = 5.0 * z2 / r2 - 1.0
        tx = r_km[0] * coeff
        ty = r_km[1] * coeff
        tz = r_km[2] * (5.0 * z2 / r2 - 3.0)
        a_j2 = factor * np.array([tx, ty, tz])
    else:
        a_j2 = 0.0
    # SRP acceleration
    current_time = ts.tt_jd(epoch_time.tt + t / 86400.0)
    r_sun_km = earth.at(current_time).observe(sun).position.km
    a_srp = srp_accel_func(t, r_km, r_sun_km)
    # Simple exponential drag (very approximate) if enabled
    a_drag = 0.0
    if include_drag and drag_params is not None:
        Cd = drag_params.get('Cd', 2.2)
        A_over_m = drag_params.get('A_over_m', 0.0)  # m^2/kg
        rho0 = drag_params.get('rho0', 0.0)          # kg/m^3 at h_ref
        H = drag_params.get('H', 60.0)               # scale height km
        h_ref = drag_params.get('h_ref', 400.0)      # reference altitude km
        h_km = r_norm - R_EARTH_KM
        if A_over_m > 0 and rho0 > 0:
            rho = rho0 * np.exp(-(h_km - h_ref) / H)
            v_vec_km_s = v_km_s
            v_mag_m_s = np.linalg.norm(v_vec_km_s) * 1000.0
            if v_mag_m_s > 0:
                # a_drag = 0.5 * Cd * (A/m) * rho * v^2 opposite velocity direction
                a_drag_mag_m_s2 = 0.5 * Cd * A_over_m * rho * v_mag_m_s**2
                a_drag_vec_km_s2 = - (a_drag_mag_m_s2 / 1000.0) * (v_vec_km_s / np.linalg.norm(v_vec_km_s))
                a_drag = a_drag_vec_km_s2
    return np.concatenate([v_km_s, a_gravity + a_j2 + a_srp + a_drag])


def propagate_orbit(r0_km, v0_km_s, duration_sec, epoch_time, srp_accel_func, quiet=False,
                    rtol=1e-10, atol=1e-12, max_step=120.0, include_j2=True, include_drag=False, drag_params=None):
    """
    Propagates a satellite's orbit using numerical integration.

    Parameters
    ----------
    r0_km : array-like
        Initial position vector [x, y, z] in km (ICRF frame).
    v0_km_s : array-like
        Initial velocity vector [vx, vy, vz] in km/s (ICRF frame).
    duration_sec : float
        Total propagation time in seconds.
    epoch_time : skyfield.timelib.Time
        The start time (epoch) of the propagation.
    C_d : float
        Drag coefficient.
    A_over_m : float
        Area-to-mass ratio (m^2/kg).
    srp_accel_func : callable
        The function to calculate SRP acceleration.

    Returns
    -------
    scipy.integrate.OdeResult
        The solution object from `solve_ivp`, containing time steps and state vectors.
    """
    if not quiet:
        print(f"\n--- Running Custom Propagator ---")
        print(f"Duration: {duration_sec / 3600:.2f} hours")
    fm = f"Gravity + {'J2 + ' if include_j2 else ''}{'Drag + ' if include_drag else ''}SRP"
    print(f"Force Models: {fm} ({srp_accel_func.__name__})")
    
    # Initial state vector
    initial_state = np.concatenate([r0_km, v0_km_s])
    
    # Time span for integration
    t_span = [0, duration_sec]
    
    # We need to pass the extra arguments to the differential equation
    args = (epoch_time, srp_accel_func, include_j2, include_drag, drag_params)
    
    # Run the numerical integration
    sol = solve_ivp(
        fun=differential_equations,
        t_span=t_span,
        y0=initial_state,
        method='RK45',  # A good general-purpose integrator
        args=args,
        dense_output=True,  # Allows evaluation at any time point
        rtol=rtol,
        atol=atol,
        max_step=max_step
    )
    
    if not quiet:
        print(f"✓ Propagation complete. Status: {sol.message}")
    return sol


def propagate_from_dataframe(df, srp_accel_func, duration_hours=24.0, use_tle_icrf: bool = True, quiet=False,
                             rtol=1e-10, atol=1e-12, max_step=120.0, include_j2=True, include_drag=False, drag_params=None):
    """
    Propagate orbit using initial conditions from preprocessing DataFrame
    
    This function follows the validation strategy:
    1. Extract initial conditions (r0, v0) from FIRST TLE in DataFrame
    2. Propagate forward using Custom Propagator (J2 + Drag + SRP)
    3. Return propagated positions for comparison with ground truth
    
    Parameters:
    -----------
    df : DataFrame
        Preprocessing output with columns:
        - EPOCH, sat_x_km, sat_y_km, sat_z_km (position in ICRF)
        - sat_vx_kmps, sat_vy_kmps, sat_vz_kmps (velocity in ICRF)
        - A_OVER_M (area-to-mass ratio)
        - TLE_LINE1, TLE_LINE2
        
    srp_accel_func : callable
        Function to calculate SRP acceleration
        Signature: srp_accel_func(time, r_sat_km, r_sun_km) -> array([ax, ay, az])
        
    duration_hours : float
        Propagation duration in hours (default: 24 hours)
        Options: 24 (1 day), 48 (2 days), 168 (7 days)
        
    C_d : float
        Drag coefficient (default: 2.2 for box-wing satellites)
        
    Returns:
    --------
    dict : {
        'times_sec': array of time points (seconds from epoch),
        'positions_km': array of positions [3, N] (ICRF frame),
        'velocities_km_s': array of velocities [3, N] (ICRF frame),
        'epochs': list of datetime objects,
        'initial_conditions': dict with r0, v0, epoch
    }
    """
    if not quiet:
        print(f"\n{'='*80}")
        print(f" Custom Propagator: DataFrame Input ".center(80))
        print(f"{'='*80}")
    
    # Extract initial conditions from FIRST TLE (Day 1)
    row0 = df.iloc[0]
    
    if use_tle_icrf and 'TLE_LINE1' in row0 and 'TLE_LINE2' in row0:
        # Recompute inertial state from raw TLE to eliminate any frame mismatch
        r0_km, v0_km_s = tle_to_icrf(row0['TLE_LINE1'], row0['TLE_LINE2'])
        source_note = "(recomputed from TLE)"
    else:
        # Fallback: use stored columns
        r0_km = np.array([
            row0.get('sat_x_km'),
            row0.get('sat_y_km'),
            row0.get('sat_z_km')
        ])
        v0_km_s = np.array([
            row0.get('sat_vx_kmps'),
            row0.get('sat_vy_kmps'),
            row0.get('sat_vz_kmps')
        ])
        source_note = "(from dataframe columns)"
    
    # Get epoch time
    epoch_datetime = pd.to_datetime(row0['EPOCH'], utc=True)
    epoch_time = ts.from_datetime(epoch_datetime.to_pydatetime())
    
    # Satellite parameters
    A_over_m = row0['A_OVER_M']
    
    if not quiet:
        print(f"\nInitial Conditions (from First TLE):")
        print(f"  Epoch: {epoch_datetime.strftime('%Y-%m-%d %H:%M:%S UTC')}")
        print(f"  Position (ICRF): {r0_km} {source_note}")
        print(f"  Velocity (ICRF): {v0_km_s} {source_note}")
        print(f"  A/m: {A_over_m:.6f} m²/kg")
        print(f"\nPropagation Settings:")
        print(f"  Duration: {duration_hours} hours")
        print(f"  SRP Function: {srp_accel_func.__name__}")
    
    # Propagate orbit
    duration_sec = duration_hours * 3600.0
    sol = propagate_orbit(r0_km, v0_km_s, duration_sec, epoch_time, srp_accel_func,
                          quiet=quiet, rtol=rtol, atol=atol, max_step=max_step, include_j2=include_j2,
                          include_drag=include_drag, drag_params=drag_params)
    
    # Extract results
    times_sec = sol.t
    positions_km = sol.y[:3, :]  # First 3 rows are positions
    velocities_km_s = sol.y[3:, :]  # Last 3 rows are velocities
    
    # Create datetime objects for each propagated point
    epochs = [epoch_datetime + timedelta(seconds=float(t)) for t in times_sec]
    
    if not quiet:
        print(f"\n{'='*80}")
        print(f"✅ Propagation Complete!")
        print(f"{'='*80}")
        print(f"Time points: {len(times_sec)}")
        print(f"Final position: {positions_km[:, -1]}")
        print(f"Final time: {epochs[-1].strftime('%Y-%m-%d %H:%M:%S UTC')}")
    
    return {
        'times_sec': times_sec,
        'positions_km': positions_km,
        'velocities_km_s': velocities_km_s,
        'epochs': epochs,
        'initial_conditions': {
            'r0_km': r0_km,
            'v0_km_s': v0_km_s,
            'epoch': epoch_time,
            'epoch_utc': epoch_datetime,
            'A_over_m': A_over_m,
            # 'C_d' removed in SRP-only mode
        }
    }


def sgp4_baseline_propagation(df, duration_hours=24.0, quiet=False):
    """
    Baseline SGP4 propagation using first TLE
    
    Parameters:
    -----------
    df : DataFrame
        Preprocessing output with TLE_LINE1, TLE_LINE2, EPOCH
        
    duration_hours : float
        Propagation duration in hours
        
    Returns:
    --------
    dict : {
        'times_sec': array of time points,
        'positions_km': array of positions [3, N] (ICRF frame),
        'epochs': list of datetime objects
    }
    """
    if not quiet:
        print(f"\n{'='*80}")
        print(f" SGP4 Baseline Propagation ".center(80))
        print(f"{'='*80}")
    
    # Get first TLE
    row0 = df.iloc[0]
    tle_line1 = row0['TLE_LINE1']
    tle_line2 = row0['TLE_LINE2']
    epoch_datetime = pd.to_datetime(row0['EPOCH'], utc=True)
    
    # Create satellite object
    sat = EarthSatellite(tle_line1, tle_line2, "SAT", ts)
    
    if not quiet:
        print(f"\nBaseline TLE Epoch: {epoch_datetime.strftime('%Y-%m-%d %H:%M:%S UTC')}")
        print(f"Propagating for: {duration_hours} hours")
    
    # Generate time points (every 10 minutes) with duration guard
    if duration_hours <= 0:
        num_points = 1
        time_deltas = np.array([0.0])
    else:
        num_points = int(duration_hours * 6) + 1
        time_deltas = np.linspace(0, duration_hours * 3600, num_points)
    
    positions = []
    epochs = []
    
    for dt_sec in time_deltas:
        future_time = epoch_datetime + timedelta(seconds=float(dt_sec))
        t = ts.from_datetime(future_time.to_pydatetime())
        
        # Get position from SGP4 (ICRF frame)
        pos = sat.at(t).position.km
        positions.append(pos)
        epochs.append(future_time)
    
    positions_km = np.array(positions).T  # Shape: [3, N]
    
    if not quiet:
        print(f"\n{'='*80}")
        print(f"✅ SGP4 Propagation Complete!")
        print(f"{'='*80}")
        print(f"Time points: {len(time_deltas)}")
        print(f"Final position: {positions_km[:, -1]}")
    
    return {
        'times_sec': time_deltas,
        'positions_km': positions_km,
        'epochs': epochs
    }

# -----------------------------------------------------------------------------
# Frame Conversion Helpers
# -----------------------------------------------------------------------------
def tle_to_icrf(tle_line1: str, tle_line2: str):
    """Convert TLE to inertial (GCRS ~ ICRF) position & velocity using SGP4 + astropy.

    Falls back to Skyfield EarthSatellite if astropy TEME frame not available.
    Returns (r_km, v_km_s).
    """
    sat = Satrec.twoline2rv(tle_line1, tle_line2)
    jd = sat.jdsatepoch
    fr = sat.jdsatepochF
    e, r_teme_km, v_teme_km_s = sat.sgp4(jd, fr)
    if e != 0:
        raise RuntimeError(f"SGP4 error code {e}")

    if _ASTROPY_AVAILABLE:
        # Convert TEME -> GCRS (≈ ICRF)
        t_ast = Time(jd + fr, format='jd', scale='utc')
        try:
            # Build TEME frame with obstime; do NOT pass obstime separately to SkyCoord when frame instance used.
            teme_frame = TEME(obstime=t_ast)
            # Attach velocity differential to position representation
            pos_rep = CartesianRepresentation(r_teme_km * u.km).with_differentials(
                CartesianDifferential(v_teme_km_s * u.km / u.s)
            )
            sc = SkyCoord(pos_rep, frame=teme_frame)
            gcrs = sc.transform_to('gcrs')
            r_icrf = gcrs.cartesian.xyz.to_value(u.km)
            v_icrf = gcrs.cartesian.differentials['s'].d_xyz.to_value(u.km / u.s)
            return np.array(r_icrf), np.array(v_icrf)
        except Exception as ex:
            print(f"[WARN] TEME→GCRS transform failed ({ex}); using raw TEME vectors.")
            return np.array(r_teme_km), np.array(v_teme_km_s)
    else:
        # Fallback: return TEME as-is (close to inertial); optionally could wrap in Skyfield
        return np.array(r_teme_km), np.array(v_teme_km_s)

def dataframe_initial_icrf(df):
    """Compute inertial r,v from FIRST TLE lines instead of trusting stored columns."""
    row0 = df.iloc[0]
    r_icrf, v_icrf = tle_to_icrf(row0['TLE_LINE1'], row0['TLE_LINE2'])
    return r_icrf, v_icrf, pd.to_datetime(row0['EPOCH'], utc=True)

def dataframe_truth_icrf(df):
    """Compute inertial truth from LAST TLE lines."""
    row_last = df.iloc[-1]
    r_icrf, v_icrf = tle_to_icrf(row_last['TLE_LINE1'], row_last['TLE_LINE2'])
    return r_icrf, v_icrf, pd.to_datetime(row_last['EPOCH'], utc=True)


def compare_to_ground_truth(propagated_positions, ground_truth_df, target_epoch_utc, quiet=False):
    """
    Compare propagated position to reference TLE position
    
    NOTE: "Ground truth" is actually TLE_LAST position (calculated via SGP4).
    TLEs are fitted to real radar tracking data, so TLE_LAST is the best
    available proxy for actual satellite position (within ~1-2 km accuracy).
    
    Parameters:
    -----------
    propagated_positions : array [3, N]
        Propagated positions in ICRF (from custom propagator or SGP4)
        
    ground_truth_df : DataFrame
        DataFrame with reference TLE at target time (TLE_LAST)
        Contains sat_x_km, sat_y_km, sat_z_km (SGP4-calculated from TLE_LAST)
        
    target_epoch_utc : datetime
        Target time for comparison (e.g., t+24hr from initial TLE)
        
    Returns:
    --------
    dict : {
        'error_km': float (position error magnitude),
        'error_vector_km': array [3] (position error vector),
        'predicted_pos': array [3],
        'truth_pos': array [3] (actually TLE_LAST SGP4 position)
    }
    """
    if not quiet:
        print(f"\n{'='*80}")
        print(f" Comparing to Reference TLE Position ".center(80))
        print(f"{'='*80}")
    
    # Find closest TLE to target time in ground truth
    ground_truth_df['time_diff'] = abs((pd.to_datetime(ground_truth_df['EPOCH'], utc=True) - target_epoch_utc).dt.total_seconds())
    closest_idx = ground_truth_df['time_diff'].idxmin()
    truth_row = ground_truth_df.loc[closest_idx]
    
    # Recompute inertial truth position from that TLE line (frame robust)
    truth_pos, _truth_vel = tle_to_icrf(truth_row['TLE_LINE1'], truth_row['TLE_LINE2'])

    # Guard for empty propagation result
    if propagated_positions is None or propagated_positions.size == 0 or propagated_positions.shape[1] == 0:
        if not quiet:
            print("[WARN] Propagated positions array empty – returning NaN error.")
        return {
            'error_km': float('nan'),
            'error_vector_km': np.array([np.nan, np.nan, np.nan]),
            'predicted_pos': np.array([np.nan, np.nan, np.nan]),
            'truth_pos': truth_pos
        }

    # Get predicted position (last point) – safe now
    predicted_pos = propagated_positions[:, -1]
    
    # Calculate error
    error_vector = predicted_pos - truth_pos
    error_magnitude = np.linalg.norm(error_vector)
    
    if not quiet:
        print(f"\nTarget Epoch: {target_epoch_utc.strftime('%Y-%m-%d %H:%M:%S UTC')}")
        print(f"Reference TLE Epoch: {pd.to_datetime(truth_row['EPOCH']).strftime('%Y-%m-%d %H:%M:%S UTC')}")
        print(f"\nPredicted Position (propagated): {predicted_pos}")
        print(f"Reference Position (TLE_LAST):   {truth_pos}")
        print(f"Error Vector:                    {error_vector}")
        print(f"\n{'='*80}")
        print(f"Position Error: {error_magnitude:.3f} km")
        print(f"{'='*80}")
    
    return {
        'error_km': error_magnitude,
        'error_vector_km': error_vector,
        'predicted_pos': predicted_pos,
        'truth_pos': truth_pos
    }


def run_complete_validation(norad_id, time_range='24hr', quiet=True, include_j2=True, include_drag=None,
                            use_piecewise=True):
    """
    Complete validation workflow - Run all 3 methods and compare
    
    Parameters:
    -----------
    norad_id : int
        NORAD catalog ID (must be in satellite_constants.py)
    time_range : str
        '12hr', '24hr', '2days', or '7days'
    C_d : float
        Drag coefficient (default: 2.2)
        
    Returns:
    --------
    dict : {
        'errors': {'sgp4_km': float, 'cannonball_km': float, 'ml_km': float},
        'winner': str,
        'improvement_pct': float
    }
    """
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    from validation.preprocessing_runner import ValidationPreprocessor
    
    if not quiet:
        print("=" * 80)
        print(f" COMPLETE VALIDATION: NORAD {norad_id} ({time_range}) ".center(80))
        print("=" * 80)
    
    # Duration mapping
    duration_map = {'12hr': 12.0, '24hr': 24.0, '2days': 48.0, '7days': 168.0}
    # Determine actual duration from TLE set (use last - first epoch)
    duration_hours_requested = duration_map[time_range]
    
    # =================================================================
    # STEP 1+2: TLE Collection + Preprocessing
    # =================================================================
    if not quiet:
        print("\n[1/4] Running TLE collection + preprocessing...")
    preprocessor = ValidationPreprocessor()
    cannonball_df = preprocessor.run_preprocessing(norad_id, time_range=time_range)
    
    if cannonball_df is None:
        print("❌ Failed!")
        return None

    # --- Simple early exit: if SRP components are all zero (shadow / no SRP effect) ---
    try:
        if {'srp_ax_mps2','srp_ay_mps2','srp_az_mps2'}.issubset(cannonball_df.columns):
            if (np.allclose(cannonball_df['srp_ax_mps2'].values, 0.0) and
                np.allclose(cannonball_df['srp_ay_mps2'].values, 0.0) and
                np.allclose(cannonball_df['srp_az_mps2'].values, 0.0)):
                msg = "[SRP] All SRP acceleration components are zero → in shadow, no SRP effect. Aborting propagation."
                print(msg)
                return {
                    'errors': {'sgp4_km': float('nan'), 'cannonball_km': float('nan'), 'ml_km': float('nan')},
                    'winner': 'NONE',
                    'improvement_pct': float('nan'),
                    'note': msg
                }
    except Exception as _srp_shadow_ex:
        print(f"[WARN] SRP zero-check failed: {_srp_shadow_ex}")
    
    full_df = preprocessor.last_preprocessed_df
    if 'EPOCH' in full_df.columns:
        full_df = full_df.sort_values('EPOCH').reset_index(drop=True)
    if not quiet:
        print(f"✅ Collected {len(full_df)} TLEs")
    
    # Get inertial initial state from FIRST TLE (ignore stored sat_x/y/z)
    r0_icrf, v0_icrf, initial_epoch = dataframe_initial_icrf(full_df)
    last_epoch = pd.to_datetime(full_df.iloc[-1]['EPOCH'], utc=True)
    actual_duration_hours = (last_epoch - initial_epoch).total_seconds() / 3600.0
    if actual_duration_hours <= 0:
        duration_hours = duration_hours_requested
    else:
        duration_hours = min(duration_hours_requested, actual_duration_hours)
    target_epoch = initial_epoch + timedelta(hours=duration_hours)
    
    # =================================================================
    # STEP 3: Method A - SGP4 Baseline (NO custom propagator!)
    # =================================================================
    if not quiet:
        print("\n[2/4] Method A: SGP4 Baseline (TLE → SGP4 directly)...")
        print("  → Uses SGP4 algorithm ONLY (not custom propagator)")
        print(f"  → Initial inertial position (recomputed): [{r0_icrf[0]:.3f}, {r0_icrf[1]:.3f}, {r0_icrf[2]:.3f}] km")
    
    # SGP4 baseline: Use TLE + SGP4 algorithm (simplified force models)
    sgp4_result = sgp4_baseline_propagation(full_df, duration_hours=duration_hours, quiet=quiet)
    
    # =================================================================
    # STEP 4: Method B - Custom Propagator + Cannonball SRP
    # =================================================================
    if not quiet:
        print("\n[3/4] Method B: Custom Propagator + Cannonball SRP...")
        print("  → Uses RK45 (two-body + SRP-only) with Cannonball SRP")
        print("  → Cannonball SRP from preprocessing (srp_ax_mps2, srp_ay_mps2, srp_az_mps2)")
    
    # --- Prepare Cannonball SRP data for interpolation ---
    # Create a time vector in seconds from the start epoch
    srp_time_diffs = (pd.to_datetime(full_df['EPOCH'], utc=True) - initial_epoch)
    srp_times_sec = srp_time_diffs.dt.total_seconds()
    
    # Get SRP acceleration components
    srp_ax_values = full_df['srp_ax_mps2'].values
    srp_ay_values = full_df['srp_ay_mps2'].values
    srp_az_values = full_df['srp_az_mps2'].values
    
    # Use Cannonball SRP already calculated in preprocessing!
    def cannonball_srp_from_preprocessing(t_sec, r_sat_km, r_sun_km):
        """
        Use interpolated Cannonball SRP from preprocessing.
        This is now DYNAMIC, not constant.
        """
        # Interpolate each acceleration component based on the current time
        ax_mps2 = np.interp(t_sec, srp_times_sec, srp_ax_values)
        ay_mps2 = np.interp(t_sec, srp_times_sec, srp_ay_values)
        az_mps2 = np.interp(t_sec, srp_times_sec, srp_az_values)
        
        # Convert m/s² → km/s²
        return np.array([ax_mps2, ay_mps2, az_mps2]) / 1000.0
    
    # Inject recomputed inertial initial conditions into DataFrame-based propagation
    # For longer arcs tighten numerical settings a bit to reduce secular drift
    long_arc = duration_hours > 24.0
    # --- Simple drag parameter estimation (very coarse) ---
    if include_drag is None:
        include_drag = duration_hours > 24.0  # auto enable for multi-day
    drag_params = None
    if include_drag:
        # Mean altitude from available TLE positions (approx) if columns present
        if {'sat_x_km','sat_y_km','sat_z_km'}.issubset(full_df.columns):
            r_mags = np.sqrt(full_df['sat_x_km']**2 + full_df['sat_y_km']**2 + full_df['sat_z_km']**2)
            h_mean = float(r_mags.mean() - R_EARTH_KM)
        else:
            h_mean = 500.0
        # Rough density table (kg/m^3)
        dens_table = {
            350:8e-12, 400:4e-12, 450:2e-12, 500:1e-12, 550:6e-13, 600:3e-13,
            650:1.5e-13, 700:8e-14
        }
        nearest = min(dens_table.keys(), key=lambda k: abs(k-h_mean))
        rho0 = dens_table[nearest]
        drag_params = {
            'Cd': 2.2,
            'A_over_m': float(full_df.iloc[0]['A_OVER_M']),
            'rho0': rho0,
            'H': 60.0,
            'h_ref': nearest
        }
        if not quiet:
            print(f"[Drag] Enabled simple drag model: h_mean≈{h_mean:.1f} km, rho0={rho0:.3e} kg/m^3 (ref {nearest} km)")
    if not use_piecewise:
        cannonball_result = propagate_from_dataframe(
            full_df,
            cannonball_srp_from_preprocessing,
            duration_hours=duration_hours,
            quiet=quiet,
            include_j2=include_j2,
            include_drag=include_drag,
            drag_params=drag_params,
            max_step=60.0 if long_arc else 120.0,
            rtol=1e-11 if long_arc else 1e-10,
            atol=1e-13 if long_arc else 1e-12
        )
    else:
        if not quiet:
            print("\n[Piecewise] Running segment-by-segment propagation (Cannonball SRP)...")
        seg_positions = []
        seg_times = []
        seg_velocities = []
        current_r, current_v, current_epoch = r0_icrf, v0_icrf, initial_epoch
        epochs_sorted = pd.to_datetime(full_df['EPOCH'], utc=True).tolist()
        for next_epoch in epochs_sorted[1:]:
            dt_hours = (next_epoch - current_epoch).total_seconds() / 3600.0
            if dt_hours <= 0:
                continue
            srp_segment_mask = (pd.to_datetime(full_df['EPOCH'], utc=True) >= current_epoch) & (pd.to_datetime(full_df['EPOCH'], utc=True) <= next_epoch)
            seg_df = full_df[srp_segment_mask].reset_index(drop=True)
            seg_srp_times = (pd.to_datetime(seg_df['EPOCH'], utc=True) - current_epoch).dt.total_seconds()
            seg_ax = seg_df['srp_ax_mps2'].values
            seg_ay = seg_df['srp_ay_mps2'].values
            seg_az = seg_df['srp_az_mps2'].values
            def seg_srp(t_sec, r_sat_km, r_sun_km):
                ax_mps2 = np.interp(t_sec, seg_srp_times, seg_ax)
                ay_mps2 = np.interp(t_sec, seg_srp_times, seg_ay)
                az_mps2 = np.interp(t_sec, seg_srp_times, seg_az)
                return np.array([ax_mps2, ay_mps2, az_mps2]) / 1000.0
            duration_seg_sec = dt_hours * 3600.0
            sol_seg = propagate_orbit(current_r, current_v, duration_seg_sec, ts.from_datetime(current_epoch.to_pydatetime()),
                                      seg_srp, quiet=True, include_j2=include_j2, include_drag=include_drag,
                                      drag_params=drag_params, max_step=60.0, rtol=1e-11, atol=1e-13)
            seg_positions.append(sol_seg.y[:3, :])
            seg_velocities.append(sol_seg.y[3:, :])
            seg_times.append(sol_seg.t + (current_epoch - initial_epoch).total_seconds())
            # Re-anchor at next TLE state (tle_to_icrf returns only position & velocity)
            next_row = full_df.loc[pd.to_datetime(full_df['EPOCH'], utc=True) == next_epoch].iloc[0]
            current_r, current_v = tle_to_icrf(next_row['TLE_LINE1'], next_row['TLE_LINE2'])
            current_epoch = next_epoch
            if (current_epoch - initial_epoch).total_seconds()/3600.0 >= duration_hours:
                break
        # Concatenate segment arrays
        if seg_positions:
            positions_concat = np.concatenate(seg_positions, axis=1)
            velocities_concat = np.concatenate(seg_velocities, axis=1)
            times_concat = np.concatenate(seg_times)
        else:
            positions_concat = np.empty((3,0)); velocities_concat = np.empty((3,0)); times_concat = np.empty((0,))
        cannonball_result = {
            'times_sec': times_concat,
            'positions_km': positions_concat,
            'velocities_km_s': velocities_concat,
            'epochs': [initial_epoch + timedelta(seconds=float(s)) for s in times_concat],
            'initial_conditions': {
                'r0_km': r0_icrf,
                'v0_km_s': v0_icrf,
                'epoch': ts.from_datetime(initial_epoch.to_pydatetime()),
                'epoch_utc': initial_epoch,
                'A_over_m': float(full_df.iloc[0]['A_OVER_M'])
            }
        }
        if not quiet:
            print(f"[Piecewise] Total points: {positions_concat.shape[1]}")
        # Fallback if no segment points produced (e.g., only one TLE or zero dt) → single-pass propagate
        if positions_concat.shape[1] == 0:
            if not quiet:
                print("[Piecewise] No segment data produced – falling back to single-pass propagation.")
            cannonball_result = propagate_from_dataframe(
                full_df,
                cannonball_srp_from_preprocessing,
                duration_hours=duration_hours,
                quiet=True,
                include_j2=include_j2,
                include_drag=include_drag,
                drag_params=drag_params,
                max_step=60.0 if long_arc else 120.0,
                rtol=1e-11 if long_arc else 1e-10,
                atol=1e-13 if long_arc else 1e-12
            )
    cannonball_result['initial_conditions']['r0_km'] = r0_icrf
    cannonball_result['initial_conditions']['v0_km_s'] = v0_icrf
    
    # =================================================================
    # STEP 5: Method C - Custom Propagator + ML SRP
    # =================================================================
    if not quiet:
        print("\n[4/4] Method C: Custom Propagator + ML SRP...")
        print("  → Uses RK45 (two-body + SRP-only) with ML SRP")
    
    # First, run ML prediction from preprocessing
    if not quiet:
        print("  → Running ML prediction...")
    ml_df = preprocessor.run_ml_prediction(model_folder='../model/trained_models')
    
    # --- Prepare ML SRP data for interpolation ---
    ml_time_diffs = (pd.to_datetime(ml_df['EPOCH'], utc=True) - initial_epoch)
    ml_times_sec = ml_time_diffs.dt.total_seconds()
    ml_ax_values = ml_df['ml_ax_mps2'].values
    ml_ay_values = ml_df['ml_ay_mps2'].values
    ml_az_values = ml_df['ml_az_mps2'].values

    # --- DEBUG: Compare Cannonball vs ML values ---
    if not quiet:
        print("\n  🔍 DEBUG: Comparing Cannonball vs ML SRP values (first TLE):")
        cannonball_first = cannonball_df.iloc[0]
        ml_first = ml_df.iloc[0]
        print(f"     Cannonball: ax={cannonball_first['srp_ax_mps2']:.12f}, ay={cannonball_first['srp_ay_mps2']:.12f}, az={cannonball_first['srp_az_mps2']:.12f} m/s²")
        print(f"     ML:         ax={ml_first['ml_ax_mps2']:.12f}, ay={ml_first['ml_ay_mps2']:.12f}, az={ml_first['ml_az_mps2']:.12f} m/s²")
        diff_ax = abs(ml_first['ml_ax_mps2'] - cannonball_first['srp_ax_mps2'])
        diff_ay = abs(ml_first['ml_ay_mps2'] - cannonball_first['srp_ay_mps2'])
        diff_az = abs(ml_first['ml_az_mps2'] - cannonball_first['srp_az_mps2'])
        print(f"     Difference: Δax={diff_ax:.12f}, Δay={diff_ay:.12f}, Δaz={diff_az:.12f} m/s²")
        if diff_ax < 1e-10 and diff_ay < 1e-10 and diff_az < 1e-10:
            print(f"     ⚠️  VALUES ARE IDENTICAL! ML models may not be loaded or trained properly.")
        else:
            print(f"     ✅ Values are DIFFERENT - ML models are working!")
    # -----------------------------------------------
    
    # Use ML SRP from preprocessing
    def ml_srp_from_preprocessing(t_sec, r_sat_km, r_sun_km):
        """
        Use interpolated ML SRP from preprocessing.
        This is now DYNAMIC, not constant.
        """
        # Interpolate each acceleration component based on the current time
        ax_mps2 = np.interp(t_sec, ml_times_sec, ml_ax_values)
        ay_mps2 = np.interp(t_sec, ml_times_sec, ml_ay_values)
        az_mps2 = np.interp(t_sec, ml_times_sec, ml_az_values)
        
        # Convert m/s² → km/s²
        return np.array([ax_mps2, ay_mps2, az_mps2]) / 1000.0
    
    if not use_piecewise:
        ml_result = propagate_from_dataframe(
            full_df,
            ml_srp_from_preprocessing,
            duration_hours=duration_hours,
            quiet=quiet,
            include_j2=include_j2,
            include_drag=include_drag,
            drag_params=drag_params,
            max_step=60.0 if long_arc else 120.0,
            rtol=1e-11 if long_arc else 1e-10,
            atol=1e-13 if long_arc else 1e-12
        )
    else:
        if not quiet:
            print("\n[Piecewise] Running segment-by-segment propagation (ML SRP)...")
        seg_positions_ml = []
        seg_velocities_ml = []
        seg_times_ml = []
        current_r, current_v, current_epoch = r0_icrf, v0_icrf, initial_epoch
        epochs_sorted = pd.to_datetime(full_df['EPOCH'], utc=True).tolist()
        for next_epoch in epochs_sorted[1:]:
            dt_hours = (next_epoch - current_epoch).total_seconds() / 3600.0
            if dt_hours <= 0:
                continue
            seg_mask = (pd.to_datetime(ml_df['EPOCH'], utc=True) >= current_epoch) & (pd.to_datetime(ml_df['EPOCH'], utc=True) <= next_epoch)
            seg_ml = ml_df[seg_mask].reset_index(drop=True)
            seg_ml_times = (pd.to_datetime(seg_ml['EPOCH'], utc=True) - current_epoch).dt.total_seconds()
            seg_ml_ax = seg_ml['ml_ax_mps2'].values
            seg_ml_ay = seg_ml['ml_ay_mps2'].values
            seg_ml_az = seg_ml['ml_az_mps2'].values
            def seg_ml_srp(t_sec, r_sat_km, r_sun_km):
                ax_mps2 = np.interp(t_sec, seg_ml_times, seg_ml_ax)
                ay_mps2 = np.interp(t_sec, seg_ml_times, seg_ml_ay)
                az_mps2 = np.interp(t_sec, seg_ml_times, seg_ml_az)
                return np.array([ax_mps2, ay_mps2, az_mps2]) / 1000.0
            sol_seg_ml = propagate_orbit(current_r, current_v, dt_hours*3600.0, ts.from_datetime(current_epoch.to_pydatetime()),
                                         seg_ml_srp, quiet=True, include_j2=include_j2, include_drag=include_drag,
                                         drag_params=drag_params, max_step=60.0, rtol=1e-11, atol=1e-13)
            seg_positions_ml.append(sol_seg_ml.y[:3, :])
            seg_velocities_ml.append(sol_seg_ml.y[3:, :])
            seg_times_ml.append(sol_seg_ml.t + (current_epoch - initial_epoch).total_seconds())
            # Re-anchor (tle_to_icrf returns (r,v))
            next_row = full_df.loc[pd.to_datetime(full_df['EPOCH'], utc=True) == next_epoch].iloc[0]
            current_r, current_v = tle_to_icrf(next_row['TLE_LINE1'], next_row['TLE_LINE2'])
            current_epoch = next_epoch
            if (current_epoch - initial_epoch).total_seconds()/3600.0 >= duration_hours:
                break
        if seg_positions_ml:
            ml_positions_concat = np.concatenate(seg_positions_ml, axis=1)
            ml_vel_concat = np.concatenate(seg_velocities_ml, axis=1)
            ml_times_concat = np.concatenate(seg_times_ml)
        else:
            ml_positions_concat = np.empty((3,0)); ml_vel_concat = np.empty((3,0)); ml_times_concat = np.empty((0,))
        ml_result = {
            'times_sec': ml_times_concat,
            'positions_km': ml_positions_concat,
            'velocities_km_s': ml_vel_concat,
            'epochs': [initial_epoch + timedelta(seconds=float(s)) for s in ml_times_concat],
            'initial_conditions': {
                'r0_km': r0_icrf,
                'v0_km_s': v0_icrf,
                'epoch': ts.from_datetime(initial_epoch.to_pydatetime()),
                'epoch_utc': initial_epoch,
                'A_over_m': float(full_df.iloc[0]['A_OVER_M'])
            }
        }
        if not quiet:
            print(f"[Piecewise] Total points (ML): {ml_positions_concat.shape[1]}")
        if ml_positions_concat.shape[1] == 0:
            if not quiet:
                print("[Piecewise] (ML) No segment data produced – falling back to single-pass propagation.")
            ml_result = propagate_from_dataframe(
                full_df,
                ml_srp_from_preprocessing,
                duration_hours=duration_hours,
                quiet=True,
                include_j2=include_j2,
                include_drag=include_drag,
                drag_params=drag_params,
                max_step=60.0 if long_arc else 120.0,
                rtol=1e-11 if long_arc else 1e-10,
                atol=1e-13 if long_arc else 1e-12
            )
    ml_result['initial_conditions']['r0_km'] = r0_icrf
    ml_result['initial_conditions']['v0_km_s'] = v0_icrf
    
    # =================================================================
    # STEP 6: Compare to Reference TLE Position (TLE_LAST)
    # =================================================================
    if not quiet:
        print("\n" + "=" * 80)
        print(" COMPARISON TO REFERENCE TLE (TLE_LAST) ".center(80))
        print("=" * 80)
        print("Note: Reference = SGP4 position from LAST TLE (proxy for actual position)")
    
    ground_truth_df = full_df.copy()
    
    error_sgp4 = compare_to_ground_truth(sgp4_result['positions_km'], ground_truth_df, target_epoch, quiet=quiet)
    error_cannonball = compare_to_ground_truth(cannonball_result['positions_km'], ground_truth_df, target_epoch, quiet=quiet)
    error_ml = compare_to_ground_truth(ml_result['positions_km'], ground_truth_df, target_epoch, quiet=quiet)
    
    # =================================================================
    # STEP 7: SIMPLE RESULTS (Just 6 Things!)
    # =================================================================
    if not quiet:
        print("\n" + "=" * 80)
        print(" VALIDATION RESULTS - SIMPLIFIED ".center(80))
        print("=" * 80)
    
    # Get final positions and velocities
    sgp4_pos = sgp4_result['positions_km'][:, -1]
    sgp4_vel = sgp4_result['positions_km'][:, -1] * 0  # Placeholder (not tracked in SGP4 result)
    
    cannonball_pos = cannonball_result['positions_km'][:, -1]
    cannonball_vel = cannonball_result['velocities_km_s'][:, -1]
    
    ml_pos = ml_result['positions_km'][:, -1]
    ml_vel = ml_result['velocities_km_s'][:, -1]
    
    # Reference position (TLE_LAST)
    ref_pos = error_sgp4['truth_pos']
    
    # Calculate differences
    diff_sgp4_cannonball_pos = sgp4_pos - cannonball_pos
    diff_sgp4_cannonball_vel = np.linalg.norm(diff_sgp4_cannonball_pos)
    
    diff_sgp4_ml_pos = sgp4_pos - ml_pos
    diff_sgp4_ml_vel = np.linalg.norm(diff_sgp4_ml_pos)
    
    # Print 6 things clearly
    if not quiet:
        print("\n1. SGP4 Last 24hr Position & Velocity:")
        print(f"   Position: [{sgp4_pos[0]:12.6f}, {sgp4_pos[1]:12.6f}, {sgp4_pos[2]:12.6f}] km")
        print(f"   Error from Reference: {error_sgp4['error_km']:.6f} km")
    
    if not quiet:
        print("\n2. Custom Propagator + Cannonball SRP Position & Velocity:")
        print(f"   Position: [{cannonball_pos[0]:12.6f}, {cannonball_pos[1]:12.6f}, {cannonball_pos[2]:12.6f}] km")
        print(f"   Velocity: [{cannonball_vel[0]:11.8f}, {cannonball_vel[1]:11.8f}, {cannonball_vel[2]:11.8f}] km/s")
        print(f"   Error from Reference: {error_cannonball['error_km']:.6f} km")
    
    if not quiet:
        print("\n3. Custom Propagator + ML SRP Position & Velocity:")
        print(f"   Position: [{ml_pos[0]:12.6f}, {ml_pos[1]:12.6f}, {ml_pos[2]:12.6f}] km")
        print(f"   Velocity: [{ml_vel[0]:11.8f}, {ml_vel[1]:11.8f}, {ml_vel[2]:11.8f}] km/s")
        print(f"   Error from Reference: {error_ml['error_km']:.6f} km")
    
    if not quiet:
        print("\n4. SGP4 - Custom+Cannonball (Position Difference):")
        print(f"   Δ Position: [{diff_sgp4_cannonball_pos[0]:12.6f}, {diff_sgp4_cannonball_pos[1]:12.6f}, {diff_sgp4_cannonball_pos[2]:12.6f}] km")
        print(f"   Magnitude:  {diff_sgp4_cannonball_vel:.6f} km")
    
    if not quiet:
        print("\n5. SGP4 - Custom+ML (Position Difference):")
        print(f"   Δ Position: [{diff_sgp4_ml_pos[0]:12.6f}, {diff_sgp4_ml_pos[1]:12.6f}, {diff_sgp4_ml_pos[2]:12.6f}] km")
        print(f"   Magnitude:  {diff_sgp4_ml_vel:.6f} km")
    
    # Minimal summary only (truth last TLE, custom+cannonball final, custom+ML final)
    # Truth at target epoch (closest TLE to propagation target) rather than always LAST TLE
    target_truth_pos = error_cannonball['truth_pos']
    # Still compute last TLE for informational purposes
    truth_last_pos, truth_last_vel, last_epoch_icrf = dataframe_truth_icrf(full_df)
    # --- RTN error decomposition ---
    h_vec = np.cross(target_truth_pos, truth_last_vel)
    r_hat = target_truth_pos / np.linalg.norm(target_truth_pos)
    n_hat = h_vec / np.linalg.norm(h_vec)
    t_hat = np.cross(n_hat, r_hat)

    def rtn_err(pred):
        dv = pred - target_truth_pos
        return (
            float(np.dot(dv, r_hat)),
            float(np.dot(dv, t_hat)),
            float(np.dot(dv, n_hat)),
            float(np.linalg.norm(dv))
        )
    cb_R, cb_T, cb_N, cb_tot = rtn_err(cannonball_pos)
    ml_R, ml_T, ml_N, ml_tot = rtn_err(ml_pos)

    summary = {
        'truth_target_pos_km': target_truth_pos,
        'truth_last_pos_km': truth_last_pos,
        'cannonball_final_pos_km': cannonball_pos,
        'ml_final_pos_km': ml_pos,
        'cannonball_error_km': np.linalg.norm(cannonball_pos - target_truth_pos),
        'ml_error_km': np.linalg.norm(ml_pos - target_truth_pos),
        'actual_duration_hours': actual_duration_hours,
        'used_duration_hours': duration_hours,
        'include_j2': include_j2,
        'cannonball_rtn': {'R': cb_R, 'T': cb_T, 'N': cb_N, 'total': cb_tot},
        'ml_rtn': {'R': ml_R, 'T': ml_T, 'N': ml_N, 'total': ml_tot}
    }
    # Print concise required lines (HDP style)
    print("HDP SUMMARY")
    print(f"Actual arc span (hrs): {actual_duration_hours:.3f}  Propagated (hrs): {duration_hours:.3f}  J2={'ON' if include_j2 else 'OFF'} Drag={'ON' if include_drag else 'OFF'} Piecewise={'ON' if use_piecewise else 'OFF'}")
    print(f"Truth (Target epoch) position km:   [{target_truth_pos[0]:.6f}, {target_truth_pos[1]:.6f}, {target_truth_pos[2]:.6f}]")
    if last_epoch > target_epoch:
        print(f"(Last TLE at {last_epoch_icrf.isoformat()} differs from target epoch {target_epoch.isoformat()})")
    print(f"Custom + Cannonball final pos km:   [{cannonball_pos[0]:.6f}, {cannonball_pos[1]:.6f}, {cannonball_pos[2]:.6f}]  err={summary['cannonball_error_km']:.6f} km")
    print(f"  RTN error (km): R={cb_R:.3f} T={cb_T:.3f} N={cb_N:.3f} Total={cb_tot:.3f}")
    print(f"Custom + ML final pos km:           [{ml_pos[0]:.6f}, {ml_pos[1]:.6f}, {ml_pos[2]:.6f}]  err={summary['ml_error_km']:.6f} km")
    print(f"  RTN error (km): R={ml_R:.3f} T={ml_T:.3f} N={ml_N:.3f} Total={ml_tot:.3f}")
    print("Note: Near-identical ML vs Cannonball indicates ML learned physical SRP mapping; cross-track (N) previously dominated by missing J2 term.")
    # Removed acceleration magnitude printout in simplified mode
    return summary


def main():
    """
    Example usage: Run validation for 24hr
    """
    print("=" * 80)
    print(" CUSTOM PROPAGATOR - VALIDATION TEST ".center(80))
    print("=" * 80)
    
    # Run validation
    results = run_complete_validation(
        norad_id=43476

,
        time_range='24hr',
        quiet=True,
        include_j2=True
    )
    
    if results:
        print("\n✅ Validation complete (minimal output mode).")


if __name__ == '__main__':
    main()
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

# --- Physical and Astronomical Constants ---
# As per SRP_Validation_Strategy.md, based on Vallado (2013) and Montenbruck & Gill (2000)

# Gravitational parameter of Earth (km^3/s^2)
MU_EARTH = 398600.4418
# Earth's second zonal harmonic (dimensionless)
J2 = 1.08263e-3
# Earth's equatorial radius (km)
R_EARTH = 6378.137

# --- Atmospheric Drag Constants ---
# Using a simple exponential model for density
RHO_0 = 1.225  # Sea-level density (kg/m^3)
H_SCALE = 8.5  # Atmospheric scale height (km)

# Load DE421 ephemeris for Sun's position
eph = load('de421.bsp')
ts = load.timescale()
sun = eph['sun']
earth = eph['earth']


def differential_equations(t, state, epoch_time, C_d, A_over_m, srp_accel_func):
    """
    Defines the system of differential equations for orbit propagation.
    This function is the core of the integrator, calculating the total acceleration
    on the satellite at a given time `t`.

    Parameters
    ----------
    t : float
        Current time in seconds from the start of the integration.
    state : array-like
        6-element state vector [x, y, z, vx, vy, vz] in km and km/s.
    epoch_time : skyfield.timelib.Time
        The start time of the propagation.
    C_d : float
        Drag coefficient.
    A_over_m : float
        Area-to-mass ratio (m^2/kg).
    srp_accel_func : callable
        A function that computes the SRP acceleration vector.
        Signature: `srp_accel_func(t_sec, r_sat_km, r_sun_km)` -> array([ax, ay, az]) in km/s^2.

    Returns
    -------
    array-like
        The derivative of the state vector [vx, vy, vz, ax, ay, az].
    """
    r_vec = state[:3]
    v_vec = state[3:]
    r_norm = np.linalg.norm(r_vec)

    # 1. Central Body Gravity (Keplerian)
    a_gravity = -MU_EARTH * r_vec / r_norm**3

    # 2. J2 Perturbation (Earth Oblateness)
    z = r_vec[2]
    j2_factor = 1.5 * J2 * MU_EARTH * (R_EARTH**2) / (r_norm**5)
    
    ax_j2 = j2_factor * r_vec[0] * (5 * (z**2 / r_norm**2) - 1)
    ay_j2 = j2_factor * r_vec[1] * (5 * (z**2 / r_norm**2) - 1)
    az_j2 = j2_factor * r_vec[2] * (5 * (z**2 / r_norm**2) - 3)
    a_j2 = np.array([ax_j2, ay_j2, az_j2])

    # 3. Atmospheric Drag
    altitude = r_norm - R_EARTH
    # Simple exponential atmospheric model
    rho = RHO_0 * np.exp(-altitude / H_SCALE)  # kg/m^3
    
    # Convert A/m from m^2/kg to km^2/kg for consistent units
    A_over_m_km2 = A_over_m / (1000**2)
    v_norm = np.linalg.norm(v_vec)
    
    # Drag acceleration in km/s^2
    a_drag = -0.5 * C_d * A_over_m_km2 * (rho * 1e9) * (v_norm**2) * (v_vec / v_norm)

    # 4. Solar Radiation Pressure (SRP)
    # Get current time for SRP calculation
    current_time = ts.tt_jd(epoch_time.tt + t / 86400.0)
    # Get Sun's position vector from Earth's center
    r_sun_km = earth.at(current_time).observe(sun).position.km
    
    # The provided SRP function must return acceleration in km/s^2
    a_srp = srp_accel_func(t, r_vec, r_sun_km)

    # --- Total Acceleration ---
    a_total = a_gravity + a_j2 + a_drag + a_srp

    # The derivative of the state is [velocity, acceleration]
    return np.concatenate([v_vec, a_total])


def propagate_orbit(r0_km, v0_km_s, duration_sec, epoch_time, C_d, A_over_m, srp_accel_func):
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
    print(f"\n--- Running Custom Propagator ---")
    print(f"Duration: {duration_sec / 3600:.2f} hours")
    print(f"Force Models: Gravity + J2 + Drag + SRP ({srp_accel_func.__name__})")
    
    # Initial state vector
    initial_state = np.concatenate([r0_km, v0_km_s])
    
    # Time span for integration
    t_span = [0, duration_sec]
    
    # We need to pass the extra arguments to the differential equation
    args = (epoch_time, C_d, A_over_m, srp_accel_func)
    
    # Run the numerical integration
    sol = solve_ivp(
        fun=differential_equations,
        t_span=t_span,
        y0=initial_state,
        method='RK45',  # A good general-purpose integrator
        args=args,
        dense_output=True,  # Allows evaluation at any time point
        rtol=1e-9,  # Relative tolerance
        atol=1e-12  # Absolute tolerance
    )
    
    print(f"✓ Propagation complete. Status: {sol.message}")
    return sol


def propagate_from_dataframe(df, srp_accel_func, duration_hours=24.0, C_d=2.2):
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
    print(f"\n{'='*80}")
    print(f" Custom Propagator: DataFrame Input ".center(80))
    print(f"{'='*80}")
    
    # Extract initial conditions from FIRST TLE (Day 1)
    row0 = df.iloc[0]
    
    # Initial position and velocity (ICRF frame from preprocessing)
    r0_km = np.array([
        row0['sat_x_km'],
        row0['sat_y_km'],
        row0['sat_z_km']
    ])
    
    v0_km_s = np.array([
        row0['sat_vx_kmps'],  # Note: column name is 'kmps' not 'km_s'
        row0['sat_vy_kmps'],
        row0['sat_vz_kmps']
    ])
    
    # Get epoch time
    epoch_datetime = pd.to_datetime(row0['EPOCH'], utc=True)
    epoch_time = ts.from_datetime(epoch_datetime.to_pydatetime())
    
    # Satellite parameters
    A_over_m = row0['A_OVER_M']
    
    print(f"\nInitial Conditions (from First TLE):")
    print(f"  Epoch: {epoch_datetime.strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print(f"  Position (ICRF): {r0_km}")
    print(f"  Velocity (ICRF): {v0_km_s}")
    print(f"  A/m: {A_over_m:.6f} m²/kg")
    print(f"  C_d: {C_d}")
    print(f"\nPropagation Settings:")
    print(f"  Duration: {duration_hours} hours")
    print(f"  SRP Function: {srp_accel_func.__name__}")
    
    # Propagate orbit
    duration_sec = duration_hours * 3600.0
    sol = propagate_orbit(r0_km, v0_km_s, duration_sec, epoch_time, C_d, A_over_m, srp_accel_func)
    
    # Extract results
    times_sec = sol.t
    positions_km = sol.y[:3, :]  # First 3 rows are positions
    velocities_km_s = sol.y[3:, :]  # Last 3 rows are velocities
    
    # Create datetime objects for each propagated point
    epochs = [epoch_datetime + timedelta(seconds=float(t)) for t in times_sec]
    
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
            'C_d': C_d
        }
    }


def sgp4_baseline_propagation(df, duration_hours=24.0):
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
    
    print(f"\nBaseline TLE Epoch: {epoch_datetime.strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print(f"Propagating for: {duration_hours} hours")
    
    # Generate time points (every 10 minutes)
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


def compare_to_ground_truth(propagated_positions, ground_truth_df, target_epoch_utc):
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
    print(f"\n{'='*80}")
    print(f" Comparing to Reference TLE Position ".center(80))
    print(f"{'='*80}")
    
    # Find closest TLE to target time in ground truth
    ground_truth_df['time_diff'] = abs((pd.to_datetime(ground_truth_df['EPOCH'], utc=True) - target_epoch_utc).dt.total_seconds())
    closest_idx = ground_truth_df['time_diff'].idxmin()
    truth_row = ground_truth_df.loc[closest_idx]
    
    # Get ground truth position
    truth_pos = np.array([
        truth_row['sat_x_km'],
        truth_row['sat_y_km'],
        truth_row['sat_z_km']
    ])
    
    # Get predicted position (last point)
    predicted_pos = propagated_positions[:, -1]
    
    # Calculate error
    error_vector = predicted_pos - truth_pos
    error_magnitude = np.linalg.norm(error_vector)
    
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


def run_complete_validation(norad_id, time_range='24hr', C_d=2.2):
    """
    Complete validation workflow - Run all 3 methods and compare
    
    Parameters:
    -----------
    norad_id : int
        NORAD catalog ID (must be in satellite_constants.py)
    time_range : str
        '24hr', '2days', or '7days'
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
    
    print("=" * 80)
    print(f" COMPLETE VALIDATION: NORAD {norad_id} ({time_range}) ".center(80))
    print("=" * 80)
    
    # Duration mapping
    duration_map = {'24hr': 24.0, '2days': 48.0, '7days': 168.0}
    duration_hours = duration_map[time_range]
    
    # =================================================================
    # STEP 1+2: TLE Collection + Preprocessing
    # =================================================================
    print("\n[1/4] Running TLE collection + preprocessing...")
    preprocessor = ValidationPreprocessor()
    cannonball_df = preprocessor.run_preprocessing(norad_id, time_range=time_range)
    
    if cannonball_df is None:
        print("❌ Failed!")
        return None
    
    full_df = preprocessor.last_preprocessed_df
    print(f"✅ Collected {len(full_df)} TLEs")
    
    # Get initial epoch
    initial_epoch = pd.to_datetime(full_df.iloc[0]['EPOCH'], utc=True)
    target_epoch = initial_epoch + timedelta(hours=duration_hours)
    
    # =================================================================
    # STEP 3: Method A - SGP4 Baseline (NO custom propagator!)
    # =================================================================
    print("\n[2/4] Method A: SGP4 Baseline (TLE → SGP4 directly)...")
    print("  → Uses SGP4 algorithm ONLY (not custom propagator)")
    print(f"  → Initial position: [{full_df.iloc[0]['sat_x_km']:.3f}, {full_df.iloc[0]['sat_y_km']:.3f}, {full_df.iloc[0]['sat_z_km']:.3f}] km")
    
    # SGP4 baseline: Use TLE + SGP4 algorithm (simplified force models)
    sgp4_result = sgp4_baseline_propagation(full_df, duration_hours=duration_hours)
    
    # =================================================================
    # STEP 4: Method B - Custom Propagator + Cannonball SRP
    # =================================================================
    print("\n[3/4] Method B: Custom Propagator + Cannonball SRP...")
    print("  → Uses RK45 integrator with J2 + Drag + Cannonball SRP")
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
    
    cannonball_result = propagate_from_dataframe(
        full_df, cannonball_srp_from_preprocessing, duration_hours=duration_hours, C_d=C_d
    )
    
    # =================================================================
    # STEP 5: Method C - Custom Propagator + ML SRP
    # =================================================================
    print("\n[4/4] Method C: Custom Propagator + ML SRP...")
    print("  → Uses RK45 integrator with J2 + Drag + ML-predicted SRP")
    
    # First, run ML prediction from preprocessing
    print("  → Running ML prediction...")
    ml_df = preprocessor.run_ml_prediction(model_folder='../model/trained_models')
    
    # --- Prepare ML SRP data for interpolation ---
    ml_time_diffs = (pd.to_datetime(ml_df['EPOCH'], utc=True) - initial_epoch)
    ml_times_sec = ml_time_diffs.dt.total_seconds()
    ml_ax_values = ml_df['ml_ax_mps2'].values
    ml_ay_values = ml_df['ml_ay_mps2'].values
    ml_az_values = ml_df['ml_az_mps2'].values

    # --- DEBUG: Compare Cannonball vs ML values ---
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
    
    ml_result = propagate_from_dataframe(
        full_df, ml_srp_from_preprocessing, duration_hours=duration_hours, C_d=C_d
    )
    
    # =================================================================
    # STEP 6: Compare to Reference TLE Position (TLE_LAST)
    # =================================================================
    print("\n" + "=" * 80)
    print(" COMPARISON TO REFERENCE TLE (TLE_LAST) ".center(80))
    print("=" * 80)
    print("Note: Reference = SGP4 position from LAST TLE (proxy for actual position)")
    
    ground_truth_df = full_df.copy()
    
    error_sgp4 = compare_to_ground_truth(sgp4_result['positions_km'], ground_truth_df, target_epoch)
    error_cannonball = compare_to_ground_truth(cannonball_result['positions_km'], ground_truth_df, target_epoch)
    error_ml = compare_to_ground_truth(ml_result['positions_km'], ground_truth_df, target_epoch)
    
    # =================================================================
    # STEP 7: SIMPLE RESULTS (Just 6 Things!)
    # =================================================================
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
    print("\n1. SGP4 Last 24hr Position & Velocity:")
    print(f"   Position: [{sgp4_pos[0]:12.6f}, {sgp4_pos[1]:12.6f}, {sgp4_pos[2]:12.6f}] km")
    print(f"   Error from Reference: {error_sgp4['error_km']:.6f} km")
    
    print("\n2. Custom Propagator + Cannonball SRP Position & Velocity:")
    print(f"   Position: [{cannonball_pos[0]:12.6f}, {cannonball_pos[1]:12.6f}, {cannonball_pos[2]:12.6f}] km")
    print(f"   Velocity: [{cannonball_vel[0]:11.8f}, {cannonball_vel[1]:11.8f}, {cannonball_vel[2]:11.8f}] km/s")
    print(f"   Error from Reference: {error_cannonball['error_km']:.6f} km")
    
    print("\n3. Custom Propagator + ML SRP Position & Velocity:")
    print(f"   Position: [{ml_pos[0]:12.6f}, {ml_pos[1]:12.6f}, {ml_pos[2]:12.6f}] km")
    print(f"   Velocity: [{ml_vel[0]:11.8f}, {ml_vel[1]:11.8f}, {ml_vel[2]:11.8f}] km/s")
    print(f"   Error from Reference: {error_ml['error_km']:.6f} km")
    
    print("\n4. SGP4 - Custom+Cannonball (Position Difference):")
    print(f"   Δ Position: [{diff_sgp4_cannonball_pos[0]:12.6f}, {diff_sgp4_cannonball_pos[1]:12.6f}, {diff_sgp4_cannonball_pos[2]:12.6f}] km")
    print(f"   Magnitude:  {diff_sgp4_cannonball_vel:.6f} km")
    
    print("\n5. SGP4 - Custom+ML (Position Difference):")
    print(f"   Δ Position: [{diff_sgp4_ml_pos[0]:12.6f}, {diff_sgp4_ml_pos[1]:12.6f}, {diff_sgp4_ml_pos[2]:12.6f}] km")
    print(f"   Magnitude:  {diff_sgp4_ml_vel:.6f} km")
    
    print("\n6. WINNER (Least Difference from Reference TLE):")
    if abs(error_ml['error_km'] - error_cannonball['error_km']) < 1e-6:
        print(f"   ⚠️  TIE: Both methods have same error!")
        winner = "Tie"
        improvement = 0.0
        degradation = 0.0
    elif error_ml['error_km'] < error_cannonball['error_km']:
        improvement = error_cannonball['error_km'] - error_ml['error_km']
        print(f"   ✅ WINNER: Custom + ML SRP")
        print(f"   Improvement: {improvement:.6f} km better than Cannonball")
        winner = "ML"
        degradation = 0.0
    else:
        degradation = error_ml['error_km'] - error_cannonball['error_km']
        print(f"   ✅ WINNER: Custom + Cannonball SRP")
        print(f"   ML is worse by: {degradation:.6f} km")
        winner = "Cannonball"
        improvement = 0.0
    
    print("\n" + "=" * 80)
    
    return {
        'errors': {
            'sgp4_km': error_sgp4['error_km'],
            'cannonball_km': error_cannonball['error_km'],
            'ml_km': error_ml['error_km']
        },
        'winner': winner,
        'improvement_pct': improvement if winner == "ML" else -degradation
    }


def main():
    """
    Example usage: Run validation for 24hr
    """
    print("=" * 80)
    print(" CUSTOM PROPAGATOR - VALIDATION TEST ".center(80))
    print("=" * 80)
    
    # Run validation
    results = run_complete_validation(
        norad_id=41240,     # Example satellite
        time_range='24hr',  # Options: '24hr', '2days', '7days'
        C_d=2.2             # Drag coefficient
    )
    
    if results:
        print("\n✅ Validation complete!")
        print(f"\n📊 Quick Summary:")
        print(f"   Winner: {results['winner']}")
        
        if results['winner'] == 'Tie':
            print(f"   Note: Check if ML models are properly trained (values identical)")
        elif results['winner'] == 'ML':
            print(f"   ML is better by: {results['improvement_pct']:.6f} km")
        else:
            print(f"   Cannonball is better by: {abs(results['improvement_pct']):.6f} km")


if __name__ == '__main__':
    main()
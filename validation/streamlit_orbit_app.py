"""Streamlit Orbit Validation Dashboard (Enhanced)

Purpose:
        Modern interactive dashboard to compare Custom Propagator (Cannonball vs ML SRP)
        against TLE / SGP4 reference using the latest capabilities of
        `validation.custom_propagator` (J2, Drag, Piecewise re‑anchoring, RTN errors).

What's New vs Old Version:
        • Added 12 hr option (short arc) in addition to 24 hr, 48 hr, 168 hr
        • Piecewise propagation toggle (re‑anchors at each successive TLE)
        • J2 and Drag toggles with auto-drag suggestion for > 24 hr arcs
        • RTN (Radial/Along-track/Cross-track) error decomposition charts
        • Rich metrics with dynamic highlighting when ML wins
        • History of runs (session) to show which satellite & arc lengths ML improved
        • Force model summary line mirroring HDP output for transparency
        • Cleaner 3D visualizations & expandable technical details

Author: Divyanshu Panday (enhanced with assistant)
Date: November 2025
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from datetime import timedelta
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from validation.preprocessing_runner import ValidationPreprocessor
from validation.custom_propagator import (
    propagate_from_dataframe,
    propagate_orbit,  # direct integrator for correct piecewise re-anchoring
    tle_to_icrf,
    dataframe_initial_icrf,
    ts,
)
from preprocessing.satellite_constants import get_params
# Reload satellite constants to avoid stale cached module without 'name' keys
try:
    import importlib
    import preprocessing.satellite_constants as _sat_mod
    _sat_mod = importlib.reload(_sat_mod)
    get_params = _sat_mod.get_params
except Exception:
    # Safe to ignore; fallback to initially imported get_params
    pass


# Page configuration
st.set_page_config(
    page_title="Orbit Validation (SRP ML vs Cannonball)",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    .metric-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .winner-ml {
        background-color: #d4edda;
        border-left-color: #28a745;
    }
    .winner-cannonball {
        background-color: #fff3cd;
        border-left-color: #ffc107;
    }
    .satcat-btn {
        display: inline-block;
        background: #1f77b4;
        color: #ffffff !important;
        padding: 0.45rem 0.75rem;
        border-radius: 4px;
        text-decoration: none;
        font-weight: 600;
        font-size: 0.85rem;
        margin-top: 0.5rem;
        float: right;
        border: 1px solid #106198;
        box-shadow: 0 1px 2px rgba(0,0,0,0.15);
    }
    .satcat-btn:hover {
        background: #106198;
        text-decoration: none;
    }
</style>
""", unsafe_allow_html=True)


def create_earth_sphere(radius_km=6378.137, resolution=20):
    """Create a sphere representing Earth"""
    theta = np.linspace(0, 2 * np.pi, resolution)
    phi = np.linspace(0, np.pi, resolution)
    theta, phi = np.meshgrid(theta, phi)
    
    x = radius_km * np.sin(phi) * np.cos(theta)
    y = radius_km * np.sin(phi) * np.sin(theta)
    z = radius_km * np.cos(phi)
    
    return {'x': x, 'y': y, 'z': z}


def create_orbit_plot(ref_positions, propagated_positions, propagated_final_pos,
                     truth_target_pos, error_km, method_name, method_color,
                     duration_hours, norad_id):
    """Create a single 3D orbit plot"""
    
    fig = go.Figure()

    # Graceful handling: if no propagated points (empty piecewise result),
    # create a single-point path at the target truth to avoid shape errors.
    if propagated_positions is None or propagated_positions.size == 0:
        propagated_positions = truth_target_pos.reshape(3, 1)
        propagated_final_pos = truth_target_pos
        if error_km is None or np.isnan(error_km):
            error_km = 0.0
    
    # 1. Add Earth sphere
    earth_sphere = create_earth_sphere(radius_km=6378.137, resolution=30)
    fig.add_trace(go.Surface(
        x=earth_sphere['x'],
        y=earth_sphere['y'],
        z=earth_sphere['z'],
        colorscale='Blues',
        showscale=False,
        opacity=0.6,
        name='Earth',
        hoverinfo='skip'
    ))
    
    # 2. Add Reference TLE orbit (SGP4 baseline)
    fig.add_trace(go.Scatter3d(
        x=ref_positions[0, :],
        y=ref_positions[1, :],
        z=ref_positions[2, :],
        mode='lines+markers',
        line=dict(color='green', width=4),
        marker=dict(size=3, color='green'),
        name=f'SGP4 Reference ({len(ref_positions[0])} TLEs)',
        hovertemplate='<b>SGP4 Reference</b><br>' +
                      'X: %{x:.3f} km<br>' +
                      'Y: %{y:.3f} km<br>' +
                      'Z: %{z:.3f} km<br>' +
                      '<extra></extra>'
    ))
    
    # 3. Add Target epoch truth / closest TLE position
    fig.add_trace(go.Scatter3d(
        x=[truth_target_pos[0]],
        y=[truth_target_pos[1]],
        z=[truth_target_pos[2]],
        mode='markers',
        marker=dict(size=12, color='darkgreen', symbol='diamond',
                   line=dict(color='white', width=2)),
        name=f'Target Truth (t={duration_hours}h)',
        hovertemplate='<b>Target Truth</b><br>' +
                      'X: %{x:.3f} km<br>' +
                      'Y: %{y:.3f} km<br>' +
                      'Z: %{z:.3f} km<br>' +
                      '<extra></extra>'
    ))
    
    # 4. Add Propagated orbit (Custom + SRP method)
    fig.add_trace(go.Scatter3d(
        x=propagated_positions[0, :],
        y=propagated_positions[1, :],
        z=propagated_positions[2, :],
        mode='lines+markers',
        line=dict(color=method_color, width=4),
        marker=dict(size=4, color=method_color),
        name=f'Custom + {method_name} (Error: {error_km:.3f} km)',
        hovertemplate=f'<b>{method_name}</b><br>' +
                      'X: %{x:.3f} km<br>' +
                      'Y: %{y:.3f} km<br>' +
                      'Z: %{z:.3f} km<br>' +
                      '<extra></extra>'
    ))
    
    # 5. Add Final propagated position
    fig.add_trace(go.Scatter3d(
        x=[propagated_final_pos[0]],
        y=[propagated_final_pos[1]],
        z=[propagated_final_pos[2]],
        mode='markers',
        marker=dict(size=12, color=method_color, symbol='diamond', 
                   line=dict(color='black', width=2)),
        name=f'{method_name} Final Position',
        hovertemplate=f'<b>{method_name} Final</b><br>' +
                      'X: %{x:.3f} km<br>' +
                      'Y: %{y:.3f} km<br>' +
                      'Z: %{z:.3f} km<br>' +
                      f'Error: {error_km:.6f} km<br>' +
                      '<extra></extra>'
    ))
    
    # Layout settings
    fig.update_layout(
        title=dict(
            text=f"<b>SGP4 vs Custom+{method_name}</b><br>" +
                 f"<sub>NORAD {norad_id} | Error: {error_km:.6f} km</sub>",
            x=0.5,
            xanchor='center',
            font=dict(size=16)
        ),
        scene=dict(
            xaxis=dict(title='X (km)', backgroundcolor="rgb(240, 240, 240)", 
                      gridcolor="white", showbackground=True),
            yaxis=dict(title='Y (km)', backgroundcolor="rgb(240, 240, 240)", 
                      gridcolor="white", showbackground=True),
            zaxis=dict(title='Z (km)', backgroundcolor="rgb(240, 240, 240)", 
                      gridcolor="white", showbackground=True),
            aspectmode='data',
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.2))
        ),
        showlegend=True,
        legend=dict(
            x=0.02,
            y=0.98,
            bgcolor='rgba(255,255,255,0.9)',
            bordercolor='black',
            borderwidth=1,
            font=dict(size=10)
        ),
        height=600,
        margin=dict(l=0, r=0, t=60, b=0)
    )
    
    return fig


@st.cache_data(show_spinner=False)
def run_validation_and_get_data(norad_id, time_range_key, include_j2=True, include_drag=None,
                                use_piecewise=True):
    """Run preprocessing + both propagations and collect visualization + metrics.

    Parameters
    ----------
    norad_id : int
    time_range_key : str ('12hr','24hr','2days','7days')
    include_j2, include_drag, use_piecewise : force model flags
    """
    duration_map = {'12hr': 12.0, '24hr': 24.0, '2days': 48.0, '7days': 168.0}
    duration_hours = duration_map[time_range_key]

    preprocessor = ValidationPreprocessor()
    cb_df = preprocessor.run_preprocessing(norad_id, time_range=time_range_key)
    if cb_df is None:
        return None
    full_df = preprocessor.last_preprocessed_df.sort_values('EPOCH').reset_index(drop=True)
    initial_epoch = pd.to_datetime(full_df.iloc[0]['EPOCH'], utc=True)
    target_epoch = initial_epoch + timedelta(hours=duration_hours)
    # Initial inertial state (for fallback when no propagation occurs)
    initial_r_icrf, initial_v_icrf, _ = dataframe_initial_icrf(full_df)

    # Build SRP interpolation arrays
    srp_times_sec = (pd.to_datetime(full_df['EPOCH'], utc=True) - initial_epoch).dt.total_seconds()
    cb_ax = full_df['srp_ax_mps2'].values
    cb_ay = full_df['srp_ay_mps2'].values
    cb_az = full_df['srp_az_mps2'].values

    # --- Minimal early exit: if cannonball SRP is effectively zero everywhere, skip propagation ---
    # Threshold chosen well below typical illuminated SRP (~4.5e-7 m/s^2) so only shadow/zero A/m cases trigger.
    if np.all(np.sqrt(cb_ax**2 + cb_ay**2 + cb_az**2) < 1e-10):  # 1e-10 m/s^2 ~ negligible
        # Determine truth (closest TLE to target epoch) for error calculation
        full_df['EPOCH_DT'] = pd.to_datetime(full_df['EPOCH'], utc=True)
        full_df['time_diff'] = abs((full_df['EPOCH_DT'] - target_epoch).dt.total_seconds())
        closest_idx = full_df['time_diff'].idxmin()
        truth_row = full_df.loc[closest_idx]
        truth_pos, truth_vel = tle_to_icrf(truth_row['TLE_LINE1'], truth_row['TLE_LINE2'])
        err_val = float(np.linalg.norm(initial_r_icrf - truth_pos))
        return {
            'duration_hours': duration_hours,
            'time_range_key': time_range_key,
            'initial_epoch': initial_epoch,
            'target_epoch': target_epoch,
            'truth_pos': truth_pos,
            'truth_epoch': truth_row['EPOCH_DT'],
            'ref_positions': np.array([
                full_df['sat_x_km'].values,
                full_df['sat_y_km'].values,
                full_df['sat_z_km'].values
            ]),
            'cannonball_positions': np.empty((3,0)),
            'ml_positions': np.empty((3,0)),
            'cannonball_final': initial_r_icrf,
            'ml_final': initial_r_icrf,
            'error_cannonball': err_val,
            'error_ml': err_val,
            'cannonball_rtn': {'R':0.0,'T':0.0,'N':0.0,'Total':err_val},
            'ml_rtn': {'R':0.0,'T':0.0,'N':0.0,'Total':err_val},
            'num_tles': len(full_df),
            'include_j2': include_j2,
            'include_drag': include_drag if include_drag is not None else (duration_hours > 24.0),
            'use_piecewise': use_piecewise,
            'drag_params': None,
            'srp_zero': True,
        }

    def cb_srp(t_sec, r_sat_km, r_sun_km):
        return np.array([
            np.interp(t_sec, srp_times_sec, cb_ax),
            np.interp(t_sec, srp_times_sec, cb_ay),
            np.interp(t_sec, srp_times_sec, cb_az),
        ]) / 1000.0

    # ML SRP
    ml_df = preprocessor.run_ml_prediction(model_folder='../model/trained_models')
    ml_times_sec = (pd.to_datetime(ml_df['EPOCH'], utc=True) - initial_epoch).dt.total_seconds()
    ml_ax = ml_df['ml_ax_mps2'].values
    ml_ay = ml_df['ml_ay_mps2'].values
    ml_az = ml_df['ml_az_mps2'].values

    def ml_srp(t_sec, r_sat_km, r_sun_km):
        return np.array([
            np.interp(t_sec, ml_times_sec, ml_ax),
            np.interp(t_sec, ml_times_sec, ml_ay),
            np.interp(t_sec, ml_times_sec, ml_az),
        ]) / 1000.0

    # Decide drag
    if include_drag is None:
        include_drag = duration_hours > 24.0
    drag_params = None
    if include_drag:
        # Estimate mean altitude from TLE positions if available
        if {'sat_x_km','sat_y_km','sat_z_km'}.issubset(full_df.columns):
            rmag = np.sqrt(full_df['sat_x_km']**2 + full_df['sat_y_km']**2 + full_df['sat_z_km']**2)
            h_mean = float(rmag.mean() - 6378.137)
        else:
            h_mean = 500.0
        dens_table = {350:8e-12, 400:4e-12, 450:2e-12, 500:1e-12, 550:6e-13, 600:3e-13, 650:1.5e-13, 700:8e-14}
        nearest = min(dens_table.keys(), key=lambda k: abs(k-h_mean))
        rho0 = dens_table[nearest]
        drag_params = {'Cd':2.2, 'A_over_m': float(full_df.iloc[0]['A_OVER_M']), 'rho0':rho0, 'H':60.0, 'h_ref':nearest}

    # Helper: piecewise propagation replicating custom_propagator logic
    def run_piecewise(srp_func):
        """Segment-by-segment propagation that correctly re-anchors state at each TLE.

        Unlike the previous implementation, this uses propagate_orbit directly with the
        current_r/current_v as initial conditions for each segment. The prior version
        mistakenly called propagate_from_dataframe (which always resets to FIRST TLE),
        causing each segment to restart from the initial epoch and masking SRP model
        differences.
        """
        r0, v0, _ = dataframe_initial_icrf(full_df)
        current_r, current_v = r0, v0
        current_epoch = initial_epoch
        epochs_list = pd.to_datetime(full_df['EPOCH'], utc=True).tolist()
        positions_segments = []
        velocities_segments = []
        times_segments = []
        for next_ep in epochs_list[1:]:
            dt_hours = (next_ep - current_epoch).total_seconds() / 3600.0
            if dt_hours <= 0:
                continue
            # Build segment SRP interpolation from appropriate dataframe (cannonball vs ML)
            if srp_func is cb_srp:
                seg_mask = (pd.to_datetime(full_df['EPOCH'], utc=True) >= current_epoch) & (pd.to_datetime(full_df['EPOCH'], utc=True) <= next_ep)
                seg_df = full_df[seg_mask].reset_index(drop=True)
                seg_times = (pd.to_datetime(seg_df['EPOCH'], utc=True) - current_epoch).dt.total_seconds()
                seg_ax = seg_df['srp_ax_mps2'].values
                seg_ay = seg_df['srp_ay_mps2'].values
                seg_az = seg_df['srp_az_mps2'].values
            else:
                seg_mask = (pd.to_datetime(ml_df['EPOCH'], utc=True) >= current_epoch) & (pd.to_datetime(ml_df['EPOCH'], utc=True) <= next_ep)
                seg_df = ml_df[seg_mask].reset_index(drop=True)
                seg_times = (pd.to_datetime(seg_df['EPOCH'], utc=True) - current_epoch).dt.total_seconds()
                seg_ax = seg_df['ml_ax_mps2'].values
                seg_ay = seg_df['ml_ay_mps2'].values
                seg_az = seg_df['ml_az_mps2'].values
            if seg_df.empty:
                # No data in this segment; skip
                current_epoch = next_ep
                next_row = full_df.loc[pd.to_datetime(full_df['EPOCH'], utc=True) == next_ep].iloc[0]
                current_r, current_v = tle_to_icrf(next_row['TLE_LINE1'], next_row['TLE_LINE2'])
                continue
            def seg_srp(t_sec, r_sat_km, r_sun_km):
                return np.array([
                    np.interp(t_sec, seg_times, seg_ax),
                    np.interp(t_sec, seg_times, seg_ay),
                    np.interp(t_sec, seg_times, seg_az),
                ]) / 1000.0
            sol_seg = propagate_orbit(
                current_r,
                current_v,
                dt_hours * 3600.0,
                ts.from_datetime(current_epoch.to_pydatetime()),
                seg_srp,
                quiet=True,
                include_j2=include_j2,
                include_drag=include_drag,
                drag_params=drag_params,
                max_step=60.0 if duration_hours > 24 else 120.0,
                rtol=1e-11 if duration_hours > 24 else 1e-10,
                atol=1e-13 if duration_hours > 24 else 1e-12,
            )
            positions_segments.append(sol_seg.y[:3, :])
            velocities_segments.append(sol_seg.y[3:, :])
            times_segments.append(sol_seg.t + (current_epoch - initial_epoch).total_seconds())
            # Re-anchor at next TLE epoch using actual TLE-derived inertial state
            next_row = full_df.loc[pd.to_datetime(full_df['EPOCH'], utc=True) == next_ep].iloc[0]
            current_r, current_v = tle_to_icrf(next_row['TLE_LINE1'], next_row['TLE_LINE2'])
            current_epoch = next_ep
            if (current_epoch - initial_epoch).total_seconds() / 3600.0 >= duration_hours:
                break
        if positions_segments:
            pos_concat = np.concatenate(positions_segments, axis=1)
            times_concat = np.concatenate(times_segments)
        else:
            pos_concat = np.empty((3, 0))
            times_concat = np.empty((0,))
        return pos_concat, times_concat

    # Run propagations
    if use_piecewise:
        cb_positions, cb_times = run_piecewise(cb_srp)
        ml_positions, ml_times = run_piecewise(ml_srp)
        # Fallback: if piecewise produced no data (e.g., only one TLE), run single-pass propagate
        if cb_positions.size == 0:
            cb_result = propagate_from_dataframe(full_df, cb_srp, duration_hours=duration_hours, quiet=True,
                                                 include_j2=include_j2, include_drag=include_drag, drag_params=drag_params,
                                                 max_step=60.0 if duration_hours>24 else 120.0)
            cb_positions = cb_result['positions_km']
        if ml_positions.size == 0:
            ml_result = propagate_from_dataframe(full_df, ml_srp, duration_hours=duration_hours, quiet=True,
                                                 include_j2=include_j2, include_drag=include_drag, drag_params=drag_params,
                                                 max_step=60.0 if duration_hours>24 else 120.0)
            ml_positions = ml_result['positions_km']
    else:
        cb_result = propagate_from_dataframe(full_df, cb_srp, duration_hours=duration_hours, quiet=True,
                                             include_j2=include_j2, include_drag=include_drag, drag_params=drag_params,
                                             max_step=60.0 if duration_hours>24 else 120.0)
        ml_result = propagate_from_dataframe(full_df, ml_srp, duration_hours=duration_hours, quiet=True,
                                             include_j2=include_j2, include_drag=include_drag, drag_params=drag_params,
                                             max_step=60.0 if duration_hours>24 else 120.0)
        cb_positions = cb_result['positions_km']
        ml_positions = ml_result['positions_km']

    # Reference (discrete TLE SGP4-derived) path
    ref_positions = np.array([
        full_df['sat_x_km'].values,
        full_df['sat_y_km'].values,
        full_df['sat_z_km'].values
    ])

    # Determine truth at target epoch (closest TLE)
    full_df['EPOCH_DT'] = pd.to_datetime(full_df['EPOCH'], utc=True)
    full_df['time_diff'] = abs((full_df['EPOCH_DT'] - target_epoch).dt.total_seconds())
    closest_idx = full_df['time_diff'].idxmin()
    truth_row = full_df.loc[closest_idx]
    truth_pos, truth_vel = tle_to_icrf(truth_row['TLE_LINE1'], truth_row['TLE_LINE2'])

    # Final position fallbacks: if no columns, use initial inertial position
    if cb_positions.shape[1] > 0:
        cb_final = cb_positions[:, -1]
    else:
        cb_final = initial_r_icrf
    if ml_positions.shape[1] > 0:
        ml_final = ml_positions[:, -1]
    else:
        ml_final = initial_r_icrf

    error_cb = float(np.linalg.norm(cb_final - truth_pos)) if cb_positions.size else float('nan')
    error_ml = float(np.linalg.norm(ml_final - truth_pos)) if ml_positions.size else float('nan')

    # RTN decomposition helper
    def rtn_err(pred):
        dv = pred - truth_pos
        if np.linalg.norm(truth_pos)==0:
            return (np.nan, np.nan, np.nan, np.nan)
        # Simple orbital frame approximation
        r_hat = truth_pos / np.linalg.norm(truth_pos)
        h_vec = np.cross(truth_pos, truth_vel)
        n_hat = h_vec / np.linalg.norm(h_vec)
        t_hat = np.cross(n_hat, r_hat)
        return (
            float(np.dot(dv, r_hat)),
            float(np.dot(dv, t_hat)),
            float(np.dot(dv, n_hat)),
            float(np.linalg.norm(dv))
        )
    cb_R, cb_T, cb_N, cb_tot = rtn_err(cb_final)
    ml_R, ml_T, ml_N, ml_tot = rtn_err(ml_final)

    return {
        'duration_hours': duration_hours,
        'time_range_key': time_range_key,
        'initial_epoch': initial_epoch,
        'target_epoch': target_epoch,
        'truth_pos': truth_pos,
        'truth_epoch': truth_row['EPOCH_DT'],
        'ref_positions': ref_positions,
        'cannonball_positions': cb_positions,
        'ml_positions': ml_positions,
        'cannonball_final': cb_final,
        'ml_final': ml_final,
        'error_cannonball': error_cb,
        'error_ml': error_ml,
        'cannonball_rtn': {'R':cb_R,'T':cb_T,'N':cb_N,'Total':cb_tot},
        'ml_rtn': {'R':ml_R,'T':ml_T,'N':ml_N,'Total':ml_tot},
        'num_tles': len(full_df),
        'include_j2': include_j2,
        'include_drag': include_drag,
        'use_piecewise': use_piecewise,
        'drag_params': drag_params,
    }


def main():
    # Header
    st.markdown('<div class="main-header">🛰️ Satellite Orbit Validation  </div>',
                unsafe_allow_html=True)
    st.markdown('<div class="sub-header">ML SRP vs Cannonball</div>',
                unsafe_allow_html=True)
    
    # Sidebar for inputs
    st.sidebar.header("Input Features")
    
    # NORAD ID dropdown
    norad_ids = [41240, 44804, 43476, 44857, 60309, 59386, 37839, 41790, 39634]
    selected_norad = st.sidebar.selectbox(
        "Select NORAD ID",
        options=norad_ids,
        index=0,
        help="Choose satellite NORAD catalog ID"
    )

    # Satellite parameters table (auto-refresh on NORAD change)
    try:
        sat_params = get_params(int(selected_norad))
        a_over_m_est = sat_params['surface_area_m2'] / sat_params['mass_kg'] if sat_params['mass_kg'] else None
        sat_df = pd.DataFrame([
            {
                'Parameter': 'Name',
                'Value': sat_params.get('name') or f"NORAD {selected_norad}"
            },
            {
                'Parameter': 'Mass (kg)',
                'Value': f"{sat_params['mass_kg']:.1f}"
            },
            {
                'Parameter': 'Surface Area (m²)',
                'Value': f"{sat_params['surface_area_m2']:.2f}"
            },
            {
                'Parameter': 'Orbit Altitude (km)',
                'Value': f"{sat_params['orbit_alt_km']:.1f}"
            },
            {
                'Parameter': 'Reflectivity Coeff (Cr)',
                'Value': f"{sat_params['cr']:.2f}"
            },
            {
                'Parameter': 'Estimated A/m (m²/kg)',
                'Value': f"{a_over_m_est:.4f}" if a_over_m_est else 'NA'
            }
        ])
        st.sidebar.markdown("### 🛰️ Satellite Details")
        st.sidebar.dataframe(sat_df, hide_index=True, use_container_width=True)
        # Styled SatCat external link button placed adjacent to details
        satcat_url = f"https://www.satcat.com/sats/{selected_norad}"
        st.sidebar.markdown(f"<a href='{satcat_url}' target='_blank' class='satcat-btn' title='Open external SatCat catalog page'>Know More</a>", unsafe_allow_html=True)
    except Exception as _e:
        st.sidebar.warning("Satellite parameters unavailable.")
    
    # Time range dropdown (include 12hr)
    time_ranges_map = {
        '12 hr':'12hr',
        '24 hr (1 day)':'24hr',
        '48 hr (2 days)':'2days',
        '168 hr (7 days)':'7days'
    }
    selected_time_range_label = st.sidebar.selectbox(
        "Propagation Duration",
        options=list(time_ranges_map.keys()),
        index=1
    )
    selected_time_range = time_ranges_map[selected_time_range_label]
    
    # Drag coefficient (optional advanced setting)
    with st.sidebar.expander("🔧 Force Model Settings"):
        include_j2 = st.checkbox("Include J2", value=True)
        user_drag_choice = st.checkbox("Include Drag (simple model)", value=(selected_time_range in ['2days','7days']))
        piecewise = st.checkbox("Piecewise Re-anchoring", value=(selected_time_range in ['2days','7days']))
        st.caption("Piecewise reduces long-arc drift by re-starting at each TLE epoch.")
    
    # Run button
    run_validation = st.sidebar.button("🚀 Run Validation", type="primary")
    
    # Information section
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📖 About")
    st.sidebar.info(
        """
        This app validates ML‑predicted Solar Radiation Pressure (SRP) against a
        classical Cannonball model with optional J2, Drag and piecewise re‑anchoring.
        
        Reference path points are derived from TLE (SGP4). Errors are measured at the
        target epoch using the closest TLE state .
        """
    )
    
    # Main content
    if run_validation:
        with st.spinner(f"🔄 Running validation for NORAD {selected_norad} ({selected_time_range})..."):
            try:
                # Run validation
                data = run_validation_and_get_data(selected_norad, selected_time_range,
                                                   include_j2=include_j2,
                                                   include_drag=user_drag_choice,
                                                   use_piecewise=piecewise)
                
                if data is None:
                    st.error("❌ Validation failed! Please check the NORAD ID and try again.")
                    return
                # Simple SRP-zero message (inherit property from summary if provided)
                if data.get('srp_zero'):
                    st.info("☀️ SRP accelerations are zero (shadow or negligible A/m) – Cannonball and ML paths identical.")
                
                # Display summary metrics
                st.success(f"✅ Validation complete! Collected {data['num_tles']} TLEs")
                
                # Calculate improvement
                improvement_ml = data['error_cannonball'] - data['error_ml']
                improvement_pct = (improvement_ml / data['error_cannonball'] * 100) if data['error_cannonball'] else 0.0
                
                # Determine winner
                if abs(improvement_ml) < 1e-6:
                    winner = "TIE"
                    winner_class = ""
                elif improvement_ml > 0:
                    winner = "ML SRP"
                    winner_class = "winner-ml"
                else:
                    winner = "Cannonball SRP"
                    winner_class = "winner-cannonball"
                
                # --- Distance formatting helper ---
                def fmt_distance(km_val: float) -> str:
                    if km_val is None or np.isnan(km_val):
                        return "NA"
                    if km_val >= 1.0:
                        return f"{km_val:.3f} km"
                    if km_val >= 0.001:  # >= 1 m
                        return f"{km_val*1000:.2f} m"
                    if km_val >= 0.000001:  # >= 1 mm (since 1e-6 km = 1 m?) Actually 1e-6 km = 1 m
                        # 0.000001 km = 1 m; branch above covers >= 1 m, so here we are <1 m
                        # Show centimeters for <1 m
                        return f"{km_val*100000:.2f} cm"  # km * 100000 = cm
                    return f"{km_val*1e6:.2f} mm"  # very tiny

                # Simplified metrics: only Improvement and Winner
                col_imp, col_win = st.columns(2)
                improve_text = fmt_distance(abs(improvement_ml))
                show_pct = improvement_pct >= 0.5
                pct_suffix = f" ({improvement_pct:.2f}%)" if show_pct else ""
                with col_imp:
                    st.metric(
                        "📊 Improvement",
                        improve_text + pct_suffix
                    )
                with col_win:
                    st.metric(
                        "🏆 Winner",
                        winner
                    )

                # Store run history in session state
                if 'run_history' not in st.session_state:
                    st.session_state['run_history'] = []
                st.session_state['run_history'].append({
                    'NORAD': selected_norad,
                    'Name': (sat_params.get('name') if 'sat_params' in locals() and sat_params.get('name') else f"NORAD {selected_norad}"),
                    'Arc': selected_time_range,
                    'J2': data['include_j2'],
                    'Drag': data['include_drag'],
                    'Piecewise': data['use_piecewise'],
                    'Err_CB_km': data['error_cannonball'],
                    'Err_ML_km': data['error_ml'],
                    'Improvement_km': improvement_ml,
                    'Winner': winner
                })
                
                # Winner announcement
                if winner != "TIE":
                    if winner == "ML SRP":
                        st.success(
                            f"🎉 **ML SRP is BETTER!** Reduces error by **{improvement_ml:.6f} km** "
                            f"({improvement_pct:.2f}%) compared to Cannonball model."
                        )
                    else:
                        st.warning(
                            f"⚠️ **Cannonball is better.** ML SRP has **{abs(improvement_ml):.6f} km** "
                            f"({abs(improvement_pct):.2f}%) more error."
                        )
                else:
                    st.info("🤝 **It's a tie!** Both methods have identical errors.")
                
                # Epoch information
                st.markdown("---")
                st.markdown("### 📅 Epoch Information")
                col_a, col_b = st.columns(2)
                with col_a:
                    st.write(f"**Initial Epoch:** {data['initial_epoch'].strftime('%Y-%m-%d %H:%M:%S UTC')}")
                with col_b:
                    st.write(f"**Target Epoch:** {data['target_epoch'].strftime('%Y-%m-%d %H:%M:%S UTC')}")
                
                st.markdown(f"**Propagation Duration:** {data['duration_hours']} hours")
                st.markdown(f"**Truth (Closest TLE) Epoch:** {data['truth_epoch'].strftime('%Y-%m-%d %H:%M:%S UTC')}")
                st.markdown(f"**Force Models:** J2={'ON' if data['include_j2'] else 'OFF'} | Drag={'ON' if data['include_drag'] else 'OFF'} | Piecewise={'ON' if data['use_piecewise'] else 'OFF'}")
                if data['include_drag'] and data['drag_params']:
                    dp = data['drag_params']
                    st.caption(f"Drag params: Cd={dp['Cd']}, A/m={dp['A_over_m']:.4f} m²/kg, rho0={dp['rho0']:.2e} kg/m³ @ {dp['h_ref']} km")
                
                st.markdown("---")
                
                # Create two columns for side-by-side plots
                st.markdown("### 🌍 3D Orbit Visualizations")
                
                col_left, col_right = st.columns(2)
                
                with col_left:
                    st.markdown("#### SGP4 vs Custom + ML SRP")
                    fig_ml = create_orbit_plot(
                        ref_positions=data['ref_positions'],
                        propagated_positions=data['ml_positions'],
                        propagated_final_pos=data['ml_final'],
                        truth_target_pos=data['truth_pos'],
                        error_km=data['error_ml'],
                        method_name="ML SRP",
                        method_color="red",
                        duration_hours=data['duration_hours'],
                        norad_id=selected_norad
                    )
                    st.plotly_chart(fig_ml, use_container_width=True)
                
                with col_right:
                    st.markdown("#### SGP4 vs Custom + Cannonball SRP")
                    fig_cannonball = create_orbit_plot(
                        ref_positions=data['ref_positions'],
                        propagated_positions=data['cannonball_positions'],
                        propagated_final_pos=data['cannonball_final'],
                        truth_target_pos=data['truth_pos'],
                        error_km=data['error_cannonball'],
                        method_name="Cannonball SRP",
                        method_color="orange",
                        duration_hours=data['duration_hours'],
                        norad_id=selected_norad
                    )
                    st.plotly_chart(fig_cannonball, use_container_width=True)
                
                # Additional details in expandable section
                with st.expander("📊 Detailed Comparison"):
                    st.markdown("#### Position & RTN Comparison")
                    comp_df = pd.DataFrame([
                        {
                            'Method':'Truth (Target TLE)',
                            'X (km)': f"{data['truth_pos'][0]:.6f}",
                            'Y (km)': f"{data['truth_pos'][1]:.6f}",
                            'Z (km)': f"{data['truth_pos'][2]:.6f}",
                            'Error (km)': '0.000000'
                        },
                        {
                            'Method':'Custom + ML SRP',
                            'X (km)': f"{data['ml_final'][0]:.6f}",
                            'Y (km)': f"{data['ml_final'][1]:.6f}",
                            'Z (km)': f"{data['ml_final'][2]:.6f}",
                            'Error (km)': f"{data['error_ml']:.6f}",
                        },
                        {
                            'Method':'Custom + Cannonball',
                            'X (km)': f"{data['cannonball_final'][0]:.6f}",
                            'Y (km)': f"{data['cannonball_final'][1]:.6f}",
                            'Z (km)': f"{data['cannonball_final'][2]:.6f}",
                            'Error (km)': f"{data['error_cannonball']:.6f}",
                        }
                    ])
                    st.dataframe(comp_df, use_container_width=True, hide_index=True)

                    # RTN table
                    rtn_df = pd.DataFrame([
                        {'Method':'Custom + ML', **data['ml_rtn']},
                        {'Method':'Custom + Cannonball', **data['cannonball_rtn']},
                    ])
                    st.markdown("#### RTN Error Components (km)")
                    st.dataframe(rtn_df, use_container_width=True, hide_index=True)

                    # (RTN bar chart removed per user request)

                # History
                with st.expander("📜 Run History (Session)"):
                    hist_df = pd.DataFrame(st.session_state['run_history'])
                    st.dataframe(hist_df, use_container_width=True)
                    best_ml = hist_df[hist_df['Winner']=="ML SRP"] if not hist_df.empty else pd.DataFrame()
                    if not best_ml.empty:
                        st.caption(f"ML SRP won {len(best_ml)} out of {len(hist_df)} runs ({len(best_ml)/len(hist_df)*100:.1f}%).")
                
            except Exception as e:
                st.error(f"❌ Error during validation: {str(e)}")
                st.exception(e)
    
    else:
        # Initial state - show instructions
        st.info(
            """
            👈 **Get Started:**
            1. Select a **NORAD ID** from the dropdown
            2. Choose a **propagation duration** (12 hr, 24 hr, 48 hr, or 168 hr)
            3. (Optional) Toggle **J2 / Drag / Piecewise** in Force Model Settings
            4. Click **"🚀 Run Validation"** to generate 3D orbit visualizations
            
            The app will show two Earth-centered 3D plots comparing:
            - **SGP4 vs ML SRP** (left)
            - **SGP4 vs Cannonball SRP** (right)
            
            You'll see which method performs better! 🎯
            """
        )
        
        # Show example visualization placeholder
        st.markdown("### Preview")
        col1, col2 = st.columns(2)
        with col1:
            st.image(
                "https://via.placeholder.com/400x300.png?text=ML+SRP+Orbit",
                caption="SGP4 vs ML SRP visualization placeholder"
            )
        with col2:
            st.image(
                "https://via.placeholder.com/400x300.png?text=Cannonball+Orbit",
                caption="SGP4 vs Cannonball visualization placeholder"
            )


if __name__ == '__main__':
    main()

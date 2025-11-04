"""
3D Orbit Visualization for Validation Results

Visualizes three orbits:
1. Reference TLE orbit (SGP4 baseline - latest ground truth)
2. Custom Propagator + Cannonball SRP orbit
3. Custom Propagator + ML SRP orbit

All orbits shown in 3D ICRF frame with Earth at center.

Author: Divyanshu Panday
Date: October 2025
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from validation.custom_propagator import run_complete_validation
from validation.preprocessing_runner import ValidationPreprocessor
from datetime import timedelta


def create_earth_sphere(radius_km=6378.137, resolution=20):
    """
    Create a sphere representing Earth
    
    Parameters:
    -----------
    radius_km : float
        Earth radius in km (default: 6378.137 km)
    resolution : int
        Number of points for sphere mesh
        
    Returns:
    --------
    dict : {'x': array, 'y': array, 'z': array} for sphere coordinates
    """
    theta = np.linspace(0, 2 * np.pi, resolution)
    phi = np.linspace(0, np.pi, resolution)
    theta, phi = np.meshgrid(theta, phi)
    
    x = radius_km * np.sin(phi) * np.cos(theta)
    y = radius_km * np.sin(phi) * np.sin(theta)
    z = radius_km * np.cos(phi)
    
    return {'x': x, 'y': y, 'z': z}


def visualize_orbits_3d(norad_id, time_range='24hr', C_d=2.2, save_html=True, comparison_mode='both'):
    """
    Create 3D visualization of validation results
    
    Parameters:
    -----------
    norad_id : int
        NORAD catalog ID
    time_range : str
        '24hr', '2days', or '7days'
    C_d : float
        Drag coefficient
    save_html : bool
        If True, saves interactive HTML file
    comparison_mode : str
        'ml' - Compare SGP4 vs Custom+ML only
        'cannonball' - Compare SGP4 vs Custom+Cannonball only
        'both' - Show all three orbits (default)
        
    Returns:
    --------
    plotly.graph_objects.Figure
    """
    
    print("=" * 80)
    mode_text = {'ml': 'SGP4 vs ML', 'cannonball': 'SGP4 vs Cannonball', 'both': 'All Methods'}
    print(f" 3D ORBIT VISUALIZATION: NORAD {norad_id} ({time_range}) - {mode_text[comparison_mode]} ".center(80))
    print("=" * 80)
    
    # Duration mapping
    duration_map = {'24hr': 24.0, '2days': 48.0, '7days': 168.0}
    duration_hours = duration_map[time_range]
    
    # =================================================================
    # STEP 1: Get TLE data and preprocessing
    # =================================================================
    print("\n[1/5] Running preprocessing...")
    preprocessor = ValidationPreprocessor()
    cannonball_df = preprocessor.run_preprocessing(norad_id, time_range=time_range)
    
    if cannonball_df is None:
        print("❌ Preprocessing failed!")
        return None
    
    full_df = preprocessor.last_preprocessed_df
    print(f"✅ Collected {len(full_df)} TLEs")
    
    # Get initial epoch
    initial_epoch = pd.to_datetime(full_df.iloc[0]['EPOCH'], utc=True)
    target_epoch = initial_epoch + timedelta(hours=duration_hours)
    
    # =================================================================
    # STEP 2: Get Reference TLE orbit (Latest TLE positions)
    # =================================================================
    print("\n[2/5] Extracting reference TLE positions...")
    
    # Get all TLE positions (these are from SGP4 calculations in preprocessing)
    ref_positions = np.array([
        full_df['sat_x_km'].values,
        full_df['sat_y_km'].values,
        full_df['sat_z_km'].values
    ])
    
    ref_epochs = pd.to_datetime(full_df['EPOCH'], utc=True)
    
    # Find the latest (last) TLE position
    latest_tle_pos = ref_positions[:, -1]
    latest_tle_epoch = ref_epochs.iloc[-1]
    
    print(f"✅ Reference orbit has {ref_positions.shape[1]} TLE points")
    print(f"   Latest TLE at: {latest_tle_epoch.strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print(f"   Latest position: [{latest_tle_pos[0]:.3f}, {latest_tle_pos[1]:.3f}, {latest_tle_pos[2]:.3f}] km")
    
    # =================================================================
    # STEP 3: Propagate with Cannonball SRP
    # =================================================================
    cannonball_positions = None
    cannonball_final_pos = None
    error_cannonball = None
    
    if comparison_mode in ['cannonball', 'both']:
        print("\n[3/5] Running Custom Propagator + Cannonball SRP...")
        
        from validation.custom_propagator import propagate_from_dataframe
        
        # Prepare Cannonball SRP function
        srp_time_diffs = (pd.to_datetime(full_df['EPOCH'], utc=True) - initial_epoch)
        srp_times_sec = srp_time_diffs.dt.total_seconds()
        srp_ax_values = full_df['srp_ax_mps2'].values
        srp_ay_values = full_df['srp_ay_mps2'].values
        srp_az_values = full_df['srp_az_mps2'].values
        
        def cannonball_srp_func(t_sec, r_sat_km, r_sun_km):
            ax_mps2 = np.interp(t_sec, srp_times_sec, srp_ax_values)
            ay_mps2 = np.interp(t_sec, srp_times_sec, srp_ay_values)
            az_mps2 = np.interp(t_sec, srp_times_sec, srp_az_values)
            return np.array([ax_mps2, ay_mps2, az_mps2]) / 1000.0
        
        cannonball_result = propagate_from_dataframe(
            full_df, cannonball_srp_func, duration_hours=duration_hours, C_d=C_d
        )
        
        cannonball_positions = cannonball_result['positions_km']
        cannonball_final_pos = cannonball_positions[:, -1]
        
        print(f"✅ Cannonball orbit has {cannonball_positions.shape[1]} propagated points")
        print(f"   Final position: [{cannonball_final_pos[0]:.3f}, {cannonball_final_pos[1]:.3f}, {cannonball_final_pos[2]:.3f}] km")
    else:
        print("\n[3/5] Skipping Cannonball SRP (not in comparison mode)")
    
    # =================================================================
    # STEP 4: Propagate with ML SRP
    # =================================================================
    ml_positions = None
    ml_final_pos = None
    error_ml = None
    
    if comparison_mode in ['ml', 'both']:
        print("\n[4/5] Running Custom Propagator + ML SRP...")
        
        from validation.custom_propagator import propagate_from_dataframe
        
        # Get ML predictions
        ml_df = preprocessor.run_ml_prediction(model_folder='../model/trained_models')
        
        # Prepare ML SRP function
        ml_time_diffs = (pd.to_datetime(ml_df['EPOCH'], utc=True) - initial_epoch)
        ml_times_sec = ml_time_diffs.dt.total_seconds()
        ml_ax_values = ml_df['ml_ax_mps2'].values
        ml_ay_values = ml_df['ml_ay_mps2'].values
        ml_az_values = ml_df['ml_az_mps2'].values
        
        def ml_srp_func(t_sec, r_sat_km, r_sun_km):
            ax_mps2 = np.interp(t_sec, ml_times_sec, ml_ax_values)
            ay_mps2 = np.interp(t_sec, ml_times_sec, ml_ay_values)
            az_mps2 = np.interp(t_sec, ml_times_sec, ml_az_values)
            return np.array([ax_mps2, ay_mps2, az_mps2]) / 1000.0
        
        ml_result = propagate_from_dataframe(
            full_df, ml_srp_func, duration_hours=duration_hours, C_d=C_d
        )
        
        ml_positions = ml_result['positions_km']
        ml_final_pos = ml_positions[:, -1]
        
        print(f"✅ ML orbit has {ml_positions.shape[1]} propagated points")
        print(f"   Final position: [{ml_final_pos[0]:.3f}, {ml_final_pos[1]:.3f}, {ml_final_pos[2]:.3f}] km")
    else:
        print("\n[4/5] Skipping ML SRP (not in comparison mode)")
    
    # =================================================================
    # STEP 5: Calculate errors
    # =================================================================
    print("\n[5/5] Calculating position errors...")
    
    if cannonball_final_pos is not None:
        error_cannonball = np.linalg.norm(cannonball_final_pos - latest_tle_pos)
        print(f"   Cannonball: {error_cannonball:.6f} km")
    
    if ml_final_pos is not None:
        error_ml = np.linalg.norm(ml_final_pos - latest_tle_pos)
        print(f"   ML:         {error_ml:.6f} km")
    
    # Determine winner
    if error_ml is not None and error_cannonball is not None:
        print(f"\n📊 Position Errors (vs Latest TLE):")
        print(f"   Cannonball: {error_cannonball:.6f} km")
        print(f"   ML:         {error_ml:.6f} km")
        
        if abs(error_ml - error_cannonball) < 1e-6:
            winner = "TIE"
            improvement = 0.0
        elif error_ml < error_cannonball:
            winner = "ML"
            improvement = error_cannonball - error_ml
            print(f"   ✅ Winner: ML (better by {improvement:.6f} km)")
        else:
            winner = "Cannonball"
            degradation = error_ml - error_cannonball
            print(f"   ✅ Winner: Cannonball (ML worse by {degradation:.6f} km)")
            improvement = -degradation
    elif error_ml is not None:
        winner = "ML"
        improvement = error_ml
        print(f"   ML Error: {error_ml:.6f} km")
    elif error_cannonball is not None:
        winner = "Cannonball"
        improvement = error_cannonball
        print(f"   Cannonball Error: {error_cannonball:.6f} km")
    else:
        winner = "Unknown"
        improvement = 0.0
    
    # =================================================================
    # STEP 6: Create 3D Plotly Visualization
    # =================================================================
    print("\n[6/6] Creating 3D visualization...")
    
    fig = go.Figure()
    
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
    
    # 2. Add Reference TLE orbit (ground truth path)
    fig.add_trace(go.Scatter3d(
        x=ref_positions[0, :],
        y=ref_positions[1, :],
        z=ref_positions[2, :],
        mode='lines+markers',
        line=dict(color='green', width=4),
        marker=dict(size=3, color='green'),
        name=f'Reference TLE Orbit ({len(ref_positions[0])} points)',
        hovertemplate='<b>Reference TLE</b><br>' +
                      'X: %{x:.3f} km<br>' +
                      'Y: %{y:.3f} km<br>' +
                      'Z: %{z:.3f} km<br>' +
                      '<extra></extra>'
    ))
    
    # 3. Add Latest TLE position (target point)
    fig.add_trace(go.Scatter3d(
        x=[latest_tle_pos[0]],
        y=[latest_tle_pos[1]],
        z=[latest_tle_pos[2]],
        mode='markers',
        marker=dict(size=10, color='darkgreen', symbol='diamond'),
        name=f'Latest TLE Position (t={duration_hours}hr)',
        hovertemplate='<b>Latest TLE (Reference)</b><br>' +
                      'X: %{x:.3f} km<br>' +
                      'Y: %{y:.3f} km<br>' +
                      'Z: %{z:.3f} km<br>' +
                      '<extra></extra>'
    ))
    
    # 4. Add Cannonball SRP orbit (only if in comparison mode)
    if cannonball_positions is not None:
        fig.add_trace(go.Scatter3d(
            x=cannonball_positions[0, :],
            y=cannonball_positions[1, :],
            z=cannonball_positions[2, :],
            mode='lines+markers',
            line=dict(color='orange', width=4),
            marker=dict(size=4, color='orange'),
            name=f'Custom + Cannonball SRP (Error: {error_cannonball:.3f} km)',
            hovertemplate='<b>Cannonball SRP</b><br>' +
                          'X: %{x:.3f} km<br>' +
                          'Y: %{y:.3f} km<br>' +
                          'Z: %{z:.3f} km<br>' +
                          '<extra></extra>'
        ))
        
        # 5. Add Cannonball final position
        fig.add_trace(go.Scatter3d(
            x=[cannonball_final_pos[0]],
            y=[cannonball_final_pos[1]],
            z=[cannonball_final_pos[2]],
            mode='markers',
            marker=dict(size=12, color='darkorange', symbol='diamond', line=dict(color='black', width=2)),
            name=f'Cannonball Final (t={duration_hours}hr)',
            hovertemplate='<b>Cannonball Final</b><br>' +
                          'X: %{x:.3f} km<br>' +
                          'Y: %{y:.3f} km<br>' +
                          'Z: %{z:.3f} km<br>' +
                          f'Error: {error_cannonball:.6f} km<br>' +
                          '<extra></extra>'
        ))
    
    # 6. Add ML SRP orbit (only if in comparison mode)
    if ml_positions is not None:
        fig.add_trace(go.Scatter3d(
            x=ml_positions[0, :],
            y=ml_positions[1, :],
            z=ml_positions[2, :],
            mode='lines+markers',
            line=dict(color='red', width=4),
            marker=dict(size=4, color='red'),
            name=f'Custom + ML SRP (Error: {error_ml:.3f} km)',
            hovertemplate='<b>ML SRP</b><br>' +
                          'X: %{x:.3f} km<br>' +
                          'Y: %{y:.3f} km<br>' +
                          'Z: %{z:.3f} km<br>' +
                          '<extra></extra>'
        ))
        
        # 7. Add ML final position
        fig.add_trace(go.Scatter3d(
            x=[ml_final_pos[0]],
            y=[ml_final_pos[1]],
            z=[ml_final_pos[2]],
            mode='markers',
            marker=dict(size=12, color='darkred', symbol='diamond', line=dict(color='black', width=2)),
            name=f'ML Final (t={duration_hours}hr)',
            hovertemplate='<b>ML Final</b><br>' +
                          'X: %{x:.3f} km<br>' +
                          'Y: %{y:.3f} km<br>' +
                          'Z: %{z:.3f} km<br>' +
                          f'Error: {error_ml:.6f} km<br>' +
                          '<extra></extra>'
        ))
    
    # Layout settings
    fig.update_layout(
        title=dict(
            text=f"<b>3D Orbit Validation - NORAD {norad_id} ({time_range})</b><br>" +
                 f"<sub>Winner: {winner} | Improvement: {improvement:.6f} km</sub>",
            x=0.5,
            xanchor='center',
            font=dict(size=18)
        ),
        scene=dict(
            xaxis=dict(title='X (km)', backgroundcolor="rgb(230, 230,230)", gridcolor="white"),
            yaxis=dict(title='Y (km)', backgroundcolor="rgb(230, 230,230)", gridcolor="white"),
            zaxis=dict(title='Z (km)', backgroundcolor="rgb(230, 230,230)", gridcolor="white"),
            aspectmode='data',
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.2)
            )
        ),
        showlegend=True,
        legend=dict(
            x=0.02,
            y=0.98,
            bgcolor='rgba(255,255,255,0.8)',
            bordercolor='black',
            borderwidth=1
        ),
        width=1200,
        height=900,
        hovermode='closest'
    )
    
    # Save HTML if requested
    if save_html:
        filename = f"orbit_3d_norad{norad_id}_{time_range}.html"
        fig.write_html(filename)
        print(f"\n✅ Visualization saved: {filename}")
    
    print(f"\n{'='*80}")
    print(f"✅ 3D Visualization Complete!")
    print(f"{'='*80}")
    
    # Show plot
    fig.show()
    
    return fig


def main():
    """
    Example usage
    """
    print("=" * 80)
    print(" 3D ORBIT VISUALIZER ".center(80))
    print("=" * 80)
    
    # Create visualization
    fig = visualize_orbits_3d(
        norad_id=39634,      # Example satellite
        time_range='24hr',   # Options: '24hr', '2days', '7days'
        C_d=2.2,             # Drag coefficient
        save_html=True       # Save interactive HTML
    )
    
    if fig:
        print("\n✅ Visualization complete!")
        print("   Check the HTML file for interactive 3D view")


if __name__ == '__main__':
    main()

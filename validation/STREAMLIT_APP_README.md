# 3D Orbit Validation - Streamlit App

## Quick Start

### 1. Install Streamlit (if not installed)
```bash
pip install streamlit
```

### 2. Run the app
```bash
cd validation
streamlit run streamlit_orbit_app.py
```

Or use PowerShell:
```powershell
cd validation
python -m streamlit run streamlit_orbit_app.py
```

## Features

### 🎯 Interactive Controls
- **NORAD ID Dropdown**: Select from 9 pre-configured satellites
  - 41240, 44804, 43476, 44857, 60309, 59386, 37839, 41790, 39634
  
- **Time Range Dropdown**: Choose propagation duration
  - 1 day (24 hours)
  - 2 days (48 hours)
  - 7 days (168 hours)

- **Advanced Settings**: Adjust drag coefficient (C_d)

### 📊 Visualizations

The app displays **TWO side-by-side 3D Earth-centered orbit plots**:

1. **Left Plot**: SGP4 vs Custom + ML SRP
   - Green: Reference TLE orbit (SGP4)
   - Red: ML-predicted SRP orbit
   - Shows ML performance

2. **Right Plot**: SGP4 vs Custom + Cannonball SRP
   - Green: Reference TLE orbit (SGP4)
   - Orange: Cannonball model orbit
   - Shows baseline physics model

### 📈 Metrics Displayed

- **ML SRP Error**: Position error in km
- **Cannonball Error**: Position error in km
- **Improvement**: Percentage and absolute improvement
- **Winner**: Which method performs better

### 🌍 3D Interactive Features

- Rotate: Click and drag
- Zoom: Scroll wheel
- Pan: Right-click and drag
- Reset: Double-click
- Hover: See coordinates and details

## How It Works

1. **Select Parameters**: Choose NORAD ID and time range
2. **Click Run**: Press "🚀 Run Validation" button
3. **Wait**: App runs preprocessing, propagation, and ML prediction
4. **View Results**: Two 3D plots appear side-by-side
5. **Compare**: See which method (ML or Cannonball) is more accurate

## Expected Output

- ✅ Success message with TLE count
- 📊 Four metrics cards showing errors and winner
- 🎉 Green success box if ML is better
- ⚠️ Yellow warning if Cannonball is better
- 🌍 Two interactive 3D orbit visualizations
- 📊 Detailed position comparison table (expandable)

## Troubleshooting

### Port Already in Use
```bash
streamlit run streamlit_orbit_app.py --server.port 8502
```

### Cache Issues
Clear cache from sidebar or use:
```bash
streamlit cache clear
```

### Import Errors
Make sure you're in the validation folder and all dependencies are installed:
```bash
pip install streamlit plotly numpy pandas scipy skyfield sgp4
```

## Notes

- First run will be slower (preprocessing + ML prediction)
- Results are cached - subsequent runs with same parameters are instant
- Interactive plots work best in modern browsers (Chrome, Firefox, Edge)
- You can run multiple validations without restarting the app

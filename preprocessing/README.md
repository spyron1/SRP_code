# Preprocessing Module

**SRP (Solar Radiation Pressure) Data Preprocessing Pipeline**

Author: Divyanshu Panday  
Date: October 2025

---

## Quick Start

### Run Complete Pipeline

```python
from preprocessing.data_selector import DataSelector
from preprocessing.position_calculator import add_skyfield_position_columns
from preprocessing.shadow_calculator import add_shadow_factor_column
from preprocessing.srp_features import SRPFeatureCalculator
from preprocessing.srp_acceleration import add_cannonball_srp_acceleration

# Step 1: Load data with satellite parameters
norad_id = 44804  # RISAT
selector = DataSelector(norad_id)
df = selector.prepare_dataset()

# Step 2: Add satellite & Sun positions
df = add_skyfield_position_columns(df)

# Step 3: Add shadow/eclipse factor
df = add_shadow_factor_column(df)

# Step 4: Add SRP features (unit vectors, beta angle, dynamic CR)
calc = SRPFeatureCalculator()
df = calc.add_all_features(df)

# Step 5: Calculate SRP acceleration (3D components)
df = add_cannonball_srp_acceleration(df)

# Save output
df.to_excel(f'NORAD_{norad_id}.xlsx', index=False)
```

---

## Module Structure

```
preprocessing/
├── satellite_constants.py    # Satellite parameters (mass, area, CR, altitude)
├── data_selector.py          # Load data + add metadata
├── position_calculator.py    # Satellite & Sun positions (Skyfield)
├── shadow_calculator.py      # Eclipse detection (umbra/penumbra)
├── srp_features.py          # SRP geometric features
├── srp_acceleration.py      # 3D SRP acceleration calculation
└── README.md                # This file
```

---

## What Each Module Does

### 1. satellite_constants.py
**Purpose:** Store satellite parameters by NORAD ID  
**Function:** `get_params(norad_id)` → Returns dict with mass_kg, surface_area_m2, orbit_alt_km, cr

### 2. data_selector.py
**Purpose:** Load TLE data and add satellite metadata  
**Input:** `../data_acquisition/raw_data/NORAD_{id}.xlsx`  
**Output Columns:** TLE columns + ORBIT_ALT_KM, CR, A_OVER_M  
**Usage:** `DataSelector(norad_id).prepare_dataset()`

### 3. position_calculator.py
**Purpose:** Calculate precise satellite & Sun positions  
**Function:** `add_skyfield_position_columns(df)`  
**Output Columns:** sat_x/y/z_km, sat_vx/vy/vz_kmps, sun_x/y/z_km  
**Method:** Skyfield + NASA JPL DE421 ephemeris + ICRF frame

### 4. shadow_calculator.py
**Purpose:** Calculate eclipse conditions (shadow factor)  
**Function:** `add_shadow_factor_column(df)`  
**Output Columns:** shadow_factor (0.0=umbra, 1.0=full sunlight)  
**Method:** Advanced cone model with oblate Earth, atmospheric expansion, limb darkening

### 5. srp_features.py
**Purpose:** Calculate SRP geometric features  
**Class Methods:**
- `add_au_over_r_column()` → AU²/R² solar flux scaling
- `add_unit_vector_column()` → sun_to_sat_ux/uy/uz
- `add_beta_angle()` → beta_angle_rad (orbit-Sun angle)
- `add_dynamic_cr()` → CR_eff (dynamic coefficient)
- `add_all_features()` → All of the above

### 6. srp_acceleration.py
**Purpose:** Calculate cannonball SRP acceleration in 3D  
**Function:** `add_cannonball_srp_acceleration(df)`  
**Output Columns:** srp_ax_mps2, srp_ay_mps2, srp_az_mps2, srp_a_mag_mps2  
**Formula:** a_srp = ν × P₀ × CR × (A/m) × (AU/R)² × û

---

## Data Flow Pipeline

```
../data_acquisition/raw_data/NORAD_{id}.xlsx
    ↓
[DataSelector] → Load TLE + add ORBIT_ALT_KM, CR, A_OVER_M
    ↓
[add_skyfield_position_columns] → + sat_x/y/z_km, sat_vx/vy/vz_kmps, sun_x/y/z_km
    ↓
[add_shadow_factor_column] → + shadow_factor
    ↓
[SRPFeatureCalculator.add_all_features] → + AU2_over_R2, sun_to_sat_ux/uy/uz, beta_angle_rad, CR_eff
    ↓
[add_cannonball_srp_acceleration] → + srp_ax_mps2, srp_ay_mps2, srp_az_mps2, srp_a_mag_mps2
    ↓
NORAD_{id}.xlsx (final output)
```

---

## Adding New Satellites

Edit `satellite_constants.py`:

```python
SATELLITE_PARAMETERS = {
    44804: {  # RISAT
        'mass_kg': 1858.0,
        'surface_area_m2': 10.5,
        'orbit_alt_km': 536.0,
        'cr': 1.3
    },
    12345: {  # Your new satellite
        'mass_kg': 500.0,
        'surface_area_m2': 10.0,
        'orbit_alt_km': 400.0,
        'cr': 1.2
    }
}
```

Then just use: `DataSelector(12345)`

---
---

## Output Columns Reference

| Column | Module | Description |
|--------|--------|-------------|
| **TLE Columns** | DataSelector | NORAD_CAT_ID, EPOCH, TLE_LINE1, TLE_LINE2, etc. |
| ORBIT_ALT_KM | DataSelector | Orbital altitude from constants |
| CR | DataSelector | Base radiation coefficient |
| A_OVER_M | DataSelector | Area/mass ratio (m²/kg) |
| sat_x/y/z_km | PositionCalculator | Satellite position (ICRF) |
| sat_vx/vy/vz_kmps | PositionCalculator | Satellite velocity (ICRF) |
| sun_x/y/z_km | PositionCalculator | Sun position (ICRF) |
| shadow_factor | ShadowCalculator | Eclipse: 0.0=umbra, 1.0=sunlit |
| AU2_over_R2 | SRPFeatureCalculator | (AU/R)² solar flux scaling |
| r_sun_sat_km | SRPFeatureCalculator | Sun-satellite distance |
| sun_to_sat_ux/uy/uz | SRPFeatureCalculator | Sun→sat unit vector |
| beta_angle_rad | SRPFeatureCalculator | Orbit-Sun angle (radians) |
| CR_eff | SRPFeatureCalculator | Dynamic CR coefficient |
| srp_ax/ay/az_mps2 | SRPAcceleration | SRP accel components (m/s²) |
| srp_a_mag_mps2 | SRPAcceleration | SRP accel magnitude (m/s²) |

---

## Key Features

### Shadow Calculation (Advanced)
- ✅ Oblate Earth model (flattened at poles)
- ✅ Atmospheric expansion (~50-80 km)
- ✅ Solar limb darkening (non-uniform Sun disk)
- ✅ Umbra/penumbra/antumbra detection

### SRP Acceleration (Cannonball Model)
- ✅ Formula: a_srp = ν × P₀ × CR × (A/m) × (AU/R)² × û
- ✅ P₀ = 4.56e-6 N/m² (solar pressure at 1 AU)
- ✅ Dynamic CR based on beta angle
- ✅ Full 3D components (x, y, z)

---

## Dependencies

```bash
pip install pandas numpy skyfield sgp4 openpyxl
```

---

## File Naming Convention

- **Input:** `../data_acquisition/raw_data/NORAD_{id}.xlsx`
- **Output:** `NORAD_{id}.xlsx` (current directory)

---

## Testing Modules

Each module can be tested independently:

```bash
cd preprocessing

# Test each module
python data_selector.py
python position_calculator.py
python shadow_calculator.py
python srp_features.py
python srp_acceleration.py  # Full pipeline test
```

---

## Notes

- **Coordinate Frame:** ICRF (International Celestial Reference Frame)
- **Ephemeris:** NASA JPL DE421 (auto-downloaded ~17MB on first run)
- **Architecture:** Simple functions, minimal classes (stateless where possible)
- **Time System:** UTC via Skyfield's Time objects


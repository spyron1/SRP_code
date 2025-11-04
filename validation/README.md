# Validation Module

This module implements the **SRP Model Validation Strategy** for comparing Machine Learning SRP predictions against the analytical Cannonball model.

---

## 📁 File Structure

```
validation/
├── README.md                    # This file
├── tle_collector.py             # Fetch TLE data (24hr, 2days, 7days)
├── preprocessing_runner.py      # Run preprocessing pipeline (Cannonball + ML SRP)
├── frame_converter.py           # Convert ICRF ↔ TEME frames
├── custom_propagator.py         # Numerical orbit propagator (J2 + Drag + SRP)
├── validation_workflow.py       # Complete validation workflow
└── srp_comparison.py            # SRP model comparison utilities
```

---

## 🎯 Validation Strategy

### Objective
Compare **3 orbit propagation methods** to determine which SRP model is most accurate:

| Method | Description |
|--------|-------------|
| **Method A: SGP4 Baseline** | Standard SGP4 propagation (from Skyfield/TLE) |
| **Method B: Custom + Cannonball SRP** | Custom propagator with analytical Cannonball SRP |
| **Method C: Custom + ML SRP** | Custom propagator with ML-predicted SRP |

### Validation Method: TLE Residuals

```
1. Fetch TLE data for time range (24hr, 2days, or 7days)
2. Extract initial conditions (r₀, v₀) from Day 1 TLE
3. Propagate orbit forward using all 3 methods
4. Compare to ground truth TLE at target time
5. Calculate position errors: ||Position_predicted - Position_truth||
6. Winner: Method with smallest RMSE error
```

**Key Point:** Day 1 TLE provides **initial conditions**, later TLEs provide **ground truth** (fitted to NEW observations). This is NOT circular logic! ✅

---

## 🚀 Quick Start

### 1. Run Complete Validation Workflow

```bash
cd validation
python validation_workflow.py
```

This will:
- Fetch latest TLE data (24hr by default)
- Run preprocessing (Cannonball + ML SRP)
- Propagate using 3 methods
- Compare to ground truth
- Display results

### 2. Run Preprocessing Only

```bash
python preprocessing_runner.py
```

### 3. Test Frame Conversions

```bash
python frame_converter.py
```

## 🚀 Quick Start

### 1️⃣ **Collect Validation TLE Data**

```python
from validation.tle_collector import ValidationTLECollector

# Initialize with Space-Track credentials
collector = ValidationTLECollector('your_username', 'your_password')

# Collect TLE for specific satellite + date
output_file = collector.collect_tle_for_date(
    norad_id=58888,
    start_date='2024-03-15',
    end_date='2024-03-16'
)
# Output: validation_data/NORAD_58888_validation_2024-03-15_to_2024-03-16.xlsx
```

---

### 2️⃣ **Run Preprocessing Pipeline**

```python
from validation.preprocessing_runner import ValidationPreprocessor

# Initialize preprocessor
preprocessor = ValidationPreprocessor(norad_id=58888)

# Run full pipeline (positions → shadow → features → Physics SRP)
output_file = preprocessor.run_preprocessing(
    input_file='validation_data/NORAD_58888_validation_2024-03-15_to_2024-03-16.xlsx'
)
# Output: validation_data/NORAD_58888_physics_srp.xlsx
```

**Pipeline Steps:**
1. **Data Selection** - Select TLE columns + add metadata
2. **Position Calculation** - Calculate satellite & Sun positions (Skyfield + DE421)
3. **Shadow Calculation** - Detect eclipses (cylindrical + conical shadow)
4. **SRP Features** - Add geometric features (distance, zenith angle, CR_eff, etc.)
5. **Physics SRP** - Calculate Cannonball model SRP acceleration (ax, ay, az)

---

### 3️⃣ **Compare Physics vs ML SRP**

```python
import pandas as pd
from validation.srp_comparison import SRPComparator

# Load preprocessed data
df = pd.read_excel('validation_data/NORAD_58888_physics_srp.xlsx')

# Initialize comparator (loads trained ML models)
comparator = SRPComparator()

# Predict ML SRP
df = comparator.predict_ml_srp(df)

# Compare Physics vs ML
metrics = comparator.compare_results(df)
# Output: validation_data/srp_comparison_results.xlsx
```

**Comparison Metrics:**
- **MAE** (Mean Absolute Error) - Average absolute difference
- **RMSE** (Root Mean Squared Error) - RMS difference
- **Residuals** - Physics - ML (for each axis: ax, ay, az)

---

## 📊 Output Files

| File | Description |
|------|-------------|
| `NORAD_{id}_validation_{dates}.xlsx` | Raw TLE data for validation |
| `NORAD_{id}_physics_srp.xlsx` | Preprocessed data with Physics SRP |
| `srp_comparison_results.xlsx` | Physics vs ML comparison with residuals |

---

## 📐 Feature Columns Used

**Input Features for ML Model:**
- `sun_sat_distance_km` - Distance from Sun to satellite
- `zenith_angle_deg` - Solar zenith angle
- `shadow_factor` - Eclipse factor (0 = full shadow, 1 = full sunlight)
- `cos_sun_angle` - Cosine of Sun angle
- `CR_eff` - Effective coefficient of reflectivity
- `SRP_force_N` - SRP force magnitude
- `BSTAR`, `ECCENTRICITY`, `INCLINATION`, `MEAN_MOTION` - TLE parameters

**Output Predictions:**
- `ax`, `ay`, `az` - Physics-based SRP acceleration (Cannonball model)
- `ax_ml`, `ay_ml`, `az_ml` - ML-predicted SRP acceleration

**Residuals:**
- `residual_ax` = `ax` - `ax_ml`
- `residual_ay` = `ay` - `ay_ml`
- `residual_az` = `az` - `az_ml`

---

## 🔧 Requirements

- Python 3.8+
- pandas >= 2.0.0
- numpy >= 1.24.0
- scikit-learn >= 1.3.0
- joblib >= 1.3.0
- skyfield >= 1.46
- sgp4 >= 2.23

---

## 🎯 Typical Use Case

**Goal:** Validate ML model accuracy on unseen data (specific date range)

**Steps:**
1. Collect TLE for target satellite + date → `tle_collector.py`
2. Run preprocessing pipeline → `preprocessing_runner.py`
3. Compare Physics vs ML SRP → `srp_comparison.py`
4. Analyze residuals to assess ML model performance

---

## ⚠️ Notes

- Ensure satellite exists in `satellite_constants.py` before running
- Space-Track credentials required for TLE collection
- Trained ML models must exist in `trained_models/` folder
- Output files saved to `validation_data/` folder (created automatically)

---

## 📖 Example Full Workflow

```python
# Step 1: Collect TLE
from validation.tle_collector import ValidationTLECollector
collector = ValidationTLECollector('username', 'password')
tle_file = collector.collect_tle_for_date(58888, '2024-03-15')

# Step 2: Preprocess
from validation.preprocessing_runner import ValidationPreprocessor
preprocessor = ValidationPreprocessor(58888)
physics_file = preprocessor.run_preprocessing(tle_file)

# Step 3: Compare
import pandas as pd
from validation.srp_comparison import SRPComparator
df = pd.read_excel(physics_file)
comparator = SRPComparator()
df = comparator.predict_ml_srp(df)
metrics = comparator.compare_results(df)

print("Validation complete! Check validation_data/ for results.")
```

---

## 📚 Next Steps

After validation:
1. **Custom Orbit Propagator** - Integrate SRP into orbit propagation
2. **TLE Residual Analysis** - Compare propagated vs actual TLE positions
3. **Error Visualization** - Plot residuals over time
4. **Model Refinement** - Retrain based on validation insights

---

**Author:** Divyanshu Panday  
**Date:** October 2025  
**Version:** 1.0.0

# Satellite SRP Hybrid Modeling Project

Simple end‑to‑end pipeline to:
1. Download TLE data
2. Compute geometric & physical SRP features
3. Generate Cannonball and ML SRP accelerations
4. Run orbit propagation validation (SGP4 vs Custom + SRP vs Custom + ML SRP)

## 1. Install

```bash
git clone <your-repo-url>


```

(Optional) create virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate      # Linux/macOS
.\.venv\Scripts\activate       # Windows
pip install -r requirements.txt
cd  validation
streamlit run streamlit_orbit_app.py
```

## 2. Space-Track Credentials

Create `.env` in project root:
```env
SPACETRACK_USERNAME=your_username
SPACETRACK_PASSWORD=your_password
```

## 3. Directory Overview

```
data_acquisition/        # TLE download client
preprocessing/           # Position + shadow + SRP features + SRP acceleration
model/                   # ML training artifacts
validation/              # Comparison workflow (SGP4 vs Cannonball vs ML SRP)
SRP_ML_Guidelines.md     # Feature list for ML
SRP_Validation_Strategy.md
```

Reference docs:
- [data_acquisition/README.md](data_acquisition/README.md)
- [preprocessing/README.md](preprocessing/README.md)
- [validation/README.md](validation/README.md)
- [SRP_ML_Guidelines.md](SRP_ML_Guidelines.md)
- [SRP_Validation_Strategy.md](SRP_Validation_Strategy.md)

## 4. Quick Start

### (A) Download TLE Data
```python
from data_acquisition import SpaceTrackClient
client = SpaceTrackClient()
df_raw = client.save_to_excel(norad_cat_id=43476, start_epoch="2024-01-01", end_epoch="2024-01-08")
```

### (B) Run Preprocessing (Positions + Features + SRP)
```python
from preprocessing import DataSelector, add_skyfield_position_columns, add_shadow_factor_column
from preprocessing import SRPFeatureCalculator, add_cannonball_srp_acceleration

selector = DataSelector(43476)
df = selector.prepare_dataset()
df = add_skyfield_position_columns(df)
df = add_shadow_factor_column(df)
calc = SRPFeatureCalculator()
df = calc.add_all_features(df)             # adds unit vectors, AU²/R², beta angle, dynamic CR
df = add_cannonball_srp_acceleration(df)   # adds srp_ax_mps2, srp_ay_mps2, srp_az_mps2
df.to_excel("NORAD_43476.xlsx", index=False)
```

### (C) (Optional) ML SRP Prediction
Model expects features defined in [SRP_ML_Guidelines.md](SRP_ML_Guidelines.md).
```python
# Placeholder – integrate your trained model placed in model/trained_models/
# Produces ml_ax_mps2, ml_ay_mps2, ml_az_mps2
```

### (D) Run Validation Workflow
```python
from validation.validation_workflow import ValidationWorkflow
workflow = ValidationWorkflow()
results = workflow.run_validation(norad_id=43476, time_range="24hr")
print(results)
```

## 5. Core Components

- Data acquisition client: [data_acquisition/spacetrack_client.py](data_acquisition/spacetrack_client.py)
- Feature calculator class: [`preprocessing.SRPFeatureCalculator`](preprocessing/srp_features.py)
- Cannonball SRP acceleration function: [`preprocessing.add_cannonball_srp_acceleration`](preprocessing/srp_acceleration.py)
- Custom propagator (numerical): [validation/custom_propagator.py](validation/custom_propagator.py)
- Simplified SRP usage pattern: [validation/SIMPLIFIED_PROPAGATOR.md](validation/SIMPLIFIED_PROPAGATOR.md)

## 6. Validation Methods

Compares three methods (see [validation/README.md](validation/README.md)):
1. SGP4 baseline
2. Custom propagator + Cannonball SRP (preprocessed)
3. Custom propagator + ML SRP

Uses TLE residual approach described in [SRP_Validation_Strategy.md](SRP_Validation_Strategy.md).

## 7. Reproducibility Notes

- Ephemeris file: `de421.bsp` (JPL)
- Frame: ICRF
- Units: km, km/s, m/s² (SRP accelerations converted internally where needed)
- Shadow factor column enables eclipse handling (0 = shadow, 1 = illuminated)

## 8. Typical Minimal Script

```python
from preprocessing import DataSelector, add_skyfield_position_columns, add_shadow_factor_column
from preprocessing import SRPFeatureCalculator, add_cannonball_srp_acceleration
from validation.validation_workflow import ValidationWorkflow

norad = 44804
df = DataSelector(norad).prepare_dataset()
df = add_skyfield_position_columns(df)
df = add_shadow_factor_column(df)
df = SRPFeatureCalculator().add_all_features(df)
df = add_cannonball_srp_acceleration(df)
df.to_excel(f"NORAD_{norad}.xlsx", index=False)

results = ValidationWorkflow().run_validation(norad_id=norad, time_range="24hr")
print(results)
```

## 9. Troubleshooting

- Missing ephemeris: ensure `de421.bsp` exists in root or preprocessing folder.
- Empty SRP columns: confirm `add_cannonball_srp_acceleration` executed after features.
- ML prediction absent: add trained model files under `model/trained_models/`.

## 10. Next Steps

- Plug in trained XGBoost / LightGBM model for SRP.
- Extend propagator with higher-order harmonics if needed.
- Add automated plotting of residuals.

## 11. License / Attribution

Academic/research use. Cite internal author (Divyanshu ) where appropriate.

---
Simplified overview only. See module READMEs for detail.

# Feature Engineering Improvements for SRP Model

To make the SRP prediction model more dynamic and accurate, we need to add features that describe the satellite's orientation and temporal context. The current features primarily describe the satellite's position, leading to overly stable predictions.

Here are the most impactful features to add, ranked by importance.

### 1. Angle Between Velocity and Sun (Most Important)

This is the single most valuable feature to add. It tells the model how "head-on" the sunlight is hitting the satellite as it moves through its orbit, providing the dynamic information it is currently missing.

*   **Feature to Add:** `sun_velocity_angle_cos`
*   **How to Calculate:**
    1.  Calculate the satellite's velocity unit vector (`v_hat`) from `sat_vx_kmps`, `sat_vy_kmps`, `sat_vz_kmps`.
    2.  You already have the sun-to-satellite unit vector (`u_hat`), which is `sun_to_sat_ux`, `sun_to_sat_uy`, `sun_to_sat_uz`.
    3.  The new feature is the **dot product** of these two vectors: `v_hat · u_hat`. The result is the cosine of the angle between them.

### 2. Time-Based Features

SRP can have cyclical patterns related to the time of day or year. Encoding time as cyclical features helps the model understand these patterns (e.g., that hour 23 is close to hour 0).

*   **Features to Add:** `day_of_year_sin`, `day_of_year_cos`, `hour_of_day_sin`, `hour_of_day_cos`
*   **How to Calculate:**
    1.  Extract `day_of_year` and `hour_of_day` from the `EPOCH` column.
    2.  Apply a sine/cosine transformation to represent the cyclical nature:
        *   `hour_sin = sin(2 * pi * hour_of_day / 24)`
        *   `hour_cos = cos(2 * pi * hour_of_day / 24)`
        *   `day_sin = sin(2 * pi * day_of_year / 365.25)`
        *   `day_cos = cos(2 * pi * day_of_year / 365.25)`

### 3. Angle Between Position and Sun

This feature helps the model understand where the satellite is in its orbit relative to the sunlit side of the Earth.

*   **Feature to Add:** `sun_position_angle_cos`
*   **How to Calculate:**
    1.  Calculate the satellite's position unit vector (`r_hat`) from `sat_x_km`, `sat_y_km`, `sat_z_km`.
    2.  Calculate the dot product with the sun-to-satellite unit vector (`u_hat`): `r_hat · u_hat`.

### Updated Feature List for Model Training

After implementing these changes, your new `feature_cols` list in `model_trainer.py` should be updated to include these new features:

```python
feature_cols = [
    "NORAD_CAT_ID",
    'A_OVER_M',
    'CR_eff',
    'AU2_over_R2',
    'shadow_factor',
    'sun_to_sat_ux',
    'sun_to_sat_uy',
    'sun_to_sat_uz',
    # --- NEW FEATURES ---
    'sun_velocity_angle_cos', # Most important
    'day_of_year_sin',
    'day_of_year_cos',
    'hour_of_day_sin',
    'hour_of_day_cos',
    'sun_position_angle_cos'
]
```

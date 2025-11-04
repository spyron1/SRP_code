# SRP Machine Learning Guidelines

## Data Preprocessing and Feature Engineering

### Random Forest for SRP Acceleration Prediction

For SRP acceleration prediction with Random Forest, **standardization/normalization is generally NOT required**.

#### Why Standardization is NOT Needed for Random Forest:
- **Tree-based models (Random Forest, Decision Tree, XGBoost) are scale-invariant**
- They make decisions based on feature splits, not distances
- Very small SRP values (10^-7 to 10^-8) won't hurt Random Forest performance
- The model will find optimal split points regardless of scale

### When Standardization/Normalization IS Required

#### Standardization (StandardScaler) needed for:
- Linear models (Linear Regression, Ridge, Lasso)
- Neural Networks
- SVM
- K-Nearest Neighbors
- Any distance-based algorithms

#### Normalization (MinMaxScaler) needed for:
- When you need bounded values [0,1]
- Neural networks with specific activation functions
- When features have very different scales AND you use distance-based methods

### Decision Matrix: When to Use Which Scaling

#### Use Standardization when:
- Features have different units (km, m/s, radians, etc.)
- Using distance-based or gradient-based algorithms
- Features have very different variances

#### Use Normalization when:
- You need bounded output [0,1]
- Neural networks with sigmoid/tanh activations
- When interpretability of scaled values matters

## Feature Selection for SRP Modeling

### Required Features for XGBoost SRP Prediction

```python
feature_cols = [
    "A_OVER_M",        # Area-to-mass ratio
    "Cr_dynamic",      # Dynamic reflection coefficient
    "AU2_over_R2",     # Solar distance scaling factor
    "shadow_factor",   # Eclipse/shadow factor (0 or 1)
    "sun_to_sat_ux",   # Unit vector components (Sun to satellite)
    "sun_to_sat_uy",
    "sun_to_sat_uz"
]

# Output labels
label_cols = ["srp_ax_mps2", "srp_ay_mps2", "srp_az_mps2"]
```

### Why Satellite Velocity is NOT Required

For standard cannonball model of SRP:
- **Velocity is not a direct input** - SRP force is caused by photons impacting the satellite
- **Instantaneous Force**: Model predicts SRP acceleration at a specific instant, depending on satellite *position* relative to sun, not velocity
- **No "SRP Drag"**: Unlike atmospheric drag, SRP is not velocity-dependent
- Including velocity could confuse the model with unnecessary features

## Model Training Guidelines

### For Tree-Based Models (Recommended Approach)

```python
# Current approach (Random Forest) - NO scaling needed
models = {
    "Random Forest": RandomForestRegressor(),
    "XGBoost": XGBRegressor(),
    "Decision Tree": DecisionTreeRegressor()
}

# Train directly on raw features
model.fit(X_train, y_train)
```

### If Using Linear Models (Alternative Approach)

```python
# If you switch to linear models - ADD scaling
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Then train linear models
linear_model.fit(X_train_scaled, y_train)
```

## Dataset Considerations

### Ideal Dataset Composition
- **4,000 rows** with shadow_factor = 1 (illuminated)
- **2,000 rows** with shadow_factor = 0 (eclipse)
- **Total: 6,000+ data points** - Perfect for XGBoost regression

### Train/Test Split Strategy

```python
from sklearn.model_selection import train_test_split

# Stratified split by shadow_factor to maintain illumination balance
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, 
    stratify=df['shadow_factor']
)
```

## Performance Validation

### Evaluation Metrics
- **RMSE**: Root Mean Square Error for overall accuracy
- **MAE**: Mean Absolute Error for average prediction error  
- **R²**: Coefficient of determination for explained variance

### Separate Validation by Shadow State
- **In shadow (shadow_factor=0)**: Target should be near zero, check MAE is very small
- **In light (shadow_factor=1)**: Check vector RMSE/MAE and direction alignment

## Physical Constants and Parameters

### Solar Radiation Pressure Constant
- **P₀ = 4.56×10⁻⁶ N/m²** (at 1 AU)
- **Not needed as ML feature** - it's a universal constant
- Effect already "baked in" to labels and handled by AU2_over_R2 feature

### AU2_over_R2 Explanation
In cannonball model: `AU2_over_R2 = (AU/R)²`
- **Purpose**: Account for variation in solar flux with distance from Sun
- **Why needed**: Solar radiation pressure decreases with square of distance
- **Formula**: `P_SRP = P₀ × (AU/R)²`
- Essential for satellites not exactly at 1 AU from Sun

### CR Values by Orbit Type

| Orbit Type | CR Range | Notes |
|------------|----------|-------|
| **LEO** | 1.0 - 1.5 | Lower due to atmospheric drag dominance |
| **MEO** | 1.2 - 1.8 | GPS satellites typically CR ≈ 1.3-1.5 |
| **GEO** | 1.5 - 2.5 | Higher due to significant SRP effects |
| **HEO** | 1.3 - 2.2 | Variable depending on satellite design |

**General Guidelines:**
- Newer satellites: CR = 1.0 - 1.3 (better materials)
- Older satellites: CR = 1.5 - 2.5 (degraded surfaces)
- Solar panel-heavy satellites: Higher CR values

## Implementation Best Practices

### GPU Acceleration
- **XGBoost**: Set `tree_method='gpu_hist'` for GPU usage
- **Random Forest/Decision Tree**: CPU only (scikit-learn doesn't support GPU)
- **Alternative**: Use RAPIDS cuML for GPU Random Forest (different API)

### Code Example for GPU XGBoost

```python
xgboost_params = {
    "learning_rate": [0.1, 0.01],
    "max_depth": [5, 8, 12, 20, 30],
    "n_estimators": [100, 200, 300],
    "colsample_bytree": [0.5, 0.8, 1, 0.3, 0.4],
    "tree_method": ["gpu_hist"]  # Enable GPU
}
```

## Summary

### Key Takeaways
1. **Random Forest works well without feature scaling** for SRP prediction
2. **Use 7 physics-based features** + shadow factor for optimal results
3. **Velocity not required** for instantaneous SRP prediction
4. **Stratified splitting by shadow_factor** ensures balanced train/test sets
5. **Tree-based models preferred** over linear models for this application
6. **GPU acceleration available** only for XGBoost, not Random Forest

### Bottom Line
Your current approach without standardization is **perfectly fine** for Random Forest. Only add scaling if switching to linear/distance-based models.
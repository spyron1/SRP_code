"""
SRP Model Training Module

Trains Random Forest model to predict SRP acceleration (3D components).
Uses combined training data from feature_engineering folder.

Author: Divyanshu Panday
Date: October 2025
"""

import os
import pandas as pd
import numpy as np
import warnings
from sklearn.model_selection import train_test_split
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

warnings.filterwarnings("ignore")


def evaluate_model(y_true, y_pred):
    """
    Calculate regression metrics.
    
    Parameters
    ----------
    y_true : array-like
        True values
    y_pred : array-like
        Predicted values
        
    Returns
    -------
    tuple
        (mae, rmse, r2)
    """
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    return mae, rmse, r2


def train_lightgbm(
    data_file: str = "../feature_engineering/training_data_combined.xlsx",
    test_size: float = 0.3,
    random_state: int = 42,
    save_model: bool = True,
    hyperparameters: dict = None
):
    """
    Train Random Forest model for SRP acceleration prediction.
    
    Parameters
    ----------
    data_file : str
        Path to combined training data
    test_size : float
        Test set ratio (default: 0.2 = 20%)
    random_state : int
        Random seed for reproducibility
    save_model : bool
        Whether to save trained models
    hyperparameters : dict, optional
        A dictionary containing specific hyperparameters for each target.
        
    Returns
    -------
    dict
        Trained models for each target (ax, ay, az)
    """
    
    print("\n" + "="*70)
    print(" SRP MODEL TRAINING - LIGHTGBM ".center(70, "="))
    print("="*70)
    
    # Load data
    print(f"\nLoading data from: {data_file}")
    df = pd.read_excel(data_file)
    print(f"✓ Loaded {len(df)} rows, {df.shape[1]} columns")
    
    # Prepare features and targets
    print("\nPreparing features and targets...")
    target_cols = ["srp_ax_mps2", "srp_ay_mps2", "srp_az_mps2"]
    
    # Define the feature columns for the model
    feature_cols = [
        "NORAD_CAT_ID",
        'A_OVER_M',
        'CR_eff',
        'AU2_over_R2',
        'shadow_factor',
        'sun_to_sat_ux',
        'sun_to_sat_uy',
        'sun_to_sat_uz',
        # New dynamic & temporal features
        'sun_velocity_angle_cos',
        'sun_position_angle_cos',
        'day_of_year_sin',
        'day_of_year_cos',
        'hour_of_day_sin',
        'hour_of_day_cos'
    ]
    
    X = df[feature_cols]
    y = df[target_cols]

    # Identify and convert categorical features
    categorical_features = ["NORAD_CAT_ID"]
    X[categorical_features] = X[categorical_features].astype('category')
    
    print(f"Features: {len(feature_cols)} columns")
    print(f"  Using: {feature_cols}")
    print(f"  Categorical Features: {categorical_features}")
    print(f"Targets: {len(target_cols)} outputs (ax, ay, az)")
    
    # Train/test split
    print(f"\nSplitting data (test_size={test_size})...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    print(f"✓ Train set: {len(X_train)} rows")
    print(f"✓ Test set:  {len(X_test)} rows")
    
    # Train models for each target
    models = {}
    results = {}
    
    print("\n" + "="*70)
    print(" TRAINING MODELS ".center(70, "="))
    print("="*70)
    
    for target_col in target_cols:
        print(f"\n{'─'*70}")
        print(f" Training for: {target_col} ".center(70, "─"))
        print(f"{'─'*70}")
        
        y_train_target = y_train[target_col]
        y_test_target = y_test[target_col]
        
        # Get the specific params for this target, or use an empty dict for defaults
        model_params = (hyperparameters or {}).get(target_col, {})
        print(f"Using hyperparameters: {model_params or 'default'}")
        
        # Create and train model
        print("Training LightGBM...")
        model = lgb.LGBMRegressor(
            random_state=random_state,
            **model_params  # Unpack the hyperparameter dictionary
        )
        model.fit(X_train, y_train_target, categorical_feature=categorical_features)
        print("✓ Training complete")
        
        # Predictions
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
        
        # Evaluate
        train_mae, train_rmse, train_r2 = evaluate_model(y_train_target, y_train_pred)
        test_mae, test_rmse, test_r2 = evaluate_model(y_test_target, y_test_pred)
        
        # Store results
        models[target_col] = model
        results[target_col] = {
            'train': {'mae': train_mae, 'rmse': train_rmse, 'r2': train_r2},
            'test': {'mae': test_mae, 'rmse': test_rmse, 'r2': test_r2}
        }
        
        # Print results
        print(f"\n📊 Performance Metrics:")
        print(f"\n  Training Set:")
        print(f"    MAE:  {train_mae:.10f}")
        print(f"    RMSE: {train_rmse:.10f}")
        print(f"    R²:   {train_r2:.10f}")
        
        print(f"\n  Test Set:")
        print(f"    MAE:  {test_mae:.10f}")
        print(f"    RMSE: {test_rmse:.10f}")
        print(f"    R²:   {test_r2:.10f}")
    
    # Save models
    if save_model:
        print("\n" + "="*70)
        print(" SAVING MODELS ".center(70, "="))
        print("="*70)
        
        model_dir = "trained_models"
        os.makedirs(model_dir, exist_ok=True)
        
        for target_col, model in models.items():
            model_file = os.path.join(model_dir, f"lgb_{target_col}.pkl")
            joblib.dump(model, model_file)
            print(f"✓ Saved: {model_file}")
    
    # Summary
    print("\n" + "="*70)
    print(" TRAINING SUMMARY ".center(70, "="))
    print("="*70)
    
    print(f"\n{'Target':<15} {'Train R²':<12} {'Test R²':<12} {'Test MAE':<15}")
    print("─" * 70)
    for target_col in target_cols:
        train_r2 = results[target_col]['train']['r2']
        test_r2 = results[target_col]['test']['r2']
        test_mae = results[target_col]['test']['mae']
        print(f"{target_col:<15} {train_r2:<12.6f} {test_r2:<12.6f} {test_mae:<15.10f}")
    
    print("\n" + "="*10)
    print("✓ Model training complete!")
    print("="*10 + "\n")
    
    return models, results


if __name__ == "__main__":
    """Train LightGBM model."""
    
    # Example hyperparameters for LGBMRegressor
    tuned_params = {
        # 'srp_ax_mps2': {
        #     'n_estimators': 400,
        #     'learning_rate': 0.05,
        #     'num_leaves': 31,
        # },
        # 'srp_ay_mps2': {
        #     'n_estimators': 400,
        #     'learning_rate': 0.05,
        #     'num_leaves': 31,
        # },
        # 'srp_az_mps2': {
        #     'n_estimators': 400,
        #     'learning_rate': 0.05,
        #     'num_leaves': 31,
        # }
        
        'srp_ax_mps2': {
            'n_estimators': 778,
            'num_leaves': 37,
            'learning_rate': 0.018303631950642548,
            'max_depth': 11,
            'min_data_in_leaf': 69,
            'feature_fraction': 0.9986531777035408,
            'bagging_fraction': 0.8574119097776769,
            'bagging_freq': 4,
            'lambda_l1': 1.5058169020871542e-09,
            'lambda_l2': 0.01630692236861022
        },
        'srp_ay_mps2': {
            'n_estimators': 977,
            'num_leaves': 50,
            'learning_rate': 0.029626981002240736,
            'max_depth': 7,
            'min_data_in_leaf': 55,
            'feature_fraction': 0.9792106351964174,
            'bagging_fraction': 0.9823163922255465,
            'bagging_freq': 1,
            'lambda_l1': 1.073934199755052e-09,
            'lambda_l2': 3.374703604061814e-07
        },
        'srp_az_mps2': {
            'n_estimators': 391,
            'num_leaves': 55,
            'learning_rate': 0.03482361164415041,
            'max_depth': 11,
            'min_data_in_leaf': 53,
            'feature_fraction': 0.9593848587695054,
            'bagging_fraction': 0.6711681972475381,
            'bagging_freq': 0,
            'lambda_l1': 4.26394757201816e-09,
            'lambda_l2': 4.140014558426177e-05
        }
    }
    
    # Train models using the tuned hyperparameters
    models, results = train_lightgbm(
        data_file="../feature_engineering/training_data_combined.xlsx",
        test_size=0.3,
        random_state=42,
        save_model=True,
        hyperparameters=tuned_params
    )

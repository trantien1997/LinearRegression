import os
import optuna
import pandas as pd
import numpy as np
import joblib  # Library to save the model
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Import configuration variables and data processing module
from constants import PATHS, FEATURES, TARGETS
from processor import TikTokDataProcessor

# ==================== OPTUNA OBJECTIVE FUNCTION ====================
def objective(trial, X_train, y_train, X_val, y_val):
    # Define the hyperparameter search space
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 500, 1500),
        'max_depth': trial.suggest_int('max_depth', 4, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
        'subsample': trial.suggest_float('subsample', 0.7, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.7, 1.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 2.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 5.0),
        'random_state': 42,
    }
    
    # Initialize and train the model with trial parameters
    model = XGBRegressor(**params)
    model.fit(X_train, y_train, verbose=False)
    
    # Predict on the validation set
    pred_val = model.predict(X_val)
    
    # Calculate the average R2 score across all 3 targets (Views, Likes, Shares)
    r2_avg = r2_score(y_val, pred_val)
    
    return r2_avg # Return average R2 for Optuna to maximize

# ==================== MAIN FUNCTION ====================
def main():
    print("1. Loading and processing raw data...")
    df_raw = pd.read_csv(PATHS["main_data"])

    # Feature Engineering
    proc = TikTokDataProcessor()
    proc.load_trends()
    df_featured = proc.process_features(df_raw)

    print("2. Splitting data into Train and Validation sets...")
    train_df, val_df = train_test_split(
        df_featured, test_size=0.2, random_state=42
    )

    # Fill missing values with 0 (if any)
    X_train = train_df[FEATURES].fillna(0)
    X_val   = val_df[FEATURES].fillna(0)
    y_train = train_df[TARGETS]
    y_val   = val_df[TARGETS]

    # ==================== OPTUNA TUNING ====================
    print("\n3. Starting Optuna Hyperparameter Tuning for Multi-Target...")
    study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=42))
    
    # Run 30 trials (can be increased to 50-100 for better optimization)
    study.optimize(lambda trial: objective(trial, X_train, y_train, X_val, y_val), n_trials=30)

    print(f"\nBest Average R2 Score: {study.best_value:.5f}")
    
    # ==================== TRAINING FINAL MODEL ====================
    print("\n4. Training the final model with the best parameters...")
    best_params = study.best_params
    best_params['random_state'] = 42

    final_model = XGBRegressor(**best_params)
    final_model.fit(X_train, y_train)

    # ==================== A. EXTRACT FEATURE IMPORTANCES ====================
    print("\n=== 5. Extracting Feature Importances ===")
    
    # Get feature importance array from the model
    importances = final_model.feature_importances_
    
    # Create a DataFrame and convert to percentage format
    fi_df = pd.DataFrame({
        'Feature': FEATURES,
        'Importance_%': importances * 100
    }).sort_values(by='Importance_%', ascending=False)
    
    # Display to console
    print(fi_df.to_string(index=False, float_format="%.2f%%"))
    
    # Save feature importances to a separate CSV file
    fi_path = PATHS["output_result"].replace(".csv", "_feature_importances.csv")
    fi_df.to_csv(fi_path, index=False, encoding="utf-8-sig")

    # ==================== B. CALCULATE METRICS & ERRORS ====================
    print("\n=== 6. Calculating Detailed Metrics & Errors ===")
    y_pred = final_model.predict(X_val)
    results = {}
    metrics = []

    # Calculate separately for each target variable
    for i, target in enumerate(TARGETS):
        y_actual_col = y_val.iloc[:, i]
        y_pred_col = y_pred[:, i]

        mae = mean_absolute_error(y_actual_col, y_pred_col)
        rmse = np.sqrt(mean_squared_error(y_actual_col, y_pred_col))
        r2 = r2_score(y_actual_col, y_pred_col)

        print(f"Target: {target:15s} -> MAE: {mae:.4f} | RMSE: {rmse:.4f} | R2: {r2:.4f}")

        # Record actual and predicted values
        results[f"{target}_actual"] = y_actual_col.values
        results[f"{target}_predict"] = y_pred_col
        
        # Calculate Accuracy % for each data point
        accuracy_percent = (1 - abs(y_actual_col - y_pred_col) / (y_actual_col + 1)) * 100
        results[f"{target}_accuracy_%"] = accuracy_percent.values

        # Store summarized metrics
        metrics.append({"Target": target, "MAE": mae, "RMSE": rmse, "R2": r2})

    # ==================== C. SAVE RESULTS AND MODEL ====================
    print("\n7. Exporting result files and saving the model...")
    
    # 7.1 Save detailed prediction file
    pd.DataFrame(results).to_csv(PATHS["output_result"], index=False, encoding="utf-8-sig")
    
    # 7.2 Save error metrics summary file
    metrics_path = PATHS["output_result"].replace(".csv", "_metrics_error.csv")
    pd.DataFrame(metrics).to_csv(metrics_path, index=False, encoding="utf-8-sig")

    # 7.3 Create 'models' directory and save the model
    base_dir = os.path.dirname(PATHS["output_result"]) # Get the directory path of the output result file
    models_dir = os.path.join(base_dir, "models")
    os.makedirs(models_dir, exist_ok=True) # Automatically create 'models' directory if it doesn't exist

    model_path = os.path.join(models_dir, "xgboost_multi_target_model.pkl")
    joblib.dump(final_model, model_path)
    
    print(f"Completed! Model successfully saved at:\n -> {model_path}")

if __name__ == "__main__":
    main()

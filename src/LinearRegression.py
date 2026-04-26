import pandas as pd
import numpy as np
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from constants import PATHS, FEATURES, TARGETS
from processor import TikTokDataProcessor

def main():
    # ---------------------------------------------------------
    # STEP 1: DATA LOADING & PREPROCESSING
    # ---------------------------------------------------------
    print(">>> Status: Loading raw data and processing features...")
    df_raw = pd.read_csv(PATHS["main_data"])

    proc = TikTokDataProcessor()
    proc.load_trends()
    df_featured = proc.process_features(df_raw)

    # ---------------------------------------------------------
    # STEP 2: DATA SPLITTING
    # ---------------------------------------------------------
    print(">>> Status: Splitting data into Train and Validation sets...")
    train_df, val_df = train_test_split(
        df_featured,
        test_size=0.2,
        random_state=42
    )

    # Export split data for consistency checks
    train_df.to_csv(PATHS["output_train"], index=False, encoding="utf-8-sig")
    val_df.to_csv(PATHS["output_val"], index=False, encoding="utf-8-sig")

    # ---------------------------------------------------------
    # STEP 3: FEATURE & TARGET PREPARATION
    # ---------------------------------------------------------
    X_train = train_df[FEATURES].fillna(0)
    X_val = val_df[FEATURES].fillna(0)

    # Apply log1p transformation to handle right-skewed target distribution
    print(f">>> Status: Applying Log transformation to targets: {TARGETS}")
    y_train = np.log1p(train_df[TARGETS]) 
    y_val_actual = val_df[TARGETS] # Keep original scale for evaluation

    # ---------------------------------------------------------
    # STEP 4: MULTI-TARGET MODEL TRAINING
    # ---------------------------------------------------------
    print(f"\n>>> Status: Training Multi-target Linear Regression...")
    
    # Encapsulate Scaler and Regressor into a single Pipeline
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LinearRegression())
    ])

    # Train the unified model
    pipeline.fit(X_train, y_train)

    # ---------------------------------------------------------
    # STEP 5: MODEL SERIALIZATION (SAVE)
    # ---------------------------------------------------------
    model_dir = "./models"
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "tiktok_multi_target_model.pkl")
    
    joblib.dump(pipeline, model_path)
    print(f"✅ Success: Model saved at: {model_path}")

    # ---------------------------------------------------------
    # STEP 6: EVALUATION & INVERSE TRANSFORMATION
    # ---------------------------------------------------------
    print("\n" + "="*50)
    print("MODEL EVALUATION REPORT")
    print("="*50)
    
    # Predict in Log Scale
    y_pred_log = pipeline.predict(X_val)
    
    # Transform predictions back to Original Scale (Inverse Log)
    y_pred_original = np.expm1(y_pred_log)
    y_pred_original = np.maximum(y_pred_original, 0) # Ensure no negative values

    y_pred_df = pd.DataFrame(y_pred_original, columns=TARGETS, index=val_df.index)
    evaluation_results = {}

    for target in TARGETS:
        actual = y_val_actual[target]
        predicted = y_pred_df[target]

        # Calculate standard regression metrics
        mae = mean_absolute_error(actual, predicted)
        rmse = np.sqrt(mean_squared_error(actual, predicted))
        r2 = r2_score(actual, predicted)

        # Calculate Adjusted MAPE (Mean Absolute Percentage Error)
        # We use (actual + 1) to prevent DivisionByZero errors
        mape = (abs(actual - predicted) / (actual + 1)).mean() * 100

        print(f"\nTarget Variable: [{target}]")
        print(f"  - MAE:   {mae:.4f}")
        print(f"  - RMSE:  {rmse:.4f}")
        print(f"  - R2:    {r2:.4f}")
        print(f"  - Error: {mape:.2f}%")

        # Prepare detailed results for CSV export
        evaluation_results[f"{target}_actual"] = actual.values
        evaluation_results[f"{target}_predicted"] = predicted.values
        evaluation_results[f"{target}_error_pct"] = (abs(actual - predicted) / (actual + 1)) * 100

    # ---------------------------------------------------------
    # STEP 7: EXPORT PREDICTIONS
    # ---------------------------------------------------------
    result_df = pd.DataFrame(evaluation_results)
    result_df.to_csv(PATHS["output_result"], index=False, encoding="utf-8-sig")
    print(f"\n✅ Success: Detailed predictions saved to: {PATHS['output_result']}")

if __name__ == "__main__":
    main()

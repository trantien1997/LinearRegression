import pandas as pd
import numpy as np
import joblib
import os

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
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

    train_df.to_csv(PATHS["output_train"], index=False, encoding="utf-8-sig")
    val_df.to_csv(PATHS["output_val"], index=False, encoding="utf-8-sig")

    # ---------------------------------------------------------
    # STEP 3: FEATURE & TARGET PREPARATION
    # ---------------------------------------------------------
    X_train = train_df[FEATURES].fillna(0)
    X_val = val_df[FEATURES].fillna(0)

    # Apply Log Transformation (Crucial for right-skewed data)
    print(f">>> Status: Applying Log transformation to targets: {TARGETS}")
    y_train = np.log1p(train_df[TARGETS]) 
    y_val_actual = val_df[TARGETS] 

    # ---------------------------------------------------------
    # STEP 4: MULTI-TARGET RANDOM FOREST WITH RANDOMIZED SEARCH
    # ---------------------------------------------------------
    print(f"\n>>> Status: Configuring Multi-target Random Forest Pipeline...")
    
    # Note: Random Forest doesn't strictly need StandardScaler, 
    # but keeping it in Pipeline is safe and standard practice.
    pipeline = Pipeline([
        ("rf", RandomForestRegressor(random_state=42, n_jobs=-1))
    ])

    # Smart, restricted parameter grid
    param_distributions = {
        "rf__n_estimators": [100, 300, 500],
        "rf__max_depth": [10, 20, 30, None],
        "rf__min_samples_split": [2, 5, 10],
        "rf__min_samples_leaf": [1, 2, 4],
        "rf__max_features": ["sqrt", "log2", 1.0]
    }

    # Use RandomizedSearchCV instead of GridSearchCV to prevent infinite runtimes
    print(">>> Status: Starting Randomized Hyperparameter Search (Max 50 iterations)...")
    search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=param_distributions,
        n_iter=50, # Chỉ thử 50 tổ hợp ngẫu nhiên để tiết kiệm thời gian
        cv=3,
        scoring="r2",
        n_jobs=-1,
        verbose=2,
        random_state=42
    )

    search.fit(X_train, y_train)

    best_model = search.best_estimator_
    print("\n✅ Success: Best Parameters Found:")
    print(search.best_params_)

    # ---------------------------------------------------------
    # STEP 5: MODEL SERIALIZATION (SAVE)
    # ---------------------------------------------------------
    model_dir = "./models"
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "tiktok_rf_multi_model.pkl")
    
    joblib.dump(best_model, model_path)
    print(f"✅ Success: Model saved at: {model_path}")

    # ---------------------------------------------------------
    # STEP 6: EVALUATION & INVERSE TRANSFORMATION
    # ---------------------------------------------------------
    print("\n" + "="*50)
    print("MODEL EVALUATION REPORT")
    print("="*50)
    
    y_pred_log = best_model.predict(X_val)
    y_pred_original = np.expm1(y_pred_log)
    y_pred_original = np.maximum(y_pred_original, 0)

    y_pred_df = pd.DataFrame(y_pred_original, columns=TARGETS, index=val_df.index)
    evaluation_results = {}

    for target in TARGETS:
        actual = y_val_actual[target]
        predicted = y_pred_df[target]

        mae = mean_absolute_error(actual, predicted)
        rmse = np.sqrt(mean_squared_error(actual, predicted))
        r2 = r2_score(actual, predicted)
        mape = (abs(actual - predicted) / (actual + 1)).mean() * 100

        print(f"\nTarget Variable: [{target}]")
        print(f"  - MAE:   {mae:.4f}")
        print(f"  - RMSE:  {rmse:.4f}")
        print(f"  - R2:    {r2:.4f}")
        print(f"  - Error: {mape:.2f}%")

        evaluation_results[f"{target}_actual"] = actual.values
        evaluation_results[f"{target}_predicted"] = predicted.values
        evaluation_results[f"{target}_error_pct"] = (abs(actual - predicted) / (actual + 1)) * 100

    # Feature Importance Printing (Averaged across all 3 targets in multi-target RF)
    print("\n>>> Status: Extracting Global Feature Importances...")
    rf_step = best_model.named_steps["rf"]
    importances = rf_step.feature_importances_
    
    # Sort and print feature importances
    feat_imp = sorted(zip(FEATURES, importances), key=lambda x: x[1], reverse=True)
    for feat, imp in feat_imp[:10]: # Print top 10 to avoid screen clutter
        print(f"  - {feat}: {imp:.4f}")

    # ---------------------------------------------------------
    # STEP 7: EXPORT PREDICTIONS
    # ---------------------------------------------------------
    result_df = pd.DataFrame(evaluation_results)
    result_df.to_csv(PATHS["output_result"], index=False, encoding="utf-8-sig")
    print(f"\n✅ Success: Detailed predictions saved to: {PATHS['output_result']}")

if __name__ == "__main__":
    main()

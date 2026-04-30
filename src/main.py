# main.py
import os
import pandas as pd
import logging
from sklearn.model_selection import train_test_split

from constants import PATHS, FEATURES, TARGETS
from processor import TikTokDataProcessor

import LinearRegression as lr_model
import RandomForestRegression as rf_model
import XGBRegressor as xgb_model

logging.basicConfig(level=logging.INFO, format='%(asctime)s - [%(levelname)s] - %(message)s')

def main():
    print("="*80)
    print("🚀 STARTING TIKTOK MODEL TRAINING PIPELINE")
    print("="*80)

    # 1. DATA CHECK & PREPARATION
    train_path = PATHS["output_train"]
    val_path = PATHS["output_val"]

    if os.path.exists(train_path) and os.path.exists(val_path):
        logging.info("Existing Train/Val datasets detected. Loading directly from CSV...")
        train_df = pd.read_csv(train_path)
        val_df = pd.read_csv(val_path)
    else:
        logging.info("Train/Val datasets not found. Initializing feature extraction process...")
        try:
            df_raw = pd.read_csv(PATHS["main_data"])
        except FileNotFoundError:
            logging.error(f"Raw data file not found at {PATHS['main_data']}. Please check the path.")
            return
        
        proc = TikTokDataProcessor()
        proc.load_trends()
        df_featured = proc.process_features(df_raw)
        
        train_df, val_df = train_test_split(df_featured, test_size=0.2, random_state=42)
        
        os.makedirs(os.path.dirname(train_path), exist_ok=True)
        train_df.to_csv(train_path, index=False, encoding="utf-8-sig")
        val_df.to_csv(val_path, index=False, encoding="utf-8-sig")
        logging.info("Datasets successfully processed and saved.")

    # 2. PREPARE FEATURES (X) AND TARGETS (y)
    X_train = train_df[FEATURES].fillna(0)
    y_train = train_df[TARGETS]
    
    X_val = val_df[FEATURES].fillna(0)
    y_val = val_df[TARGETS]

    # 3. EXECUTE INDEPENDENT MODELS & COLLECT METRICS
    print("\n" + "="*80)
    print("🧠 TRAINING MODELS IN PROGRESS...")
    print("="*80)
    
    lr_metrics = lr_model.run_linear_regression(X_train, y_train, X_val, y_val)
    rf_metrics = rf_model.run_random_forest(X_train, y_train, X_val, y_val)
    xgb_metrics = xgb_model.run_xgboost(X_train, y_train, X_val, y_val)

    # 4. CENTRALIZED REPORTING
    print("\n" + "="*85)
    print(" 🏆 FINAL MODEL PERFORMANCE COMPARISON REPORT ".center(84, "="))
    print("="*85)

    all_metrics = {
        "Linear Regression": lr_metrics,
        "Random Forest": rf_metrics,
        "XGBoost": xgb_metrics
    }

    for target in TARGETS:
        print(f"\n📌 TARGET VARIABLE: {target.upper()}")
        print("-" * 85)
        print(f"{'Model Name':<25} | {'R2 Score':<12} | {'MAE':<12} | {'RMSE':<12} | {'Avg Error (%)':<12}")
        print("-" * 85)
        
        for model_name, metrics in all_metrics.items():
            data = metrics[target]
            print(f"{model_name:<25} | {data['R2']:<12.4f} | {data['MAE']:<12.4f} | {data['RMSE']:<12.4f} | {data['Error_Pct']:<10.2f}%")
            
    print("\n" + "="*85)
    print("🎉 ENTIRE PIPELINE COMPLETED SUCCESSFULLY!")

if __name__ == "__main__":
    main()

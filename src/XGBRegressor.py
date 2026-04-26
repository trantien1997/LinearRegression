# main.py
import os
import joblib
import warnings
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from xgboost import XGBRegressor
from sklearn.multioutput import MultiOutputRegressor

# Custom imports
from constants import PATHS, TARGETS, FEATURES
from processor import TikTokDataProcessor

warnings.filterwarnings("ignore")

class UnifiedTikTokModule:
    """
    Consolidated Expert System using Multi-Output Regression.
    Provides a single interface for Likes, Views, and Shares.
    """
    def __init__(self, model_path="models/xgboost_multioutput_model.pkl"):
        self.model_path = model_path
        self.model = None
        self.metrics_report = []

    def train(self, X_train, y_train, X_val, y_val):
        """
        Trains the Unified Expert Model and computes comprehensive error metrics.
        Includes Max Error expressed as a percentage.
        """
        print("[Unified Module] Starting specialized Multi-Target training...")
        
        # Hyperparameters optimized for balanced multi-output stability
        base_params = {
            'n_estimators': 1500,
            'learning_rate': 0.008,
            'max_depth': 6,
            'gamma': 0.5,
            'reg_alpha': 2.0,
            'reg_lambda': 5.0,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42,
            'n_jobs': -1
        }

        # MultiOutputRegressor ensures each target in TARGETS is predicted simultaneously
        self.model = MultiOutputRegressor(XGBRegressor(**base_params))
        self.model.fit(X_train, y_train)

        # Predicting for validation
        y_pred = self.model.predict(X_val)
        
        # Comprehensive Error Analysis
        for i, target in enumerate(TARGETS):
            actual = y_val.iloc[:, i]
            pred = y_pred[:, i]
            
            # Metric Calculation
            r2 = r2_score(actual, pred)
            mae = mean_absolute_error(actual, pred)
            mse = mean_squared_error(actual, pred)
            rmse = np.sqrt(mse)
            
            # MAPE (Mean Absolute Percentage Error)
            mape = mean_absolute_percentage_error(actual + 1, pred + 1) 
            
            # Max Error as a Percentage (MaxAPE - Maximum Absolute Percentage Error)
            # Calculated relative to (actual + 1) to avoid division by zero
            max_error_pct = np.max(np.abs(actual - pred) / (actual + 1)) * 100
            
            self.metrics_report.append({
                "Target Metric": target,
                "R2 Score": f"{r2:.5f}",
                "MAE": f"{mae:.4f}",
                "RMSE": f"{rmse:.4f}",
                "MAPE (%)": f"{mape * 100:.4f}%",
                "Max Error (%)": f"{max_error_pct:.4f}%"
            })

        # Persist model
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        joblib.dump(self.model, self.model_path)
        print(f"[Unified Module] Master model saved at: {self.model_path}")

    def display_report(self):
        """
        Outputs a clean, instruction-ready performance summary.
        """
        print("\n" + "="*95)
        print("📊 UNIFIED MULTI-TARGET SYSTEM PERFORMANCE REPORT")
        print("="*95)
        report_df = pd.DataFrame(self.metrics_report)
        print(report_df.to_string(index=False))
        print("="*95)
        print("💡 Note: Percentage errors are relative to log1p-transformed scales.")
        print("✅ System Ready for Inference.\n")

def main():
    # 1. Pipeline Initialization
    print("[System] Loading data and trends reference...")
    df_raw = pd.read_csv(PATHS["main_data"])
    processor = TikTokDataProcessor()
    processor.load_trends()
    
    # Feature Engineering
    df_featured = processor.process_features(df_raw)

    # 2. Data Preparation
    print("[System] Preparing Train/Val datasets (80/20 split)...")
    train_df, val_df = train_test_split(df_featured, test_size=0.2, random_state=42)
    
    X_train = train_df[FEATURES].fillna(0)
    X_val = val_df[FEATURES].fillna(0)
    y_train = train_df[TARGETS]
    y_val = val_df[TARGETS]

    # 3. Module Execution
    expert_system = UnifiedTikTokModule()
    expert_system.train(X_train, y_train, X_val, y_val)

    # 4. Final Output
    expert_system.display_report()

if __name__ == "__main__":
    main()

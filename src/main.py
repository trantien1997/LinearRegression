# main.py
import os
import joblib
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
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
        self.feature_importance_df = None
        self.detailed_predictions = None

    def train(self, X_train, y_train, X_val, y_val):
        """
        Trains the Unified Expert Model, computes error metrics, 
        extracts feature importance and prepares detailed results.
        """
        print("[Unified Module] Starting specialized Multi-Target training...")
        
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

        self.model = MultiOutputRegressor(XGBRegressor(**base_params))
        self.model.fit(X_train, y_train)

        # Performance Evaluation
        y_pred = self.model.predict(X_val)
        
        # Prepare dictionary for detailed CSV export
        results_dict = {}

        for i, target in enumerate(TARGETS):
            actual = y_val.iloc[:, i].values
            pred = y_pred[:, i]
            
            # Metric Calculation for Report
            r2 = r2_score(actual, pred)
            mae = mean_absolute_error(actual, pred)
            rmse = np.sqrt(mean_squared_error(actual, pred))
            
            # Row-level % Error: |actual - pred| / (actual + 1) * 100
            # (We use actual+1 to avoid division by zero)
            error_pct = (np.abs(actual - pred) / (actual + 1)) * 100
            
            self.metrics_report.append({
                "Target Metric": target,
                "R2 Score": f"{r2:.5f}",
                "MAE": f"{mae:.4f}",
                "RMSE": f"{rmse:.4f}",
                "Avg Error (%)": f"{error_pct.mean():.4f}%",
                "Max Error (%)": f"{error_pct.max():.4f}%"
            })

            # Store columns for CSV
            results_dict[f"{target}_actual"] = actual
            results_dict[f"{target}_predicted"] = pred
            results_dict[f"{target}_error_pct"] = error_pct

        self.detailed_predictions = pd.DataFrame(results_dict)

        # Calculate Average Feature Importance across all targets
        importances = np.mean([est.feature_importances_ for est in self.model.estimators_], axis=0)
        self.feature_importance_df = pd.DataFrame({
            "Feature": FEATURES,
            "Contribution": importances
        }).sort_values(by="Contribution", ascending=False)

        # Persist model
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        joblib.dump(self.model, self.model_path)
        print(f"[Unified Module] Master model saved at: {self.model_path}")

    def save_results_to_csv(self, output_path):
        """Saves row-level predictions and errors to CSV."""
        if self.detailed_predictions is not None:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            self.detailed_predictions.to_csv(output_path, index=False, encoding="utf-8-sig")
            print(f"[System] Detailed predictions saved to '{output_path}'")

    def plot_feature_importance(self):
        """Generates a professional Donut Chart for Feature Importance."""
        if self.feature_importance_df is None: return

        top_n = 8
        df_plot = self.feature_importance_df.head(top_n).copy()
        others_val = self.feature_importance_df.iloc[top_n:]["Contribution"].sum()
        
        if others_val > 0:
            df_plot = pd.concat([df_plot, pd.DataFrame([{"Feature": "Others", "Contribution": others_val}])], ignore_index=True)

        plt.figure(figsize=(12, 8))
        colors = sns.color_palette("viridis", len(df_plot))
        
        plt.pie(df_plot["Contribution"], labels=df_plot["Feature"], autopct='%1.1f%%', 
                startangle=140, colors=colors, pctdistance=0.85, explode=[0.05] * len(df_plot))
        
        centre_circle = plt.Circle((0,0), 0.70, fc='white')
        plt.gcf().gca().add_artist(centre_circle)
        plt.title("Global Feature Importance Distribution (%)", fontsize=15, pad=20)
        plt.axis('equal') 
        plt.tight_layout()
        plt.savefig("output/feature_importance_pie_xgboost.png", dpi=300)
        print("[System] Pie chart saved as 'output/feature_importance_pie_xgboost.png'")
        plt.show()

    def display_report(self):
        """Outputs a clean performance summary."""
        print("\n" + "="*95)
        print("📊 UNIFIED MULTI-TARGET SYSTEM PERFORMANCE REPORT")
        print("="*95)
        print(pd.DataFrame(self.metrics_report).to_string(index=False))
        print("="*95)

def main():
    # 1. Pipeline Initialization
    df_raw = pd.read_csv(PATHS["main_data"])
    processor = TikTokDataProcessor()
    processor.load_trends()
    df_featured = processor.process_features(df_raw)

    # 2. Data Preparation
    train_df, val_df = train_test_split(df_featured, test_size=0.2, random_state=42)
    X_train, X_val = train_df[FEATURES].fillna(0), val_df[FEATURES].fillna(0)
    
    # Save Train and Validation sets to CSV ---
    print(f"[System] Saving processed datasets to CSV...")
    # Ensure directories exist
    os.makedirs(os.path.dirname(PATHS["output_train"]), exist_ok=True)
    os.makedirs(os.path.dirname(PATHS["output_val"]), exist_ok=True)

    # Save files (using utf-8-sig to support Vietnamese characters in captions)
    train_df.to_csv(PATHS["output_train"], index=False, encoding="utf-8-sig")
    val_df.to_csv(PATHS["output_val"], index=False, encoding="utf-8-sig")
    print(f"✅ Train set saved: {PATHS['output_train']}")
    print(f"✅ Val set saved: {PATHS['output_val']}")
    # --------------------------------------------------

    # Use log1p for training as usual
    y_train = train_df[TARGETS]
    y_val = val_df[TARGETS]

    # 3. Module Execution
    expert_system = UnifiedTikTokModule()
    expert_system.train(X_train, y_train, X_val, y_val)

    # 4. Final Output, CSV Export & Visualization
    expert_system.display_report()
    expert_system.save_results_to_csv(PATHS["output_result_xgboost"])
    expert_system.plot_feature_importance()

if __name__ == "__main__":
    os.makedirs("output", exist_ok=True)
    main()

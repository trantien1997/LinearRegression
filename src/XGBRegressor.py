import os
import joblib
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor
from sklearn.multioutput import MultiOutputRegressor

from constants import PATHS, TARGETS, FEATURES

warnings.filterwarnings("ignore")

class UnifiedTikTokModule:
    def __init__(self, model_path=PATHS["output_model_xgboost"]):
        self.model_path = model_path
        self.model = None
        self.raw_metrics = {}
        self.feature_importance_df = None
        self.detailed_predictions = None

    def train(self, X_train, y_train, X_val, y_val):
        print(">>> Training Multi-Target XGBoost...")
        base_params = {
            'n_estimators': 1500, 'learning_rate': 0.008, 'max_depth': 6,
            'gamma': 0.5, 'reg_alpha': 2.0, 'reg_lambda': 5.0,
            'subsample': 0.8, 'colsample_bytree': 0.8,
            'random_state': 42, 'n_jobs': -1
        }
        self.model = MultiOutputRegressor(XGBRegressor(**base_params))
        self.model.fit(X_train, y_train)

        y_pred = self.model.predict(X_val)
        results_dict = {}

        for i, target in enumerate(TARGETS):
            actual = y_val.iloc[:, i].values
            pred = y_pred[:, i]
            
            r2 = r2_score(actual, pred)
            mae = mean_absolute_error(actual, pred)
            rmse = np.sqrt(mean_squared_error(actual, pred))
            error_pct = (np.abs(actual - pred) / (actual + 1)) * 100
            
            # Lưu metrics chuẩn dạng float
            self.raw_metrics[target] = {
                "R2": r2, "MAE": mae, "RMSE": rmse, "Error_Pct": error_pct.mean()
            }

            results_dict[f"{target}_actual"] = actual
            results_dict[f"{target}_predicted"] = pred
            results_dict[f"{target}_error_pct"] = error_pct

        self.detailed_predictions = pd.DataFrame(results_dict)

        importances = np.mean([est.feature_importances_ for est in self.model.estimators_], axis=0)
        self.feature_importance_df = pd.DataFrame({
            "Feature": FEATURES, "Contribution": importances
        }).sort_values(by="Contribution", ascending=False)

        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        joblib.dump(self.model, self.model_path)

    def save_results_to_csv(self, output_path):
        if self.detailed_predictions is not None:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            self.detailed_predictions.to_csv(output_path, index=False, encoding="utf-8-sig")

    def plot_feature_importance(self):
        if self.feature_importance_df is None: return
        top_n = 8
        df_plot = self.feature_importance_df.head(top_n).copy()
        others_val = self.feature_importance_df.iloc[top_n:]["Contribution"].sum()
        if others_val > 0:
            df_plot = pd.concat([df_plot, pd.DataFrame([{"Feature": "Others", "Contribution": others_val}])], ignore_index=True)

        plt.figure(figsize=(10, 7))
        colors = sns.color_palette("viridis", len(df_plot))
        plt.pie(df_plot["Contribution"], labels=df_plot["Feature"], autopct='%1.1f%%', 
                startangle=140, colors=colors, pctdistance=0.85, explode=[0.05] * len(df_plot))
        centre_circle = plt.Circle((0,0), 0.70, fc='white')
        plt.gcf().gca().add_artist(centre_circle)
        plt.title("XGBoost: Global Feature Contribution", fontsize=15, pad=20)
        plt.axis('equal') 
        plt.tight_layout()
        os.makedirs(PATHS["output_feature_importance_xgb"].parent, exist_ok=True)
        plt.savefig(PATHS["output_feature_importance_xgb"], dpi=300)
        plt.close()

def run_xgboost(X_train, y_train, X_val, y_val):
    expert_system = UnifiedTikTokModule()
    expert_system.train(X_train, y_train, X_val, y_val)
    expert_system.save_results_to_csv(PATHS["output_result_xgboost"])
    expert_system.plot_feature_importance()
    
    print("✅ Success: XGBoost completed.")
    return expert_system.raw_metrics

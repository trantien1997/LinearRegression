import pandas as pd
import numpy as np
import joblib
import os
import warnings
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from constants import PATHS, FEATURES, TARGETS

warnings.filterwarnings("ignore")

def plot_feature_importance_pie(features, coefficients, save_path):
    abs_coefs = np.abs(coefficients)
    df_fi = pd.DataFrame({'Feature': features, 'Importance': abs_coefs}).sort_values(by='Importance', ascending=False)
    
    top_n = 8
    df_plot = df_fi.head(top_n).copy()
    others_val = df_fi.iloc[top_n:]['Importance'].sum()
    if others_val > 0:
        df_plot = pd.concat([df_plot, pd.DataFrame([{"Feature": "Others", "Importance": others_val}])], ignore_index=True)

    plt.figure(figsize=(10, 7))
    colors = sns.color_palette("viridis", len(df_plot))
    wedges, texts, autotexts = plt.pie(df_plot['Importance'], labels=df_plot['Feature'], autopct='%1.1f%%', 
                                       startangle=140, colors=colors, pctdistance=0.85, explode=[0.03] * len(df_plot))
    centre_circle = plt.Circle((0,0), 0.70, fc='white')
    plt.gcf().gca().add_artist(centre_circle)
    plt.title("Linear Regression: Global Feature Contribution", fontsize=15, pad=20)
    plt.axis('equal') 
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()

def run_linear_regression(X_train, y_train, X_val, y_val_actual):
    print(">>> Training Linear Regression...")
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LinearRegression())
    ])
    pipeline.fit(X_train, y_train)

    os.makedirs("./models", exist_ok=True)
    joblib.dump(pipeline, "./models/tiktok_linear_regression_multi.pkl")

    y_pred = pipeline.predict(X_val)
    y_pred_df = pd.DataFrame(y_pred, columns=TARGETS, index=X_val.index)

    evaluation_results = {}
    summary_metrics = {}
    
    for target in TARGETS:
        actual = y_val_actual[target].values
        predicted = y_pred_df[target].values
        
        r2 = r2_score(actual, predicted)
        mae = mean_absolute_error(actual, predicted)
        rmse = np.sqrt(mean_squared_error(actual, predicted))
        error_pct = (np.abs(actual - predicted) / (actual + 1)) * 100

        # Lưu metrics trả về cho file main.py
        summary_metrics[target] = {
            "R2": r2, "MAE": mae, "RMSE": rmse, "Error_Pct": error_pct.mean()
        }

        evaluation_results[f"{target}_actual"] = actual
        evaluation_results[f"{target}_predicted"] = predicted
        evaluation_results[f"{target}_error_pct"] = error_pct

    os.makedirs(os.path.dirname(PATHS["output_result_linear_regression"]), exist_ok=True)
    pd.DataFrame(evaluation_results).to_csv(PATHS["output_result_linear_regression"], index=False, encoding="utf-8-sig")
    
    lr_model = pipeline.named_steps["lr"]
    plot_feature_importance_pie(FEATURES, np.mean(np.abs(lr_model.coef_), axis=0), "./output/feature_impact_pie_linear_regression.png")
    
    print("✅ Success: Linear Regression completed.")
    return summary_metrics

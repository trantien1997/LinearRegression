import pandas as pd
import numpy as np
import joblib
import os
import warnings
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from constants import PATHS, FEATURES, TARGETS
from processor import TikTokDataProcessor

warnings.filterwarnings("ignore")

def plot_feature_importance_pie(features, coefficients, save_path):
    """
    Generates a professional Donut Chart for Linear Regression Coefficients.
    """
    abs_coefs = np.abs(coefficients)
    df_fi = pd.DataFrame({'Feature': features, 'Importance': abs_coefs})
    df_fi = df_fi.sort_values(by='Importance', ascending=False)

    top_n = 8
    df_plot = df_fi.head(top_n).copy()
    others_val = df_fi.iloc[top_n:]['Importance'].sum()
    
    if others_val > 0:
        new_row = pd.DataFrame([{"Feature": "Others", "Importance": others_val}])
        df_plot = pd.concat([df_plot, new_row], ignore_index=True)

    plt.figure(figsize=(10, 7))
    colors = sns.color_palette("viridis", len(df_plot))
    
    wedges, texts, autotexts = plt.pie(
        df_plot['Importance'], 
        labels=df_plot['Feature'], 
        autopct='%1.1f%%', 
        startangle=140, 
        colors=colors,
        pctdistance=0.85,
        explode=[0.03] * len(df_plot)
    )

    centre_circle = plt.Circle((0,0), 0.70, fc='white')
    plt.gcf().gca().add_artist(centre_circle)

    plt.setp(autotexts, size=10, weight="bold")
    plt.title("Global Feature Contribution Analysis (%)", fontsize=15, pad=20)
    
    plt.axis('equal') 
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    print(f"✅ Success: Feature Impact chart saved to: {save_path}")
    plt.show()

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
    # STEP 2: DATA SPLITTING & EXPORTING TRAIN/VAL SETS
    # ---------------------------------------------------------
    print(">>> Status: Splitting data and saving Train/Val sets...")
    train_df, val_df = train_test_split(df_featured, test_size=0.2, random_state=42)
    
    os.makedirs(os.path.dirname(PATHS["output_train"]), exist_ok=True)
    train_df.to_csv(PATHS["output_train"], index=False, encoding="utf-8-sig")
    val_df.to_csv(PATHS["output_val"], index=False, encoding="utf-8-sig")

    # ---------------------------------------------------------
    # STEP 3: FEATURE & TARGET PREPARATION
    # ---------------------------------------------------------
    X_train = train_df[FEATURES].fillna(0)
    X_val = val_df[FEATURES].fillna(0)
    y_train = train_df[TARGETS] 
    y_val_actual = val_df[TARGETS] 

    # ---------------------------------------------------------
    # STEP 4: MODEL TRAINING (PIPELINE)
    # ---------------------------------------------------------
    print(f"\n>>> Status: Training Multi-target Linear Regression...")
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LinearRegression())
    ])
    pipeline.fit(X_train, y_train)

    # ---------------------------------------------------------
    # STEP 5: MODEL SERIALIZATION
    # ---------------------------------------------------------
    model_dir = "./models"
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "tiktok_linear_regression_multi_model.pkl")
    joblib.dump(pipeline, model_path)
    print(f"✅ Success: Model pipeline saved at: {model_path}")

    # ---------------------------------------------------------
    # STEP 6: EVALUATION & DETAILED RESULTS EXPORT
    # ---------------------------------------------------------
    print("\n" + "="*50)
    print("MODEL EVALUATION REPORT")
    print("="*50)
    
    y_pred = pipeline.predict(X_val)
    y_pred_df = pd.DataFrame(y_pred, columns=TARGETS, index=val_df.index)

    evaluation_results = {}
    for target in TARGETS:
        actual = y_val_actual[target].values
        predicted = y_pred_df[target].values
        
        # Calculate row-level percentage error
        error_pct = (np.abs(actual - predicted) / (actual + 1)) * 100

        # Metrics
        r2 = r2_score(actual, predicted)
        mae = mean_absolute_error(actual, predicted)
        mse = mean_squared_error(actual, predicted)
        rmse = np.sqrt(mse) # Added RMSE
        mape = error_pct.mean()

        print(f"\nTarget Variable: [{target}]")
        print(f"  - R2 Score: {r2:.4f}")
        print(f"  - MAE:      {mae:.4f}")
        print(f"  - RMSE:     {rmse:.4f}") # Displaying RMSE
        print(f"  - Avg Err:  {mape:.2f}%")

        evaluation_results[f"{target}_actual"] = actual
        evaluation_results[f"{target}_predicted"] = predicted
        evaluation_results[f"{target}_error_pct"] = error_pct

    # Save detailed CSV
    result_df = pd.DataFrame(evaluation_results)
    os.makedirs(os.path.dirname(PATHS["output_result_linear_regression"]), exist_ok=True)
    result_df.to_csv(PATHS["output_result_linear_regression"], index=False, encoding="utf-8-sig")
    print(f"\n✅ Success: Detailed CSV predictions saved to: {PATHS['output_result_linear_regression']}")

    # ---------------------------------------------------------
    # STEP 7: FEATURE IMPACT VISUALIZATION
    # ---------------------------------------------------------
    print("\n>>> Status: Generating Feature Impact Analysis...")
    lr_model = pipeline.named_steps["lr"]
    avg_abs_coeffs = np.mean(np.abs(lr_model.coef_), axis=0)
    
    plot_feature_importance_pie(FEATURES, avg_abs_coeffs, "./output/feature_impact_pie_linear_regression.png")

if __name__ == "__main__":
    main()

import pandas as pd
import numpy as np
import joblib
import os
import warnings
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from constants import PATHS, FEATURES, TARGETS
from processor import TikTokDataProcessor

warnings.filterwarnings("ignore")

def plot_feature_importance_pie(features, importances, save_path):
    """
    Generates a professional Donut Chart for Feature Importance.
    """
    df_fi = pd.DataFrame({'Feature': features, 'Importance': importances})
    df_fi = df_fi.sort_values(by='Importance', ascending=False)

    top_n = 8
    df_plot = df_fi.head(top_n).copy()
    others_val = df_fi.iloc[top_n:]['Importance'].sum()
    
    if others_val > 0:
        new_row = pd.DataFrame([{"Feature": "Others", "Importance": others_val}])
        df_plot = pd.concat([df_plot, new_row], ignore_index=True)

    plt.figure(figsize=(10, 7))
    colors = sns.color_palette("pastel", len(df_plot))
    
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
    fig = plt.gcf()
    fig.gca().add_artist(centre_circle)

    plt.setp(autotexts, size=10, weight="bold")
    plt.title("Global Feature Contribution Analysis (%)", fontsize=15, pad=20)
    
    plt.axis('equal') 
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    print(f"\n✅ Success: Pie chart saved to: {save_path}")
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
    # STEP 2: DATA SPLITTING
    # ---------------------------------------------------------
    print(">>> Status: Splitting data into Train and Validation sets...")
    train_df, val_df = train_test_split(
        df_featured,
        test_size=0.2,
        random_state=42
    )

    # ---------------------------------------------------------
    # STEP 3: FEATURE & TARGET PREPARATION
    # ---------------------------------------------------------
    X_train = train_df[FEATURES].fillna(0)
    X_val = val_df[FEATURES].fillna(0)

    print(f">>> Status: Applying Log transformation to targets: {TARGETS}")
    y_train = np.log1p(train_df[TARGETS]) 
    y_val_actual = val_df[TARGETS] 

    # ---------------------------------------------------------
    # STEP 4: MULTI-TARGET RANDOM FOREST SEARCH
    # ---------------------------------------------------------
    pipeline = Pipeline([
        ("rf", RandomForestRegressor(random_state=42, n_jobs=-1))
    ])

    param_distributions = {
        "rf__n_estimators": [100, 300, 500],
        "rf__max_depth": [10, 20, 30, None],
        "rf__min_samples_split": [2, 5, 10],
        "rf__min_samples_leaf": [1, 2, 4],
        "rf__max_features": ["sqrt", 1.0]
    }

    print(">>> Status: Starting Randomized Hyperparameter Search (n_iter=50)...")
    search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=param_distributions,
        n_iter=50,
        cv=3,
        scoring="r2",
        n_jobs=-1,
        verbose=1,
        random_state=42
    )

    search.fit(X_train, y_train)
    best_model = search.best_estimator_

    # ---------------------------------------------------------
    # STEP 5: MODEL SERIALIZATION & EVALUATION
    # ---------------------------------------------------------
    model_dir = "./models"
    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(best_model, os.path.join(model_dir, "tiktok_random_forest_multi_model.pkl"))

    print("\n" + "="*50)
    print("MODEL EVALUATION REPORT")
    print("="*50)
    
    y_pred_log = best_model.predict(X_val)
    y_pred_original = np.expm1(y_pred_log)
    y_pred_original = np.maximum(y_pred_original, 0) # Ensure no negative values
    y_pred_df = pd.DataFrame(y_pred_original, columns=TARGETS, index=val_df.index)

    # Dictionary to store data for CSV export
    evaluation_results = {}

    for target in TARGETS:
        actual = y_val_actual[target].values
        predicted = y_pred_df[target].values

        # Metric calculation for Console
        mae = mean_absolute_error(actual, predicted)
        rmse = np.sqrt(mean_squared_error(actual, predicted))
        r2 = r2_score(actual, predicted)
        
        # Row-level percentage error: |actual - predicted| / (actual + 1) * 100
        error_pct = (np.abs(actual - predicted) / (actual + 1)) * 100

        print(f"\nTarget Variable: [{target}]")
        print(f"  - MAE:   {mae:.4f}")
        print(f"  - RMSE:  {rmse:.4f}")
        print(f"  - R2:    {r2:.4f}")
        print(f"  - Error: {error_pct.mean():.2f}%")

        # Storing for Step 7 CSV export
        evaluation_results[f"{target}_actual"] = actual
        evaluation_results[f"{target}_predicted"] = predicted
        evaluation_results[f"{target}_error_pct"] = error_pct

    # ---------------------------------------------------------
    # STEP 6: FEATURE IMPORTANCE & VISUALIZATION
    # ---------------------------------------------------------
    print("\n>>> Status: Extracting & Plotting Feature Importances...")
    rf_step = best_model.named_steps["rf"]
    importances = rf_step.feature_importances_
    
    feat_imp = sorted(zip(FEATURES, importances), key=lambda x: x[1], reverse=True)
    print("\nTop 10 Global Features (Contribution %):")
    for feat, imp in feat_imp[:10]:
        print(f"  - {feat}: {imp*100:.2f}%")

    plot_feature_importance_pie(FEATURES, importances, "./output/feature_importance_pie_random_forest.png")

    # ---------------------------------------------------------
    # STEP 7: EXPORT RESULTS (WITH DETAILED % ERROR)
    # ---------------------------------------------------------
    result_df = pd.DataFrame(evaluation_results)
    
    # Ensure the output directory exists
    os.makedirs(os.path.dirname(PATHS["output_result_random_forest"]), exist_ok=True)
    
    result_df.to_csv(PATHS["output_result_random_forest"], index=False, encoding="utf-8-sig")
    print(f"\n✅ Success: Detailed predictions with % error saved to: {PATHS['output_result_random_forest']}")

if __name__ == "__main__":
    main()

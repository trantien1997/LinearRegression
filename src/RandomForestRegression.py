import pandas as pd
import numpy as np
import joblib
import os
import warnings
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from constants import PATHS, FEATURES, TARGETS

warnings.filterwarnings("ignore")

def plot_feature_importance_pie(features, importances, save_path):
    df_fi = pd.DataFrame({'Feature': features, 'Importance': importances}).sort_values(by='Importance', ascending=False)
    
    top_n = 8
    df_plot = df_fi.head(top_n).copy()
    others_val = df_fi.iloc[top_n:]['Importance'].sum()
    if others_val > 0:
        df_plot = pd.concat([df_plot, pd.DataFrame([{"Feature": "Others", "Importance": others_val}])], ignore_index=True)

    plt.figure(figsize=(10, 7))
    colors = sns.color_palette("pastel", len(df_plot))
    wedges, texts, autotexts = plt.pie(df_plot['Importance'], labels=df_plot['Feature'], autopct='%1.1f%%', 
                                       startangle=140, colors=colors, pctdistance=0.85, explode=[0.03] * len(df_plot))
    centre_circle = plt.Circle((0,0), 0.70, fc='white')
    plt.gcf().gca().add_artist(centre_circle)
    plt.title("Random Forest: Global Feature Contribution", fontsize=15, pad=20)
    plt.axis('equal') 
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()

def run_random_forest(X_train, y_train, X_val, y_val):
    print(">>> Training Random Forest (Randomized Search)...")
    pipeline = Pipeline([("rf", RandomForestRegressor(random_state=42, n_jobs=-1))])
    param_distributions = {
        "rf__n_estimators": [100, 300, 500],
        "rf__max_depth": [10, 20, 30, None],
        "rf__min_samples_split": [2, 5, 10],
        "rf__min_samples_leaf": [1, 2, 4],
        "rf__max_features": ["sqrt", 1.0]
    }

    search = RandomizedSearchCV(estimator=pipeline, param_distributions=param_distributions,
                                n_iter=20, cv=3, scoring="r2", n_jobs=-1, random_state=42)
    search.fit(X_train, y_train)
    best_model = search.best_estimator_

    os.makedirs(PATHS["output_model_random_forest"].parent, exist_ok=True)
    joblib.dump(best_model, PATHS["output_model_random_forest"])

    y_pred = best_model.predict(X_val)
    y_pred_df = pd.DataFrame(y_pred, columns=TARGETS, index=X_val.index)
    
    evaluation_results = {}
    summary_metrics = {}

    for target in TARGETS:
        actual = y_val[target].values
        predicted = y_pred_df[target].values

        r2 = r2_score(actual, predicted)
        mae = mean_absolute_error(actual, predicted)
        rmse = np.sqrt(mean_squared_error(actual, predicted))
        error_pct = (np.abs(actual - predicted) / (actual + 1)) * 100

        summary_metrics[target] = {
            "R2": r2, "MAE": mae, "RMSE": rmse, "Error_Pct": error_pct.mean()
        }

        evaluation_results[f"{target}_actual"] = actual
        evaluation_results[f"{target}_predicted"] = predicted
        evaluation_results[f"{target}_error_pct"] = error_pct

    plot_feature_importance_pie(FEATURES, best_model.named_steps["rf"].feature_importances_, PATHS["output_feature_importance_rf"])
    
    os.makedirs(PATHS["output_result_random_forest"].parent, exist_ok=True)
    pd.DataFrame(evaluation_results).to_csv(PATHS["output_result_random_forest"], index=False, encoding="utf-8-sig")
    
    print("✅ Success: Random Forest completed.")
    return summary_metrics

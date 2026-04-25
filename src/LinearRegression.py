import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from constants import PATHS, FEATURES, TARGETS
from processor import TikTokDataProcessor


def main():
    # 1. Load & Process
    print("Loading data and processing features...")
    df_raw = pd.read_csv(PATHS["main_data"])

    proc = TikTokDataProcessor()
    proc.load_trends()

    df_featured = proc.process_features(df_raw)

    # 2. Split Data
    print("Splitting data into Train/Val...")
    train_df, val_df = train_test_split(
        df_featured,
        test_size=0.2,
        random_state=42
    )

    train_df.to_csv(PATHS["output_train"], index=False, encoding="utf-8-sig")
    val_df.to_csv(PATHS["output_val"], index=False, encoding="utf-8-sig")

    # 3. Prepare Features
    X_train = train_df[FEATURES].fillna(0)
    X_val = val_df[FEATURES].fillna(0)

    results = {}

    print("Training Ridge Regression with Auto Tuning...")

    for target in TARGETS:
        if target in train_df.columns:
            print(f"\nTraining target: {target}")

            # log transform target
            y_train = np.log1p(train_df[target])
            y_val = val_df[target]

            # pipeline
            pipeline = Pipeline([
                ("scaler", StandardScaler()),
                ("ridge", Ridge())
            ])

            # hyperparameter search space
            param_grid = {
                "ridge__alpha": [0.1, 1, 10, 50, 100]
            }

            # grid search
            grid = GridSearchCV(
                estimator=pipeline,
                param_grid=param_grid,
                cv=5,
                scoring="r2",
                n_jobs=-1
            )

            grid.fit(X_train, y_train)

            best_model = grid.best_estimator_

            print("Best params:", grid.best_params_)

            # predict
            y_pred_log = best_model.predict(X_val)

            # inverse transform
            y_pred = np.expm1(y_pred_log)

            # remove negative values
            y_pred = np.maximum(y_pred, 0)

            # metrics
            mae = mean_absolute_error(y_val, y_pred)
            rmse = np.sqrt(mean_squared_error(y_val, y_pred))
            r2 = r2_score(y_val, y_pred)

            print(f"{target} -> MAE={mae:.4f}")
            print(f"{target} -> RMSE={rmse:.4f}")
            print(f"{target} -> R2={r2:.4f}")

            results[f"{target}_actual"] = y_val.values
            results[f"{target}_predicted"] = y_pred
            results[f"{target}_error_%"] = (
                abs(y_val - y_pred) / (y_val + 1)
            ) * 100

    # 4. Save Results
    result_df = pd.DataFrame(results)

    result_df.to_csv(
        PATHS["output_result"],
        index=False,
        encoding="utf-8-sig"
    )

    print(f"\nResults saved to: {PATHS['output_result']}")


if __name__ == "__main__":
    main()

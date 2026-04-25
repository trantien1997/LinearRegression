import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
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

    print("Training Random Forest with Large Grid Search...")

    for target in TARGETS:
        if target in train_df.columns:
            print(f"\nTraining target: {target}")

            y_train = train_df[target]
            y_actual = val_df[target]

            model = RandomForestRegressor(
                random_state=42,
                n_jobs=-1
            )

            # Large search space
            param_grid = {
                "n_estimators": list(range(100, 1001, 100)),
                "max_depth": list(range(5, 51, 5)) + [None],
                "min_samples_split": list(range(2, 11)),
                "min_samples_leaf": list(range(1, 6)),
                "max_features": ["sqrt", "log2", None]
            }

            grid = GridSearchCV(
                estimator=model,
                param_grid=param_grid,
                cv=3,
                scoring="r2",
                n_jobs=-1,
                verbose=2
            )

            print("Start Grid Search...")
            grid.fit(X_train, y_train)

            best_model = grid.best_estimator_

            print("\nBest Parameters:")
            print(grid.best_params_)

            y_pred = best_model.predict(X_val)

            mae = mean_absolute_error(y_actual, y_pred)
            rmse = np.sqrt(mean_squared_error(y_actual, y_pred))
            r2 = r2_score(y_actual, y_pred)

            print(f"\nFinal {target} Results")
            print(f"MAE  = {mae:.4f}")
            print(f"RMSE = {rmse:.4f}")
            print(f"R2   = {r2:.4f}")

            # Feature importance
            print("\nFeature Importance:")
            for feature, importance in zip(FEATURES, best_model.feature_importances_):
                print(f"{feature}: {importance:.4f}")

            results[f"{target}_actual"] = y_actual.values
            results[f"{target}_predicted"] = y_pred
            results[f"{target}_error_%"] = (
                abs(y_actual - y_pred) / (y_actual + 1)
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

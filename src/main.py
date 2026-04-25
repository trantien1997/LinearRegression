import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

from constants import PATHS, FEATURES, TARGETS
from processor import TikTokDataProcessor


def main():
    # =====================================
    # 1. Load & Process
    # =====================================
    print("Loading data and processing features...")

    df_raw = pd.read_csv(PATHS["main_data"])

    proc = TikTokDataProcessor()
    proc.load_trends()

    df_featured = proc.process_features(df_raw)

    print("Tổng số samples ban đầu:", len(df_featured))

    # =====================================
    # 2. Split Data
    # =====================================
    print("Splitting data into Train/Val...")

    train_df, val_df = train_test_split(
        df_featured,
        test_size=0.2,
        random_state=42
    )

    print("Train trước khi lọc:", len(train_df))
    print("Validation:", len(val_df))

    # =====================================
    # 3. FILTER TRAIN ONLY
    # =====================================
    train_df = train_df[train_df["views"] >= 1000].copy()

    print("Train sau khi lọc views >= 1000:", len(train_df))

    # =====================================
    # 4. Save train/val
    # =====================================
    train_df.to_csv(
        PATHS["output_train"],
        index=False,
        encoding="utf-8-sig"
    )

    val_df.to_csv(
        PATHS["output_val"],
        index=False,
        encoding="utf-8-sig"
    )

    # =====================================
    # 5. Prepare Features
    # =====================================
    print("Training and Predicting...")

    X_train = train_df[FEATURES].fillna(0)
    X_val = val_df[FEATURES].fillna(0)

    results = {}
    metrics = []

    # =====================================
    # 6. Train per target
    # =====================================
    for target in TARGETS:
        if target in train_df.columns:
            print(f"\nTraining target: {target}")

            model = XGBRegressor(
                n_estimators=500,
                max_depth=4,
                learning_rate=0.03,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=1,
                reg_lambda=5,
                random_state=42
            )

            model.fit(X_train, train_df[target])

            y_actual = val_df[target]
            y_pred = model.predict(X_val)

            # Save prediction
            results[f"{target}_actual"] = y_actual.values
            results[f"{target}_predict"] = y_pred
            results[f"{target}_error_%"] = (
                abs(y_actual - y_pred) / (y_actual + 1)
            ) * 100

            # Metrics
            mae = mean_absolute_error(y_actual, y_pred)
            rmse = np.sqrt(mean_squared_error(y_actual, y_pred))
            r2 = r2_score(y_actual, y_pred)

            metrics.append({
                "target": target,
                "MAE": mae,
                "RMSE": rmse,
                "R2": r2
            })

            print(f"{target} -> MAE={mae:.4f}")
            print(f"{target} -> RMSE={rmse:.4f}")
            print(f"{target} -> R2={r2:.4f}")

    # =====================================
    # 7. Save Results
    # =====================================
    result_df = pd.DataFrame(results)

    result_df.to_csv(
        PATHS["output_result"],
        index=False,
        encoding="utf-8-sig"
    )

    metric_df = pd.DataFrame(metrics)

    metric_df.to_csv(
        PATHS["output_result"].replace(".csv", "_metrics.csv"),
        index=False,
        encoding="utf-8-sig"
    )

    print("\nResults saved successfully")
    print(f"Prediction file: {PATHS['output_result']}")
    print(f"Metric file: {PATHS['output_result'].replace('.csv', '_metrics.csv')}")


if __name__ == "__main__":
    main()
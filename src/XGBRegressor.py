import optuna
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
from constants import PATHS, FEATURES, TARGETS
from processor import TikTokDataProcessor

# ==================== OBJECTIVE FUNCTION ====================
def objective(trial, X_train, y_train, X_val, y_val, target_name):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 800, 2000),
        'max_depth': trial.suggest_int('max_depth', 5, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.08, log=True),
        'subsample': trial.suggest_float('subsample', 0.75, 0.95),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.75, 0.95),
        'colsample_bylevel': trial.suggest_float('colsample_bylevel', 0.75, 0.95),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 2.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 5.0),
        'gamma': trial.suggest_float('gamma', 0.0, 0.4),
        'random_state': 42,
        'early_stopping_rounds': 100,      # tăng lên một chút
        'eval_metric': 'rmse'
    }
    
    model = XGBRegressor(**params)
    
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False
    )
    
    pred_val = model.predict(X_val)
    
    r2 = r2_score(y_val, pred_val)
    
    # In ra theo dõi tiến trình
    print(f"Trial {trial.number:3d} | {target_name:15s} -> R2: {r2:.5f} | "
          f"MAE: {mean_absolute_error(y_val, pred_val):.4f} | "
          f"RMSE: {np.sqrt(mean_squared_error(y_val, pred_val)):.4f}")
    
    return r2   # Optuna sẽ tối ưu R² (càng cao càng tốt)


# ==================== MAIN FUNCTION ====================
def main():
    print("Loading data and processing features...")
    df_raw = pd.read_csv(PATHS["main_data"])

    proc = TikTokDataProcessor()
    proc.load_trends()
    df_featured = proc.process_features(df_raw)

    # Split data
    print("Splitting data into Train/Val...")
    train_df, val_df = train_test_split(
        df_featured, test_size=0.2, random_state=42, stratify=None
    )

    print(f"Train size: {len(train_df)}, Validation size: {len(val_df)}")

    # Save train/val (giữ nguyên)
    train_df.to_csv(PATHS["output_train"], index=False, encoding="utf-8-sig")
    val_df.to_csv(PATHS["output_val"], index=False, encoding="utf-8-sig")

    X_train = train_df[FEATURES].fillna(0)
    X_val   = val_df[FEATURES].fillna(0)

    results = {}
    metrics = []

    # ==================== OPTUNA TUNING ====================
    print("\n=== Bắt đầu Optuna Tuning ===\n")

    for target in TARGETS:
        if target not in train_df.columns:
            continue

        print(f"\n=== Tuning cho target: {target} ===")
        
        study = optuna.create_study(
            direction='maximize',
            sampler=optuna.samplers.TPESampler(seed=42)
        )
        
        # Số trials bạn muốn thử (bắt đầu với 40-60 là hợp lý)
        study.optimize(
            lambda trial: objective(trial, X_train, train_df[target], 
                                  X_val, val_df[target], target),
            n_trials=60,           # ← Bạn có thể chỉnh số này (30~100)
            timeout=7200           # tối đa 2 tiếng (7200 giây), có thể bỏ nếu muốn chạy hết
        )

        print(f"\nBest R2 for {target}: {study.best_value:.5f}")
        print("Best params:", study.best_params)

        # Train lại model tốt nhất
        best_params = study.best_params
        best_params.update({
            'random_state': 42,
            'early_stopping_rounds': 100,
            'eval_metric': 'rmse'
        })

        best_model = XGBRegressor(**best_params)
        best_model.fit(X_train, train_df[target], 
                      eval_set=[(X_val, val_df[target])], 
                      verbose=False)

        # Dự đoán và tính metrics
        y_pred = best_model.predict(X_val)
        y_actual = val_df[target]

        mae = mean_absolute_error(y_actual, y_pred)
        rmse = np.sqrt(mean_squared_error(y_actual, y_pred))
        r2 = r2_score(y_actual, y_pred)

        print(f"Final {target} -> MAE={mae:.4f} | RMSE={rmse:.4f} | R2={r2:.4f}\n")

        # Lưu kết quả
        results[f"{target}_actual"] = y_actual.values
        results[f"{target}_predict"] = y_pred
        metrics.append({"target": target, "MAE": mae, "RMSE": rmse, "R2": r2})

    # Save results
    pd.DataFrame(results).to_csv(PATHS["output_result"], index=False, encoding="utf-8-sig")
    pd.DataFrame(metrics).to_csv(
        PATHS["output_result"].replace(".csv", "_metrics.csv"), 
        index=False, encoding="utf-8-sig"
    )

    print("Hoàn thành tuning và training!")

if __name__ == "__main__":
    main()

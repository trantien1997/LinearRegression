import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
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
    train_df, val_df = train_test_split(df_featured, test_size=0.2, random_state=42)
    train_df.to_csv(PATHS["output_train"], index=False, encoding="utf-8-sig")
    val_df.to_csv(PATHS["output_val"], index=False, encoding="utf-8-sig")

    # 3. Train & Predict
    print("Training and Predicting...")
    X_train = train_df[FEATURES].fillna(0)
    X_val = val_df[FEATURES].fillna(0)
    
    results = {}
    for target in TARGETS:
        if target in train_df.columns:
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X_train, train_df[target])
            
            y_actual = val_df[target]
            y_pred = model.predict(X_val)
            
            results[f"{target}_thucte"] = y_actual.values
            results[f"{target}_predict"] = y_pred
            results[f"{target}_error_%"] = (abs(y_actual - y_pred) / (y_actual + 1)) * 100

    # 4. Save Results
    result_df = pd.DataFrame(results)
    result_df.to_csv(PATHS["output_result"], index=False, encoding="utf-8-sig")
    print(f"Results saved to: {PATHS['output_result']}")

if __name__ == "__main__":
    main()

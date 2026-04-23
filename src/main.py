import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)
from xgboost import XGBRegressor
from scipy.sparse import hstack


# =====================================
# 1. FILE PATH
# =====================================
train_path = r"C:\Users\Admin\OneDrive\Desktop\LuanVan\DB_tiktok\tiktok_train.csv"
val_path = r"C:\Users\Admin\OneDrive\Desktop\LuanVan\DB_tiktok\tiktok_validate.csv"
result_path = r"C:\Users\Admin\OneDrive\Desktop\LuanVan\DB_tiktok\Result.csv"


# =====================================
# 2. LOAD DATA
# =====================================
df_train = pd.read_csv(train_path, encoding="utf-8-sig")
df_val = pd.read_csv(val_path, encoding="utf-8-sig")


# =====================================
# 3. FEATURE COLUMNS
# =====================================
num_cols = [
    "caption_length",
    "word_count",
    "emoji_count",
    "hashtag_count",
    "followers_log1p",
    "time",
    "is_weekend",
    "has_trend_keyword",
    "has_trend_hashtag",
    "has_trend_song",
    "is_related_gameshow"
]

target_cols = [
    "likes_log1p",
    "views_log1p",
    "shares_log1p"
]


# =====================================
# 4. TRAIN FEATURES
# =====================================
X_train_num = df_train[num_cols].fillna(0)

tfidf = TfidfVectorizer(max_features=300)

X_train_text = tfidf.fit_transform(
    df_train["caption_clean"].fillna("")
)

X_train = hstack([X_train_text, X_train_num])

y_train = df_train[target_cols]


# =====================================
# 5. TRAIN MODEL
# =====================================
model = MultiOutputRegressor(
    XGBRegressor(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.05,
        random_state=42
    )
)

model.fit(X_train, y_train)

print("Train model thành công")


# =====================================
# 6. VALIDATION PREDICT
# =====================================
X_val_num = df_val[num_cols].fillna(0)

X_val_text = tfidf.transform(
    df_val["caption_clean"].fillna("")
)

X_val = hstack([X_val_text, X_val_num])

y_val = df_val[target_cols]

pred = model.predict(X_val)


# =====================================
# 7. METRICS
# =====================================
target_names = ["likes", "views", "shares"]

metric_rows = []

for i, name in enumerate(target_names):
    mae = mean_absolute_error(y_val.iloc[:, i], pred[:, i])
    rmse = np.sqrt(mean_squared_error(y_val.iloc[:, i], pred[:, i]))
    r2 = r2_score(y_val.iloc[:, i], pred[:, i])

    # accuracy %
    actual = y_val.iloc[:, i].replace(0, 1e-6)
    accuracy = (
        100 - np.mean(np.abs((actual - pred[:, i]) / actual)) * 100
    )

    metric_rows.append({
        "target": name,
        "MAE": round(mae, 4),
        "RMSE": round(rmse, 4),
        "R2_score": round(r2, 4),
        "accuracy_percent": round(accuracy, 2)
    })

    print(f"\n===== {name.upper()} =====")
    print(f"MAE: {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R²: {r2:.4f}")
    print(f"Accuracy: {accuracy:.2f}%")


# =====================================
# 8. CREATE RESULT FILE
# =====================================
result_df = pd.DataFrame({
    "likes_log1p_thucte": y_val["likes_log1p"].values,
    "views_log1p_thucte": y_val["views_log1p"].values,
    "shares_log1p_thucte": y_val["shares_log1p"].values,

    "likes_log1p_predict": pred[:, 0],
    "views_log1p_predict": pred[:, 1],
    "shares_log1p_predict": pred[:, 2]
})

# accuracy từng sample
result_df["likes_accuracy_%"] = (
    100 - np.abs(
        (
            result_df["likes_log1p_thucte"]
            - result_df["likes_log1p_predict"]
        )
        / result_df["likes_log1p_thucte"].replace(0, 1e-6)
    ) * 100
)

result_df["views_accuracy_%"] = (
    100 - np.abs(
        (
            result_df["views_log1p_thucte"]
            - result_df["views_log1p_predict"]
        )
        / result_df["views_log1p_thucte"].replace(0, 1e-6)
    ) * 100
)

result_df["shares_accuracy_%"] = (
    100 - np.abs(
        (
            result_df["shares_log1p_thucte"]
            - result_df["shares_log1p_predict"]
        )
        / result_df["shares_log1p_thucte"].replace(0, 1e-6)
    ) * 100
)

# save result
result_df.to_csv(
    result_path,
    index=False,
    encoding="utf-8-sig"
)

print(f"\nĐã lưu file result: {result_path}")


# =====================================
# 9. SAVE METRIC SUMMARY
# =====================================
metric_df = pd.DataFrame(metric_rows)

metric_summary_path = r"C:\Users\Admin\OneDrive\Desktop\LuanVan\DB_tiktok\Model_Metrics.csv"

metric_df.to_csv(
    metric_summary_path,
    index=False,
    encoding="utf-8-sig"
)

print(f"Đã lưu metric summary: {metric_summary_path}")

import pandas as pd
from sklearn.model_selection import train_test_split

# =====================================
# 1. Đọc file gốc
# =====================================
file_path = r"C:\Users\Admin\OneDrive\Desktop\LuanVan\DB_tiktok\Bản sao của tiktok_preprocessed_final.csv"

df = pd.read_csv(file_path, encoding="utf-8-sig")

print("Tổng số samples:", len(df))


# =====================================
# 2. Chia train / validate
# =====================================
train_df, val_df = train_test_split(
    df,
    test_size=0.2,
    random_state=42,
    shuffle=True
)

print("Train size:", len(train_df))
print("Validate size:", len(val_df))


# =====================================
# 3. Lưu file
# =====================================
train_path = r"C:\Users\Admin\OneDrive\Desktop\LuanVan\DB_tiktok\tiktok_train.csv"
val_path = r"C:\Users\Admin\OneDrive\Desktop\LuanVan\DB_tiktok\tiktok_validate.csv"

train_df.to_csv(train_path, index=False, encoding="utf-8-sig")
val_df.to_csv(val_path, index=False, encoding="utf-8-sig")

print("Đã tạo:")
print(train_path)
print(val_path)

# main.py
import os, joblib, warnings
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor

from constants import PATHS, TARGETS
from processor import TikTokDataProcessor

# Tắt các cảnh báo không cần thiết để log sạch hơn
warnings.filterwarnings("ignore")

SELECTED_FEATURES = [
    "followers", "ema_views_last_3", "avg_views_last_3_videos", "hist_like_rate",
    "days_since_last_post", "is_related_gameshow", "count_hashtag_famous", 
    "is_original_sound", "hashtag_count", "emoji_count", "hashtag_density", 
    "score_caption", "time_sin", "time_cos", "is_weekend"
]

class TikTokExpertSystem:
    """Hệ thống module chuyên gia dự đoán đa mục tiêu"""
    def __init__(self, model_dir="models"):
        self.model_dir = model_dir
        self.models = {}
        self.metrics = []
        os.makedirs(self.model_dir, exist_ok=True)

    def train_expert(self, target, X_train, y_train, X_val, y_val, params):
        """Huấn luyện một module chuyên gia cho một Target cụ thể"""
        model = XGBRegressor(**params)
        model.fit(X_train, y_train)
        
        # Dự đoán và tính toán hệ số
        y_pred = model.predict(X_val)
        r2 = r2_score(y_val, y_pred)
        mae = mean_absolute_error(y_val, y_pred)
        rmse = np.sqrt(mean_squared_error(y_val, y_pred))
        
        # Lưu kết quả
        self.models[target] = model
        self.metrics.append({
            "Target": target,
            "R2": f"{r2:.5f}",
            "MAE": f"{mae:.4f}",
            "RMSE": f"{rmse:.4f}"
        })
        
        # Lưu file model
        joblib.dump(model, os.path.join(self.model_dir, f"expert_{target}.pkl"))

    def show_final_report(self):
        """In bảng thông báo cuối cùng sạch sẽ"""
        print("\n" + "="*60)
        print("📊 FINAL MODULE SYSTEM PERFORMANCE REPORT")
        print("="*60)
        report_df = pd.DataFrame(self.metrics)
        print(report_df.to_string(index=False))
        print("="*60)
        print(f"✅ All models saved in: {self.model_dir}\n")

def main():
    # 1. Khởi tạo Processor và Manager
    print("[System] Loading data and initializing experts...")
    df_raw = pd.read_csv(PATHS["main_data"])
    proc = TikTokDataProcessor()
    proc.load_trends()
    
    # Xử lý Feature (Tắt log chi tiết bên trong nếu muốn sạch hơn)
    df_featured = proc.process_features(df_raw)
    
    # 2. Chia dữ liệu
    train_df, val_df = train_test_split(df_featured, test_size=0.2, random_state=42)
    X_train = train_df[SELECTED_FEATURES].fillna(0)
    X_val = val_df[SELECTED_FEATURES].fillna(0)

    manager = TikTokExpertSystem()

    # 3. Định nghĩa tham số tối ưu (Lấy từ kết quả Optuna tốt nhất của bạn)
    # Lưu ý: Bạn có thể chạy lại Optuna trước đó để lấy các bộ params này
    configs = {
        "likes_log1p": {
            'n_estimators': 1223, 'learning_rate': 0.0058, 'max_depth': 7, 
            'gamma': 0.49, 'reg_alpha': 3.14, 'reg_lambda': 2.17, 'random_state': 42
        },
        "views_log1p": {
            'n_estimators': 1703, 'learning_rate': 0.0061, 'max_depth': 5, 
            'gamma': 0.77, 'reg_alpha': 2.17, 'reg_lambda': 2.70, 'random_state': 42
        },
        "shares_log1p": {
            'n_estimators': 1500, 'learning_rate': 0.01, 'max_depth': 5, 
            'random_state': 42 # Shares khó lên nên dùng param cơ bản hoặc tune riêng
        }
    }

    # 4. Huấn luyện từng chuyên gia
    for target in TARGETS:
        params = configs.get(target, {'n_estimators': 1000, 'learning_rate': 0.01})
        manager.train_expert(target, X_train, train_df[target], X_val, val_df[target], params)

    # 5. Thông báo kết quả cuối cùng
    manager.show_final_report()

if __name__ == "__main__":
    main()
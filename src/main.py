import pandas as pd
import numpy as np
import joblib
import os
import re
from datetime import datetime

# Importing custom project components
from constants import PATHS, FEATURES, TARGETS, EMOJI_PATTERN
from processor import TikTokDataProcessor

# --- SECTION 1: PREPROCESSING FUNCTIONS ---

def force_clean_id(x):
    """Converts Scientific, Float, or String ID to a clean integer string."""
    if pd.isna(x) or x == "" or x is None: return ""
    try:
        return str(int(float(str(x).strip())))
    except:
        return re.sub(r'\D', '', str(x))

def extract_features_from_caption(caption):
    """Extract hashtags with #, word count, emoji count, and clean length."""
    # Tìm hashtag và nối lại kèm dấu #
    hashtags = re.findall(r'#(\w+)', caption)
    hashtag_str = " ".join([f"#{h}" for h in hashtags]).lower()
    
    hashtag_count = len(hashtags)
    emoji_count = len(EMOJI_PATTERN.findall(caption))
    
    # Làm sạch text để tính độ dài (67 ký tự)
    clean_text = re.sub(r'#\w+', '', caption) 
    clean_text = EMOJI_PATTERN.sub('', clean_text) 
    clean_text = re.sub(r'[^\w\s]', '', clean_text) 
    clean_text = " ".join(clean_text.split()) 
    
    word_count = len(clean_text.split()) if clean_text else 0
    clean_len = len(clean_text) 
    
    return {
        "hashtag_count": hashtag_count,
        "word_count": word_count,
        "caption_clean": clean_text.lower(),
        "hashtag_str": hashtag_str,
        "emoji_count": emoji_count,
        "caption_clean_len": clean_len
    }

# --- SECTION 2: PREDICTION & EVALUATION ---

def predict_and_save(raw_input, output_file="results.csv", debug_file="debug_features.csv"):
    print("\n" + "="*85)
    print("🚀 TIKTOK PREDICTION SYSTEM (FINAL VERSION)")
    print("="*85)

    # 1. Khởi tạo mặc định các tham số không truyền vào
    post_id = raw_input.get('post_id', "")
    actual_likes, actual_views, actual_shares = 0, 0, 0
    found_actual = False
    
    # 2. Tìm Actual nếu có post_id
    if post_id != "":
        try:
            df_val = pd.read_csv(PATHS["output_val"], dtype={'post_id': str})
            target_id = force_clean_id(post_id)
            df_val['post_id_match'] = df_val['post_id'].apply(force_clean_id)
            target_row = df_val[df_val['post_id_match'] == target_id]
            
            if not target_row.empty:
                actual_likes = float(target_row['likes'].values[0])
                actual_views = float(target_row['views'].values[0])
                actual_shares = float(target_row['shares'].values[0])
                found_actual = True
                print(f"✅ FOUND ACTUAL DATA for ID: {target_id}")
        except Exception as e:
            print(f"⚠️ Error accessing validation file: {e}")

    # 3. Trích xuất đặc trưng & Set mặc định
    caption_info = extract_features_from_caption(raw_input['caption'])
    full_case = {**raw_input, **caption_info}
    
    # Ép các cột cần thiết để Processor không báo lỗi KeyError
    full_case['media_url'] = "" 
    full_case['author_username'] = "unknown_user"
    full_case['avg_views_last_3_videos'] = 1219066.66666667
    full_case['ema_views_last_3'] = 1376992.53594568
    full_case['hist_like_rate'] = 0.0911047489889104
    full_case['days_since_last_post'] = 0.996342592592593
    full_case['views'] = actual_views
    full_case['likes'] = actual_likes
    
    df_input = pd.DataFrame([full_case])
    
    # 4. Chạy Processor
    processor = TikTokDataProcessor()
    processor.load_trends()
    processed_df = processor.process_features(df_input)
    
    # 5. Khôi phục/Ghi đè lại các giá trị để đúng logic single-case
    processed_df['caption_length'] = processed_df['caption_clean_len']
    processed_df['author_username'] = "unknown_user"
    for col in ["avg_views_last_3_videos", "ema_views_last_3", "hist_like_rate", "days_since_last_post"]:
        processed_df[col] = 0

    # 6. Lưu file Debug (Quan trọng)
    file_exists = os.path.isfile(debug_file)
    processed_df.to_csv(debug_file, mode='a', index=False, header=not file_exists, encoding='utf-8-sig')
    print(f"📁 Processed features logged to: {debug_file}")

    # 7. Dự đoán
    X_input = processed_df[FEATURES].reindex(columns=FEATURES)
    models_to_run = {
        "Linear Regression": r"C:\Users\Admin\OneDrive\Desktop\LinearRegression\src\models\tiktok_linear_regression_multi_model.pkl",
        "Random Forest": r"C:\Users\Admin\OneDrive\Desktop\LinearRegression\src\models\tiktok_random_forest_multi_model.pkl",
        "XGBoost": r"C:\Users\Admin\OneDrive\Desktop\LinearRegression\src\models\xgboost_multioutput_model.pkl"
    }
    
    all_predictions = []
    for name, path in models_to_run.items():
        if os.path.exists(path):
            try:
                model = joblib.load(path)
                log_pred = model.predict(X_input).flatten()
                
                # Convert log1p -> Real values
                p_l, p_v, p_s = np.expm1(log_pred[0]), np.expm1(log_pred[1]), np.expm1(log_pred[2])

                print(f"\n🔹 MODULE: {name}")
                if found_actual:
                    print(f"{'Metric':<10} | {'Actual':<15} | {'Predict':<15} | {'% Error':<10}")
                    print("-" * 65)
                    metrics = [("Likes", actual_likes, p_l), ("Views", actual_views, p_v), ("Shares", actual_shares, p_s)]
                    for label, act, pre in metrics:
                        err = (abs(act - pre) / act * 100) if act > 0 else 0
                        print(f"{label:<10} | {int(act):<15,} | {int(pre):<15,} | {err:.2f}%")
                else:
                    print(f"{'Metric':<15} | {'Predict (Value)':<20}")
                    print("-" * 40)
                    print(f"{'Likes':<15} | {int(p_l):<20,}")
                    print(f"{'Views':<15} | {int(p_v):<20,}")
                    print(f"{'Shares':<15} | {int(p_s):<20,}")

                all_predictions.append({
                    "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "Model": name, "pred_views": int(p_v), "pred_likes": int(p_l)
                })
            except Exception as e:
                print(f"⚠️ Model {name} error: {e}")

    if all_predictions:
        res_exists = os.path.isfile(output_file)
        pd.DataFrame(all_predictions).to_csv(output_file, mode='a', index=False, header=not res_exists, encoding='utf-8-sig')
        print(f"\n🚀 DONE! Prediction results added to {output_file}")

# --- SECTION 3: RUN ---

if __name__ == "__main__":
    my_input = {
        "caption": "Video mới nè mọi người xem nha #fyp #video",
        "music_name": "Original sound",
        "followers": 8500,
        "created_at": "2026-04-25 03:15:00+07:00"
    }
    
    predict_and_save(my_input)

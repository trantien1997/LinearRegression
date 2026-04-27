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
    """
    Forcefully converts any ID format (Scientific, Float, String) to a clean integer string.
    Example: '7.04488E+18' -> '7044882725949525249' (approx)
    """
    if pd.isna(x) or x == "": return ""
    try:
        # Convert to float first (to handle E+18), then to int, then to string
        return str(int(float(str(x).strip())))
    except:
        # If conversion fails, return cleaned string version
        return re.sub(r'\D', '', str(x))

def extract_username_smart(url):
    """Extract username from TikTok or Apify URL formats."""
    if not url or not isinstance(url, str):
        return "unknown_user"
    try:
        if '@' in url:
            return url.split('@')[1].split('/')[0]
        last_part = url.split('/')[-1]
        if '-' in last_part:
            parts = last_part.split('-')
            return parts[1] if len(parts) > 1 else last_part
        return last_part
    except:
        return "unknown_user"

def extract_features_from_caption(caption):
    """Extract hashtags, word count, emoji count, and clean length (67 chars)."""
    hashtags = re.findall(r'#(\w+)', caption)
    hashtag_str = " ".join(hashtags).lower()
    hashtag_count = len(hashtags)
    emoji_count = len(EMOJI_PATTERN.findall(caption))
    
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
    print("🚀 TIKTOK PREDICTION: SCIENTIFIC ID & ERROR ANALYSIS")
    print("="*85)

    # 1. Load Actual Values from Validation File
    actual_likes, actual_views, actual_shares = 0, 0, 0
    found_actual = False
    
    try:
        # Load validation data (Keep original IDs as object to prevent data loss)
        df_val = pd.read_csv(PATHS["output_val"], dtype={'post_id': str})
        
        # Clean both the input ID and the file's ID column
        target_id = force_clean_id(raw_input['post_id'])
        df_val['post_id_match'] = df_val['post_id'].apply(force_clean_id)
        
        # Exact match attempt
        target_row = df_val[df_val['post_id_match'] == target_id]
        
        # If exact match fails (due to precision loss in CSV), use fuzzy prefix match
        if target_row.empty and len(target_id) > 10:
            print(f"⚠️ Exact match failed for {target_id}. Attempting prefix match...")
            target_row = df_val[df_val['post_id_match'].str.startswith(target_id[:12])].head(1)

        if not target_row.empty:
            actual_likes = float(target_row['likes'].values[0])
            actual_views = float(target_row['views'].values[0])
            actual_shares = float(target_row['shares'].values[0])
            found_actual = True
            print(f"✅ SUCCESS: Found Actual data for ID: {target_id}")
            print(f"📊 Real Data -> Views: {actual_views:,.0f} | Likes: {actual_likes:,.0f}")
        else:
            print(f"❌ NOT FOUND: post_id {target_id} is missing in validation file.")
            print(f"💡 Sample IDs in file: {df_val['post_id_match'].head(3).tolist()}")
            
    except Exception as e:
        print(f"⚠️ Error accessing validation file: {e}")

    # 2. Feature Extraction
    caption_info = extract_features_from_caption(raw_input['caption'])
    full_case = {**raw_input, **caption_info}
    full_case['author_username'] = extract_username_smart(raw_input['media_url'])
    
    # Inject values for Processor stability
    full_case['views'] = actual_views
    full_case['likes'] = actual_likes
    
    df_input = pd.DataFrame([full_case])
    
    # 3. Momentum & Processor
    momentum_cols = ["avg_views_last_3_videos", "ema_views_last_3", "hist_like_rate", "days_since_last_post"]
    original_momentum = df_input[momentum_cols].copy()

    processor = TikTokDataProcessor()
    processor.load_trends()
    processed_df = processor.process_features(df_input)
    
    # 4. Restore manual values & fix length
    for col in momentum_cols:
        processed_df[col] = original_momentum[col].values
    processed_df['caption_length'] = processed_df['caption_clean_len']
    processed_df['author_username'] = full_case['author_username']

    # Save to Debug CSV
    processed_df.to_csv(debug_file, mode='a', index=False, header=not os.path.isfile(debug_file), encoding='utf-8-sig')

    # 5. Model Input Preparation
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
                
                # Convert log1p predictions back to Real Values
                p_l_real = np.expm1(log_pred[0])
                p_v_real = np.expm1(log_pred[1])
                p_s_real = np.expm1(log_pred[2])

                print(f"\n🔹 MODULE: {name}")
                print(f"{'Metric':<10} | {'Actual':<15} | {'Predict':<15} | {'% Error':<10}")
                print("-" * 65)
                
                def calc_error(act, pre):
                    if not found_actual or act == 0: return 0.0
                    return (abs(act - pre) / act * 100)

                metrics = [
                    ("Likes", actual_likes, p_l_real),
                    ("Views", actual_views, p_v_real),
                    ("Shares", actual_shares, p_s_real)
                ]

                for label, act, pre in metrics:
                    err = calc_error(act, pre)
                    act_str = f"{int(act):,}" if found_actual else "N/A"
                    err_str = f"{err:.2f}%" if found_actual else "N/A"
                    print(f"{label:<10} | {act_str:<15} | {int(pre):<15,} | {err_str:<10}")

                all_predictions.append({
                    "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "post_id": raw_input['post_id'],
                    "Model": name,
                    "pred_views": int(p_v_real),
                    "pred_likes": int(p_l_real),
                    "error_views_pct": round(calc_error(actual_views, p_v_real), 2) if found_actual else None
                })
            except Exception as e:
                print(f"⚠️ Model {name} error: {e}")

    if all_predictions:
        res_df = pd.DataFrame(all_predictions)
        res_df.to_csv(output_file, mode='a', index=False, header=not os.path.isfile(output_file), encoding='utf-8-sig')
        print(f"\n🚀 SUCCESS: Final results recorded in {output_file}")

# --- SECTION 3: RUN ---

if __name__ == "__main__":
    my_input = {
        # Using scientific notation string as per your request
        "post_id": "7.04488272594953E+018", 
        "caption": "#sponsored Nhảy 1 chút ở SG,cực nhiều sp như tui mặc đag sale ở Cotton On nhé😆 #mycottonon #cottononvietnam #endofseasonsale #streetfashion #fashion",
        "music_name": "Own brand freestyle transition",
        "followers": 16600000,
        "created_at": "2021-12-23 13:03:00+00:00",
        "media_url": "https://api.apify.com/v2/key-value-stores/awoyBmj2vRJYsw0RR/records/video-ciin-20211223130300-7044882725949525249.mp4",
        "avg_views_last_3_videos": 1219066.66666667,
        "ema_views_last_3": 1376992.53594568,
        "hist_like_rate": 0.0911047489889104,
        "days_since_last_post": 0.996342592592593
    }
    
    predict_and_save(my_input)

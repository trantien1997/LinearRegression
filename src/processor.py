# processor.py
import pandas as pd
import warnings
import re
import numpy as np
from transformers import pipeline
from constants import PATHS, EMOJI_PATTERN

warnings.filterwarnings("ignore")

def normalize_text(text):
    return str(text).lower().strip() if pd.notna(text) else ""

def count_emojis(text):
    if pd.isna(text): return 0
    return sum(len(match) for match in EMOJI_PATTERN.findall(str(text)))

def extract_username(url):
    if pd.isna(url): return "unknown"
    match = re.search(r'video-(.*?)-\d{14}', str(url))
    return match.group(1) if match else "unknown"

class TikTokDataProcessor:
    def __init__(self):
        self.trend_keywords = set()
        self.trend_hashtags = set()
        self.trend_songs = set()
        self.gameshow_hashtags = []
        self.famous_hashtags = [] 
        
        print("[Processor] Loading PhoBERT Sentiment Pipeline...")
        self.sentiment_analyzer = pipeline(
            "sentiment-analysis", 
            model="wonrax/phobert-base-vietnamese-sentiment", 
            tokenizer="wonrax/phobert-base-vietnamese-sentiment"
        )

    def get_phobert_score(self, text):
        if pd.isna(text) or str(text).strip() == "": return 0.0
        try:
            result = self.sentiment_analyzer(str(text), truncation=True, max_length=256)[0]
            label, confidence = result['label'], result['score']
            return confidence if label == 'POS' else (-confidence if label == 'NEG' else 0.0)
        except: return 0.0

    def load_trends(self):
        print("[Processor] Loading trend data...")
        for key in ["keyword_trend", "hashtag_trend", "song_trend"]:
            df = pd.read_csv(PATHS[key])
            for col in [c for c in df.columns if 'days' in c]:
                data = df[col].dropna().apply(normalize_text).tolist()
                if key == "keyword_trend": self.trend_keywords.update(data)
                elif key == "hashtag_trend": self.trend_hashtags.update(data)
                else: self.trend_songs.update(data)
        self.gameshow_hashtags = pd.read_csv(PATHS["gameshow"])["hashtag_gameshow"].dropna().astype(str).str.lower().tolist()
        self.famous_hashtags = pd.read_csv(PATHS["kol_to_gameshow"])["hashtag_famous"].dropna().astype(str).str.lower().tolist()

    def process_features(self, df):
        df = df.copy()
        print("[Processor] Extracting features...")
        
        df["author_username"] = df["media_url"].apply(extract_username)
        df["is_original_sound"] = df["music_name"].apply(lambda x: 1 if pd.notna(x) and any(k in str(x).lower() for k in ["nhạc nền -", "original sound"]) else 0)
        df["hashtag_density"] = df["hashtag_count"].fillna(0) / df["word_count"].fillna(1).clip(lower=1)
        df["emoji_count"] = df["caption"].apply(count_emojis)
        df["is_related_gameshow"] = df["hashtag_str"].apply(lambda x: 1 if any(t in str(x).lower() for t in self.gameshow_hashtags) else 0)
        df["count_hashtag_famous"] = df["hashtag_str"].apply(lambda x: sum(1 for t in self.famous_hashtags if t in str(x).lower()) if pd.notna(x) else 0)
        df["score_caption"] = df["caption_clean"].apply(self.get_phobert_score)

        df["created_at"] = pd.to_datetime(df["created_at"], utc=True, errors="coerce")
        df_vn = df["created_at"].dt.tz_convert("Asia/Ho_Chi_Minh")
        df["time_sin"] = np.sin(2 * np.pi * df_vn.dt.hour / 24)
        df["time_cos"] = np.cos(2 * np.pi * df_vn.dt.hour / 24)
        df["is_weekend"] = (df_vn.dt.dayofweek >= 5).astype(int)
        
        # --- DYNAMIC MOMENTUM (Đảm bảo có đủ 3 cột views) ---
        df = df.sort_values(by=["author_username", "created_at"])
        # 1. Simple Moving Average
        df["avg_views_last_3_videos"] = df.groupby("author_username")["views"].transform(lambda x: x.shift(1).rolling(3, min_periods=1).mean()).fillna(0)
        # 2. Exponential Moving Average
        df["ema_views_last_3"] = df.groupby("author_username")["views"].transform(lambda x: x.shift(1).ewm(span=3, adjust=False).mean()).fillna(0)
        # 3. Historical Like Rate
        df["like_rate_temp"] = df["likes"] / df["views"].replace(0, 1)
        df["hist_like_rate"] = df.groupby("author_username")["like_rate_temp"].transform(lambda x: x.shift(1).expanding().mean()).fillna(0)
        df["days_since_last_post"] = df.groupby("author_username")["created_at"].diff().dt.total_seconds().fillna(0) / 86400
        
        return df
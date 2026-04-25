# processor.py
import pandas as pd
from constants import PATHS, EMOJI_PATTERN

def normalize_text(text):
    return str(text).lower().strip() if pd.notna(text) else ""

def count_emojis(text):
    if pd.isna(text): return 0
    return sum(len(match) for match in EMOJI_PATTERN.findall(str(text)))

class TikTokDataProcessor:
    def __init__(self):
        self.trend_keywords = set()
        self.trend_hashtags = set()
        self.trend_songs = set()
        self.gameshow_hashtags = []

    def load_trends(self):
        # Load Keywords
        df_k = pd.read_csv(PATHS["keyword_trend"])
        for col in ["last_7_days", "last_30_days", "last_120_days"]:
            self.trend_keywords.update(df_k[col].dropna().apply(normalize_text).tolist())
        
        # Load Hashtags, Songs tương tự...
        df_h = pd.read_csv(PATHS["hashtag_trend"])
        for col in ["last_7_days", "last_30_days", "last_120_days"]:
            self.trend_hashtags.update(df_h[col].dropna().apply(normalize_text).tolist())

        df_s = pd.read_csv(PATHS["song_trend"])
        for col in ["song_last_7_days", "song_last_30_days", "song_last_120_days"]:
            self.trend_songs.update(df_s[col].dropna().apply(normalize_text).tolist())

        self.gameshow_hashtags = pd.read_csv(PATHS["gameshow"])["hashtag_gameshow"].dropna().astype(str).str.lower().tolist()

    def process_features(self, df):
        df = df.copy()
        df["emoji_count"] = df["caption"].apply(count_emojis)
        df["has_trend_keyword"] = df["caption_clean"].apply(lambda x: 1 if any(kw in normalize_text(x) for kw in self.trend_keywords if kw) else 0)
        df["has_trend_hashtag"] = df["hashtag_str"].apply(lambda x: 1 if any(ht in normalize_text(x) for ht in self.trend_hashtags if ht) else 0)
        df["has_trend_song"] = df["music_name"].apply(lambda x: 1 if normalize_text(x) in self.trend_songs else 0)
        df["is_related_gameshow"] = df["hashtag_str"].apply(lambda x: 1 if any(tag in str(x).lower() for tag in self.gameshow_hashtags) else 0)

        # Xử lý thời gian
        df["created_at"] = pd.to_datetime(df["created_at"], utc=True, errors="coerce")
        df_vn = df["created_at"].dt.tz_convert("Asia/Ho_Chi_Minh")
        df["time"] = df_vn.dt.hour
        df["is_weekend"] = (df_vn.dt.dayofweek >= 5).astype(int)
        
        return df

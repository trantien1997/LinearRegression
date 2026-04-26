# processor.py
import pandas as pd
import numpy as np
import warnings
import re
from transformers import pipeline
from constants import PATHS, EMOJI_PATTERN

warnings.filterwarnings("ignore")

# --- Utility Functions ---
def normalize_text(text):
    """Normalize text for consistency in trend matching."""
    return str(text).lower().strip() if pd.notna(text) else ""

def count_emojis(text):
    """Count number of emojis in a given text."""
    if pd.isna(text): return 0
    return sum(len(match) for match in EMOJI_PATTERN.findall(str(text)))

def extract_username(url):
    """Extract author username from TikTok media URL."""
    if pd.isna(url): return "unknown"
    match = re.search(r'video-(.*?)-\d{14}', str(url))
    return match.group(1) if match else "unknown"

class TikTokDataProcessor:
    def __init__(self):
        # Trend containers
        self.trend_keywords = set()
        self.trend_hashtags = set()
        self.trend_songs = set()
        self.gameshow_hashtags = []
        self.famous_hashtags = [] 
        
        # Initialize PhoBERT Sentiment Analysis
        print("[Processor] Loading PhoBERT Sentiment Pipeline...")
        self.sentiment_analyzer = pipeline(
            "sentiment-analysis", 
            model="wonrax/phobert-base-vietnamese-sentiment", 
            tokenizer="wonrax/phobert-base-vietnamese-sentiment"
        )

    def get_phobert_score(self, text):
        """Calculate sentiment score using PhoBERT (-1 to 1)."""
        if pd.isna(text) or str(text).strip() == "": return 0.0
        try:
            result = self.sentiment_analyzer(str(text), truncation=True, max_length=256)[0]
            label, confidence = result['label'], result['score']
            return confidence if label == 'POS' else (-confidence if label == 'NEG' else 0.0)
        except: return 0.0

    def load_trends(self):
        """Load and consolidate all trend data from CSV files."""
        print("[Processor] Loading trend data...")
        
        # 1. Load Keywords
        df_k = pd.read_csv(PATHS["keyword_trend"])
        for col in ["last_7_days", "last_30_days", "last_120_days"]:
            if col in df_k.columns:
                self.trend_keywords.update(df_k[col].dropna().apply(normalize_text).tolist())
        
        # 2. Load Hashtags
        df_h = pd.read_csv(PATHS["hashtag_trend"])
        for col in ["last_7_days", "last_30_days", "last_120_days"]:
            if col in df_h.columns:
                self.trend_hashtags.update(df_h[col].dropna().apply(normalize_text).tolist())

        # 3. Load Songs
        df_s = pd.read_csv(PATHS["song_trend"])
        for col in ["song_last_7_days", "song_last_30_days", "song_last_120_days"]:
            if col in df_s.columns:
                self.trend_songs.update(df_s[col].dropna().apply(normalize_text).tolist())

        # 4. Load Gameshow and KOL lists
        self.gameshow_hashtags = pd.read_csv(PATHS["gameshow"])["hashtag_gameshow"].dropna().astype(str).str.lower().tolist()
        if "kol_to_gameshow" in PATHS:
            self.famous_hashtags = pd.read_csv(PATHS["kol_to_gameshow"])["hashtag_famous"].dropna().astype(str).str.lower().tolist()

    def process_features(self, df):
        """Main pipeline to merge old and new features."""
        df = df.copy()
        print("[Processor] Extracting features...")
        
        # --- Basic Content Features ---
        df["author_username"] = df["media_url"].apply(extract_username)
        df["emoji_count"] = df["caption"].apply(count_emojis)
        df["hashtag_density"] = df["hashtag_count"].fillna(0) / df["word_count"].fillna(1).clip(lower=1)
        
        # --- Trend Matching (Old Features) ---
        df["has_trend_keyword"] = df["caption_clean"].apply(lambda x: 1 if any(kw in normalize_text(x) for kw in self.trend_keywords if kw) else 0)
        df["has_trend_hashtag"] = df["hashtag_str"].apply(lambda x: 1 if any(ht in normalize_text(x) for ht in self.trend_hashtags if ht) else 0)
        df["has_trend_song"] = df["music_name"].apply(lambda x: 1 if normalize_text(x) in self.trend_songs else 0)
        
        # --- Special Tags (New Features) ---
        df["is_original_sound"] = df["music_name"].apply(lambda x: 1 if pd.notna(x) and any(k in str(x).lower() for k in ["nhạc nền -", "original sound"]) else 0)
        df["is_related_gameshow"] = df["hashtag_str"].apply(lambda x: 1 if any(tag in str(x).lower() for tag in self.gameshow_hashtags) else 0)
        df["count_hashtag_famous"] = df["hashtag_str"].apply(lambda x: sum(1 for t in self.famous_hashtags if t in str(x).lower()) if pd.notna(x) else 0)

        # --- NLP Sentiment (Advanced) ---
        print("[Processor] Running PhoBERT analysis...")
        df["score_caption"] = df["caption_clean"].apply(self.get_phobert_score)

        # --- Time Engineering (Cyclical & Categorical) ---
        df["created_at"] = pd.to_datetime(df["created_at"], utc=True, errors="coerce")
        df_vn = df["created_at"].dt.tz_convert("Asia/Ho_Chi_Minh")
        
        df["hour"] = df_vn.dt.hour # Linear hour
        df["time_sin"] = np.sin(2 * np.pi * df_vn.dt.hour / 24) # Cyclical time
        df["time_cos"] = np.cos(2 * np.pi * df_vn.dt.hour / 24)
        df["is_weekend"] = (df_vn.dt.dayofweek >= 5).astype(int)
        
        # --- Dynamic Momentum (Grouped by Author) ---
        print("[Processor] Calculating author momentum...")
        df = df.sort_values(by=["author_username", "created_at"])
        
        # 1. Simple Moving Average
        df["avg_views_last_3_videos"] = df.groupby("author_username")["views"].transform(lambda x: x.shift(1).rolling(3, min_periods=1).mean()).fillna(0)
        # 2. Exponential Moving Average (Better for Trends)
        df["ema_views_last_3"] = df.groupby("author_username")["views"].transform(lambda x: x.shift(1).ewm(span=3, adjust=False).mean()).fillna(0)
        # 3. Historical Like Rate (Author's past quality)
        df["like_rate_temp"] = df["likes"] / df["views"].replace(0, 1)
        df["hist_like_rate"] = df.groupby("author_username")["like_rate_temp"].transform(lambda x: x.shift(1).expanding().mean()).fillna(0)
        # 4. Posting frequency
        df["days_since_last_post"] = df.groupby("author_username")["created_at"].diff().dt.total_seconds().fillna(0) / 86400
        
        # Drop temporary columns
        df = df.drop(columns=["like_rate_temp"])
        
        return df
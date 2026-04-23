import pandas as pd
import re

# Read main TikTok posts data
df_posts = pd.read_csv(r"C:\Users\Admin\OneDrive\Desktop\LuanVan\DB_tiktok\Bản sao của tiktok_preprocessed_final.csv")

# Read keyword trend data
df_keyword_trend = pd.read_csv(r"C:\Users\Admin\OneDrive\Desktop\LuanVan\DB_Trend_keywords_tiktok\tiktok_keyword_insights_vn_rank_keyword_7_30_120_21-04-2026.csv")

# Read song trend data
df_song_trend = pd.read_csv(r"C:\Users\Admin\OneDrive\Desktop\LuanVan\DB_trend_song_tiktok\songs_rank_7_30_120_12-01-2026.csv")

# Read hashtag trend data
df_hashtag_trend = pd.read_csv(r"C:\Users\Admin\OneDrive\Desktop\LuanVan\DB_trend_hastag_tiktok\hashtags_rank_7_30_120_21-04-2026.csv")

# Read gameshow data
df_gameshow = pd.read_csv(r"C:\Users\Admin\OneDrive\Desktop\LuanVan\Gameshow\Data_gameshow.csv")

# =================================
# 1. Helper normalize
# =================================
def normalize_text(text):
    if pd.isna(text):
        return ""
    return str(text).lower().strip()


# =================================
# 2. Emoji count
# =================================
EMOJI_PATTERN = re.compile(
    "["
    "\U0001F600-\U0001F64F"
    "\U0001F300-\U0001F5FF"
    "\U0001F680-\U0001F6FF"
    "\U0001F1E0-\U0001F1FF"
    "\U00002700-\U000027BF"
    "\U0001F900-\U0001F9FF"
    "\U00002600-\U000026FF"
    "]+",
    flags=re.UNICODE
)

def count_emojis(text):
    if pd.isna(text):
        return 0

    text = str(text)
    matches = EMOJI_PATTERN.findall(text)

    return sum(len(match) for match in matches)


# =================================
# 3. Build trend keyword set
# =================================
trend_keywords = set()

for col in ["last_7_days", "last_30_days", "last_120_days"]:
    trend_keywords.update(
        df_keyword_trend[col]
        .dropna()
        .apply(normalize_text)
        .tolist()
    )


# =================================
# 4. Build trend hashtag set
# =================================
trend_hashtags = set()

for col in ["last_7_days", "last_30_days", "last_120_days"]:
    trend_hashtags.update(
        df_hashtag_trend[col]
        .dropna()
        .apply(normalize_text)
        .tolist()
    )


# =================================
# 5. Build trend song set
# =================================
trend_songs = set()

for col in ["song_last_7_days", "song_last_30_days", "song_last_120_days"]:
    trend_songs.update(
        df_song_trend[col]
        .dropna()
        .apply(normalize_text)
        .tolist()
    )


# =================================
# 6. Build trend gameshow set
# =================================
gameshow_hashtags = (
    df_gameshow["hashtag_gameshow"]
    .dropna()
    .astype(str)
    .str.lower()
    .tolist()
)

# =================================
# 7. Feature functions
# =================================
def has_trend_keyword(row):
    text = (
        normalize_text(row["caption_clean"])
    )

    for keyword in trend_keywords:
        if keyword and keyword in text:
            return 1

    return 0


def has_trend_hashtag(row):
    hashtag_text = normalize_text(row["hashtag_str"])

    for hashtag in trend_hashtags:
        if hashtag and hashtag in hashtag_text:
            return 1

    return 0


def has_trend_song(music_name):
    music_name = normalize_text(music_name)
    return 1 if music_name in trend_songs else 0

def is_related_gameshow(hashtag_str):
    if pd.isna(hashtag_str):
        return 0

    text = str(hashtag_str).lower()

    for tag in gameshow_hashtags:
        if tag in text:
            return 1

    return 0

# =================================
# 8. Add new columns
# =================================
df_posts["emoji_count"] = df_posts["caption"].apply(count_emojis)

df_posts["has_trend_keyword"] = df_posts.apply(
    has_trend_keyword,
    axis=1
)

df_posts["has_trend_hashtag"] = df_posts.apply(
    has_trend_hashtag,
    axis=1
)

df_posts["has_trend_song"] = df_posts["music_name"].apply(
    has_trend_song
)

df_posts["is_related_gameshow"] = df_posts["hashtag_str"].apply(
    is_related_gameshow
)

# =================================
# 9 Time features
# =================================
df_posts["created_at"] = pd.to_datetime(
    df_posts["created_at"],
    utc=True,
    errors="coerce"
)

df_posts["created_at_vn"] = df_posts["created_at"].dt.tz_convert(
    "Asia/Ho_Chi_Minh"
)

df_posts["time"] = df_posts["created_at_vn"].dt.hour

df_posts["is_weekend"] = (
    df_posts["created_at_vn"].dt.dayofweek >= 5
).astype(int)

# ==========================================
# 10. SAVE RESULT
# ==========================================
df_posts.to_csv(
    r"C:\Users\Admin\OneDrive\Desktop\LuanVan\DB_tiktok\tiktok_with_features.csv",
    index=False,
    encoding="utf-8-sig"
)

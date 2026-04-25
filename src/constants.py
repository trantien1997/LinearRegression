import re

# Đường dẫn dữ liệu
PATHS = {
    "main_data": r"..\DB_tiktok\Tiktok_preprocessed_final.csv",
    "keyword_trend": r"..\DB_Trend_keywords_tiktok\tiktok_keyword_insights_vn_rank_keyword_7_30_120_21-04-2026.csv",
    "song_trend": r"..\DB_trend_song_tiktok\songs_rank_7_30_120_12-01-2026.csv",
    "hashtag_trend": r"..\DB_trend_hastag_tiktok\hashtags_rank_7_30_120_21-04-2026.csv",
    "gameshow": r"..\Gameshow\Data_gameshow.csv",
    "output_train": r"..\DB_tiktok\tiktok_train.csv",
    "output_val": r"..\DB_tiktok\tiktok_validate.csv",
    "output_result": r"..\Result\Result_100_42.csv"
}

# Regex cho Emoji
EMOJI_PATTERN = re.compile(
    r"[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF"
    r"\U0001F1E0-\U0001F1FF\U00002700-\U000027BF\U0001F900-\U0001F9FF"
    r"\U00002600-\U000026FF]+", 
    flags=re.UNICODE
)

# Danh sách feature dùng để train model
FEATURES = [
    "caption_length",
    "word_count",
    "emoji_count",
    "hashtag_count",
    "followers",
    "has_trend_keyword",
    "has_trend_hashtag", 
    "has_trend_song", 
    "is_related_gameshow", 
    "time", 
    "is_weekend"
]

# Các cột mục tiêu (Target)
TARGETS = [
    "likes_log1p",
    "views_log1p",
    "shares_log1p"
]

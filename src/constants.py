import re

# Đường dẫn dữ liệu
PATHS = {
    "main_data": r"..\DB_tiktok\Tiktok_preprocessed_final.csv",
    "keyword_trend": r"..\DB_Trend_keywords_tiktok\tiktok_keyword_insights_vn_rank_keyword_7_30_120_21-04-2026.csv",
    "song_trend": r"..\DB_trend_song_tiktok\songs_rank_7_30_120_12-01-2026.csv",
    "hashtag_trend": r"..\DB_trend_hastag_tiktok\hashtags_rank_7_30_120_21-04-2026.csv",
    "gameshow": r"..\Gameshow\Data_gameshow.csv",
    "kol_to_gameshow": r"..\Gameshow\kol_to_gameshows.csv",
    "output_train": r"..\DB_tiktok\tiktok_train.csv",
    "output_val": r"..\DB_tiktok\tiktok_validate.csv",
    "output_result_xgboost": r"..\Result\Result_xgboost.csv",
    "output_result_random_forest": r"..\Result\Result_random_forest.csv",
    "output_result_linear_regression": r"..\Result\Result_linear_regression.csv"
}

# Regex pattern to match emojis in text
EMOJI_PATTERN = re.compile(
    r"[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF"
    r"\U0001F1E0-\U0001F1FF\U00002700-\U000027BF\U0001F900-\U0001F9FF"
    r"\U00002600-\U000026FF]+", 
    flags=re.UNICODE
)

# High-impact feature set for multi-target prediction
FEATURES = [
    "followers",
    "ema_views_last_3",
    "avg_views_last_3_videos",
    "hist_like_rate",
    "days_since_last_post",
    "is_related_gameshow",
    "count_hashtag_famous", 
    "is_original_sound",
    "hashtag_count",
    "emoji_count",
    "hashtag_density", 
    "score_caption",
    "time_sin",
    "time_cos",
    "is_weekend",
    # Old features for trend matching (for interpretability)
    "caption_length",
    "word_count",
    "has_trend_keyword",
    "has_trend_hashtag", 
    "has_trend_song",
    "hour"
]
# Objective targets
TARGETS = [
    "likes_log1p",
    "views_log1p",
    "shares_log1p"
]

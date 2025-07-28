import pandas as pd

# 讀取資料
df = pd.read_csv("youtube_trending.csv")

# 把時間轉換成時間格式
df["collected_at"] = pd.to_datetime(df["collected_at"])

# 只保留必要欄位
df = df[["video_id", "title", "views", "likes", "comments", "collected_at"]]

# 排序方便比對
df = df.sort_values(by=["video_id", "collected_at"])

# 計算成長率
df["view_growth"] = df.groupby("video_id")["views"].pct_change().fillna(0)
df["like_growth"] = df.groupby("video_id")["likes"].pct_change().fillna(0)
df["comment_growth"] = df.groupby("video_id")["comments"].pct_change().fillna(0)

# 加總成一個簡單趨勢分數（可以調整公式）
df["trend_score"] = (
    df["view_growth"] * 0.6 +
    df["like_growth"] * 0.2 +
    df["comment_growth"] * 0.2
)

# 把最高分的前幾筆列出來
top_trending = df.sort_values(by="trend_score", ascending=False).head(10)

print("🔥 熱度成長最快的影片：")
print(top_trending[["title", "trend_score", "view_growth", "like_growth", "comment_growth", "collected_at"]])

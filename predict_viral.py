import pandas as pd
import numpy as np
import joblib
from sentence_transformers import SentenceTransformer

# 1. 載入資料
print("🚀 載入資料與模型...")
df = pd.read_csv("youtube_trending.csv")
df["collected_at"] = pd.to_datetime(df["collected_at"])
df = df[["video_id", "title", "views", "likes", "comments", "collected_at"]]
df = df.sort_values(by=["video_id", "collected_at"])

# 2. 計算成長率
df["view_growth"] = df.groupby("video_id")["views"].pct_change().fillna(0)
df["like_growth"] = df.groupby("video_id")["likes"].pct_change().fillna(0)
df["comment_growth"] = df.groupby("video_id")["comments"].pct_change().fillna(0)

# 3. 計算推薦分數
print("📡 計算推薦相似度...")
model = SentenceTransformer('all-MiniLM-L6-v2')
df["full_text"] = df["title"].fillna('')
df["recommend_score"] = df["full_text"].apply(
    lambda text: np.inner(model.encode(text), model.encode("最新科技產品、3C開箱、TikTok爆紅、Amazon好物"))
)

# 4. 準備特徵
features = ["view_growth", "like_growth", "comment_growth", "recommend_score"]
X = df[features]

# 5. 載入模型
clf = joblib.load("model.pkl")

# 6. 預測爆紅機率與標籤
df["predict_is_viral"] = clf.predict(X)
df["predict_proba"] = clf.predict_proba(X)[:, 1]  # 預測機率（爆紅的可能性）

# 7. 篩選高機率影片（可以自行調整門檻）
viral_predictions = df[df["predict_proba"] > 0.8].sort_values(by="predict_proba", ascending=False)

# 8. 輸出
print("\n🔥 預測有機會爆紅的影片：\n")
print(viral_predictions[["title", "predict_proba", "view_growth", "collected_at"]])

viral_predictions.to_csv("predicted_viral_candidates.csv", index=False)
print("\n✅ 結果已儲存為 predicted_viral_candidates.csv")

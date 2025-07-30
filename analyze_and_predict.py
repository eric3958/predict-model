import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# ✅ 載入 Sentence-Transformer 模型（語意向量）
print("🚀 載入本地語意模型中...")
model = SentenceTransformer('all-MiniLM-L6-v2')  # 僅需載一次，自動快取

# ---------- STEP 1：讀取資料 ----------
df = pd.read_csv("youtube_trending.csv")
df["collected_at"] = pd.to_datetime(df["collected_at"])
df = df[["video_id", "title", "views", "likes", "comments", "collected_at"]]
df = df.sort_values(by=["video_id", "collected_at"])

# ---------- STEP 2：計算 trend_score ----------
df["view_growth"] = df.groupby("video_id")["views"].pct_change().fillna(0)
df["like_growth"] = df.groupby("video_id")["likes"].pct_change().fillna(0)
df["comment_growth"] = df.groupby("video_id")["comments"].pct_change().fillna(0)

df["trend_score"] = (
    df["view_growth"] * 0.6 +
    df["like_growth"] * 0.2 +
    df["comment_growth"] * 0.2
)

# ---------- STEP 3：將影片轉為語意向量 ----------
print("📡 產生影片語意向量中...")
df["full_text"] = df["title"].fillna('')
df["embedding"] = df["full_text"].apply(lambda text: model.encode(text))

# ---------- STEP 4：定義「科技族群興趣」語意向量 ----------
tech_interest_text = """
最新科技產品、智慧型手機、藍牙耳機、充電設備、Amazon 熱賣、TikTok 推薦、行動配件、科技開箱、3C新品
"""
tech_vector = model.encode(tech_interest_text)

# ---------- STEP 5：計算推薦潛力（語意相似度） ----------
def compute_similarity(vec1, vec2):
    try:
        return cosine_similarity([vec1], [vec2])[0][0]
    except:
        return 0

df["recommend_score"] = df["embedding"].apply(lambda vec: compute_similarity(vec, tech_vector))

# ---------- STEP 6：找出高熱度 + 高相似度影片 ----------
hot_candidates = df[
    (df["trend_score"] > 0.2) &
    (df["recommend_score"] > 0.85)
].sort_values(by="trend_score", ascending=False)

# ---------- 顯示與輸出 ----------
print("\n🔥 潛在推薦熱門影片（科技產品類）：\n")
print(hot_candidates[["title", "trend_score", "recommend_score", "view_growth", "collected_at"]])

from datetime import datetime

today = datetime.now().strftime("%Y%m%d")
filename = f"viral_candidates_tech_{today}.csv"
hot_candidates.to_csv(filename, index=False)
print(f"\n✅ 結果已儲存到 {filename}")


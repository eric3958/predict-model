import pandas as pd
import numpy as np
import joblib
from sentence_transformers import SentenceTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

print("🚀 載入資料中...")
df = pd.read_csv("youtube_trending.csv")
df["collected_at"] = pd.to_datetime(df["collected_at"])
df = df[["video_id", "title", "views", "likes", "comments", "collected_at"]]
df = df.sort_values(by=["video_id", "collected_at"])

# 計算成長率
df["view_growth"] = df.groupby("video_id")["views"].pct_change().fillna(0)
df["like_growth"] = df.groupby("video_id")["likes"].pct_change().fillna(0)
df["comment_growth"] = df.groupby("video_id")["comments"].pct_change().fillna(0)

# 趨勢分數
df["trend_score"] = (
    df["view_growth"] * 0.6 +
    df["like_growth"] * 0.2 +
    df["comment_growth"] * 0.2
)

# 語意模型
print("📡 載入語意模型...")
model = SentenceTransformer('all-MiniLM-L6-v2')
df["full_text"] = df["title"].fillna('')
df["recommend_score"] = df["full_text"].apply(
    lambda text: np.inner(model.encode(text), model.encode("最新科技產品、3C開箱、TikTok爆紅、Amazon好物"))
)

# 建立 label：視為爆紅的定義（你可以調整條件）
df["label"] = (df["view_growth"] > 0.3).astype(int)

# 特徵欄位
features = ["view_growth", "like_growth", "comment_growth", "recommend_score"]
X = df[features]
y = df["label"]

# 分割資料
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 建立並訓練模型
print("🧠 訓練模型中...")
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# 模型評估
y_pred = clf.predict(X_test)
print("\n📊 模型評估結果：\n")
print(classification_report(y_test, y_pred))

# 儲存模型
joblib.dump(clf, "model.pkl")
print("\n✅ 模型已儲存為 model.pkl")


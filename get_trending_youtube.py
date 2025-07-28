from googleapiclient.discovery import build
import pandas as pd
from datetime import datetime

api_key = "AIzaSyDLbzP1NJrrwYKe7imwUxXbLkouwpB7du8"  # 改回你剛才的 key
youtube = build('youtube', 'v3', developerKey=api_key)

request = youtube.videos().list(
    part="snippet,statistics",
    chart="mostPopular",
    maxResults=10,
    regionCode="US"
)

response = request.execute()

video_data = []

for item in response['items']:
    video_id = item['id']
    title = item['snippet']['title']
    channel = item['snippet']['channelTitle']
    published = item['snippet']['publishedAt']
    views = item['statistics'].get('viewCount', 0)
    likes = item['statistics'].get('likeCount', 0)
    comments = item['statistics'].get('commentCount', 0)
    link = f"https://www.youtube.com/watch?v={video_id}"
    timestamp = datetime.now().isoformat()

    video_data.append({
        "video_id": video_id,
        "title": title,
        "channel": channel,
        "published_at": published,
        "views": int(views),
        "likes": int(likes),
        "comments": int(comments),
        "url": link,
        "collected_at": timestamp
    })

# 存成 CSV（追加模式）
df = pd.DataFrame(video_data)
df.to_csv("youtube_trending.csv", index=False, mode='a', header=not pd.io.common.file_exists("youtube_trending.csv"))

print("✅ 已儲存熱門影片資料！")

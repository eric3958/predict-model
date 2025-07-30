import subprocess

print("🧠 開始執行 YouTube 熱門影片抓取...")
subprocess.run(["python", "get_trending_youtube.py"])

print("\n🔍 執行趨勢分析與語意推薦...")
subprocess.run(["python", "analyze_and_predict.py"])

print("\n🎉 所有流程完成，請查看 viral_candidates_xxx.csv 檔案")


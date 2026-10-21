from ultralytics import YOLO

def main():
    # 1. 載入模型（初次執行會自動下載輕量版的預訓練模型 yolo11n.pt 或對應版本）
    model = YOLO("yolo11n.pt")

    # 2. 開始訓練
    results = model.train(
        data="data.yaml",    # 指向根目錄的設定檔
        epochs=50,           # 訓練輪數（可先設 50 測試，之後再依收斂狀況調整）
        imgsz=640,           # 輸入圖片解析度
        batch=16             # 批次大小（如果電腦顯示卡 VRAM 較小，可改為 8）
    )

if __name__ == '__main__':
    main()
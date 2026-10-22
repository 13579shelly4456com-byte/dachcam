from ultralytics import YOLO

def main():
    # 載入你上次訓練好的 best.pt 當作起手式，讓模型「溫故知新」
    model = YOLO("runs/detect/train-3/weights/best.pt")

    # 開始新一輪訓練
    results = model.train(
        data="data.yaml",    
        epochs=30,           # 增量訓練通常不需要設太多輪（例如 30~50 輪即可）
        imgsz=640,           
        batch=16             
    )

if __name__ == '__main__':
    main()
from ultralytics import YOLO

def main():
    # 1. 載入你最新訓練好的權重（請確認資料夾編號是 train-3）
    model = YOLO("runs/detect/train-3/weights/best.pt")

    # 2. 對 test 資料夾裡面的圖片進行預測
    # save=True 會把框好物件的照片自動存到 runs/detect/predict 資料夾中
    results = model(source="dataset/test/images", save=True) 

    print("測試完成！結果已經儲存至 runs/detect/predict 底下！")

if __name__ == '__main__':
    main()
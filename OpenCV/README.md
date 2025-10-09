# OpenCV 學習專案使用指南

## 📚 專案概述

此專案包含 OpenCV 計算機視覺學習教材，包括教學筆記本、練習題、預訓練模型和 dlib 機器學習專案。

## 📁 目錄結構

```
OpenCV/
├── 📚 教學筆記本
│   ├── Day0_py_np.ipynb       # Python 與 NumPy 基礎
│   ├── Day1_OpenCV.ipynb      # OpenCV 基礎操作
│   ├── Day2_OpenCV.ipynb      # 進階圖像處理
│   └── Day3_OpenCV.ipynb      # 特徵檢測與描述
│
├── 📝 作業練習 (HW/)
│   ├── Q_02.ipynb             # 色彩空間與繪圖練習
│   ├── Q_04_08.ipynb          # 圖像變換練習
│   ├── Q_09_12.ipynb          # 濾波與邊緣檢測
│   └── Q_15.ipynb             # 綜合應用練習
│
├── 🖼️ 範例圖片 (image/)
│   ├── 人臉檢測圖片          # dlib*.jpg, face*.jpg
│   ├── 圖像處理範例          # lena*.jpg, test*.jpg
│   ├── 特徵檢測圖片          # box.png, coins.jpg
│   └── 機器學習訓練圖片      # 各種物件圖片
│
├── 🎬 範例影片 (video/)
│   ├── car_chase_*.mp4        # 車輛追蹤範例
│   ├── Alec_Baldwin.mp4       # 人臉識別範例
│   └── overpass.mp4           # 場景分析範例
│
├── 🤖 預訓練模型 (model/)
│   ├── dlib 模型
│   ├── Haar 級聯分類器
│   └── DNN 模型
│
├── 📦 安裝檔案 (install/)
│   ├── dlib-19.22.99-cp39-cp39-win_amd64.whl
│   ├── chi_sim.traineddata    # 簡體中文 OCR 模型
│   └── chi_tra.traineddata    # 繁體中文 OCR 模型
│
└── 🔬 dlib 機器學習專案
    ├── dlib_ObjectCategories10/  # 訓練資料集
    ├── dlib_Annotations10/       # 標註檔案
    └── dlib_output/              # 輸出模型
```

## 🚀 快速開始

### 1. 環境需求
```bash
# 基本依賴
pip install opencv-python numpy matplotlib jupyter

# 可選依賴
pip install dlib face-recognition pytesseract
```

### 2. 運行教學筆記本
```bash
# 啟動 Jupyter Notebook
jupyter notebook

# 開啟教學檔案
# Day0_py_np.ipynb -> Python 基礎
# Day1_OpenCV.ipynb -> OpenCV 入門
```

### 3. 完成練習作業
```bash
# 按順序完成 HW/ 資料夾中的練習
# Q_02.ipynb -> 基礎圖像操作
# Q_04_08.ipynb -> 圖像變換
# Q_09_12.ipynb -> 濾波處理
# Q_15.ipynb -> 綜合應用
```

## 📖 學習路徑

### 初學者路徑
1. **Day0_py_np.ipynb** - Python 與 NumPy 基礎
2. **Day1_OpenCV.ipynb** - OpenCV 基本操作
3. **Q_02.ipynb** - 色彩空間練習
4. **Q_04_08.ipynb** - 圖像變換練習

### 進階路徑
1. **Day2_OpenCV.ipynb** - 圖像處理技術
2. **Q_09_12.ipynb** - 濾波與邊緣檢測
3. **Day3_OpenCV.ipynb** - 特徵檢測
4. **Q_15.ipynb** - 綜合應用

### 專案實戰
1. **dlib 人臉識別專案** - 使用 model/ 中的人臉模型
2. **物件分類專案** - 使用 dlib_ObjectCategories10 資料集
3. **影片處理專案** - 使用 video/ 中的範例影片

## 🔧 常用功能

### 圖像讀取與顯示
```python
import cv2
import numpy as np

# 讀取圖片
img = cv2.imread('image/lenaColor.png')

# 顯示圖片
cv2.imshow('Image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

### 人臉檢測
```python
# 載入人臉檢測器
face_cascade = cv2.CascadeClassifier('model/haarcascade_frontalface_default.xml')

# 檢測人臉
faces = face_cascade.detectMultiScale(gray, 1.3, 5)
```

### dlib 人臉特徵點
```python
import dlib

# 載入模型
predictor = dlib.shape_predictor('model/shape_predictor_68_face_landmarks.dat')
detector = dlib.get_frontal_face_detector()
```

## 📝 注意事項

1. **路徑問題**: 確保圖片和模型路徑正確
2. **依賴安裝**: 某些功能需要額外安裝 dlib 或 face-recognition
3. **記憶體使用**: 處理大圖片或影片時注意記憶體使用
4. **中文路徑**: 避免使用中文路徑名稱

## 🆘 常見問題

### Q: dlib 安裝失敗？
A: 使用提供的 wheel 檔案：`pip install install/dlib-19.22.99-cp39-cp39-win_amd64.whl`

### Q: 圖片無法顯示？
A: 檢查圖片路徑，使用相對路徑：`cv2.imread('image/filename.jpg')`

### Q: 中文 OCR 無法使用？
A: 將 tessdata 檔案複製到 Tesseract 安裝目錄

## 📚 相關資源

- [OpenCV 官方文檔](https://docs.opencv.org/)
- [dlib 官方網站](http://dlib.net/)
- [計算機視覺學習資源](https://github.com/jbhuang0604/awesome-computer-vision)

## 📞 技術支援

如有問題，請參考：
1. `INSTALLATION.md` - 安裝指南
2. `DLIB_GUIDE.md` - dlib 專項說明
3. `REORGANIZATION.md` - 資料夾重組指南
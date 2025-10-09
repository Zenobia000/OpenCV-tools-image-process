# dlib 機器學習專項使用指南

## 🎯 概述

dlib 是一個包含機器學習算法和工具的 C++ 工具包，特別擅長人臉檢測、人臉識別和物件分類。本專案包含完整的 dlib 學習資源和實際應用範例。

## 📁 dlib 相關檔案結構

### 🗂️ 資料集檔案
```
dlib_ObjectCategories10/          # 物件分類訓練資料集
├── accordion/                   # 手風琴類別 (55張圖片)
│   └── image_*.jpg
├── camera/                      # 相機類別 (49張圖片)
│   └── image_*.jpg
└── [其他8個類別...]
```

### 🏷️ 標註檔案
```
dlib_Annotations10/              # 對應的物件位置標註
├── accordion/                   # 手風琴標註檔案
├── camera/                      # 相機標註檔案
└── [其他類別標註...]
```

### 🤖 輸出模型
```
dlib_output/
└── model.svm                    # 訓練完成的 SVM 分類器
```

## 🚀 dlib 核心功能

### 1. 人臉檢測與識別

#### 基本人臉檢測
```python
import dlib
import cv2
import numpy as np

# 初始化檢測器
detector = dlib.get_frontal_face_detector()

# 讀取圖片
img = cv2.imread('image/dlib01.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 檢測人臉
faces = detector(gray)

# 繪製檢測結果
for face in faces:
    x, y, w, h = face.left(), face.top(), face.width(), face.height()
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

cv2.imshow('Face Detection', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

#### 68個人臉關鍵點檢測
```python
# 載入68點預測器
predictor = dlib.shape_predictor('model/shape_predictor_68_face_landmarks.dat')

# 檢測關鍵點
for face in faces:
    landmarks = predictor(gray, face)

    # 繪製68個關鍵點
    for n in range(68):
        x = landmarks.part(n).x
        y = landmarks.part(n).y
        cv2.circle(img, (x, y), 2, (0, 255, 0), -1)
```

#### 人臉特徵向量提取
```python
# 載入人臉識別模型
face_rec = dlib.face_recognition_model_v1('model/dlib_face_recognition_resnet_model_v1.dat')

# 提取128維特徵向量
face_descriptor = face_rec.compute_face_descriptor(img, landmarks)
face_vector = np.array(face_descriptor)
print(f"特徵向量維度: {face_vector.shape}")  # (128,)
```

### 2. 物件分類訓練

#### 準備訓練資料
```python
import dlib
import glob
import os

# 掃描訓練圖片
def load_training_data():
    images = []
    labels = []

    # 載入所有類別
    categories = ['accordion', 'camera', ...]  # 10個類別

    for i, category in enumerate(categories):
        pattern = f'dlib_ObjectCategories10/{category}/*.jpg'
        for img_path in glob.glob(pattern):
            img = dlib.load_rgb_image(img_path)
            images.append(img)
            labels.append(i)

    return images, labels

images, labels = load_training_data()
print(f"總共載入 {len(images)} 張圖片，{len(set(labels))} 個類別")
```

#### 特徵提取器訓練
```python
# 使用 dlib 的 scan_fhog_pyramid 進行特徵提取
detector_options = dlib.simple_object_detector_training_options()
detector_options.add_left_right_image_flips = True
detector_options.C = 5  # SVM 正則化參數

# 訓練物件檢測器
detector = dlib.train_simple_object_detector(images, boxes, detector_options)

# 儲存模型
detector.save('dlib_output/custom_detector.svm')
```

### 3. HOG + SVM 物件檢測

#### HOG 特徵提取
```python
# 提取 HOG 特徵
def extract_hog_features(image):
    # 轉換為灰階
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image

    # 計算 HOG 特徵
    hog = cv2.HOGDescriptor()
    features = hog.compute(gray)

    return features.flatten()

# 批量提取特徵
features_list = []
for img in images:
    features = extract_hog_features(img)
    features_list.append(features)

X = np.array(features_list)
y = np.array(labels)
```

#### SVM 分類器訓練
```python
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# 分割訓練測試集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 訓練 SVM
svm = SVC(kernel='linear', C=1.0)
svm.fit(X_train, y_train)

# 評估模型
y_pred = svm.predict(X_test)
print(classification_report(y_test, y_pred))

# 儲存模型
import joblib
joblib.dump(svm, 'dlib_output/svm_classifier.pkl')
```

## 🔧 實用工具函數

### 圖像預處理
```python
def preprocess_image(image_path, target_size=(64, 64)):
    """標準化圖像預處理"""
    img = cv2.imread(image_path)
    img = cv2.resize(img, target_size)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def normalize_image(image):
    """圖像歸一化"""
    return image.astype(np.float32) / 255.0
```

### 資料增強
```python
def augment_dataset(images, labels):
    """資料增強：翻轉、旋轉、亮度調整"""
    augmented_images = []
    augmented_labels = []

    for img, label in zip(images, labels):
        # 原始圖片
        augmented_images.append(img)
        augmented_labels.append(label)

        # 水平翻轉
        flipped = cv2.flip(img, 1)
        augmented_images.append(flipped)
        augmented_labels.append(label)

        # 旋轉 ±15 度
        for angle in [-15, 15]:
            rotated = rotate_image(img, angle)
            augmented_images.append(rotated)
            augmented_labels.append(label)

    return augmented_images, augmented_labels

def rotate_image(image, angle):
    """圖像旋轉"""
    height, width = image.shape[:2]
    center = (width//2, height//2)

    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, rotation_matrix, (width, height))

    return rotated
```

## 📊 模型評估與視覺化

### 混淆矩陣
```python
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

def plot_confusion_matrix(y_true, y_pred, class_names):
    """繪製混淆矩陣"""
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title('Confusion Matrix')
    plt.show()
```

### 檢測結果視覺化
```python
def visualize_detection_results(image, detections, class_names):
    """視覺化檢測結果"""
    result_img = image.copy()

    for detection in detections:
        x, y, w, h = detection['bbox']
        confidence = detection['confidence']
        class_id = detection['class_id']

        # 繪製邊界框
        cv2.rectangle(result_img, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # 添加標籤
        label = f"{class_names[class_id]}: {confidence:.2f}"
        cv2.putText(result_img, label, (x, y-10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    return result_img
```

## 🎯 完整專案範例

### 端到端物件分類專案
```python
class ObjectClassifier:
    def __init__(self, model_path='dlib_output/model.svm'):
        self.detector = dlib.simple_object_detector(model_path)
        self.classes = ['accordion', 'camera', ...]  # 10個類別

    def predict(self, image_path):
        """預測圖片中的物件"""
        img = dlib.load_rgb_image(image_path)
        detections = self.detector(img)

        results = []
        for detection in detections:
            results.append({
                'bbox': (detection.left(), detection.top(),
                        detection.width(), detection.height()),
                'confidence': detection.confidence
            })

        return results

    def train(self, dataset_path, output_path):
        """訓練新的分類器"""
        # 載入訓練資料
        images, boxes = self.load_training_data(dataset_path)

        # 設定訓練參數
        options = dlib.simple_object_detector_training_options()
        options.add_left_right_image_flips = True
        options.C = 5

        # 訓練
        detector = dlib.train_simple_object_detector(images, boxes, options)
        detector.save(output_path)

        return detector

# 使用範例
classifier = ObjectClassifier()
results = classifier.predict('image/test.jpg')
print(f"檢測到 {len(results)} 個物件")
```

## 🔍 故障排除

### 常見問題

1. **模型載入失敗**
   ```python
   # 檢查檔案路徑和權限
   import os
   model_path = 'model/shape_predictor_68_face_landmarks.dat'
   if not os.path.exists(model_path):
       print(f"模型檔案不存在: {model_path}")
   ```

2. **記憶體不足**
   ```python
   # 批次處理大資料集
   def process_in_batches(images, batch_size=32):
       for i in range(0, len(images), batch_size):
           batch = images[i:i+batch_size]
           yield batch
   ```

3. **訓練速度慢**
   ```python
   # 使用多執行緒
   import concurrent.futures

   def parallel_feature_extraction(images):
       with concurrent.futures.ThreadPoolExecutor() as executor:
           features = list(executor.map(extract_hog_features, images))
       return features
   ```

## 📚 進階主題

- **深度學習整合**: 將 dlib 與 TensorFlow/PyTorch 結合
- **即時處理**: 網路攝影機即時人臉識別
- **模型優化**: 量化和壓縮模型以提高效能
- **部署方案**: 將模型部署到行動裝置或邊緣設備

## 🔗 相關資源

- [dlib 官方文檔](http://dlib.net/)
- [人臉識別教學](http://dlib.net/face_recognition.py.html)
- [物件檢測範例](http://dlib.net/train_object_detector.py.html)
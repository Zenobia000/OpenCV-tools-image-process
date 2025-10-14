# 5.1.2 物體分類模組 (Object Classification Module)

## 概述

本模組實作完整的HOG+SVM物體分類系統，涵蓋從特徵工程到模型部署的完整機器學習流程。

**檔案**: `5.1.2_WBS_object_classification.ipynb`

## 模組結構

### 1. HOG特徵提取基礎 (20%)
- HOG原理與實作
- 參數影響分析
- 特徵視覺化
- 維度計算

### 2. 數據集準備與載入 (10%)
- dlib ObjectCategories10數據集
- 圖像預處理
- 標籤編碼
- 數據可視化

### 3. HOG特徵提取Pipeline (10%)
- 批量特徵提取
- 性能優化
- 特徵標準化
- 數據分割

### 4. SVM分類器訓練 (15%)
- SVM基礎理論
- 核函數選擇
- 基線模型訓練
- 分類報告與混淆矩陣

### 5. 超參數調整 (15%)
- GridSearchCV實作
- 交叉驗證
- 參數網格搜索
- 結果視覺化

### 6. 模型評估與分析 (15%)
- K-fold交叉驗證
- 學習曲線分析
- 偏差/方差診斷
- 錯誤分析

### 7. 模型保存與載入 (5%)
- Joblib序列化
- 完整Pipeline保存
- 模型驗證

### 8. 實時分類應用 (5%)
- 推理Pipeline
- 批量預測
- 置信度計算

### 9. 進階技巧與優化 (5%)
- 數據增強
- 集成學習
- 性能對比

### 10. 總結與延伸 (5%)
- 特徵方法對比
- 應用場景分析
- 延伸學習方向

## 技術要點

### HOG特徵
```python
features = hog(
    image,
    orientations=9,        # 方向bin數量
    pixels_per_cell=(8, 8),  # 細胞大小
    cells_per_block=(2, 2),  # 區塊大小
    block_norm='L2-Hys'      # 歸一化方法
)
```

### SVM訓練
```python
svm = SVC(kernel='rbf', C=10, gamma=0.01)
svm.fit(X_train_scaled, y_train)
```

### 超參數調整
```python
param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': [0.001, 0.01, 0.1, 1]
}
grid_search = GridSearchCV(svm, param_grid, cv=5)
```

## 性能指標

### 模型評估
- **準確率 (Accuracy)**: 整體正確率
- **精確率 (Precision)**: 預測為正的樣本中實際為正的比例
- **召回率 (Recall)**: 實際為正的樣本中被正確預測的比例
- **F1-score**: 精確率和召回率的調和平均

### 交叉驗證
- 10-fold CV確保結果可靠
- 計算置信區間
- 分析方差

### 學習曲線
- 診斷過擬合/欠擬合
- 確定數據集大小需求
- 評估模型複雜度

## 實作細節

### 數據準備
1. 圖像灰階化
2. 統一尺寸 (64x128)
3. 特徵提取
4. 標準化 (zero mean, unit variance)

### 特徵工程
- HOG特徵維度: 3780 (標準配置)
- 計算時間: ~15ms/image (CPU)
- 內存占用: ~15KB/feature vector

### 模型訓練
- LinearSVC: 快速訓練
- RBF SVM: 更好性能
- 訓練時間: 數秒到數分鐘
- 模型大小: <1MB

## 使用方式

### 1. 環境準備
```bash
# 激活虛擬環境
source .venv/bin/activate

# 確認依賴已安裝
pip install opencv-python scikit-learn scikit-image joblib
```

### 2. 執行Notebook
```bash
jupyter lab 05_machine_learning/5.1.2_WBS_object_classification.ipynb
```

### 3. 快速測試
```bash
python3 05_machine_learning/test_notebook.py
```

## 數據集要求

### dlib ObjectCategories10
- **路徑**: `../assets/datasets/dlib_ObjectCategories10/`
- **結構**:
  ```
  dlib_ObjectCategories10/
  ├── accordion/
  ├── camera/
  └── .../
  ```
- **樣本數**: ~50-100 images/class
- **格式**: JPG/PNG

### 替代方案
如數據集不可用，代碼會自動生成合成圖像用於演示。

## 預期結果

### 基線模型
- Training accuracy: ~0.95
- Testing accuracy: ~0.85

### 調優模型
- Cross-validation score: ~0.90
- Testing accuracy: ~0.88-0.92

### 集成模型
- Bagging improvement: +2-5%

## 常見問題

### Q1: GridSearch太慢？
**A**: 減少參數網格：
```python
param_grid = {
    'C': [1, 10],
    'gamma': [0.01, 0.1]
}
```

### Q2: 內存不足？
**A**: 限制數據量：
```python
images, labels, class_names = load_dataset(
    DATASET_PATH, 
    max_samples_per_class=20  # 限制每類20個樣本
)
```

### Q3: 特徵維度過大？
**A**: 調整HOG參數：
```python
features = hog(
    image,
    pixels_per_cell=(16, 16),  # 增大細胞 → 減少維度
    cells_per_block=(2, 2)
)
```

## 進階優化

### 1. 數據增強
- 翻轉、旋轉、亮度調整
- 噪聲注入
- 彈性變形

### 2. 特徵選擇
- PCA降維
- 特徵重要性分析
- 相關性過濾

### 3. 集成方法
- Bagging: 多個模型投票
- Boosting: 順序訓練強化弱點
- Stacking: 多層模型組合

### 4. 部署優化
- 模型量化
- 特徵緩存
- 並行推理

## 延伸學習

### 相關模組
- `5.1.1` - 人臉檢測 (Haar Cascade, dlib)
- `5.1.3` - 深度學習入門 (CNN基礎)
- `5.2.1` - 物體檢測 (YOLO, SSD)

### 推薦資源
- **論文**: Dalal & Triggs (2005) - HOG原始論文
- **書籍**: "Hands-On Machine Learning" - Scikit-learn實戰
- **課程**: Andrew Ng - Machine Learning (SVM章節)

## 版本歷史

- **v1.0** (2025-10-14): 初始版本
  - 完整HOG+SVM實作
  - 10個主要章節
  - 20個代碼單元
  - 21個說明單元

## 授權

本教學模組為OpenCV Computer Vision Toolkit專案的一部分，遵循專案整體授權。

---

**維護者**: OpenCV Toolkit Team  
**更新日期**: 2025-10-14  
**模組狀態**: ✅ Production Ready

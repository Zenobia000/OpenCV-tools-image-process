# 階段三：前處理技術模組 (Stage 3: Preprocessing Techniques)

## 📋 模組概覽

本階段涵蓋影像前處理的核心技術，為後續特徵檢測與機器學習奠定基礎。

**WBS編號**: 3.x.x
**學習時長**: 8-10小時
**難度等級**: ⭐⭐⭐ (中級)

---

## 📚 模組列表

### 3.1 濾波與平滑 (Filtering & Smoothing)

#### [3.1.1 濾波與平滑](./3.1.1_filtering_smoothing.ipynb) ✅
**內容**:
- 高斯濾波器 (Gaussian Filter)
- 平均濾波器 (Mean Filter)
- 中值濾波器 (Median Filter)
- 雙邊濾波器 (Bilateral Filter)
- 自定義卷積核 (Custom Kernels)

**學習目標**: 理解不同濾波器的特性與應用場景

---

#### [3.1.2 形態學操作](./3.1.2_morphological_ops.ipynb) ✅
**內容**:
- 侵蝕 (Erosion)
- 膨脹 (Dilation)
- 開運算/閉運算 (Opening/Closing)
- 形態學梯度 (Morphological Gradient)
- 頂帽/黑帽變換 (Top-hat/Black-hat)
- 自定義結構元素 (Custom Kernels)

**學習目標**: 掌握形態學操作在影像處理中的應用

---

#### [3.1.3 邊緣檢測](./3.1.3_edge_detection.ipynb) ✅
**內容**:
- Canny 邊緣檢測
- Sobel 算子
- Scharr 濾波器
- Laplacian 算子
- 邊緣檢測參數調整
- DoG (Difference of Gaussian)
- 輪廓檢測與分析 (findContours)
- 邊界矩形與矩計算
- 輪廓面積與周長

**學習目標**: 精通各種邊緣檢測算法及其參數調整

---

### 3.2 影像增強技術 (Image Enhancement)

#### [3.2.1 直方圖處理](./3.2.1_histogram_processing.ipynb) ✅
**內容**:
- 直方圖計算與顯示
- 直方圖均化 (Histogram Equalization)
- 對比度限制自適應直方圖均化 (CLAHE)
- BGR三通道處理
- 灰階拉伸 (Contrast Stretching)

**學習目標**: 使用直方圖技術改善影像品質

---

#### [3.2.2 噪聲處理](./3.2.2_noise_reduction.ipynb) ✅
**內容**:
- 噪聲類型識別 (高斯/椒鹽/斑點/週期性)
- 空間域降噪 (Mean/Gaussian/Median/Bilateral)
- 進階降噪 (NLM非局部均值)
- 形態學降噪
- 效能基準測試
- 自適應降噪演算法
- 降噪方法選擇決策樹

**學習目標**: 掌握多種降噪技術並能根據場景選擇

---

#### [3.2.3 閾值處理](./3.2.3_thresholding.ipynb) ✅
**內容**:
- Threshold 二值化處理
- 自適應閾值 (Adaptive Threshold)
- Otsu 自動閾值
- 多種閾值類型比較

**學習目標**: 理解二值化技術在影像分割中的應用

---

## 🎯 學習路徑

```
推薦學習順序:

1. 濾波與平滑 (3.1.1)     ← 基礎
   ↓
2. 形態學操作 (3.1.2)     ← 結構處理
   ↓
3. 閾值處理 (3.2.3)       ← 二值化
   ↓
4. 邊緣檢測 (3.1.3)       ← 特徵提取
   ↓
5. 直方圖處理 (3.2.1)     ← 影像增強
   ↓
6. 噪聲處理 (3.2.2)       ← 品質改善
```

---

## 📊 階段完成度

| 模組編號 | 模組名稱 | 狀態 | 檔案大小 | Cells數 |
|---------|---------|------|----------|---------|
| 3.1.1 | 濾波與平滑 | ✅ | 86KB | - |
| 3.1.2 | 形態學操作 | ✅ | 21KB | 31 |
| 3.1.3 | 邊緣檢測 | ✅ | 216KB | 66 |
| 3.2.1 | 直方圖處理 | ✅ | 15KB | 17 |
| 3.2.2 | 噪聲處理 | ✅ | 45KB | 31 |
| 3.2.3 | 閾值處理 | ✅ | 85KB | - |

**總計**: 6/6 模組完成 = **100%** ✅

---

## 🔧 技術棧

### 核心函數

#### 濾波
```python
cv2.GaussianBlur()      # 高斯濾波
cv2.blur()              # 平均濾波
cv2.medianBlur()        # 中值濾波
cv2.bilateralFilter()   # 雙邊濾波
cv2.filter2D()          # 自定義卷積
```

#### 形態學
```python
cv2.erode()             # 侵蝕
cv2.dilate()            # 膨脹
cv2.morphologyEx()      # 形態學操作
cv2.getStructuringElement()  # 結構元素
```

#### 邊緣檢測
```python
cv2.Canny()             # Canny邊緣檢測
cv2.Sobel()             # Sobel算子
cv2.Scharr()            # Scharr濾波器
cv2.Laplacian()         # Laplacian算子
cv2.findContours()      # 輪廓檢測
```

#### 直方圖
```python
cv2.calcHist()          # 計算直方圖
cv2.equalizeHist()      # 直方圖均化
cv2.createCLAHE()       # CLAHE
```

#### 降噪
```python
cv2.fastNlMeansDenoising()       # NLM灰階
cv2.fastNlMeansDenoisingColored()  # NLM彩色
```

#### 閾值
```python
cv2.threshold()         # 固定閾值
cv2.adaptiveThreshold() # 自適應閾值
```

---

## 💡 實用技巧

### 1. 降噪流程
```python
# 混合噪聲處理流程
image_noisy = add_mixed_noise(image)
step1 = cv2.medianBlur(image_noisy, 5)      # 移除椒鹽噪聲
step2 = cv2.bilateralFilter(step1, 9, 75, 75)  # 移除高斯噪聲
result = cv2.fastNlMeansDenoising(step2, None, 10, 7, 21)  # 深度降噪
```

### 2. 邊緣檢測優化
```python
# 邊緣檢測前的預處理
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)  # 降噪
edges = cv2.Canny(blurred, 50, 150)  # 邊緣檢測
```

### 3. 直方圖均化
```python
# 彩色影像CLAHE
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
lab[:, :, 0] = clahe.apply(lab[:, :, 0])  # 僅處理亮度通道
result = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
```

---

## 🎓 實戰應用

### 應用場景

| 技術 | 應用場景 | 效果 |
|------|---------|------|
| **高斯濾波** | 平滑影像、去除高頻噪聲 | 模糊但保留大致輪廓 |
| **中值濾波** | 去除椒鹽噪聲 | 保留邊緣細節 |
| **形態學** | 去除小雜點、連接斷裂 | 改善結構 |
| **Canny** | 物體邊界檢測 | 清晰邊緣 |
| **CLAHE** | 低對比度影像增強 | 提升細節可見度 |
| **NLM** | 保邊降噪 | 高品質降噪 |

### 典型工作流程

```
原始影像
   ↓
降噪處理 (Bilateral/NLM)
   ↓
對比度增強 (CLAHE)
   ↓
邊緣檢測 (Canny)
   ↓
形態學處理 (Opening/Closing)
   ↓
閾值分割 (Adaptive Threshold)
   ↓
輪廓分析 (findContours)
```

---

## 📈 效能基準

### 典型操作耗時 (512x512影像)

| 操作 | 時間 (ms) | 速度等級 |
|------|-----------|----------|
| Gaussian Blur (5x5) | ~2-3 | ⚡⚡⚡⚡⚡ |
| Median Blur (5x5) | ~5-8 | ⚡⚡⚡⚡ |
| Bilateral Filter | ~15-20 | ⚡⚡⚡ |
| Canny Edge | ~5-10 | ⚡⚡⚡⚡ |
| CLAHE | ~10-15 | ⚡⚡⚡⚡ |
| NLM Denoising | ~100-200 | ⚡ |

> 測試環境: CPU處理, 無GPU加速

---

## ✅ 里程碑檢查

- [✅] 所有算法範例可執行
- [✅] 前處理效果明顯
- [✅] 效能基準測試通過
- [✅] 實戰應用案例完整
- [✅] 文檔說明清晰

**階段三狀態**: ✅ **100%完成**

---

## 📖 參考資源

- [OpenCV Image Filtering](https://docs.opencv.org/4.x/d4/d86/group__imgproc__filter.html)
- [OpenCV Morphological Operations](https://docs.opencv.org/4.x/d9/d61/tutorial_py_morphological_ops.html)
- [OpenCV Edge Detection](https://docs.opencv.org/4.x/da/d22/tutorial_py_canny.html)
- [OpenCV Histograms](https://docs.opencv.org/4.x/d1/db7/tutorial_py_histogram_begins.html)
- [OpenCV Denoising](https://docs.opencv.org/4.x/d5/d69/tutorial_py_non_local_means.html)

---

## 🚀 下一步

完成階段三後，建議進入：

**階段四：特徵檢測模組** (`04_feature_detection/`)
- 角點檢測 (Harris, Shi-Tomasi, FAST)
- 特徵描述子 (SIFT, ORB, BRISK)
- 特徵匹配與追蹤

---

**建立日期**: 2024-10-13
**維護者**: OpenCV Toolkit Team
**版本**: v1.0
**狀態**: ✅ 完成

# 階段四：特徵檢測模組 (Stage 4: Feature Detection)

## 📋 模組概覽

本階段涵蓋電腦視覺中的特徵檢測與匹配技術，是物體識別、影像配準、3D重建的核心技術。

**WBS編號**: 4.x.x
**學習時長**: 10-12小時
**難度等級**: ⭐⭐⭐⭐ (進階)

---

## 📚 模組列表

### 4.1 角點與特徵檢測

#### [4.1.1 角點檢測](./4.1.1_corner_detection.ipynb) ✅
**內容**:
- Harris 角點檢測（數學原理、響應函數）
- Shi-Tomasi 角點檢測（最小特徵值法）
- FAST 角點檢測（圓形分割測試、NMS）
- 角點檢測器比較分析（速度、準確度、魯棒性）
- 實戰應用：特徵追蹤示範

**檔案大小**: 46KB | **Cells數**: 30 | **學習時間**: 90分鐘

**學習目標**: 掌握三種主流角點檢測算法，能根據應用場景選擇適當方法

---

#### [4.1.2 特徵描述子](./4.1.2_feature_descriptors.ipynb) ✅
**內容**:
- SIFT 特徵檢測器（DoG、128維描述子、尺度不變性）
- SURF 特徵檢測器（積分影像、Haar小波、專利限制）
- ORB 特徵檢測器（Oriented FAST、rBRIEF、開源最快）
- BRIEF 特徵描述子（二進位描述、速度優勢）
- BRISK 特徵檢測器（同心圓採樣、512位元）
- 特徵匹配算法（BFMatcher、FLANN、KNN）
- Homography 變換與影像配準
- 旋轉不變性實驗與驗證

**檔案大小**: 48KB | **Cells數**: 38 | **學習時間**: 120分鐘

**學習目標**: 理解五種特徵描述子的原理與應用，掌握特徵匹配技術

---

#### [4.1.3 物體追蹤](./4.1.3_object_tracking.ipynb) ✅
**內容**:
- 光流法理論基礎（亮度恆定假設、光流方程）
- Lucas-Kanade 稀疏光流（金字塔、追蹤軌跡）
- Farneback 稠密光流（HSV視覺化、向量場）
- 多目標追蹤基礎（數據關聯、軌跡管理）
- 卡爾曼濾波器（預測與更新、噪聲抑制）
- 稀疏 vs 稠密光流比較
- 實戰練習：視頻特徵點追蹤

**檔案大小**: 71KB | **Cells數**: 37 | **學習時間**: 150分鐘

**學習目標**: 掌握光流法與卡爾曼濾波，能實現基本的物體追蹤系統

---

### 4.2 進階特徵檢測

#### [4.2.1 模板匹配](./4.2.1_template_matching.ipynb) ✅
**內容**:
- 模板匹配基礎原理（滑動窗口、相似度計算）
- 六種匹配方法詳解（TM_CCOEFF_NORMED最推薦）
- 光照變化魯棒性測試
- 多目標檢測（NMS非極大值抑制）
- 多尺度模板匹配（尺度不變性）
- 實戰應用：Logo檢測、UI自動化測試
- 性能基準測試與優化

**檔案大小**: 56KB | **Cells數**: 34 | **學習時間**: 120分鐘

**學習目標**: 掌握模板匹配技術及其局限性，能應用於實際檢測任務

---

## 🎯 學習路徑

```
推薦學習順序:

1. 角點檢測 (4.1.1)           ← 特徵檢測基礎
   ↓
2. 特徵描述子 (4.1.2)         ← 核心技術
   ↓
3. 模板匹配 (4.2.1)           ← 簡單但實用
   ↓
4. 物體追蹤 (4.1.3)           ← 進階應用
```

**學習建議**:
- 4.1.1 + 4.1.2 是核心，必須精通
- 4.2.1 相對獨立，可穿插學習
- 4.1.3 需要前面的基礎，建議最後學習

---

## 📊 階段完成度

| 模組編號 | 模組名稱 | 狀態 | 檔案大小 | Cells數 | 學習時間 |
|---------|---------|------|----------|---------|----------|
| 4.1.1 | 角點檢測 | ✅ | 46KB | 30 | 90分鐘 |
| 4.1.2 | 特徵描述子 | ✅ | 48KB | 38 | 120分鐘 |
| 4.1.3 | 物體追蹤 | ✅ | 71KB | 37 | 150分鐘 |
| 4.2.1 | 模板匹配 | ✅ | 56KB | 34 | 120分鐘 |

**總計**: 4/4 模組完成 = **100%** ✅

---

## 🔧 技術棧

### 角點檢測
```python
cv2.cornerHarris()          # Harris角點
cv2.goodFeaturesToTrack()   # Shi-Tomasi角點
cv2.FastFeatureDetector_create()  # FAST角點
```

### 特徵描述子
```python
cv2.SIFT_create()           # SIFT (OpenCV 4.4+)
cv2.ORB_create()            # ORB
cv2.BRISK_create()          # BRISK
cv2.BFMatcher()             # 暴力匹配器
cv2.FlannBasedMatcher()     # 快速近似匹配
cv2.findHomography()        # 計算透視變換矩陣
```

### 光流法
```python
cv2.calcOpticalFlowPyrLK()      # Lucas-Kanade光流
cv2.calcOpticalFlowFarneback()  # Farneback稠密光流
```

### 模板匹配
```python
cv2.matchTemplate()         # 模板匹配
cv2.minMaxLoc()             # 找出最佳匹配位置
```

---

## 💡 實用技巧

### 1. 特徵匹配流程
```python
# Step 1: 檢測特徵點
detector = cv2.SIFT_create()
kp1, des1 = detector.detectAndCompute(img1, None)
kp2, des2 = detector.detectAndCompute(img2, None)

# Step 2: 匹配特徵
matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
matches = matcher.knnMatch(des1, des2, k=2)

# Step 3: Lowe's Ratio Test
good_matches = []
for m, n in matches:
    if m.distance < 0.75 * n.distance:
        good_matches.append(m)

# Step 4: 計算Homography
if len(good_matches) > 10:
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches])
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches])
    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
```

### 2. 光流追蹤優化
```python
# 使用Shi-Tomasi檢測特徵點
feature_params = dict(
    maxCorners=100,
    qualityLevel=0.3,
    minDistance=7,
    blockSize=7
)
p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)

# 使用LK光流追蹤
lk_params = dict(
    winSize=(15, 15),
    maxLevel=2,
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
)
p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

# 只保留追蹤成功的點
good_new = p1[st==1]
good_old = p0[st==1]
```

### 3. 多尺度模板匹配
```python
def multi_scale_template_matching(image, template, scales=np.linspace(0.5, 2.0, 20)):
    best_match = None
    best_val = -1

    for scale in scales:
        resized = cv2.resize(template, None, fx=scale, fy=scale)
        if resized.shape[0] > image.shape[0] or resized.shape[1] > image.shape[1]:
            continue

        result = cv2.matchTemplate(image, resized, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(result)

        if max_val > best_val:
            best_val = max_val
            best_match = (max_loc, resized.shape, scale)

    return best_match, best_val
```

---

## 🎓 實戰應用

### 應用場景對比

| 技術 | 應用場景 | 優勢 | 局限性 |
|------|---------|------|--------|
| **Harris角點** | 影像配準、相機標定 | 魯棒性高、理論完善 | 計算較慢 |
| **Shi-Tomasi** | 特徵追蹤、光流法 | 追蹤性能最佳 | 對光照敏感 |
| **FAST** | 實時應用、SLAM | 速度最快 | 準確度中等 |
| **SIFT** | 物體識別、3D重建 | 最魯棒、尺度不變 | 速度慢、有專利 |
| **ORB** | 移動設備、實時應用 | 開源、速度快 | 準確度略低 |
| **模板匹配** | Logo檢測、UI測試 | 簡單直觀 | 無尺度/旋轉不變性 |
| **光流法** | 運動分析、追蹤 | 密集運動場 | 對大位移敏感 |

### 典型工作流程

```
場景: 物體識別與追蹤

步驟1: 特徵檢測
   ↓ (使用 SIFT/ORB)
步驟2: 特徵匹配
   ↓ (使用 BFMatcher + Lowe's Ratio Test)
步驟3: 幾何驗證
   ↓ (使用 RANSAC + Homography)
步驟4: 物體定位
   ↓ (透視變換映射)
步驟5: 追蹤初始化
   ↓ (選擇追蹤算法)
步驟6: 持續追蹤
   ↓ (光流法 + 卡爾曼濾波)
```

---

## 📈 效能基準

### 典型操作耗時 (640x480影像, Intel Core i7)

| 操作 | 時間 (ms) | FPS | 速度等級 |
|------|-----------|-----|----------|
| Harris 角點 (500點) | ~20-30 | 30-50 | ⚡⚡⚡ |
| FAST 角點 (500點) | ~5-10 | 100-200 | ⚡⚡⚡⚡⚡ |
| SIFT 檢測+描述 (500點) | ~150-200 | 5-7 | ⚡ |
| ORB 檢測+描述 (500點) | ~15-25 | 40-60 | ⚡⚡⚡⚡ |
| BFMatcher (500對) | ~5-10 | 100-200 | ⚡⚡⚡⚡⚡ |
| FLANN (500對) | ~2-5 | 200-500 | ⚡⚡⚡⚡⚡ |
| LK 光流 (100點) | ~5-10 | 100-200 | ⚡⚡⚡⚡⚡ |
| Farneback 光流 (全圖) | ~30-50 | 20-30 | ⚡⚡⚡ |
| 模板匹配 (100x100) | ~10-20 | 50-100 | ⚡⚡⚡⚡ |

> 測試環境: CPU處理, 無GPU加速

### 速度 vs 準確度權衡

```
準確度 ↑
    │
    │  SIFT ●
    │         BRISK ●
    │              ORB ●
    │                    FAST ●
    │
    └──────────────────────────→ 速度
                               ↑
```

---

## ✅ 里程碑檢查

- [✅] 所有算法範例可執行
- [✅] 特徵匹配demo成功
- [✅] 追蹤算法穩定運行
- [✅] 實戰應用案例完整
- [✅] 效能基準測試通過
- [✅] 文檔說明清晰

**階段四狀態**: ✅ **100%完成**

---

## 🚧 常見問題與解決方案

### 問題1: SIFT/SURF 不可用
**錯誤**: `AttributeError: module 'cv2' has no attribute 'xfeatures2d'`

**解決方案**:
```bash
# 安裝 opencv-contrib-python
pip install opencv-contrib-python>=4.8.0

# 或使用 ORB 替代
detector = cv2.ORB_create()
```

### 問題2: 特徵匹配誤匹配過多
**症狀**: 匹配結果中有很多錯誤的對應關係

**解決方案**:
```python
# 1. 使用 Lowe's Ratio Test
good_matches = []
for m, n in matches:
    if m.distance < 0.75 * n.distance:  # 調整閾值 (0.6-0.8)
        good_matches.append(m)

# 2. 使用 RANSAC 過濾外點
H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
```

### 問題3: 光流追蹤漂移
**症狀**: 追蹤點逐漸偏離目標

**解決方案**:
```python
# 1. 定期重新檢測特徵點 (每N幀)
if frame_count % 30 == 0:
    p0 = cv2.goodFeaturesToTrack(gray, mask=None, **feature_params)

# 2. 使用雙向光流驗證
p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, new_gray, p0, None)
p0r, st, err = cv2.calcOpticalFlowPyrLK(new_gray, old_gray, p1, None)
d = abs(p0-p0r).reshape(-1, 2).max(-1)
good = d < 1  # 雙向誤差小於1像素
```

### 問題4: 模板匹配對尺度/旋轉敏感
**症狀**: 目標稍微變化就無法檢測

**解決方案**:
```python
# 1. 使用多尺度匹配
best_match = multi_scale_template_matching(image, template)

# 2. 改用特徵匹配方法
detector = cv2.ORB_create()
kp1, des1 = detector.detectAndCompute(template, None)
kp2, des2 = detector.detectAndCompute(image, None)
matches = matcher.knnMatch(des1, des2, k=2)
```

---

## 📖 參考資源

### 官方文檔
- [OpenCV Feature Detection](https://docs.opencv.org/4.x/db/d27/tutorial_py_table_of_contents_feature2d.html)
- [OpenCV Video Analysis](https://docs.opencv.org/4.x/d7/df3/group__imgproc__motion.html)
- [Feature Matching Tutorial](https://docs.opencv.org/4.x/dc/dc3/tutorial_py_matcher.html)

### 經典論文
- **Harris Corner** (1988): Harris & Stephens - "A Combined Corner and Edge Detector"
- **SIFT** (2004): David Lowe - "Distinctive Image Features from Scale-Invariant Keypoints"
- **SURF** (2006): Bay et al. - "SURF: Speeded Up Robust Features"
- **FAST** (2006): Rosten & Drummond - "Machine Learning for High-Speed Corner Detection"
- **ORB** (2011): Rublee et al. - "ORB: An efficient alternative to SIFT or SURF"
- **Lucas-Kanade** (1981): Lucas & Kanade - "An Iterative Image Registration Technique"

### 在線資源
- [Learn OpenCV](https://learnopencv.com/) - 豐富的實戰教程
- [PyImageSearch](https://pyimagesearch.com/) - 電腦視覺部落格
- [OpenCV Python Tutorials](https://opencv-python-tutroals.readthedocs.io/)

---

## 🚀 下一步

完成階段四後，建議進入：

**階段五：機器學習整合** (`05_machine_learning/`)
- 人臉檢測（Haar Cascade, HOG + SVM）
- 物體分類（特徵提取 + 機器學習）
- dlib 整合（68點面部特徵、人臉識別）
- 深度學習模型載入（OpenCV DNN）

**進階主題**:
- Deep Learning 特徵（SuperPoint, D2-Net）
- SLAM（Simultaneous Localization and Mapping）
- 3D Reconstruction
- Visual Odometry

---

**建立日期**: 2025-10-13
**維護者**: OpenCV Toolkit Team
**版本**: v1.0
**狀態**: ✅ 完成

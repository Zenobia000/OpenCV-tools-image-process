# 📝 檔案重新命名計畫 (File Renaming Plan)

## 🎯 目標

將所有教學模組檔案名稱加上 WBS 小節編號前綴，以對齊 ULTIMATE_PROJECT_GUIDE.md 的結構。

**命名格式**: `{WBS編號}_{原檔名}.ipynb`
**範例**: `2.2.3_color_spaces.ipynb`

---

## 📋 重新命名對照表

### Stage 2.1: 基礎知識模組 (01_fundamentals/)

| WBS編號 | 原檔名 | 新檔名 | 狀態 |
|---------|--------|--------|------|
| 2.1.1 | `python_numpy_basics.ipynb` | `2.1.1_python_numpy_basics.ipynb` | ✅ 存在 |
| 2.1.2 | `opencv_installation.md` | `2.1.2_opencv_installation.md` | ✅ 存在 |
| 2.1.3 | `computer_vision_concepts.ipynb` | `2.1.3_computer_vision_concepts.ipynb` | ✅ 存在 |
| N/A | `opencv_fundamentals.ipynb` | ❌ **刪除或合併** | 重複檔案 |

**說明**:
- `opencv_fundamentals.ipynb` 不在 WBS 中，應該合併到 2.1.2 或刪除
- 如果 `opencv_fundamentals.ipynb` 有獨特內容，建議合併到 `2.1.2_opencv_installation.md` 中作為 "OpenCV 基礎概念" 章節

---

### Stage 2.2: 核心操作模組 (02_core_operations/)

| WBS編號 | 原檔名 | 新檔名 | 狀態 |
|---------|--------|--------|------|
| 2.2.1 | `image_io_display.ipynb` | `2.2.1_image_io_display.ipynb` | ✅ 存在 |
| 2.2.2 | `geometric_transformations.ipynb` | `2.2.2_geometric_transformations.ipynb` | ✅ 存在 |
| 2.2.3 | **[需創建]** | `2.2.3_color_spaces.ipynb` | ⏳ 需創建 |
| 2.2.4 | `arithmetic_operations.ipynb` | `2.2.4_arithmetic_operations.ipynb` | ✅ 存在 |
| N/A | `image_io_display_modern.ipynb` | ❌ **刪除** | 重複檔案 |
| N/A | `image_processing_techniques.ipynb` | ❌ **審查後處理** | 可能重複 |

**說明**:
- `image_io_display_modern.ipynb` (34K) 應該合併到 `2.2.1_image_io_display.ipynb` 或刪除
- `image_processing_techniques.ipynb` (251K) 很大，需要審查內容是否應該拆分到其他 WBS 項目

---

### Stage 3.1: 濾波與平滑 (03_preprocessing/)

| WBS編號 | 原檔名 | 新檔名 | 狀態 |
|---------|--------|--------|------|
| 3.1.1 | `filtering_smoothing.ipynb` | `3.1.1_filtering_smoothing.ipynb` | ✅ 存在 |
| 3.1.2 | **[需創建]** | `3.1.2_morphological_ops.ipynb` | ⏳ 需創建 |
| 3.1.3 | **[需創建]** | `3.1.3_edge_detection.ipynb` | ⏳ 需創建 |
| N/A | `advanced_image_operations.ipynb` | ❌ **審查後處理** | 可能屬於 3.2.x |

**說明**:
- `advanced_image_operations.ipynb` (84K) 可能屬於 Stage 3.2 (影像增強技術)，需審查內容

---

### Stage 3.2: 影像增強技術 (03_preprocessing/)

| WBS編號 | 原檔名 | 新檔名 | 狀態 |
|---------|--------|--------|------|
| 3.2.1 | **[需創建]** | `3.2.1_histogram_processing.ipynb` | ⏳ 需創建 |
| 3.2.2 | **[需創建]** | `3.2.2_noise_reduction.ipynb` | ⏳ 需創建 |

---

### Stage 4.1: 角點檢測 (04_feature_detection/)

| WBS編號 | 原檔名 | 新檔名 | 狀態 |
|---------|--------|--------|------|
| 4.1.1 | **[需創建]** | `4.1.1_corner_detection.ipynb` | ⏳ 需創建 |
| 4.1.2 | **[需創建]** | `4.1.2_feature_descriptors.ipynb` | ⏳ 需創建 |
| 4.1.3 | **[需創建]** | `4.1.3_object_tracking.ipynb` | ⏳ 需創建 |
| N/A | `feature_detection_basics.ipynb` | ❌ **審查後處理** | 可能合併到 4.1.x |

---

### Stage 4.2: 進階特徵檢測 (04_feature_detection/)

| WBS編號 | 原檔名 | 新檔名 | 狀態 |
|---------|--------|--------|------|
| 4.2.1 | **[需創建]** | `4.2.1_template_matching.ipynb` | ⏳ 需創建 |

---

### Stage 5.1: 傳統機器學習 (05_machine_learning/)

| WBS編號 | 原檔名 | 新檔名 | 狀態 |
|---------|--------|--------|------|
| 5.1.1 | **[需創建]** | `5.1.1_face_detection.ipynb` | ⏳ 需創建 |
| 5.1.2 | **[需創建]** | `5.1.2_object_classification.ipynb` | ⏳ 需創建 |
| 5.1.3 | **[需創建]** | `5.1.3_dlib_integration.ipynb` | ⏳ 需創建 |

---

### Stage 5.2: 深度學習整合 (05_machine_learning/)

| WBS編號 | 原檔名 | 新檔名 | 狀態 |
|---------|--------|--------|------|
| 5.2.1 | **[需創建]** | `5.2.1_dnn_integration.ipynb` | ⏳ 需創建 |
| 5.2.2 | **[需創建]** | `5.2.2_real_time_detection.ipynb` | ⏳ 需創建 |

---

### Stage 6.1: 初學者練習 (06_exercises/beginner/)

| WBS編號 | 原檔名 | 新檔名 | 狀態 |
|---------|--------|--------|------|
| 6.1.1 | `bgr_channel_operations.ipynb` | `6.1.1_bgr_channel_operations.ipynb` | ✅ 存在 |
| 6.1.2 | `drawing_functions_practice.ipynb` | `6.1.2_drawing_functions_practice.ipynb` | ✅ 存在 |
| 6.1.3 | `filtering_applications.ipynb` | `6.1.3_filtering_applications.ipynb` | ✅ 存在 |
| 6.1.4 | `comprehensive_basics.ipynb` | `6.1.4_comprehensive_basics.ipynb` | ✅ 存在 |

---

### Stage 6.2: 中級練習 (06_exercises/intermediate/)

| WBS編號 | 原檔名 | 新檔名 | 狀態 |
|---------|--------|--------|------|
| 6.2.1 | **[需創建]** | `6.2.1_feature_matching_challenge.ipynb` | ⏳ 需創建 |
| 6.2.2 | **[需創建]** | `6.2.2_image_stitching_project.ipynb` | ⏳ 需創建 |
| 6.2.3 | **[需創建]** | `6.2.3_video_analysis_tasks.ipynb` | ⏳ 需創建 |

---

### Stage 6.3: 高級挑戰 (06_exercises/advanced/)

| WBS編號 | 原檔名 | 新檔名 | 狀態 |
|---------|--------|--------|------|
| 6.3.1 | **[需創建]** | `6.3.1_custom_algorithm_implementation.ipynb` | ⏳ 需創建 |
| 6.3.2 | **[需創建]** | `6.3.2_performance_optimization_challenge.ipynb` | ⏳ 需創建 |
| 6.3.3 | **[需創建]** | `6.3.3_research_project_template.ipynb` | ⏳ 需創建 |

---

## 🚀 執行計畫

### Phase 1: 立即執行 (重新命名現有檔案)

#### Step 1.1: 01_fundamentals/
```bash
cd /home/sunny/python_workstation/github/OpenCV-tools-image-process/01_fundamentals

# 重新命名
mv python_numpy_basics.ipynb 2.1.1_python_numpy_basics.ipynb
mv opencv_installation.md 2.1.2_opencv_installation.md
mv computer_vision_concepts.ipynb 2.1.3_computer_vision_concepts.ipynb

# 審查 opencv_fundamentals.ipynb 是否需要保留
# 建議: 合併內容到 2.1.2 後刪除
```

#### Step 1.2: 02_core_operations/
```bash
cd /home/sunny/python_workstation/github/OpenCV-tools-image-process/02_core_operations

# 重新命名
mv image_io_display.ipynb 2.2.1_image_io_display.ipynb
mv geometric_transformations.ipynb 2.2.2_geometric_transformations.ipynb
mv arithmetic_operations.ipynb 2.2.4_arithmetic_operations.ipynb

# 待處理檔案
# - image_io_display_modern.ipynb (可能合併或刪除)
# - image_processing_techniques.ipynb (需審查內容歸屬)
```

#### Step 1.3: 03_preprocessing/
```bash
cd /home/sunny/python_workstation/github/OpenCV-tools-image-process/03_preprocessing

# 重新命名
mv filtering_smoothing.ipynb 3.1.1_filtering_smoothing.ipynb

# 待處理檔案
# - advanced_image_operations.ipynb (可能屬於 3.2.x)
```

#### Step 1.4: 04_feature_detection/
```bash
cd /home/sunny/python_workstation/github/OpenCV-tools-image-process/04_feature_detection

# 審查 feature_detection_basics.ipynb 內容
# 可能拆分到 4.1.1, 4.1.2, 4.1.3
```

#### Step 1.5: 06_exercises/beginner/
```bash
cd /home/sunny/python_workstation/github/OpenCV-tools-image-process/06_exercises/beginner

# 重新命名
mv bgr_channel_operations.ipynb 6.1.1_bgr_channel_operations.ipynb
mv drawing_functions_practice.ipynb 6.1.2_drawing_functions_practice.ipynb
mv filtering_applications.ipynb 6.1.3_filtering_applications.ipynb
mv comprehensive_basics.ipynb 6.1.4_comprehensive_basics.ipynb
```

---

### Phase 2: 創建缺失檔案

**優先級排序**:
1. **P0 - 立即創建**:
   - `2.2.3_color_spaces.ipynb` (WBS M2 milestone 需要)

2. **P1 - 本週創建**:
   - `3.1.2_morphological_ops.ipynb`
   - `3.1.3_edge_detection.ipynb`

3. **P2 - 下週創建**:
   - `3.2.1_histogram_processing.ipynb`
   - `3.2.2_noise_reduction.ipynb`

4. **P3 - 未來創建**:
   - Stage 4, 5, 6.2, 6.3 的其他檔案

---

### Phase 3: 處理重複/模糊檔案

**需要審查的檔案**:
1. `01_fundamentals/opencv_fundamentals.ipynb` - 合併到 2.1.2 或刪除
2. `02_core_operations/image_io_display_modern.ipynb` - 合併到 2.2.1 或刪除
3. `02_core_operations/image_processing_techniques.ipynb` - 審查內容，可能拆分到多個 WBS 項目
4. `03_preprocessing/advanced_image_operations.ipynb` - 可能屬於 3.2.x
5. `04_feature_detection/feature_detection_basics.ipynb` - 可能拆分到 4.1.x

---

### Phase 4: 更新文檔引用

**需要更新的檔案**:
1. `ULTIMATE_PROJECT_GUIDE.md` - 更新所有檔案名稱引用
2. `CLAUDE.md` - 更新範例路徑
3. `README.md` - 更新教學路徑
4. `FILENAME_MIGRATION_PLAN.md` - 更新對照表
5. 各目錄的 `README.md` - 更新檔案列表

---

## 📊 統計摘要

### 需要重新命名的檔案數量
- **Stage 2.1**: 3 個檔案 ✅
- **Stage 2.2**: 3 個檔案 ✅ + 1 個待創建 ⏳
- **Stage 3.1**: 1 個檔案 ✅ + 2 個待創建 ⏳
- **Stage 6.1**: 4 個檔案 ✅

**總計**: 11 個現有檔案需重新命名

### 需要創建的檔案數量
- **Stage 2.2**: 1 個 (color_spaces)
- **Stage 3.1**: 2 個 (morphological_ops, edge_detection)
- **Stage 3.2**: 2 個 (histogram_processing, noise_reduction)
- **Stage 4.x**: 5 個
- **Stage 5.x**: 5 個
- **Stage 6.2**: 3 個
- **Stage 6.3**: 3 個

**總計**: 21 個新檔案需創建

### 需要審查處理的檔案
- `opencv_fundamentals.ipynb`
- `image_io_display_modern.ipynb`
- `image_processing_techniques.ipynb`
- `advanced_image_operations.ipynb`
- `feature_detection_basics.ipynb`

**總計**: 5 個檔案需審查

---

## ⚠️ 注意事項

1. **Git 版本控制**: 所有重新命名應該使用 `git mv` 保留檔案歷史
2. **測試驗證**: 重新命名後需驗證所有 notebook 仍可執行
3. **導入路徑**: 檢查是否有 Python 檔案 import 這些 notebooks
4. **文檔同步**: 確保所有文檔引用都已更新
5. **備份**: 重新命名前建議先建立 git branch 或備份

---

## ✅ 驗證清單

重新命名完成後，檢查以下項目:

- [ ] 所有教學檔案都有 WBS 編號前綴
- [ ] 沒有重複或矛盾的檔案名稱
- [ ] 所有 notebooks 可以正常執行
- [ ] ULTIMATE_PROJECT_GUIDE.md 中的檔案名稱已更新
- [ ] README.md 中的路徑已更新
- [ ] CLAUDE.md 中的範例路徑已更新
- [ ] Git 提交記錄清晰完整
- [ ] 文檔引用已全部更新

---

**建立日期**: 2025-10-12
**版本**: v1.0
**狀態**: 待執行

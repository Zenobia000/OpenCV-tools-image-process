# 📝 檔案名稱重構計劃 (Filename Migration Plan)

## 🎯 重構目標

基於新的專案架構和 WBS，將現有檔案重新命名為更具語義性和組織性的名稱，以提升專案的可維護性和學習體驗。

---

## 📋 檔案重構對應表

### 🔄 主要教學檔案重構

| 原始檔案名稱 | 新檔案名稱 | 目標位置 | 重構原因 |
|-------------|------------|----------|----------|
| `Day0_py_np.ipynb` | `python_numpy_basics.ipynb` | `01_fundamentals/` | 更具描述性，明確表達內容 |
| `Day1_OpenCV.ipynb` | `opencv_fundamentals.ipynb` | `01_fundamentals/` | 符合模組化命名規範 |
| `Day2_OpenCV.ipynb` | `image_processing_techniques.ipynb` | `02_core_operations/` | 反映實際內容重點 |
| `Day2_OpenCV2.ipynb` | `advanced_image_operations.ipynb` | `03_preprocessing/` | 避免重複命名，明確難度級別 |
| `Day3_OpenCV.ipynb` | `feature_detection_basics.ipynb` | `04_feature_detection/` | 明確功能定位 |

### 📝 練習檔案重構

| 原始檔案名稱 | 新檔案名稱 | 目標位置 | 內容描述 |
|-------------|------------|----------|----------|
| `HW/Q_02.ipynb` | `bgr_channel_operations.ipynb` | `06_exercises/beginner/` | BGR色彩通道操作練習 |
| `HW/Q_04_08.ipynb` | `drawing_functions_practice.ipynb` | `06_exercises/beginner/` | 繪圖函數與幾何變換練習 |
| `HW/Q_09_12.ipynb` | `filtering_applications.ipynb` | `06_exercises/beginner/` | 濾波器應用實作練習 |
| `HW/Q_15.ipynb` | `comprehensive_basics.ipynb` | `06_exercises/beginner/` | 基礎技術綜合應用 |

### 📚 文檔檔案重構

| 原始檔案名稱 | 新檔案名稱 | 目標位置 | 用途說明 |
|-------------|------------|----------|----------|
| `OpenCV/README.md` | `legacy_opencv_readme.md` | `docs/legacy/` | 保留原始說明作為參考 |
| `DLIB_GUIDE.md` | `dlib_integration_guide.md` | `docs/guides/` | dlib專項整合指南 |
| `INSTALLATION.md` | `installation_guide.md` | `docs/setup/` | 安裝配置指南 |
| `REORGANIZATION.md` | `project_reorganization_history.md` | `docs/legacy/` | 專案重組歷史記錄 |

---

## 🏗️ 新增檔案建議

### 基礎教學模組檔案

```
01_fundamentals/
├── python_numpy_basics.ipynb          # 已存在，需移動
├── opencv_fundamentals.ipynb          # 已存在，需重構
├── computer_vision_concepts.ipynb     # 新增：電腦視覺概念
├── opencv_installation.md             # 新增：安裝指南
└── environment_setup.py               # 新增：環境驗證腳本
```

### 核心操作模組檔案

```
02_core_operations/
├── image_io_display.ipynb             # 新增：圖像I/O操作
├── basic_transformations.ipynb        # 新增：基礎變換
├── color_spaces.ipynb                 # 新增：色彩空間
├── image_processing_techniques.ipynb  # 重構自Day2_OpenCV.ipynb
└── pixel_operations.ipynb             # 新增：像素級操作
```

### 前處理技術模組檔案

```
03_preprocessing/
├── filtering_smoothing.ipynb          # 已存在
├── morphological_ops.ipynb            # 新增：形態學操作
├── edge_detection.ipynb               # 新增：邊緣檢測
├── histogram_processing.ipynb         # 新增：直方圖處理
├── noise_reduction.ipynb              # 新增：降噪技術
└── advanced_image_operations.ipynb    # 重構自Day2_OpenCV2.ipynb
```

### 特徵檢測模組檔案

```
04_feature_detection/
├── corner_detection.ipynb             # 新增：角點檢測
├── feature_descriptors.ipynb          # 新增：特徵描述子
├── object_tracking.ipynb              # 新增：物體追蹤
├── template_matching.ipynb            # 新增：模板匹配
└── feature_detection_basics.ipynb     # 重構自Day3_OpenCV.ipynb
```

### 機器學習整合模組檔案

```
05_machine_learning/
├── face_detection.ipynb               # 新增：人臉檢測
├── object_classification.ipynb        # 新增：物體分類
├── dlib_integration.ipynb             # 新增：dlib整合
├── dnn_integration.ipynb              # 新增：深度學習整合
└── real_time_detection.ipynb          # 新增：實時檢測
```

---

## 🔧 檔案重構執行腳本

### 自動化重構腳本 (migrate_files.py)

```python
#!/usr/bin/env python3
"""
檔案名稱重構自動化腳本
用途：根據重構計劃自動移動和重命名檔案
"""

import os
import shutil
from pathlib import Path

# 重構對應表
MIGRATION_MAP = {
    # 教學檔案重構
    "OpenCV/Day0_py_np.ipynb": "01_fundamentals/python_numpy_basics.ipynb",
    "OpenCV/Day1_OpenCV.ipynb": "01_fundamentals/opencv_fundamentals.ipynb",
    "OpenCV/Day2_OpenCV.ipynb": "02_core_operations/image_processing_techniques.ipynb",
    "OpenCV/Day2_OpenCV2.ipynb": "03_preprocessing/advanced_image_operations.ipynb",
    "OpenCV/Day3_OpenCV.ipynb": "04_feature_detection/feature_detection_basics.ipynb",

    # 練習檔案重構
    "OpenCV/HW/Q_02.ipynb": "06_exercises/beginner/bgr_channel_operations.ipynb",
    "OpenCV/HW/Q_04_08.ipynb": "06_exercises/beginner/drawing_functions_practice.ipynb",
    "OpenCV/HW/Q_09_12.ipynb": "06_exercises/beginner/filtering_applications.ipynb",
    "OpenCV/HW/Q_15.ipynb": "06_exercises/beginner/comprehensive_basics.ipynb",

    # 文檔檔案重構
    "OpenCV/README.md": "docs/legacy/legacy_opencv_readme.md",
    "OpenCV/DLIB_GUIDE.md": "docs/guides/dlib_integration_guide.md",
    "OpenCV/INSTALLATION.md": "docs/setup/installation_guide.md",
    "OpenCV/REORGANIZATION.md": "docs/legacy/project_reorganization_history.md",
}

def create_directories():
    """建立必要的目錄結構"""
    directories = [
        "01_fundamentals", "02_core_operations", "03_preprocessing",
        "04_feature_detection", "05_machine_learning", "06_exercises/beginner",
        "06_exercises/intermediate", "06_exercises/advanced",
        "07_projects", "docs/legacy", "docs/guides", "docs/setup"
    ]

    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✅ 建立目錄: {directory}")

def migrate_files():
    """執行檔案重構"""
    print("🔄 開始檔案重構...")

    for old_path, new_path in MIGRATION_MAP.items():
        if os.path.exists(old_path):
            # 確保目標目錄存在
            os.makedirs(os.path.dirname(new_path), exist_ok=True)

            # 移動檔案
            shutil.move(old_path, new_path)
            print(f"✅ 重構: {old_path} -> {new_path}")
        else:
            print(f"⚠️  檔案不存在: {old_path}")

def update_imports_and_paths():
    """更新檔案中的路徑引用"""
    # 這個函數需要根據實際檔案內容進行客製化
    print("🔄 更新檔案路徑引用...")

    # 範例：更新圖片路徑
    path_updates = {
        r"image/": "../assets/images/",
        r"model/": "../assets/models/",
        r"video/": "../assets/videos/",
    }

    # 實際實現需要遍歷所有 .ipynb 檔案並更新路徑

def main():
    """主執行函數"""
    print("🚀 開始檔案名稱重構...")

    # 1. 建立目錄結構
    create_directories()

    # 2. 執行檔案重構
    migrate_files()

    # 3. 更新路徑引用
    update_imports_and_paths()

    print("✅ 檔案重構完成！")

if __name__ == "__main__":
    main()
```

---

## 📋 重構檢查清單

### 重構前準備
- [ ] 備份整個專案目錄
- [ ] 確認所有檔案都已提交到 Git
- [ ] 檢查檔案是否正在使用中
- [ ] 記錄當前目錄結構

### 重構執行步驟
- [ ] 執行目錄建立腳本
- [ ] 運行檔案重構腳本
- [ ] 驗證所有檔案都已正確移動
- [ ] 更新檔案內部路徑引用
- [ ] 更新 README.md 中的路徑連結

### 重構後驗證
- [ ] 測試所有 Jupyter Notebook 是否可正常開啟
- [ ] 驗證圖片/模型路徑是否正確
- [ ] 檢查匯入語句是否需要更新
- [ ] 運行基本功能測試
- [ ] 更新專案文檔

---

## 🎯 命名規範

### 檔案命名原則
1. **小寫字母 + 底線**: 如 `image_processing.ipynb`
2. **動詞 + 名詞**: 如 `detect_faces.py`
3. **功能導向**: 檔名清楚表達功能
4. **避免縮寫**: 使用完整單詞，提高可讀性
5. **分層命名**: 依難度/模組分層組織

### 目錄命名原則
1. **數字前綴**: 表達學習順序，如 `01_fundamentals`
2. **動詞導向**: 如 `preprocessing`, `detection`
3. **模組化**: 每個目錄代表一個學習模組
4. **一致性**: 所有目錄遵循相同命名格式

### 變數與函數命名
1. **蛇形命名法**: `load_image()`, `face_detector`
2. **描述性**: 名稱清楚表達用途
3. **避免單字母**: 除了迴圈變數外
4. **統一術語**: 相同概念使用相同術語

---

## 🔄 段階性實施

### Phase 1: 核心檔案重構 (Week 1)
- 移動並重命名主要教學檔案
- 建立新的目錄結構
- 更新主要路徑引用

### Phase 2: 練習檔案重構 (Week 1)
- 重構所有練習檔案
- 建立練習分級系統
- 添加自動評分機制

### Phase 3: 文檔整理 (Week 2)
- 重組所有文檔檔案
- 建立文檔分類系統
- 更新所有文檔連結

### Phase 4: 驗證與測試 (Week 2)
- 全面功能測試
- 路徑驗證
- 使用者體驗測試

---

## ⚠️ 注意事項

1. **備份重要性**: 執行重構前務必完整備份
2. **Git 追蹤**: 使用 `git mv` 保留檔案歷史
3. **相依性檢查**: 注意檔案間的相依關係
4. **段階實施**: 分階段執行，每階段都要驗證
5. **回滾計劃**: 準備回滾腳本以防出錯

---

## 📊 重構效益

### 短期效益
- **提升可讀性**: 檔名更直觀易懂
- **改善組織**: 模組化結構更清晰
- **減少混淆**: 避免 Day1/Day2 等模糊命名
- **標準化**: 統一命名規範

### 長期效益
- **維護便利**: 更容易找到和修改檔案
- **擴展性**: 新內容更容易整合
- **協作效率**: 團隊成員更容易理解結構
- **學習體驗**: 學習者更容易導航

---

此重構計劃將大幅提升專案的組織性和可維護性，為後續的開發和使用奠定良好基礎。
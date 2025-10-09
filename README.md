# 🎯 OpenCV Computer Vision Toolkit

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.8%2B-green.svg)](https://opencv.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**現代化電腦視覺與影像處理學習工具包** - 基於2025年最新技術標準重新設計

## 🌟 專案特色

- **🏗️ 模組化架構**: 清晰的學習路徑，從基礎到進階
- **🛠️ 實用工具**: 內建常用影像處理函數庫
- **📊 性能評估**: 完整的效能分析與比較工具
- **🎯 實戰導向**: 真實場景應用專案
- **📚 豐富資源**: 345+測試圖片、預訓練模型

## 📁 專案架構

```
OpenCV-Computer-Vision-Toolkit/
├── 📚 01_fundamentals/              # 基礎知識
├── 🎯 02_core_operations/           # 核心操作
├── 🔧 03_preprocessing/             # 前處理技術
├── 🧠 04_feature_detection/         # 特徵檢測
├── 🤖 05_machine_learning/          # 機器學習
├── 📝 06_exercises/                 # 練習與作業
├── 🎬 07_projects/                  # 實戰專案
├── 📊 assets/                       # 資源檔案
└── 🛠️ utils/                        # 工具函數
```

## 🚀 快速開始

### 環境安裝

```bash
# 克隆專案
git clone https://github.com/your-username/OpenCV-Computer-Vision-Toolkit.git
cd OpenCV-Computer-Vision-Toolkit

# 建立虛擬環境 (推薦)
python -m venv cv_env
source cv_env/bin/activate  # Linux/Mac
# cv_env\\Scripts\\activate  # Windows

# 安裝依賴
pip install -r requirements.txt
```

### 驗證安裝

```python
import cv2
import numpy as np
from utils import image_utils, visualization, performance

print(f"OpenCV版本: {cv2.__version__}")
print(f"NumPy版本: {np.__version__}")
print("✅ 環境設置完成！")
```

## 📚 學習路徑

### 🎯 Level 1: 基礎入門 (1-2週)
- **Python/NumPy基礎**: `01_fundamentals/python_numpy_basics.ipynb`
- **OpenCV安裝與配置**: `01_fundamentals/opencv_installation.md`
- **圖像讀取與顯示**: `02_core_operations/image_io_display.ipynb`
- **基本變換操作**: `02_core_operations/basic_transformations.ipynb`

### 🔧 Level 2: 核心技術 (2-3週)
- **色彩空間轉換**: `02_core_operations/color_spaces.ipynb`
- **濾波與平滑**: `03_preprocessing/filtering_smoothing.ipynb`
- **形態學操作**: `03_preprocessing/morphological_ops.ipynb`
- **邊緣檢測**: `03_preprocessing/edge_detection.ipynb`

### 🧠 Level 3: 進階應用 (3-4週)
- **特徵檢測**: `04_feature_detection/corner_detection.ipynb`
- **物體檢測**: `05_machine_learning/face_detection.ipynb`
- **機器學習整合**: `05_machine_learning/object_classification.ipynb`

### 🎬 Level 4: 實戰專案 (4-6週)
- **監控系統**: `07_projects/security_camera/`
- **文檔掃描**: `07_projects/document_scanner/`
- **醫學影像分析**: `07_projects/medical_imaging/`

## 🛠️ 工具函數庫

### 影像處理工具 (`utils/image_utils.py`)
```python
from utils.image_utils import load_image, resize_image, normalize_image

# 載入並處理圖像
img = load_image('assets/images/basic/sample.jpg')
img_resized = resize_image(img, (640, 480), keep_aspect=True)
img_normalized = normalize_image(img_resized, range_type='0-1')
```

### 視覺化工具 (`utils/visualization.py`)
```python
from utils.visualization import display_image, display_multiple_images

# 顯示單張圖像
display_image(img, title="原始圖像")

# 並排比較
display_multiple_images([original, processed],
                       ["原始", "處理後"], rows=1)
```

### 性能評估 (`utils/performance.py`)
```python
from utils.performance import time_function, benchmark_function

# 函數計時
@time_function
def my_processing_function(image):
    return cv2.GaussianBlur(image, (15, 15), 0)

# 演算法比較
results = benchmark_function(cv2.GaussianBlur,
                           args=(img, (15, 15), 0),
                           iterations=100)
```

## 📊 資源檔案

### 測試圖片
- **`assets/images/basic/`**: 基礎測試圖片 (200+張)
- **`assets/images/faces/`**: 人臉檢測圖片
- **`assets/images/objects/`**: 物體識別圖片

### 預訓練模型
- **人臉檢測**: Haar Cascade, DNN模型
- **物體分類**: dlib分類器
- **深度學習**: 預訓練權重檔案

### 資料集
- **dlib ObjectCategories**: 10類物體分類資料集
- **Caltech101**: 物體識別基準資料集

## 🎯 實戰專案

### 1. 智能監控系統 (`07_projects/security_camera/`)
- 即時人臉檢測
- 動作檢測與追蹤
- 異常行為警報

### 2. 文檔掃描器 (`07_projects/document_scanner/`)
- 文檔邊界檢測
- 透視變換校正
- OCR文字識別

### 3. 醫學影像分析 (`07_projects/medical_imaging/`)
- X光影像增強
- 病灶區域分割
- 量化分析工具

### 4. 擴增實境 (`07_projects/augmented_reality/`)
- 標記追蹤
- 3D物體渲染
- 實時互動

## 📝 練習系統

### 初學者 (`06_exercises/beginner/`)
- Q_02.ipynb: BGR通道操作
- Q_04_08.ipynb: 繪圖函數練習
- Q_09_12.ipynb: 濾波器應用
- Q_15.ipynb: 綜合練習

### 中級練習 (`06_exercises/intermediate/`)
- 特徵點檢測與匹配
- 圖像拼接
- 物體追蹤

### 高級挑戰 (`06_exercises/advanced/`)
- 深度學習整合
- 實時處理優化
- 自定義演算法實現

## 🔧 進階配置

### Jupyter Lab 擴展
```bash
pip install jupyterlab
jupyter labextension install @jupyterlab/widgets
```

### GPU 加速 (可選)
```bash
# CUDA 支援
pip install opencv-contrib-python-headless[cuda]

# 檢查GPU支援
python -c "import cv2; print('CUDA:', cv2.cuda.getCudaEnabledDeviceCount())"
```

## 📈 效能基準

| 操作類型 | CPU (Intel i7) | GPU (RTX 3070) | 加速比 |
|---------|----------------|----------------|--------|
| 高斯模糊 | 15.2 ms | 2.3 ms | 6.6x |
| 邊緣檢測 | 12.8 ms | 1.9 ms | 6.7x |
| 形態學操作 | 8.4 ms | 1.2 ms | 7.0x |
| DNN推理 | 45.6 ms | 5.2 ms | 8.8x |

## 🤝 貢獻指南

1. **Fork** 本專案
2. 建立功能分支 (`git checkout -b feature/AmazingFeature`)
3. 提交變更 (`git commit -m 'Add some AmazingFeature'`)
4. 推送分支 (`git push origin feature/AmazingFeature`)
5. 開啟 **Pull Request**

## 📄 授權條款

本專案採用 MIT 授權條款 - 詳見 [LICENSE](LICENSE) 檔案

## 🎯 技術支援

- **文檔**: [完整API文檔](docs/api.md)
- **範例**: [程式碼範例庫](examples/)
- **FAQ**: [常見問題解答](docs/faq.md)
- **Issues**: [GitHub Issues](https://github.com/your-username/OpenCV-Computer-Vision-Toolkit/issues)

## 🌟 致謝

- OpenCV 開發團隊
- 原始教學材料貢獻者
- 開源社群支持

---

<div align="center">

**🚀 立即開始你的電腦視覺學習之旅！**

[開始學習](01_fundamentals/python_numpy_basics.ipynb) | [查看範例](06_exercises/beginner/) | [實戰專案](07_projects/)

Made with ❤️ by OpenCV Community

</div>
# OpenCV 安裝指南 - Poetry 環境版

## 🎯 本專案推薦方式：使用 Poetry

本專案已配置完整的 Poetry 環境，**推薦使用 Poetry 管理依賴**，無需手動安裝。

### ✅ 快速開始（推薦）

```bash
# 1. 確認 Poetry 已安裝
poetry --version

# 2. 安裝所有依賴（包含 OpenCV）
poetry install

# 3. 啟動環境
poetry shell

# 4. 驗證安裝
python -c "import cv2; print('OpenCV version:', cv2.__version__)"
```

**已安裝版本**：
- ✅ OpenCV: 4.12.0
- ✅ OpenCV Contrib: 4.12.0
- ✅ Python: 3.10.12

---

## 📚 傳統安裝方式（參考）

如果你不使用 Poetry，可以參考以下傳統安裝方式。

### 方法 1: 使用 pip（推薦）

#### Windows

```bash
# 基本安裝
pip install opencv-python

# 完整版本（包含額外模組）
pip install opencv-contrib-python

# 指定版本
pip install opencv-python==4.12.0
pip install opencv-contrib-python==4.12.0
```

#### Linux (Ubuntu/Debian)

```bash
# 更新套件清單
sudo apt update

# 安裝系統依賴
sudo apt install python3-dev python3-pip

# 安裝 OpenCV
pip3 install opencv-python opencv-contrib-python

# 如果需要 GUI 功能，安裝額外依賴
sudo apt install libgtk-3-dev
```

#### macOS

```bash
# 使用 Homebrew 安裝依賴
brew install python

# 安裝 OpenCV
pip3 install opencv-python opencv-contrib-python
```

### 方法 2: 使用 Anaconda

```bash
# 創建虛擬環境
conda create -n cv_env python=3.10

# 啟動環境
conda activate cv_env

# 安裝 OpenCV
conda install -c conda-forge opencv

# 或使用 pip
pip install opencv-python opencv-contrib-python
```

### 方法 3: 從源碼編譯（進階）

適合需要自定義配置或 GPU 加速的情況。

```bash
# Ubuntu 依賴
sudo apt install build-essential cmake git pkg-config
sudo apt install libgtk-3-dev libavcodec-dev libavformat-dev libswscale-dev
sudo apt install libv4l-dev libxvidcore-dev libx264-dev
sudo apt install libjpeg-dev libpng-dev libtiff-dev

# 下載源碼
git clone https://github.com/opencv/opencv.git
git clone https://github.com/opencv/opencv_contrib.git

# 編譯安裝
cd opencv
mkdir build && cd build
cmake -DOPENCV_EXTRA_MODULES_PATH=../../opencv_contrib/modules ..
make -j$(nproc)
sudo make install
```

---

## 🧪 驗證安裝

### 基本驗證

```python
import cv2
print("OpenCV version:", cv2.__version__)
```

### 完整驗證

```python
import cv2
import numpy as np

# 版本資訊
print(f"OpenCV version: {cv2.__version__}")
print(f"NumPy version: {np.__version__}")

# 測試圖像讀取
img = np.zeros((100, 100, 3), dtype=np.uint8)
print(f"Image shape: {img.shape}")

# 測試顯示功能（需要 GUI 環境）
# cv2.imshow('Test', img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

print("\n✅ OpenCV 安裝成功！")
```

### 使用本專案工具函數驗證

```python
from utils.image_utils import load_image, resize_image
from utils.visualization import display_image
import numpy as np

# 創建測試圖像
img = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)

# 測試工具函數
resized = resize_image(img, (50, 50))
print(f"Resized shape: {resized.shape}")

print("✅ 工具函數測試通過！")
```

---

## 🔧 常見問題排除

### 問題 1: ImportError: libGL.so.1

**Linux 環境缺少 GUI 依賴**

```bash
# Ubuntu/Debian
sudo apt install libgl1-mesa-glx libglib2.0-0

# CentOS/RHEL
sudo yum install mesa-libGL
```

### 問題 2: cv2.imshow() 無法顯示

**遠端 SSH 環境沒有 GUI**

```python
# 解決方案：使用 matplotlib 顯示
import matplotlib.pyplot as plt
import cv2

img = cv2.imread('image.jpg')
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.imshow(img_rgb)
plt.show()
```

或使用本專案的 visualization 模組：

```python
from utils.visualization import display_image
display_image(img, title="My Image")
```

### 問題 3: 版本衝突

```bash
# 卸載舊版本
pip uninstall opencv-python opencv-contrib-python

# 重新安裝
pip install opencv-python==4.12.0 opencv-contrib-python==4.12.0
```

### 問題 4: Poetry 環境中的問題

```bash
# 清除緩存
poetry cache clear pypi --all

# 重新安裝
rm -rf .venv poetry.lock
poetry install
```

### 問題 5: CUDA/GPU 支援

OpenCV pip 版本**不包含** CUDA 支援，需要從源碼編譯：

```bash
# 編譯時啟用 CUDA
cmake -DWITH_CUDA=ON \
      -DCUDA_ARCH_BIN=7.5 \
      -DWITH_CUDNN=ON \
      ..
```

---

## 📦 套件版本說明

### opencv-python vs opencv-contrib-python

| 套件 | 說明 | 適用情況 |
|------|------|---------|
| `opencv-python` | 核心模組 | 基本圖像處理 |
| `opencv-contrib-python` | 核心 + 額外模組 | 需要 SIFT/SURF 等進階功能 |

### 本專案使用

```toml
# pyproject.toml
opencv-python = ">=4.8.0"
opencv-contrib-python = ">=4.8.0"
```

**建議**: 使用 contrib 版本，包含完整功能。

---

## 🎓 學習資源

### 官方文檔
- OpenCV 官網: https://opencv.org/
- OpenCV Python 教程: https://docs.opencv.org/4.x/d6/d00/tutorial_py_root.html
- GitHub: https://github.com/opencv/opencv

### 本專案資源
- `01_fundamentals/opencv_fundamentals.ipynb` - OpenCV 基礎
- `01_fundamentals/python_numpy_basics.ipynb` - Python/NumPy 基礎
- `utils/` - 工具函數庫
- `tests/` - 單元測試範例

---

## ✅ 確認清單

完成安裝後，確認以下項目：

- [ ] Python 3.8+ 已安裝
- [ ] OpenCV 4.8+ 已安裝
- [ ] NumPy 已安裝
- [ ] Matplotlib 已安裝（用於顯示圖像）
- [ ] 可以正常 `import cv2`
- [ ] 測試圖像讀取與顯示功能
- [ ] 本專案工具函數可正常使用

---

## 🚀 下一步

安裝完成後，建議依序學習：

1. **Python/NumPy 基礎** → `01_fundamentals/python_numpy_basics.ipynb`
2. **OpenCV 基礎** → `01_fundamentals/opencv_fundamentals.ipynb`
3. **圖像 I/O** → `02_core_operations/image_io_display_modern.ipynb`
4. **前處理技術** → `03_preprocessing/filtering_smoothing.ipynb`

---

**最後更新**: 2025-10-12
**OpenCV 版本**: 4.12.0
**Python 版本**: 3.10.12
**環境管理**: Poetry 2.2.1

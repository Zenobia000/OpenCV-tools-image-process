# OpenCV 與 dlib 安裝指南

## 🚀 快速安裝

### 📋 系統需求

- **Python 3.7+** (建議 3.8 或 3.9)
- **pip** 最新版本
- **C++ 編譯器** (Windows: Visual Studio, Linux/Mac: gcc)
- **CMake 3.12+** (dlib 編譯需要)

## 🔧 基礎環境安裝

### 1. Python 環境設置

#### 建立虛擬環境 (建議)
```bash
# 建立虛擬環境
python -m venv opencv_env

# 啟動虛擬環境
# Windows
opencv_env\\Scripts\\activate
# Linux/Mac
source opencv_env/bin/activate

# 升級 pip
pip install --upgrade pip
```

#### 安裝基礎套件
```bash
# 核心套件
pip install numpy matplotlib jupyter

# OpenCV
pip install opencv-python opencv-contrib-python

# 圖像處理相關
pip install Pillow scikit-image

# 機器學習
pip install scikit-learn
```

### 2. OpenCV 驗證安裝

```python
import cv2
import numpy as np

print(f"OpenCV 版本: {cv2.__version__}")

# 測試基本功能
img = np.zeros((100, 100, 3), dtype=np.uint8)
cv2.circle(img, (50, 50), 30, (0, 255, 0), -1)
print("OpenCV 基本功能正常！")
```

## 🤖 dlib 安裝指南

### 方案一：使用預編譯 Wheel (推薦)

#### Windows (Python 3.9)
```bash
# 使用專案提供的預編譯檔案
pip install install/dlib-19.22.99-cp39-cp39-win_amd64.whl

# 驗證安裝
python -c "import dlib; print(f'dlib 版本: {dlib.version}')"
```

#### 其他版本下載
如果您的 Python 版本不是 3.9，請從以下來源下載對應版本：
- [官方 PyPI](https://pypi.org/project/dlib/#files)
- [Christoph Gohlke 預編譯檔案](https://www.lfd.uci.edu/~gohlke/pythonlibs/#dlib)

### 方案二：從源碼編譯

#### 安裝編譯依賴
```bash
# Windows (需要 Visual Studio)
# 下載並安裝 Visual Studio Community
# 確保勾選 "C++ 桌面開發" 工作負載

# Linux (Ubuntu/Debian)
sudo apt-get update
sudo apt-get install build-essential cmake pkg-config
sudo apt-get install libx11-dev libatlas-base-dev
sudo apt-get install libgtk-3-dev libboost-python-dev

# macOS
brew install cmake boost
```

#### 編譯安裝
```bash
# 安裝 dlib
pip install dlib

# 如果上述方法失敗，嘗試從源碼安裝
git clone https://github.com/davisking/dlib.git
cd dlib
python setup.py build
python setup.py install
```

### 方案三：使用 conda (推薦給 Anaconda 用戶)
```bash
# 使用 conda-forge 頻道
conda install -c conda-forge dlib

# 或使用 mamba (更快的套件管理器)
mamba install -c conda-forge dlib
```

## 📦 專案特定套件安裝

### face-recognition (基於 dlib)
```bash
# Windows 用戶建議先安裝 dlib，再安裝 face-recognition
pip install face-recognition

# 如果安裝失敗，嘗試：
pip install cmake
pip install dlib
pip install face-recognition
```

### Tesseract OCR (中文識別)

#### 安裝 Tesseract 引擎
```bash
# Windows
# 下載安裝程式：https://github.com/UB-Mannheim/tesseract/wiki
# 安裝後將路徑添加到 PATH 環境變數

# Linux (Ubuntu)
sudo apt-get install tesseract-ocr tesseract-ocr-chi-sim tesseract-ocr-chi-tra

# macOS
brew install tesseract
```

#### 安裝 Python 介面
```bash
pip install pytesseract
```

#### 配置中文語言包
```bash
# 將專案中的語言包複製到 Tesseract 資料夾
# Windows 預設路徑：C:\\Program Files\\Tesseract-OCR\\tessdata\\
# Linux 預設路徑：/usr/share/tesseract-ocr/4.00/tessdata/

# 複製語言包檔案
cp install/chi_sim.traineddata /usr/share/tesseract-ocr/4.00/tessdata/
cp install/chi_tra.traineddata /usr/share/tesseract-ocr/4.00/tessdata/
```

## 🔍 驗證完整安裝

### 建立測試腳本
```python
# test_installation.py
import sys

def test_basic_packages():
    """測試基礎套件"""
    try:
        import numpy as np
        import matplotlib.pyplot as plt
        import cv2
        print("✅ 基礎套件安裝成功")
        print(f"   NumPy: {np.__version__}")
        print(f"   OpenCV: {cv2.__version__}")
        return True
    except ImportError as e:
        print(f"❌ 基礎套件安裝失敗: {e}")
        return False

def test_dlib():
    """測試 dlib"""
    try:
        import dlib
        print(f"✅ dlib 安裝成功 (版本: {dlib.version})")

        # 測試人臉檢測器
        detector = dlib.get_frontal_face_detector()
        print("✅ dlib 人臉檢測器正常")
        return True
    except ImportError:
        print("❌ dlib 未安裝或安裝失敗")
        return False
    except Exception as e:
        print(f"❌ dlib 功能測試失敗: {e}")
        return False

def test_optional_packages():
    """測試可選套件"""
    optional_packages = {
        'face_recognition': 'face-recognition',
        'pytesseract': 'pytesseract',
        'sklearn': 'scikit-learn'
    }

    for module, package in optional_packages.items():
        try:
            __import__(module)
            print(f"✅ {package} 安裝成功")
        except ImportError:
            print(f"⚠️  {package} 未安裝 (可選)")

def test_models():
    """測試模型檔案"""
    import os
    models = [
        'model/haarcascade_frontalface_default.xml',
        'model/shape_predictor_68_face_landmarks.dat',
        'model/dlib_face_recognition_resnet_model_v1.dat'
    ]

    for model in models:
        if os.path.exists(model):
            print(f"✅ 模型檔案存在: {model}")
        else:
            print(f"⚠️  模型檔案不存在: {model}")

if __name__ == "__main__":
    print("🔍 開始檢查安裝狀態...")
    print(f"Python 版本: {sys.version}")
    print("=" * 50)

    test_basic_packages()
    print()
    test_dlib()
    print()
    test_optional_packages()
    print()
    test_models()

    print("=" * 50)
    print("🎉 安裝檢查完成!")
```

### 執行測試
```bash
python test_installation.py
```

## 🐛 常見問題解決

### 問題 1: dlib 編譯失敗
```bash
# 錯誤訊息: Microsoft Visual C++ 14.0 is required

# 解決方案：
# 1. 下載並安裝 Visual Studio Community
# 2. 或下載 Microsoft C++ Build Tools
# 3. 確保勾選 "MSVC v143 - VS 2022 C++ x64/x86"
```

### 問題 2: OpenCV 圖片顯示失敗
```python
# 錯誤：cv2.imshow() 無反應

# 解決方案：
import cv2
img = cv2.imread('test.jpg')
if img is not None:
    cv2.imshow('Test', img)
    cv2.waitKey(0)  # 重要：等待按鍵
    cv2.destroyAllWindows()
else:
    print("圖片載入失敗，檢查路徑")
```

### 問題 3: 路徑問題
```python
# Windows 路徑問題解決
import os

# 使用 os.path.join 避免路徑分隔符問題
image_path = os.path.join('image', 'test.jpg')

# 或使用原始字串
image_path = r'image\test.jpg'

# 使用正斜線 (推薦)
image_path = 'image/test.jpg'
```

### 問題 4: 中文路徑問題
```python
# 避免中文路徑
# 錯誤：cv2.imread('圖片/測試.jpg')
# 正確：cv2.imread('images/test.jpg')

# 如果必須使用中文路徑：
import cv2
import numpy as np

def cv2_imread_chinese(filepath):
    """支援中文路徑的圖片讀取"""
    return cv2.imdecode(np.fromfile(filepath, dtype=np.uint8), cv2.IMREAD_COLOR)
```

## 📝 安裝檢查清單

- [ ] Python 3.7+ 已安裝
- [ ] 虛擬環境已建立並啟動
- [ ] OpenCV 基礎功能正常
- [ ] dlib 安裝成功並可載入模型
- [ ] 測試腳本全部通過
- [ ] 中文 OCR (如需要) 配置完成
- [ ] 專案路徑設定正確

## 🔗 額外資源

### 下載連結
- [Visual Studio Community](https://visualstudio.microsoft.com/zh-hant/vs/community/)
- [CMake](https://cmake.org/download/)
- [Tesseract](https://github.com/UB-Mannheim/tesseract/wiki)

### 替代安裝方式
- [Docker 容器](https://hub.docker.com/r/spmallick/opencv-docker)
- [Anaconda 整合包](https://anaconda.org/conda-forge/opencv)

## 💡 效能優化建議

### 1. 使用預編譯的 OpenCV
```bash
# 安裝包含優化版本的 OpenCV
pip install opencv-contrib-python-headless  # 無 GUI 版本，適合伺服器
```

### 2. 啟用多執行緒
```python
import cv2
# 設定 OpenCV 執行緒數
cv2.setNumThreads(4)
```

### 3. 記憶體管理
```python
# 釋放大物件
del large_image
import gc
gc.collect()
```

完成安裝後，請參考 `README.md` 開始學習之旅！
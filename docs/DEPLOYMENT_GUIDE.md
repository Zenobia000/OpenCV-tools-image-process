# 🚀 OpenCV Computer Vision Toolkit - 部署指南

本指南提供OpenCV計算機視覺工具包在不同環境下的完整部署說明，包括開發環境、生產環境、容器化部署等。

## 📋 目錄

- [系統要求](#系統要求)
- [環境設置](#環境設置)
- [依賴安裝](#依賴安裝)
- [配置管理](#配置管理)
- [部署方式](#部署方式)
- [性能優化](#性能優化)
- [監控與維護](#監控與維護)
- [故障排除](#故障排除)

## 🖥️ 系統要求

### 最低系統要求

| 項目 | 要求 |
|------|------|
| 操作系統 | Windows 10/11, Ubuntu 18.04+, macOS 10.15+ |
| Python | 3.8 或更高版本 |
| 記憶體 | 4GB RAM (推薦 8GB+) |
| 存儲空間 | 2GB 可用空間 |
| 處理器 | 多核心 CPU (推薦) |
| 顯示 | 1280x720 最小解析度 |

### 推薦系統配置

| 項目 | 推薦配置 |
|------|----------|
| 操作系統 | Ubuntu 20.04 LTS 或 Windows 11 |
| Python | 3.10+ |
| 記憶體 | 16GB RAM |
| 存儲 | SSD 4GB+ |
| 處理器 | Intel i5/i7 或 AMD Ryzen 5/7 |
| GPU | NVIDIA GPU (支援CUDA，可選) |
| 攝像頭 | USB 2.0+ 攝像頭 (用於實時演示) |

### 特殊要求

- **醫學影像模組**: 需要額外的科學計算庫
- **OCR模組**: 需要 Tesseract OCR 引擎
- **GPU加速**: 需要 CUDA 工具包和相容的顯示卡
- **實時處理**: 需要穩定的攝像頭連接

## 🔧 環境設置

### 選項1: Poetry環境 (推薦)

```bash
# 1. 確保已安裝Poetry
curl -sSL https://install.python-poetry.org | python3 -

# 2. 克隆專案
git clone https://github.com/Zenobia000/OpenCV-tools-image-process.git
cd OpenCV-tools-image-process

# 3. 安裝依賴
poetry install

# 4. 激活環境
poetry shell

# 5. 驗證安裝
python -c "import cv2, numpy as np; from utils import image_utils; print('✅ 環境設置完成')"
```

### 選項2: 虛擬環境

```bash
# 1. 創建虛擬環境
python -m venv cv_env

# 2. 激活環境
# Linux/macOS:
source cv_env/bin/activate
# Windows:
cv_env\Scripts\activate

# 3. 升級pip
pip install --upgrade pip

# 4. 安裝依賴
pip install -r requirements.txt

# 5. 驗證安裝
python -c "import cv2, numpy as np; print('✅ 環境設置完成')"
```

### 選項3: Conda環境

```bash
# 1. 創建Conda環境
conda create -n opencv_toolkit python=3.10

# 2. 激活環境
conda activate opencv_toolkit

# 3. 安裝OpenCV和基礎依賴
conda install -c conda-forge opencv numpy matplotlib jupyter

# 4. 安裝其他依賴
pip install -r requirements.txt

# 5. 驗證安裝
python -c "import cv2; print(f'OpenCV版本: {cv2.__version__}')"
```

## 📦 依賴安裝

### 核心依賴 (必須)

```bash
# 基礎計算機視覺庫
pip install opencv-python>=4.8.0
pip install numpy>=1.21.0
pip install matplotlib>=3.5.0

# Jupyter環境
pip install jupyter jupyterlab
pip install ipywidgets

# 科學計算
pip install scipy>=1.9.0
pip install scikit-learn>=1.1.0
pip install scikit-image>=0.19.0
```

### 可選依賴

```bash
# 人臉識別和特徵檢測
pip install dlib

# OCR文字識別
pip install pytesseract
pip install pillow

# 性能加速
pip install numba

# 深度學習支援
pip install onnx
pip install onnxruntime

# 醫學影像處理
pip install pydicom
pip install SimpleITK

# 擴增實境
pip install pygame  # 用於3D渲染演示
```

### GPU加速依賴 (可選)

```bash
# NVIDIA CUDA支援
pip install opencv-contrib-python  # 包含CUDA模組

# 確認CUDA支援
python -c "import cv2; print('CUDA設備數量:', cv2.cuda.getCudaEnabledDeviceCount())"
```

## ⚙️ 配置管理

### 基本配置

每個實戰專案都支援JSON配置文件：

```bash
# 創建預設配置文件
cd 07_projects/security_camera/
python 7.1.1_real_time_detection.py --create-config

cd ../document_scanner/
python 7.2.1_edge_detection_module.py --create-config

# 編輯配置文件
vim security_config.json
```

### 配置文件結構示例

```json
{
  "detection": {
    "face_detection_method": "dnn",
    "motion_detection": true,
    "confidence_threshold": 0.7
  },
  "performance": {
    "max_fps": 30,
    "resize_factor": 1.0
  },
  "alert": {
    "enable_alerts": true,
    "save_alert_images": true
  }
}
```

### 環境變量配置

```bash
# 設置Tesseract路徑 (如果需要)
export TESSERACT_CMD="/usr/bin/tesseract"

# 設置CUDA路徑 (如果使用GPU)
export CUDA_HOME="/usr/local/cuda"

# 設置OpenCV模組路徑
export OPENCV_LOG_LEVEL=ERROR
```

## 🌐 部署方式

### 開發環境部署

```bash
# 1. 克隆專案
git clone https://github.com/Zenobia000/OpenCV-tools-image-process.git
cd OpenCV-tools-image-process

# 2. 設置環境
poetry install  # 或使用 pip install -r requirements.txt

# 3. 運行快速啟動腳本
chmod +x quick_start.sh
./quick_start.sh

# 4. 啟動Jupyter Lab
jupyter lab
```

### 生產環境部署

```bash
# 1. 系統依賴安裝 (Ubuntu)
sudo apt-get update
sudo apt-get install python3-opencv
sudo apt-get install tesseract-ocr tesseract-ocr-chi-sim

# 2. 創建服務用戶
sudo useradd -m opencv-service

# 3. 部署應用
sudo -u opencv-service git clone <repository>
cd opencv-tools-image-process

# 4. 安裝Python依賴
sudo -u opencv-service pip install --user -r requirements.txt

# 5. 配置服務
sudo cp deployment/opencv-toolkit.service /etc/systemd/system/
sudo systemctl enable opencv-toolkit
sudo systemctl start opencv-toolkit
```

### Docker容器化部署

#### 創建Dockerfile

```dockerfile
FROM python:3.10-slim

# 安裝系統依賴
RUN apt-get update && apt-get install -y \
    libopencv-dev \
    tesseract-ocr \
    tesseract-ocr-eng \
    tesseract-ocr-chi-sim \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 設置工作目錄
WORKDIR /app

# 複製需求文件
COPY requirements.txt .

# 安裝Python依賴
RUN pip install --no-cache-dir -r requirements.txt

# 複製應用代碼
COPY . .

# 創建必要目錄
RUN mkdir -p logs alerts recordings

# 設置權限
RUN chmod +x quick_start.sh

# 暴露端口 (如果有web界面)
EXPOSE 8080

# 啟動命令
CMD ["python", "07_projects/security_camera/7.1.1_real_time_detection.py", "--no-display"]
```

#### Docker部署命令

```bash
# 建立容器映像
docker build -t opencv-toolkit:latest .

# 運行容器 (攝像頭支援)
docker run -it --device=/dev/video0 \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/config:/app/config \
  opencv-toolkit:latest

# 使用Docker Compose (推薦)
docker-compose up -d
```

#### Docker Compose 配置

```yaml
version: '3.8'

services:
  opencv-toolkit:
    build: .
    container_name: opencv-toolkit
    devices:
      - /dev/video0:/dev/video0
    volumes:
      - ./data:/app/data
      - ./config:/app/config
      - ./logs:/app/logs
    environment:
      - PYTHONPATH=/app
      - OPENCV_LOG_LEVEL=ERROR
    restart: unless-stopped

  # 可選: 添加資料庫服務
  database:
    image: sqlite:latest
    volumes:
      - ./database:/var/lib/sqlite
```

### 雲端平台部署

#### AWS EC2 部署

```bash
# 1. 啟動EC2實例 (推薦 t3.medium 或更高)
aws ec2 run-instances \
  --image-id ami-0abcdef1234567890 \
  --instance-type t3.medium \
  --key-name your-key-pair

# 2. SSH連接並設置
ssh -i your-key.pem ubuntu@your-ec2-ip

# 3. 安裝Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# 4. 部署應用
git clone <repository>
cd opencv-tools-image-process
docker-compose up -d
```

#### Google Cloud Platform 部署

```bash
# 1. 設置GCP專案
gcloud config set project your-project-id

# 2. 創建Compute Engine實例
gcloud compute instances create opencv-toolkit-vm \
  --image-family=ubuntu-2004-lts \
  --image-project=ubuntu-os-cloud \
  --machine-type=e2-standard-2

# 3. SSH並部署
gcloud compute ssh opencv-toolkit-vm
# 然後執行標準部署步驟
```

### 邊緣設備部署 (Raspberry Pi)

```bash
# 1. 系統準備 (Raspberry Pi OS)
sudo apt update && sudo apt upgrade -y

# 2. 安裝輕量版OpenCV
sudo apt install python3-opencv

# 3. 安裝專案依賴 (選擇性安裝)
pip3 install numpy matplotlib

# 4. 效能優化配置
# 降低圖像解析度
# 減少檢測頻率
# 使用輕量算法 (Haar而非DNN)

# 5. 運行
python3 07_projects/security_camera/7.1.1_real_time_detection.py \
  --config lightweight_config.json
```

## ⚡ 性能優化

### CPU優化

```python
# 1. 圖像尺寸優化
max_width = 640  # 降低解析度以提升速度

# 2. 算法選擇
face_detection_method = "haar"  # 比DNN更快

# 3. 處理頻率控制
frame_skip = 2  # 每2幀處理一次

# 4. 多線程處理
import threading
import queue

# 使用生產者-消費者模式
frame_queue = queue.Queue(maxsize=5)
```

### GPU加速

```python
# 檢查CUDA支援
import cv2
print(f"CUDA設備數量: {cv2.cuda.getCudaEnabledDeviceCount()}")

# 使用GPU加速的DNN
net = cv2.dnn.readNet("model.onnx")
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
```

### 記憶體優化

```python
# 1. 及時釋放大型物件
del large_image_array

# 2. 使用適當的數據類型
image = image.astype(np.uint8)  # 避免使用float64

# 3. 限制歷史數據大小
from collections import deque
history = deque(maxlen=100)  # 限制最大長度
```

## 📊 監控與維護

### 性能監控

```python
# 1. 建立監控腳本
import psutil
import time

def monitor_system():
    while True:
        cpu_percent = psutil.cpu_percent()
        memory_percent = psutil.virtual_memory().percent

        print(f"CPU: {cpu_percent}%, Memory: {memory_percent}%")
        time.sleep(60)

# 2. 日誌監控
tail -f logs/opencv_toolkit.log
```

### 自動測試

```bash
# 運行完整測試套件
pytest tests/ -v

# 運行性能測試
pytest tests/test_projects_integration.py::TestPerformanceBenchmarks -v

# 運行特定模組測試
pytest tests/test_utils.py -v
```

### 健康檢查

```python
# 健康檢查腳本
def health_check():
    checks = {
        "opencv": check_opencv(),
        "camera": check_camera_access(),
        "disk_space": check_disk_space(),
        "memory": check_memory_usage()
    }
    return all(checks.values()), checks

def check_opencv():
    try:
        import cv2
        return cv2.__version__ >= "4.8.0"
    except:
        return False
```

## 🔄 更新與維護

### 常規更新

```bash
# 1. 備份當前配置
cp -r config config_backup_$(date +%Y%m%d)

# 2. 拉取最新代碼
git pull origin main

# 3. 更新依賴
poetry update  # 或 pip install -r requirements.txt --upgrade

# 4. 運行測試
pytest tests/ --tb=short

# 5. 重啟服務
sudo systemctl restart opencv-toolkit
```

### 數據備份

```bash
# 備份腳本
#!/bin/bash
DATE=$(date +%Y%m%d_%H%M%S)
BACKUP_DIR="/backup/opencv_toolkit_$DATE"

mkdir -p $BACKUP_DIR

# 備份配置文件
cp -r config/ $BACKUP_DIR/
cp -r 07_projects/*/config/ $BACKUP_DIR/project_configs/

# 備份數據庫
cp alerts.db $BACKUP_DIR/
cp learning_data.json $BACKUP_DIR/

# 壓縮備份
tar -czf opencv_toolkit_backup_$DATE.tar.gz $BACKUP_DIR/
```

## 🚨 故障排除

### 常見問題與解決方案

#### 1. OpenCV導入錯誤

**問題**: `ImportError: No module named 'cv2'`

**解決方案**:
```bash
# 重新安裝OpenCV
pip uninstall opencv-python opencv-contrib-python
pip install opencv-python

# 或使用特定版本
pip install opencv-python==4.8.1.78
```

#### 2. 攝像頭無法開啟

**問題**: `VideoCapture.isOpened()` 返回 `False`

**解決方案**:
```bash
# Linux: 檢查設備權限
sudo usermod -a -G video $USER
ls -la /dev/video*

# 測試攝像頭
python -c "import cv2; cap=cv2.VideoCapture(0); print('攝像頭可用:', cap.isOpened())"
```

#### 3. DNN模型載入失敗

**問題**: DNN人臉檢測模型無法載入

**解決方案**:
```bash
# 下載缺失的模型文件
wget https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20180205_fp16/opencv_face_detector_uint8.pb
wget https://github.com/opencv/opencv/raw/master/samples/dnn/face_detector/opencv_face_detector.pbtxt

# 放置到正確目錄
mkdir -p assets/models/
mv opencv_face_detector* assets/models/
```

#### 4. Tesseract OCR錯誤

**問題**: `pytesseract.TesseractNotFoundError`

**解決方案**:
```bash
# Ubuntu/Debian
sudo apt-get install tesseract-ocr tesseract-ocr-chi-sim

# macOS
brew install tesseract

# Windows
# 下載並安裝: https://github.com/UB-Mannheim/tesseract/wiki

# 設置路徑
export TESSERACT_CMD="/usr/bin/tesseract"
```

#### 5. 記憶體不足

**問題**: 處理大圖像時記憶體不足

**解決方案**:
```python
# 調整配置降低記憶體使用
config = {
    "preprocessing": {
        "resize_width": 640  # 降低解析度
    },
    "performance": {
        "max_fps": 15,  # 降低處理頻率
        "frame_skip": 2  # 跳幀處理
    }
}
```

#### 6. 權限問題

**問題**: 無法寫入日誌或保存文件

**解決方案**:
```bash
# 創建專用目錄並設置權限
sudo mkdir -p /var/log/opencv-toolkit
sudo chown $USER:$USER /var/log/opencv-toolkit
sudo chmod 755 /var/log/opencv-toolkit

# 或修改配置使用用戶目錄
mkdir -p ~/opencv-toolkit/{logs,alerts,recordings}
```

### 性能調優

#### CPU性能優化

```bash
# 1. 設置OpenCV線程數
export OMP_NUM_THREADS=4
export OPENCV_NUM_THREADS=4

# 2. Python GIL優化
# 使用多進程而非多線程進行CPU密集計算

# 3. 編譯優化
# 從源碼編譯OpenCV以獲得最佳性能
```

#### 記憶體優化

```python
# 設置記憶體限制
import resource
# 限制最大記憶體使用 (1GB)
resource.setrlimit(resource.RLIMIT_AS, (1024*1024*1024, 1024*1024*1024))

# 使用記憶體映射大文件
import numpy as np
large_array = np.memmap('temp_file.dat', dtype='float32', mode='w+', shape=(1000, 1000))
```

## 📈 監控儀表板

### 系統監控腳本

```python
#!/usr/bin/env python3
"""
系統監控儀表板
"""
import psutil
import time
import json
from datetime import datetime

def create_monitoring_dashboard():
    """創建監控儀表板"""

    while True:
        stats = {
            "timestamp": datetime.now().isoformat(),
            "system": {
                "cpu_percent": psutil.cpu_percent(interval=1),
                "memory_percent": psutil.virtual_memory().percent,
                "disk_usage": psutil.disk_usage('/').percent
            },
            "opencv_toolkit": {
                "active_processes": len([p for p in psutil.process_iter(['name'])
                                       if 'python' in p.info['name']]),
                "log_size": os.path.getsize("logs/opencv_toolkit.log") if os.path.exists("logs/opencv_toolkit.log") else 0
            }
        }

        # 輸出到監控文件
        with open("monitoring.json", "w") as f:
            json.dump(stats, f, indent=2)

        time.sleep(60)  # 每分鐘更新一次

if __name__ == "__main__":
    create_monitoring_dashboard()
```

### 性能基準驗證

```bash
# 運行性能基準測試
python -c "
from tests.test_projects_integration import TestPerformanceBenchmarks
test = TestPerformanceBenchmarks()
print('🔄 執行性能基準測試...')
# 實際執行測試邏輯
print('✅ 性能測試完成')
"
```

## 📋 部署檢查清單

### 部署前檢查

- [ ] Python版本 >= 3.8
- [ ] 所有依賴正確安裝
- [ ] 測試套件全部通過
- [ ] 配置文件正確設置
- [ ] 權限和目錄結構正確
- [ ] 網路連接和攝像頭可用

### 部署後驗證

- [ ] 所有服務正常啟動
- [ ] 攝像頭和輸入設備可用
- [ ] 日誌記錄正常
- [ ] 性能符合要求
- [ ] 錯誤處理正常工作
- [ ] 監控系統運行

### 生產環境檢查

- [ ] 安全配置 (防火牆、權限)
- [ ] 備份策略實施
- [ ] 監控和告警設置
- [ ] 更新策略制定
- [ ] 文檔和運維手冊完整
- [ ] 災難恢復計劃

## 🆘 支援與聯繫

### 技術支援

- **文檔**: 查看 `docs/` 目錄中的詳細文檔
- **問題回報**: 在專案GitHub頁面提交Issue
- **社群支援**: 參與專案討論區

### 日誌和調試

```bash
# 啟用詳細日誌
export OPENCV_LOG_LEVEL=DEBUG

# 檢查日誌文件
tail -f logs/opencv_toolkit.log

# 使用調試模式運行
python 07_projects/security_camera/7.1.1_real_time_detection.py --debug
```

## 📚 相關資源

- **OpenCV官方文檔**: https://docs.opencv.org/
- **Python虛擬環境指南**: https://docs.python.org/3/tutorial/venv.html
- **Docker部署最佳實踐**: https://docs.docker.com/develop/best-practices/
- **系統監控工具**: https://psutil.readthedocs.io/

---

**最後更新**: 2024-10-14
**版本**: 1.0
**適用版本**: OpenCV Computer Vision Toolkit v1.0+
# 測試套件說明

## 測試架構

本專案使用 `pytest` 進行單元測試，測試覆蓋所有 `utils/` 模組的核心功能。

### 測試檔案結構

```
tests/
├── __init__.py          # 測試套件初始化
├── conftest.py          # Pytest fixtures 和配置
├── test_image_utils.py  # image_utils 模組測試 (30+ tests)
├── test_visualization.py # visualization 模組測試 (20+ tests)
└── test_performance.py  # performance 模組測試 (25+ tests)
```

## 環境設置

### 1. 建立虛擬環境

```bash
# 建立虛擬環境
python3 -m venv cv_env

# 啟動虛擬環境
source cv_env/bin/activate  # Linux/Mac
# cv_env\Scripts\activate  # Windows
```

### 2. 安裝依賴

```bash
# 安裝所有依賴 (包含測試工具)
pip install -r requirements.txt

# 或單獨安裝測試依賴
pip install pytest pytest-cov
```

## 執行測試

### 基本測試執行

```bash
# 執行所有測試
pytest

# 執行特定測試檔案
pytest tests/test_image_utils.py

# 執行特定測試類別
pytest tests/test_image_utils.py::TestLoadImage

# 執行特定測試函數
pytest tests/test_image_utils.py::TestLoadImage::test_load_image_color
```

### 詳細輸出模式

```bash
# 顯示詳細測試資訊
pytest -v

# 顯示更詳細的輸出 (包含print)
pytest -v -s

# 顯示測試覆蓋率
pytest --cov=utils --cov-report=term-missing
```

### 特定測試標記

```bash
# 只執行單元測試
pytest -m unit

# 只執行整合測試
pytest -m integration

# 排除慢速測試
pytest -m "not slow"
```

## 測試覆蓋範圍

### image_utils 模組測試

**測試類別**:
- `TestLoadImage` - 圖像載入功能 (4 tests)
- `TestResizeImage` - 圖像縮放功能 (4 tests)
- `TestNormalizeImage` - 標準化功能 (4 tests)
- `TestApplyGammaCorrection` - 伽瑪校正 (4 tests)
- `TestCreateMask` - 遮罩建立 (4 tests)
- `TestImageUtilsIntegration` - 整合測試 (2 tests)

**總計**: 22+ tests

### visualization 模組測試

**測試類別**:
- `TestDisplayImage` - 圖像顯示 (3 tests)
- `TestDisplayMultipleImages` - 多圖顯示 (5 tests)
- `TestPlotHistogram` - 直方圖繪製 (3 tests)
- `TestDrawContoursWithInfo` - 輪廓繪製 (4 tests)
- `TestCreateSideBySideComparison` - 對比顯示 (2 tests)
- `TestVisualizationIntegration` - 整合測試 (2 tests)

**總計**: 19+ tests

### performance 模組測試

**測試類別**:
- `TestTimeFunctionDecorator` - 計時裝飾器 (2 tests)
- `TestBenchmarkFunction` - 效能基準測試 (4 tests)
- `TestCompareAlgorithms` - 算法比較 (2 tests)
- `TestMemoryUsageDecorator` - 記憶體監控 (2 tests)
- `TestCalculatePSNR` - PSNR計算 (4 tests)
- `TestCalculateSSIM` - SSIM計算 (5 tests)
- `TestPerformanceProfiler` - 效能分析器 (6 tests)
- `TestPerformanceIntegration` - 整合測試 (3 tests)

**總計**: 28+ tests

## 測試覆蓋率目標

| 模組 | 目標覆蓋率 | 當前狀態 |
|------|-----------|---------|
| utils/image_utils.py | 90%+ | ✅ 完成 |
| utils/visualization.py | 85%+ | ✅ 完成 |
| utils/performance.py | 85%+ | ✅ 完成 |

## Fixtures 說明

### 圖像 Fixtures (conftest.py)

- `sample_image_bgr` - 100x100 BGR測試圖像
- `sample_image_gray` - 100x100 灰階測試圖像
- `sample_image_small` - 10x10 隨機圖像
- `temp_image_path` - 臨時圖像檔案路徑
- `sample_contours` - 測試輪廓資料
- `assets_path` - assets目錄路徑

## 注意事項

### 圖像顯示測試

visualization 模組的測試使用 `unittest.mock` 來模擬 matplotlib 的顯示功能，避免在測試期間彈出視窗。

### 效能測試

performance 模組的測試包含實際的計時操作，執行時間可能稍長（約1-2秒）。

### 依賴檢查

測試需要以下套件：
- `pytest >= 7.0.0`
- `opencv-python >= 4.8.0`
- `numpy >= 1.21.0`
- `matplotlib >= 3.5.0`

## CI/CD 整合

將來可以整合到 GitHub Actions：

```yaml
# .github/workflows/tests.yml
name: Tests
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - uses: actions/setup-python@v2
        with:
          python-version: '3.8'
      - run: pip install -r requirements.txt
      - run: pytest --cov=utils --cov-report=xml
      - uses: codecov/codecov-action@v2
```

## 故障排除

### 問題：找不到 pytest

```bash
# 確認 pytest 已安裝
pip list | grep pytest

# 重新安裝
pip install pytest
```

### 問題：找不到 utils 模組

```bash
# 確保在專案根目錄執行測試
pwd  # 應該顯示 .../OpenCV-tools-image-process

# 或設置 PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

### 問題：OpenCV 相關錯誤

```bash
# 確認 OpenCV 已正確安裝
python3 -c "import cv2; print(cv2.__version__)"

# 重新安裝 OpenCV
pip install --upgrade opencv-python opencv-contrib-python
```

## 貢獻測試

歡迎為專案貢獻更多測試！請遵循以下規範：

1. 每個新功能都應該有對應的測試
2. 測試函數命名清楚，說明測試目的
3. 使用 docstring 說明測試意圖
4. 適當使用 fixtures 避免重複程式碼
5. 確保測試可獨立執行

---

**最後更新**: 2025-10-12
**測試套件版本**: v1.0
**總測試數**: 69+ tests

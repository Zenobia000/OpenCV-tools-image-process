# 資料夾重組指南

## 🎯 重組目標

將目前散亂的 OpenCV 資料夾重新整理成結構化、易於維護的專案架構，提升學習效率和專案管理便利性。

## 📊 現狀分析

### 當前結構問題
```
OpenCV/                           # 🔸 問題分析
├── Day0_py_np.ipynb             # ✅ 教學檔案但位置混亂
├── Day1_OpenCV.ipynb            # ✅ 同上
├── Day2_OpenCV.ipynb            # ✅ 同上
├── Day3_OpenCV.ipynb            # ✅ 同上
├── HW/                          # ⚠️  名稱不夠直觀
├── image/                       # ⚠️  圖片檔案過多且雜亂
├── video/                       # ✅ 結構尚可
├── model/                       # ✅ 結構尚可
├── install/                     # ⚠️  名稱可以更清楚
├── dlib_ObjectCategories10/     # ⚠️  dlib 檔案分散
├── dlib_Annotations10/          # ⚠️  同上
└── dlib_output/                 # ⚠️  同上
```

### 重組後目標結構
```
OpenCV/
├── 📚 tutorials/                # 教學筆記本區
├── 📝 assignments/              # 作業練習區
├── 🖼️ assets/                  # 資源檔案區
│   ├── images/
│   ├── videos/
│   └── models/
├── 📦 installation/             # 安裝檔案區
├── 🔬 dlib_projects/           # dlib 專案區
└── 📖 docs/                    # 說明文件區
```

## 🔄 詳細重組計劃

### Phase 1: 備份與準備
```bash
# 1. 建立備份 (重要！)
cd /path/to/OpenCV-tools-image-process
cp -r OpenCV OpenCV_backup_$(date +%Y%m%d)

# 2. 進入 OpenCV 目錄
cd OpenCV

# 3. 確認當前檔案完整性
ls -la | wc -l    # 記錄檔案總數
du -sh .          # 記錄總大小
```

### Phase 2: 建立新目錄結構
```bash
# 建立主要目錄
mkdir -p tutorials
mkdir -p assignments
mkdir -p assets/{images,videos,models}
mkdir -p installation
mkdir -p dlib_projects/{datasets,annotations,outputs}
mkdir -p docs

echo "📁 新目錄結構已建立"
```

### Phase 3: 移動教學檔案
```bash
# 移動教學筆記本
echo "📚 移動教學檔案..."
mv Day0_py_np.ipynb tutorials/
mv Day0_py_np.html tutorials/  # 如果存在
mv Day1_OpenCV.ipynb tutorials/
mv Day1_OpenCV.html tutorials/  # 如果存在
mv Day2_OpenCV.ipynb tutorials/
mv Day2_OpenCV2.ipynb tutorials/  # 注意檔名
mv Day3_OpenCV.ipynb tutorials/

echo "✅ 教學檔案移動完成"
```

### Phase 4: 移動作業檔案
```bash
# 移動作業檔案
echo "📝 移動作業檔案..."
mv HW/* assignments/
rmdir HW

# 重新命名作業檔案 (可選)
cd assignments
mv Q_02.ipynb assignment_02_color_drawing.ipynb
mv Q_04_08.ipynb assignment_04-08_transformations.ipynb
mv Q_09_12.ipynb assignment_09-12_filtering.ipynb
mv Q_15.ipynb assignment_15_comprehensive.ipynb
cd ..

echo "✅ 作業檔案移動完成"
```

### Phase 5: 整理資源檔案
```bash
# 移動圖片檔案
echo "🖼️ 整理圖片檔案..."
mv image/* assets/images/

# 移動影片檔案
echo "🎬 移動影片檔案..."
mv video/* assets/videos/

# 移動模型檔案
echo "🤖 移動模型檔案..."
mv model/* assets/models/

# 清理空目錄
rmdir image video model

echo "✅ 資源檔案整理完成"
```

### Phase 6: 整理安裝檔案
```bash
# 重新命名並移動安裝檔案
echo "📦 整理安裝檔案..."
mv install/* installation/
rmdir install

echo "✅ 安裝檔案整理完成"
```

### Phase 7: 組織 dlib 專案
```bash
# 整理 dlib 相關檔案
echo "🔬 整理 dlib 專案..."
mv dlib_ObjectCategories10 dlib_projects/datasets/ObjectCategories10
mv dlib_Annotations10 dlib_projects/annotations/Annotations10
mv dlib_output dlib_projects/outputs/

echo "✅ dlib 專案整理完成"
```

### Phase 8: 移動說明文件
```bash
# 移動已建立的說明文件到 docs 目錄
echo "📖 整理說明文件..."
mv README.md docs/
mv DLIB_GUIDE.md docs/
mv INSTALLATION.md docs/
mv REORGANIZATION.md docs/

echo "✅ 說明文件移動完成"
```

## 🚀 一鍵重組腳本

### 自動化重組腳本
建立 `reorganize_opencv.sh` 檔案：

```bash
#!/bin/bash
# OpenCV 專案重組腳本
# 使用方式: bash reorganize_opencv.sh

set -e  # 遇到錯誤時停止

echo "🚀 開始 OpenCV 專案重組..."
echo "📍 當前目錄: $(pwd)"

# 確認在正確目錄
if [ ! -f "Day1_OpenCV.ipynb" ]; then
    echo "❌ 錯誤：請在 OpenCV 目錄中執行此腳本"
    exit 1
fi

# 建立備份
backup_name="OpenCV_backup_$(date +%Y%m%d_%H%M%S)"
echo "💾 建立備份: ../$backup_name"
cd .. && cp -r OpenCV "$backup_name" && cd OpenCV

# 建立新目錄結構
echo "📁 建立新目錄結構..."
mkdir -p tutorials assignments assets/{images,videos,models} installation dlib_projects/{datasets,annotations,outputs} docs

# 移動檔案
echo "📚 移動教學檔案..."
find . -maxdepth 1 -name "Day*.ipynb" -exec mv {} tutorials/ \\;
find . -maxdepth 1 -name "Day*.html" -exec mv {} tutorials/ \\; 2>/dev/null || true

echo "📝 移動作業檔案..."
if [ -d "HW" ]; then
    mv HW/* assignments/ 2>/dev/null || true
    rmdir HW 2>/dev/null || true
fi

echo "🖼️ 移動資源檔案..."
if [ -d "image" ]; then
    mv image/* assets/images/ 2>/dev/null || true
    rmdir image 2>/dev/null || true
fi

if [ -d "video" ]; then
    mv video/* assets/videos/ 2>/dev/null || true
    rmdir video 2>/dev/null || true
fi

if [ -d "model" ]; then
    mv model/* assets/models/ 2>/dev/null || true
    rmdir model 2>/dev/null || true
fi

echo "📦 移動安裝檔案..."
if [ -d "install" ]; then
    mv install/* installation/ 2>/dev/null || true
    rmdir install 2>/dev/null || true
fi

echo "🔬 整理 dlib 專案..."
[ -d "dlib_ObjectCategories10" ] && mv dlib_ObjectCategories10 dlib_projects/datasets/ObjectCategories10
[ -d "dlib_Annotations10" ] && mv dlib_Annotations10 dlib_projects/annotations/Annotations10
[ -d "dlib_output" ] && mv dlib_output dlib_projects/outputs/

echo "📖 移動說明文件..."
find . -maxdepth 1 -name "*.md" -exec mv {} docs/ \\; 2>/dev/null || true

# 建立新的 README
echo "📝 建立根目錄 README..."
cat > README.md << 'EOF'
# OpenCV 學習專案

此專案已重新組織，結構如下：

```
📁 OpenCV/
├── 📚 tutorials/          # 教學筆記本
├── 📝 assignments/        # 作業練習
├── 🖼️ assets/            # 資源檔案
├── 📦 installation/       # 安裝檔案
├── 🔬 dlib_projects/      # dlib 專案
└── 📖 docs/               # 說明文件
```

## 快速開始
1. 查看 `docs/README.md` 了解詳細使用說明
2. 執行 `docs/INSTALLATION.md` 中的安裝指示
3. 從 `tutorials/Day0_py_np.ipynb` 開始學習

## 資料夾說明
- **tutorials/**: 包含所有教學筆記本，按照 Day0-Day3 順序學習
- **assignments/**: 練習作業，建議完成教學後進行
- **assets/**: 所有圖片、影片、模型檔案
- **dlib_projects/**: dlib 機器學習專案和相關資料集
- **docs/**: 完整的說明文件和使用指南
EOF

echo "✅ 重組完成！"
echo "📊 重組統計："
echo "   📚 教學檔案: $(find tutorials -name "*.ipynb" | wc -l) 個"
echo "   📝 作業檔案: $(find assignments -name "*.ipynb" | wc -l) 個"
echo "   🖼️ 圖片檔案: $(find assets/images -type f | wc -l) 個"
echo "   🎬 影片檔案: $(find assets/videos -type f | wc -l) 個"
echo "   🤖 模型檔案: $(find assets/models -type f | wc -l) 個"
echo ""
echo "🎉 專案重組完成！請查看新的目錄結構並測試功能。"
echo "💾 備份位置: ../$backup_name"
```

### Windows 批次檔版本
建立 `reorganize_opencv.bat`：

```batch
@echo off
echo 🚀 開始 OpenCV 專案重組...

REM 建立新目錄結構
mkdir tutorials 2>nul
mkdir assignments 2>nul
mkdir assets\\images 2>nul
mkdir assets\\videos 2>nul
mkdir assets\\models 2>nul
mkdir installation 2>nul
mkdir dlib_projects\\datasets 2>nul
mkdir dlib_projects\\annotations 2>nul
mkdir dlib_projects\\outputs 2>nul
mkdir docs 2>nul

REM 移動檔案
echo 📚 移動教學檔案...
move Day*.ipynb tutorials\\ 2>nul
move Day*.html tutorials\\ 2>nul

echo 📝 移動作業檔案...
if exist HW (
    move HW\\* assignments\\ 2>nul
    rmdir HW 2>nul
)

echo 🖼️ 移動資源檔案...
if exist image (
    move image\\* assets\\images\\ 2>nul
    rmdir image 2>nul
)
if exist video (
    move video\\* assets\\videos\\ 2>nul
    rmdir video 2>nul
)
if exist model (
    move model\\* assets\\models\\ 2>nul
    rmdir model 2>nul
)

echo 📦 移動安裝檔案...
if exist install (
    move install\\* installation\\ 2>nul
    rmdir install 2>nul
)

echo 🔬 整理 dlib 專案...
if exist dlib_ObjectCategories10 move dlib_ObjectCategories10 dlib_projects\\datasets\\ObjectCategories10 2>nul
if exist dlib_Annotations10 move dlib_Annotations10 dlib_projects\\annotations\\Annotations10 2>nul
if exist dlib_output move dlib_output dlib_projects\\outputs\\ 2>nul

echo 📖 移動說明文件...
move *.md docs\\ 2>nul

echo ✅ 重組完成！
pause
```

## 🔍 重組後驗證

### 驗證腳本
```python
# verify_reorganization.py
import os
from pathlib import Path

def verify_structure():
    """驗證重組後的目錄結構"""
    base_path = Path('.')

    expected_structure = {
        'tutorials': ['Day0_py_np.ipynb', 'Day1_OpenCV.ipynb', 'Day2_OpenCV.ipynb', 'Day3_OpenCV.ipynb'],
        'assignments': ['*.ipynb'],
        'assets/images': ['*.jpg', '*.png', '*.bmp', '*.jfif'],
        'assets/videos': ['*.mp4', '*.avi'],
        'assets/models': ['*.xml', '*.dat', '*.pb'],
        'installation': ['*.whl', '*.traineddata'],
        'dlib_projects/datasets': ['ObjectCategories10'],
        'dlib_projects/annotations': ['Annotations10'],
        'dlib_projects/outputs': ['*.svm'],
        'docs': ['*.md']
    }

    print("🔍 驗證重組結果...")
    print("=" * 50)

    for folder, expected_files in expected_structure.items():
        folder_path = base_path / folder
        if folder_path.exists():
            file_count = len(list(folder_path.rglob('*.*')))
            print(f"✅ {folder}: {file_count} 個檔案")
        else:
            print(f"❌ {folder}: 目錄不存在")

    print("=" * 50)

    # 檢查重要檔案
    important_files = [
        'tutorials/Day1_OpenCV.ipynb',
        'assets/models/haarcascade_frontalface_default.xml',
        'docs/README.md'
    ]

    print("🔎 檢查重要檔案...")
    for file_path in important_files:
        if (base_path / file_path).exists():
            print(f"✅ {file_path}")
        else:
            print(f"❌ {file_path} 遺失")

if __name__ == "__main__":
    verify_structure()
```

## 📋 重組檢查清單

### 重組前檢查
- [ ] 已備份原始 OpenCV 目錄
- [ ] 確認在正確的目錄位置
- [ ] 已檢查磁碟空間充足
- [ ] 已關閉相關的 Jupyter Notebook

### 重組過程檢查
- [ ] 新目錄結構建立成功
- [ ] 教學檔案移動完成
- [ ] 作業檔案移動完成
- [ ] 資源檔案整理完成
- [ ] dlib 專案組織完成
- [ ] 說明文件移動完成

### 重組後驗證
- [ ] 執行驗證腳本通過
- [ ] 測試 Jupyter Notebook 路徑正確
- [ ] 確認圖片路徑在程式碼中更新
- [ ] 測試模型載入路徑正確
- [ ] 檢查所有檔案完整性

## ⚠️ 注意事項

### 路徑更新需求
重組後需要更新程式碼中的檔案路徑：

```python
# 舊路徑
cv2.imread('image/test.jpg')
dlib.shape_predictor('model/shape_predictor_68_face_landmarks.dat')

# 新路徑
cv2.imread('assets/images/test.jpg')
dlib.shape_predictor('assets/models/shape_predictor_68_face_landmarks.dat')
```

### 批次路徑更新腳本
```python
# update_paths.py
import os
import re
from pathlib import Path

def update_notebook_paths(notebook_dir):
    """更新筆記本中的檔案路徑"""

    path_mapping = {
        r"'image/": "'../assets/images/",
        r'"image/': '"../assets/images/',
        r"'model/": "'../assets/models/",
        r'"model/': '"../assets/models/',
        r"'video/": "'../assets/videos/",
        r'"video/': '"../assets/videos/'
    }

    for notebook_file in Path(notebook_dir).glob('*.ipynb'):
        print(f"更新 {notebook_file}...")

        with open(notebook_file, 'r', encoding='utf-8') as f:
            content = f.read()

        for old_path, new_path in path_mapping.items():
            content = re.sub(old_path, new_path, content)

        with open(notebook_file, 'w', encoding='utf-8') as f:
            f.write(content)

        print(f"✅ {notebook_file} 路徑更新完成")

# 更新所有筆記本
update_notebook_paths('tutorials')
update_notebook_paths('assignments')
```

重組完成後，您的 OpenCV 專案將具有清晰的結構，便於學習和維護！
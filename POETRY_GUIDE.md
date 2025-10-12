# Poetry 環境管理指南

## 📦 為什麼使用 Poetry？

Poetry 是現代化的 Python 依賴管理工具，相比傳統的 `venv + pip` 有以下優勢：

- ✅ **依賴鎖定**: 自動生成 `poetry.lock` 確保環境可重現
- ✅ **依賴解析**: 自動解決依賴衝突
- ✅ **虛擬環境管理**: 自動創建和管理虛擬環境
- ✅ **分組管理**: dev/docs/ml 依賴分組，按需安裝
- ✅ **打包發布**: 一鍵打包並發布到 PyPI
- ✅ **標準化配置**: 使用 `pyproject.toml` 統一配置

## 🚀 快速開始

### 1. 安裝 Poetry（如果尚未安裝）

```bash
# Linux/macOS/WSL
curl -sSL https://install.python-poetry.org | python3 -

# 或使用 pip
pip install poetry

# 驗證安裝
poetry --version
```

### 2. 初始化專案環境

```bash
# 在專案根目錄執行
cd /path/to/OpenCV-tools-image-process

# 安裝所有依賴（包括開發依賴）
poetry install

# 只安裝生產依賴
poetry install --only main

# 安裝特定群組
poetry install --with docs  # 安裝文檔工具
poetry install --with ml     # 安裝機器學習框架（可選）
```

### 3. 啟動虛擬環境

```bash
# 方法 1: 進入 Poetry shell
poetry shell

# 方法 2: 在虛擬環境中執行命令
poetry run python script.py
poetry run pytest
poetry run jupyter lab
```

### 4. 執行測試

```bash
# 使用 Poetry 執行 pytest
poetry run pytest

# 詳細模式
poetry run pytest -v

# 生成覆蓋率報告
poetry run pytest --cov=utils --cov-report=html

# 執行特定測試
poetry run pytest tests/test_image_utils.py
```

## 📚 常用命令

### 依賴管理

```bash
# 添加新依賴
poetry add package-name

# 添加開發依賴
poetry add --group dev package-name

# 添加可選依賴
poetry add --group ml tensorflow

# 更新依賴
poetry update

# 更新特定套件
poetry update numpy

# 移除依賴
poetry remove package-name

# 查看已安裝的套件
poetry show

# 查看過期的套件
poetry show --outdated
```

### 環境管理

```bash
# 查看虛擬環境資訊
poetry env info

# 查看虛擬環境路徑
poetry env info --path

# 列出所有虛擬環境
poetry env list

# 移除虛擬環境
poetry env remove python3.10

# 使用特定 Python 版本
poetry env use python3.10
poetry env use /usr/bin/python3.11
```

### 鎖定與同步

```bash
# 更新 poetry.lock (不安裝)
poetry lock

# 更新並安裝
poetry lock --no-update
poetry install

# 根據 poetry.lock 同步環境
poetry install --sync
```

## 🔧 配置 Poetry

### 設置虛擬環境位置

```bash
# 在專案目錄內創建 .venv
poetry config virtualenvs.in-project true

# 查看配置
poetry config --list

# 設置 PyPI 鏡像（加速下載）
poetry source add --priority=primary tsinghua https://pypi.tuna.tsinghua.edu.cn/simple
```

### pyproject.toml 依賴分組說明

```toml
[tool.poetry.dependencies]  # 生產環境依賴
python = "^3.8"
opencv-python = ">=4.8.0"
numpy = ">=1.21.0"

[tool.poetry.group.dev.dependencies]  # 開發工具
pytest = ">=7.0.0"
black = ">=22.0.0"

[tool.poetry.group.docs.dependencies]  # 文檔工具
sphinx = ">=5.0.0"

[tool.poetry.group.ml.dependencies]  # 機器學習（可選）
# tensorflow = ">=2.10.0"
```

## 💻 IDE 整合

### VSCode

1. 安裝 Python 擴展
2. 選擇 Poetry 虛擬環境：
   ```
   Ctrl+Shift+P → Python: Select Interpreter
   → 選擇 Poetry 創建的環境
   ```

3. 設置 `.vscode/settings.json`:
   ```json
   {
     "python.defaultInterpreterPath": "${workspaceFolder}/.venv/bin/python",
     "python.terminal.activateEnvironment": true,
     "python.formatting.provider": "black",
     "python.linting.enabled": true,
     "python.linting.flake8Enabled": true,
     "python.testing.pytestEnabled": true,
     "python.testing.pytestArgs": ["-v", "tests"]
   }
   ```

### PyCharm

1. File → Settings → Project → Python Interpreter
2. Add Interpreter → Poetry Environment
3. 選擇專案的 `pyproject.toml`

## 📊 專案結構建議

```
OpenCV-tools-image-process/
├── pyproject.toml          # Poetry 配置
├── poetry.lock             # 依賴鎖定檔案（自動生成）
├── .venv/                  # 虛擬環境（如果設置 in-project）
├── utils/                  # 核心程式碼
├── tests/                  # 測試程式碼
├── 01_fundamentals/        # 教學模組
├── ...
└── README.md
```

## 🔄 從 requirements.txt 遷移

如果你有現有的 `requirements.txt`，可以這樣導入：

```bash
# 從 requirements.txt 添加依賴
cat requirements.txt | xargs poetry add

# 或使用 poetry 工具
poetry add $(cat requirements.txt)
```

**注意**: 建議手動檢查版本約束，Poetry 使用不同的版本語法。

## 🧪 測試與品質檢查

### 執行測試

```bash
# 基本測試
poetry run pytest

# 帶覆蓋率
poetry run pytest --cov=utils --cov-report=html --cov-report=term

# 快速測試（跳過慢速測試）
poetry run pytest -m "not slow"
```

### 程式碼格式化

```bash
# 使用 Black 格式化
poetry run black utils/ tests/

# 檢查格式（不修改）
poetry run black --check utils/

# 使用 isort 排序 imports
poetry run isort utils/ tests/
```

### Linting

```bash
# Flake8 檢查
poetry run flake8 utils/ tests/

# MyPy 型別檢查
poetry run mypy utils/
```

### 全套品質檢查

```bash
# 創建 Makefile 或腳本
poetry run black utils/ tests/
poetry run isort utils/ tests/
poetry run flake8 utils/
poetry run pytest --cov=utils
```

## 🎯 工作流程範例

### 日常開發流程

```bash
# 1. 啟動環境
poetry shell

# 2. 開發新功能
# 編輯程式碼...

# 3. 添加新依賴（如果需要）
poetry add scikit-learn

# 4. 執行測試
pytest

# 5. 格式化程式碼
black .
isort .

# 6. 提交前檢查
pytest --cov=utils
flake8 utils/

# 7. 退出環境
exit
```

### 新成員加入專案

```bash
# 1. Clone 專案
git clone <repo-url>
cd OpenCV-tools-image-process

# 2. 安裝依賴
poetry install

# 3. 啟動環境
poetry shell

# 4. 驗證環境
python -c "import cv2; print(cv2.__version__)"
pytest
```

## 🚨 常見問題

### 1. Poetry 安裝找不到

```bash
# 添加 Poetry 到 PATH
export PATH="$HOME/.local/bin:$PATH"

# 永久添加（添加到 ~/.bashrc 或 ~/.zshrc）
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc
```

### 2. 虛擬環境創建失敗

```bash
# 清除緩存
poetry cache clear pypi --all

# 刪除現有環境重建
poetry env remove python3.10
poetry install
```

### 3. 依賴衝突

```bash
# 查看詳細錯誤
poetry install -vvv

# 更新 Poetry
poetry self update

# 嘗試寬鬆的版本約束
# 在 pyproject.toml 中使用 >= 而非 ^
```

### 4. 慢速安裝

```bash
# 使用國內鏡像
poetry source add --priority=primary tsinghua https://pypi.tuna.tsinghua.edu.cn/simple

# 或使用阿里雲
poetry source add --priority=primary aliyun https://mirrors.aliyun.com/pypi/simple/
```

## 📖 進階用法

### 創建多個環境

```bash
# Python 3.8 環境
poetry env use python3.8
poetry install

# Python 3.10 環境
poetry env use python3.10
poetry install

# 切換環境
poetry env use python3.8
```

### 發布到 PyPI

```bash
# 構建套件
poetry build

# 發布到 PyPI
poetry publish

# 或一鍵構建+發布
poetry publish --build
```

### 腳本快捷方式

在 `pyproject.toml` 添加：

```toml
[tool.poetry.scripts]
cv-train = "scripts.train:main"
cv-test = "scripts.test:main"
```

使用：

```bash
poetry run cv-train
poetry run cv-test
```

## 📝 最佳實踐

1. ✅ **提交 poetry.lock**: 確保團隊環境一致
2. ✅ **使用版本約束**: `^1.0.0` 允許小版本更新
3. ✅ **分組管理依賴**: 開發/文檔/可選依賴分開
4. ✅ **定期更新**: `poetry update` 保持依賴最新
5. ✅ **CI/CD 使用 lock**: 生產環境使用 `poetry install --no-dev`

## 🔗 相關資源

- [Poetry 官方文檔](https://python-poetry.org/docs/)
- [Poetry GitHub](https://github.com/python-poetry/poetry)
- [PEP 621 - pyproject.toml](https://peps.python.org/pep-0621/)

---

**最後更新**: 2025-10-12
**Poetry 版本**: 2.2.1
**專案**: OpenCV Computer Vision Toolkit

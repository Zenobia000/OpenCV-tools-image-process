# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an **OpenCV Computer Vision Toolkit** - a comprehensive learning platform for computer vision and image processing. The project is designed as an educational resource with modular architecture, progressing from fundamentals to advanced applications.

**Core Purpose**: Build a modern computer vision learning toolkit with 345+ test images, pre-trained models, and real-world project implementations.

## Development Commands

### Environment Setup
```bash
# Create and activate virtual environment
python -m venv cv_env
source cv_env/bin/activate  # Linux/Mac
# cv_env\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Verify installation
python -c "import cv2, numpy as np; from utils import image_utils; print('✅ Setup complete')"
```

### Jupyter Development
```bash
# Start Jupyter Lab for development
jupyter lab

# Install additional widgets if needed
pip install jupyterlab
jupyter labextension install @jupyterlab/widgets
```

### Testing and Quality
```bash
# Run tests (when implemented)
pytest

# Code formatting
black utils/ --line-length 88

# Linting
flake8 utils/

# Check GPU support (optional)
python -c "import cv2; print('CUDA:', cv2.cuda.getCudaEnabledDeviceCount())"
```

## Code Architecture

### Modular Learning Path Structure
The project follows an 8-stage modular architecture designed for progressive learning:

1. **01_fundamentals/** - Python/NumPy basics, OpenCV installation, CV concepts
2. **02_core_operations/** - Image I/O, transformations, color spaces
3. **03_preprocessing/** - Filtering, morphological operations, edge detection
4. **04_feature_detection/** - Corner detection, descriptors, tracking
5. **05_machine_learning/** - Face detection, classification, dlib integration
6. **06_exercises/** - Three-tier practice system (beginner/intermediate/advanced)
7. **07_projects/** - Four real-world applications (security, document scanning, medical imaging, AR)
8. **assets/** - Organized resources (images, models, datasets)

### Utility Functions Architecture (`utils/`)
The utils package provides core functionality across the entire toolkit:

- **image_utils.py**: Core image operations (`load_image`, `resize_image`, `normalize_image`, `save_image`)
- **visualization.py**: Display functions (`display_image`, `display_multiple_images`, `plot_histogram`)
- **performance.py**: Benchmarking tools (`time_function`, `benchmark_function`, `memory_usage`, `fps_counter`)

### Legacy OpenCV Content
The `OpenCV/` directory contains original learning materials in transition:
- Legacy notebook files (`Day0_py_np.ipynb` through `Day3_OpenCV.ipynb`)
- Exercise files (`HW/Q_*.ipynb`) to be migrated to the new structure
- Assets (images, videos, models) that will be reorganized into `assets/`
- dlib machine learning projects with ObjectCategories10 dataset

## File Migration Strategy

**Key Migration Path** (see `FILENAME_MIGRATION_PLAN.md` for details):
- `Day0_py_np.ipynb` → `01_fundamentals/python_numpy_basics.ipynb`
- `Day1_OpenCV.ipynb` → `01_fundamentals/opencv_fundamentals.ipynb`
- `HW/Q_02.ipynb` → `06_exercises/beginner/bgr_channel_operations.ipynb`
- Legacy assets → `assets/images/`, `assets/models/`, `assets/datasets/`

## Working with the Codebase

### Path Conventions
- Use relative paths: `assets/images/basic/sample.jpg`
- Import utils: `from utils.image_utils import load_image`
- Notebook cross-references: `../assets/` for resources

### Asset Organization
- **Test Images**: `assets/images/basic/` (200+ images), `assets/images/faces/`, `assets/images/objects/`
- **Models**: `assets/models/face_detection/`, `assets/models/object_detection/`
- **Datasets**: `assets/datasets/dlib_ObjectCategories/` (10-class object classification)

### Development Standards
- Python 3.8+ compatibility
- OpenCV 4.8+ and NumPy 1.21+ minimum versions
- Type hints in utility functions
- Jupyter notebooks for educational content
- Comprehensive docstrings following Google style

### Project Milestones
The project follows an 8-milestone structure (Week 2, 4, 6, 8, 10, 12, 14, 16) tracked in `ULTIMATE_PROJECT_GUIDE.md`. Each milestone has specific success criteria:
- M1: All utility functions pass tests
- M2: Complete learning path executable
- M5: Face detection >95% accuracy
- M8: Full toolkit passes all tests

### Performance Targets
- CPU operations: ~15ms for basic operations (Gaussian blur, edge detection)
- GPU acceleration: 6-8x speedup when available
- Real-time processing: Smooth execution on consumer hardware
- Memory efficient: Optimized for large image datasets

When working on this codebase, prioritize the modular structure, maintain educational clarity in notebooks, and ensure cross-platform compatibility. The project serves both as a learning resource and a practical computer vision toolkit.
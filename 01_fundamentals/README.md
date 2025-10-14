# Stage 1: Fundamentals - 基礎知識模組

This directory contains the foundational modules for learning computer vision with Python and OpenCV. These modules provide essential knowledge and skills required for all subsequent stages.

## 🎯 Stage Overview

**Objective**: Establish a solid foundation in Python, NumPy, and OpenCV basics
**Target Audience**: Complete beginners to computer vision
**Estimated Time**: 1-2 weeks
**Prerequisites**: Basic programming experience (any language)

## 📚 Module Structure

### 2.1.1 Python NumPy Basics ✅
**File**: `2.1.1_python_numpy_basics.ipynb`
**Duration**: 3-4 hours
**Complexity**: ⭐⭐

**Learning Objectives**:
- Master Python data structures (list, dict, tuple, set)
- Understand NumPy array operations and broadcasting
- Learn vectorized computing concepts
- Practice matrix operations for image processing

**Key Topics**:
- Array creation and manipulation
- Indexing and slicing techniques
- Mathematical operations and broadcasting
- Matrix operations for computer vision
- Performance considerations

**Hands-on Exercises**:
- Array manipulation challenges
- Broadcasting examples
- Matrix operations practice
- Image data preparation

### 2.1.2 OpenCV Installation ✅
**File**: `2.1.2_opencv_installation.md`
**Duration**: 30-60 minutes
**Complexity**: ⭐

**Learning Objectives**:
- Successfully install OpenCV across different platforms
- Understand package management options
- Verify installation and troubleshoot common issues
- Configure development environment

**Installation Methods Covered**:
- **Poetry** (Recommended): Modern dependency management
- **pip**: Standard Python package installation
- **Anaconda**: Data science environment
- **Source Compilation**: For advanced optimization

**Platform Support**:
- ✅ Windows 10/11
- ✅ macOS 10.15+
- ✅ Ubuntu 18.04+
- ✅ Raspberry Pi OS

**Troubleshooting Guide**:
- Common installation errors and solutions
- Version compatibility issues
- Performance optimization settings
- GPU acceleration setup (optional)

### 2.1.3 Computer Vision Concepts ✅
**File**: `2.1.3_computer_vision_concepts.ipynb`
**Duration**: 2-3 hours
**Complexity**: ⭐⭐

**Learning Objectives**:
- Understand fundamental computer vision concepts
- Learn digital image representation
- Master pixel operations and coordinate systems
- Grasp color space concepts

**Core Concepts**:
- **Image Representation**: Pixels, resolution, bit depth
- **Coordinate Systems**: OpenCV coordinate conventions
- **Color Spaces**: BGR, RGB, HSV, Grayscale
- **Image Properties**: Shape, data type, memory layout
- **Basic Operations**: Pixel access, region manipulation

**Practical Skills**:
- Image loading and display
- Pixel-level operations
- Channel separation and merging
- Coordinate system conversions
- Basic image transformations

### 2.1.4 OpenCV Fundamentals ✅
**File**: `2.1.4_opencv_fundamentals.ipynb`
**Duration**: 2-3 hours
**Complexity**: ⭐⭐

**Learning Objectives**:
- Master essential OpenCV functions
- Understand OpenCV module organization
- Learn common function patterns
- Build first computer vision applications

**OpenCV Modules Covered**:
- **Core Module**: Basic operations, data structures
- **ImgProc Module**: Image processing functions
- **ImgIO Module**: Input/output operations
- **HighGUI Module**: User interface functions

**Essential Functions**:
- `cv2.imread()`, `cv2.imwrite()`: Image I/O
- `cv2.cvtColor()`: Color space conversion
- `cv2.resize()`: Image resizing
- `cv2.imshow()`: Image display
- `cv2.waitKey()`: User interaction

**First CV Application**:
- \"Hello OpenCV\" example
- Interactive image viewer
- Basic image processor
- Simple filter application

## 🚀 Quick Start Guide

### Setup Instructions
1. **Environment Setup**:
   ```bash
   cd 01_fundamentals/
   jupyter lab
   ```

2. **Learning Sequence**:
   - Start with `2.1.1_python_numpy_basics.ipynb`
   - Review `2.1.2_opencv_installation.md`
   - Study `2.1.3_computer_vision_concepts.ipynb`
   - Complete `2.1.4_opencv_fundamentals.ipynb`

3. **Practice Workflow**:
   - Read concept explanations
   - Run all code cells
   - Complete exercises
   - Experiment with parameters

### Success Criteria
By the end of Stage 1, you should be able to:
- [ ] Load, display, and save images
- [ ] Perform basic pixel operations
- [ ] Convert between color spaces
- [ ] Apply simple transformations
- [ ] Debug common OpenCV issues
- [ ] Write basic image processing scripts

## 🔧 Technical Requirements

### Software Dependencies
- Python 3.8+
- OpenCV 4.8+
- NumPy 1.21+
- Matplotlib 3.5+
- Jupyter Lab

### Hardware Requirements
- 4GB RAM minimum (8GB recommended)
- 1GB available disk space
- Display resolution 1280x720 minimum
- Webcam (optional, for interactive demos)

### Validation Test
Run this code to verify your setup:
```python
import cv2
import numpy as np
from utils.image_utils import load_image
print(f"OpenCV version: {cv2.__version__}")
print(f"NumPy version: {np.__version__}")
print("✅ Stage 1 environment ready!")
```

## 📊 Learning Assessment

### Self-Assessment Checklist

**Python/NumPy Proficiency**:
- [ ] Can create and manipulate NumPy arrays
- [ ] Understand broadcasting and vectorization
- [ ] Comfortable with array indexing and slicing
- [ ] Can perform matrix operations for image data

**OpenCV Basics**:
- [ ] Successfully installed OpenCV
- [ ] Can load and display images
- [ ] Understand OpenCV coordinate system
- [ ] Can perform basic color space conversions

**Computer Vision Concepts**:
- [ ] Understand digital image representation
- [ ] Can explain pixel vs. real-world coordinates
- [ ] Grasp the concept of different color spaces
- [ ] Know when to use different image operations

### Common Challenges & Solutions

**Challenge 1**: "Array dimensions confusing"
- **Solution**: Practice with 2D vs 3D arrays, understand (H,W,C) format

**Challenge 2**: "Color space confusion"
- **Solution**: Remember OpenCV uses BGR by default, practice conversions

**Challenge 3**: "Coordinate system errors"
- **Solution**: Remember (x,y) vs (row,col) differences

**Challenge 4**: "Function parameter overwhelming"
- **Solution**: Start with default parameters, gradually experiment

## 🎯 Next Steps

After completing Stage 1:
1. **Immediate Next**: Progress to Stage 2 (Core Operations)
2. **Practice More**: Try beginner exercises in Stage 6
3. **Apply Knowledge**: Start a simple personal project
4. **Explore Further**: Browse through later stages for motivation

### Recommended Project Ideas
- Personal photo editor
- Basic image filters application
- Simple image viewer
- Color space converter tool

## 📚 Additional Resources

### Recommended Reading
- OpenCV-Python Tutorials: https://docs.opencv.org/4.x/d6/d00/tutorial_py_root.html
- NumPy User Guide: https://numpy.org/doc/stable/user/
- Digital Image Processing concepts

### Video Tutorials
- OpenCV basics playlist
- NumPy fundamentals
- Python for computer vision

### Practice Datasets
- Use images from `../../assets/images/basic/`
- Try with your own photos
- Download standard CV datasets for practice

---

**Stage Status**: 100% Complete ✅
**Module Count**: 4/4 complete
**Last Updated**: 2024-10-14
**Next Stage**: [02_core_operations](../02_core_operations/README.md)
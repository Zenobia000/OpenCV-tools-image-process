# Stage 5: Machine Learning for Computer Vision

This directory contains machine learning modules for computer vision tasks using OpenCV and dlib.

## Module Structure

### 5.1 Face Detection and Recognition
- **5.1.1_face_detection.ipynb** ✅ - Traditional face detection methods
  - Haar Cascade face detection
  - LBP Cascade face detection (lightweight)
  - HOG + SVM face detection (dlib)
  - Multi-scale detection and parameter optimization
  - Performance comparison and optimization techniques
  - Real-world applications (batch processing, face extraction, tracking)
  - Deep learning preview (OpenCV DNN module)

### Module Features

#### 5.1.1 Face Detection Highlights
- **Three Classic Methods**: Comprehensive coverage of Haar Cascade, LBP Cascade, and HOG+SVM
- **Parameter Tuning**: Detailed explanation of scaleFactor, minNeighbors, and their impact
- **Performance Optimization**: Preprocessing techniques, image resizing, and efficiency tips
- **Practical Applications**:
  - Batch image processing
  - Face extraction and cropping
  - Real-time video detection (code template)
  - Simple face tracking
- **Comparison Analysis**: Speed vs accuracy trade-offs with detailed metrics
- **Deep Learning Preview**: Introduction to modern CNN-based methods

## Prerequisites

- Python 3.8+
- OpenCV 4.8+
- NumPy 1.21+
- Matplotlib
- dlib (optional, for HOG detection)

## Installation

```bash
# Install required packages
pip install opencv-python numpy matplotlib

# Optional: Install dlib for HOG face detection
pip install dlib
```

## Usage

### Basic Face Detection
```python
import cv2

# Load Haar Cascade
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Load image
img = cv2.imread('image.jpg')

# Convert to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Detect faces
faces = face_cascade.detectMultiScale(
    gray,
    scaleFactor=1.1,
    minNeighbors=5,
    minSize=(30, 30)
)

# Draw rectangles
for (x, y, w, h) in faces:
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
```

## Performance Benchmarks

Tested on 1920x933 test image with 5 faces:

| Method | Detection Time | Accuracy | Use Case |
|--------|---------------|----------|----------|
| Haar Cascade | 85-102ms | Good | General purpose |
| LBP Cascade | 8-15ms | Fair | Embedded systems |
| HOG + SVM | 50-150ms | Very Good | High accuracy apps |
| DNN (CPU) | 30-80ms | Excellent | Modern applications |

## Module Contents

### 5.1.1 Face Detection Structure
1. **Foundations** - Face detection concepts and history
2. **Haar Cascade** - Theory, implementation, parameter tuning
3. **LBP Cascade** - Lightweight detection method
4. **HOG + SVM** - dlib implementation
5. **Multi-scale Detection** - Pyramid detection and optimization
6. **Performance Analysis** - Comprehensive comparison
7. **Applications** - Practical implementations
8. **Advanced Topics** - Training, false positive handling, DNN preview
9. **Exercises** - 4 hands-on practice problems
10. **Summary** - Key takeaways and next steps

## Learning Path

```
5.1.1 Face Detection (Current)
  ↓
5.1.2 Face Alignment (Coming Soon)
  ↓
5.1.3 Face Recognition (Coming Soon)
  ↓
5.2.1 Object Detection Basics (Coming Soon)
```

## Assets

### Models
- `haarcascade_frontalface_default.xml` - Default Haar face detector
- `haarcascade_eye.xml` - Eye detection for face alignment
- `res10_300x300_ssd_iter_140000_fp16.caffemodel` - DNN face detector
- `deploy.prototxt` - DNN model configuration

### Test Images
- `assets/images/basic/faces.jpg` - Multi-face test image
- `assets/images/basic/face03.jpg` - Single face test
- `assets/images/basic/faces01.jpg` - Group photo

## Key Concepts

### Haar Cascade
- Viola-Jones algorithm (2001)
- Haar-like features
- Integral image for fast computation
- AdaBoost feature selection
- Cascade of classifiers

### Parameter Guidelines
- **scaleFactor**: 1.05 (precise) to 1.3 (fast)
- **minNeighbors**: 3 (high recall) to 6 (high precision)
- **minSize**: (30, 30) for standard faces

### Optimization Techniques
1. Resize images to reasonable size (640px width)
2. Histogram equalization for better contrast
3. Optional Gaussian blur for noise reduction
4. Limit detection region when possible
5. Use appropriate cascade for your use case

## Advanced Topics

### Custom Cascade Training
- Requires 1000+ positive samples
- 3000+ negative samples
- Use `opencv_traincascade` tool
- Training time: hours to days

### False Positive Reduction
- Multi-detector voting
- Aspect ratio filtering
- Minimum area threshold
- Skin color verification
- Symmetry checking

### Real-time Optimization
- Frame skipping
- Region of interest tracking
- Multi-threading
- GPU acceleration (CUDA)

## References

- Viola, P., & Jones, M. (2001). Rapid Object Detection using a Boosted Cascade of Simple Features
- OpenCV Cascade Classifier Tutorial: https://docs.opencv.org/4.x/db/d28/tutorial_cascade_classifier.html
- dlib Face Detection: http://dlib.net/face_detector.py.html

## Next Steps

After completing face detection:
1. Face alignment and landmark detection (5.1.2)
2. Face recognition and verification (5.1.3)
3. Object detection with YOLO/SSD (5.2.1)
4. Real-world project implementation (Stage 7)

---

**Last Updated**: 2025-10-14
**Module Status**: ✅ 5.1.1 Complete

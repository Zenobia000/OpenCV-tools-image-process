# Stage 2: Core Operations - 核心操作模組

This directory contains essential image processing operations that form the foundation of all computer vision applications. These modules cover the core OpenCV functions you'll use daily.

## 🎯 Stage Overview

**Objective**: Master fundamental image operations and transformations
**Target Audience**: Learners with basic OpenCV knowledge
**Estimated Time**: 1-2 weeks
**Prerequisites**: Completed Stage 1 (Fundamentals)

## 📚 Module Structure

### 2.2.1 Image I/O and Display ✅
**File**: `2.2.1_image_io_display.ipynb`
**Duration**: 2-3 hours
**Complexity**: ⭐⭐

**Learning Objectives**:
- Master image reading, writing, and display operations
- Understand different file formats and their properties
- Learn proper image display techniques
- Handle file I/O errors gracefully

**Key Functions Covered**:
- `cv2.imread()` - Image loading with different modes
- `cv2.imwrite()` - Image saving with quality control
- `cv2.imshow()` - Native OpenCV display
- `matplotlib.pyplot` - Scientific visualization display

**File Format Support**:
- **JPEG**: Lossy compression, adjustable quality
- **PNG**: Lossless compression, transparency support
- **BMP**: Uncompressed, large file size
- **TIFF**: High-quality, scientific imaging
- **WebP**: Modern web format

**Display Techniques**:
- OpenCV native windows
- Matplotlib integration
- Jupyter notebook inline display
- Multi-image comparison layouts

### 2.2.2 Geometric Transformations ✅
**File**: `2.2.2_geometric_transformations.ipynb`
**Duration**: 3-4 hours
**Complexity**: ⭐⭐⭐

**Learning Objectives**:
- Understand affine and perspective transformations
- Master image resizing, rotation, and translation
- Learn document scanning correction techniques
- Implement custom transformation matrices

**Transformation Types**:
- **Scaling**: Resize images while preserving aspect ratio
- **Rotation**: Rotate images around arbitrary points
- **Translation**: Move images in 2D space
- **Affine**: Parallel lines remain parallel
- **Perspective**: Full 3D-to-2D projection

**Practical Applications**:
- Document perspective correction
- Image registration and alignment
- Augmentation for machine learning
- Panoramic image creation
- Camera calibration correction

**Advanced Topics**:
- Custom transformation matrices
- Interpolation methods comparison
- Border handling strategies
- Performance optimization techniques

### 2.2.3 Color Spaces ✅
**File**: `2.2.3_color_spaces.ipynb`
**Duration**: 2-3 hours
**Complexity**: ⭐⭐

**Learning Objectives**:
- Master color space conversions and applications
- Understand when to use different color spaces
- Implement color-based object detection
- Perform advanced color analysis

**Color Spaces Covered**:
- **BGR/RGB**: Default OpenCV and standard display
- **HSV**: Hue, Saturation, Value - intuitive color selection
- **LAB**: Perceptually uniform color space
- **YCrCb**: Luma and chroma separation
- **Grayscale**: Single channel intensity

**Practical Applications**:
- Skin tone detection in YCrCb
- Object detection by color in HSV
- Color constancy in LAB space
- Artistic effects with color manipulation
- Performance testing across color spaces

**Real-world Examples**:
- Traffic light detection
- Fruit ripeness assessment
- Medical imaging color analysis
- Quality control in manufacturing

### 2.2.4 Arithmetic Operations ✅
**File**: `2.2.4_arithmetic_operations.ipynb`
**Duration**: 2-3 hours
**Complexity**: ⭐⭐

**Learning Objectives**:
- Master image arithmetic operations
- Understand blending and compositing techniques
- Learn bitwise operations for masking
- Implement advanced image combinations

**Operations Covered**:
- **Addition/Subtraction**: Image math with overflow handling
- **Multiplication/Division**: Scaling and normalization
- **Bitwise Operations**: AND, OR, XOR, NOT for masking
- **Alpha Blending**: Weighted image combination
- **Masking**: Selective image operations

**Advanced Techniques**:
- **Background Replacement**: Green screen effects
- **Logo Overlay**: Watermarking with transparency
- **Image Morphing**: Smooth transitions between images
- **HDR Processing**: High dynamic range imaging
- **Difference Analysis**: Change detection between images

**Performance Optimization**:
- NumPy vs OpenCV arithmetic comparison
- Memory-efficient operations
- Vectorized computations
- GPU acceleration options

## 🚀 Learning Path

### Recommended Study Order
1. **Start Here**: `2.2.1_image_io_display.ipynb`
   - Essential for all subsequent work
   - Practice with your own images

2. **Build Skills**: `2.2.3_color_spaces.ipynb`
   - Crucial for object detection and analysis
   - Many practical applications

3. **Add Power**: `2.2.4_arithmetic_operations.ipynb`
   - Enables complex image combinations
   - Foundation for advanced techniques

4. **Master Geometry**: `2.2.2_geometric_transformations.ipynb`
   - Most mathematically demanding
   - Highly useful for real applications

### Hands-on Projects
After completing the modules, try these projects:
- **Photo Editor**: Combine all core operations
- **Document Scanner**: Use geometric transformations
- **Color Detective**: Apply color space analysis
- **Image Mixer**: Practice arithmetic operations

## 🔧 Technical Details

### Performance Benchmarks
All operations optimized for real-time performance:

| Operation | Target Time (640x480) | Typical Use Case |
|-----------|----------------------|------------------|
| Color Conversion | <2ms | Real-time video |
| Image Resize | <5ms | Display optimization |
| Rotation | <10ms | Document correction |
| Arithmetic Blend | <3ms | Overlay effects |

### Memory Management
- Efficient memory usage patterns
- Avoiding unnecessary copies
- Understanding reference vs copy semantics
- Garbage collection considerations

### Error Handling
- Robust file I/O error handling
- Graceful degradation for missing files
- Type and dimension validation
- User-friendly error messages

## 📊 Assessment & Validation

### Module Completion Checklist

**2.2.1 Image I/O**:
- [ ] Can load images in different modes (color/grayscale/unchanged)
- [ ] Successfully save images with different formats and quality
- [ ] Comfortable with both OpenCV and matplotlib display
- [ ] Handle file path errors appropriately

**2.2.2 Geometric Transformations**:
- [ ] Can resize images maintaining aspect ratios
- [ ] Successfully rotate images around custom points
- [ ] Implement basic perspective correction
- [ ] Understand transformation matrix concepts

**2.2.3 Color Spaces**:
- [ ] Convert between major color spaces fluently
- [ ] Apply color-based object detection
- [ ] Understand when to use each color space
- [ ] Implement color analysis algorithms

**2.2.4 Arithmetic Operations**:
- [ ] Perform safe arithmetic with proper overflow handling
- [ ] Create smooth blending effects
- [ ] Use masking for selective operations
- [ ] Implement background replacement techniques

### Knowledge Validation
```python
# Run this comprehensive test
def validate_stage2_knowledge():
    import cv2
    import numpy as np

    # Test 1: Load and display
    img = cv2.imread('test_image.jpg')
    assert img is not None, "Image loading failed"

    # Test 2: Color conversion
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    assert hsv.shape[:2] == img.shape[:2], "Color conversion failed"

    # Test 3: Geometric transformation
    rotated = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    assert rotated.shape[:2] == img.shape[1::-1], "Rotation failed"

    # Test 4: Arithmetic operation
    blended = cv2.addWeighted(img, 0.7, img, 0.3, 0)
    assert blended.shape == img.shape, "Blending failed"

    print("✅ All Stage 2 knowledge validation tests passed!")

# Run validation (uncomment to test)
# validate_stage2_knowledge()
```

## 🎯 Common Applications

### Real-World Use Cases

**Photography & Media**:
- Photo editing and enhancement
- Batch image processing
- Format conversion utilities
- Artistic filter applications

**Document Processing**:
- Scan quality improvement
- Perspective correction
- Multi-page document assembly
- Text extraction preparation

**Scientific Imaging**:
- Microscopy image analysis
- Astronomical image processing
- Medical imaging preprocessing
- Research data visualization

**Industrial Applications**:
- Quality control imaging
- Product inspection systems
- Measurement and calibration
- Automated visual testing

## 🤝 Tips for Success

### Best Practices
1. **Practice Regularly**: Work with different image types and sizes
2. **Experiment Freely**: Modify parameters to see effects
3. **Handle Errors**: Always check for None returns and exceptions
4. **Optimize Early**: Learn efficient coding patterns from the start

### Common Mistakes to Avoid
- Forgetting BGR vs RGB color order
- Not checking image loading success
- Ignoring data type and range issues
- Excessive memory allocation in loops

### Debugging Tips
- Use `print()` statements to check array shapes and types
- Visualize intermediate results frequently
- Compare outputs with expected results
- Use try-catch blocks for robust error handling

## 📈 Progress Tracking

### Competency Levels

**Novice**: Can perform basic operations with guidance
**Competent**: Comfortable with all core operations
**Proficient**: Can combine operations for complex tasks
**Expert**: Optimizes for performance and handles edge cases

### Next Steps
Upon completing Stage 2:
1. **Advance to Stage 3**: [Preprocessing Techniques](../03_preprocessing/README.md)
2. **Practice More**: Try [beginner exercises](../06_exercises/beginner/README.md)
3. **Build Something**: Create a simple image processing application

---

**Stage Status**: 100% Complete ✅
**Module Count**: 4/4 complete
**Last Updated**: 2024-10-14
**Prerequisites**: [Stage 1 Fundamentals](../01_fundamentals/README.md)
**Next Stage**: [Stage 3 Preprocessing](../03_preprocessing/README.md)
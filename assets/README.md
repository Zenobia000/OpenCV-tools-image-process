# Assets Directory - 資源文件目錄

This directory contains all multimedia resources, datasets, and models used throughout the OpenCV Computer Vision Toolkit. All assets are organized by type and usage scenario.

## 📁 Directory Structure

```
assets/
├── images/          # Test images and visual resources
│   ├── basic/       # General purpose test images (234 files)
│   ├── faces/       # Face detection specific images
│   ├── objects/     # Object recognition test images
│   └── medical/     # Medical imaging samples (simulated)
├── models/          # Pre-trained models and classifiers
│   ├── face_detection/    # Face detection models
│   ├── object_detection/  # Object detection models
│   ├── deep_learning/     # DNN models (ONNX, TensorFlow)
│   └── custom/           # Custom trained models
├── datasets/        # Training and evaluation datasets
│   ├── dlib_ObjectCategories10/  # 10-class object dataset
│   └── custom_annotations/       # Custom annotation data
└── videos/          # Video samples for motion analysis
    ├── security/    # Security camera footage samples
    ├── medical/     # Medical video sequences
    └── general/     # General purpose video clips
```

## 🖼️ Image Assets

### Basic Test Images (`images/basic/`)
**Total**: 234 high-quality test images
**Resolution Range**: 240x320 to 1920x1080
**Formats**: JPEG, PNG, BMP
**Use Cases**: General algorithm testing and development

**Categories**:
- Portrait photos (30+ images)
- Landscape and outdoor scenes (50+ images)
- Indoor and architectural (40+ images)
- Objects and still life (35+ images)
- Technical and scientific (25+ images)
- Text and document samples (20+ images)
- Artistic and creative (30+ images)

**Sample Images**:
- `faces01.jpg` - Group photo for face detection
- `faces02.png` - High resolution portrait
- `dlib68.jpg` - 68-point landmark reference
- `faces.png` - Multiple faces in complex scene

### Face Detection Images (`images/faces/`)
**Purpose**: Specialized images for face detection algorithm testing
**Contents**: Curated collection of facial images with various:
- Lighting conditions (bright, dim, mixed)
- Face orientations (frontal, profile, tilted)
- Demographics (age, gender, ethnicity diversity)
- Scene complexity (single face, crowds, partial occlusion)
- Image quality (high-res, low-res, compressed)

### Object Recognition Images (`images/objects/`)
**Purpose**: Object detection and classification testing
**Contents**: Common objects in various contexts:
- Household items and furniture
- Vehicles and transportation
- Animals and pets
- Food items and beverages
- Tools and equipment
- Sports and recreation items

## 🤖 Model Assets

### Face Detection Models (`models/face_detection/`)
**Included Models**:
- `haarcascade_frontalface_default.xml` - OpenCV Haar classifier
- `haarcascade_eye.xml` - Eye detection classifier
- `shape_predictor_68_face_landmarks.dat` - dlib 68-point predictor
- `dlib_face_recognition_resnet_model_v1.dat` - Face recognition model

### Object Detection Models (`models/object_detection/`)
**Model Types**:
- YOLO pre-trained weights and configurations
- SSD MobileNet models for mobile deployment
- Custom trained models for specific applications
- Model conversion utilities and scripts

### Deep Learning Models (`models/deep_learning/`)
**Framework Support**:
- **ONNX**: Cross-platform model format
- **TensorFlow**: `.pb` and SavedModel formats
- **PyTorch**: TorchScript `.pt` models
- **OpenVINO**: Optimized inference models

### Custom Models (`models/custom/`)
**Purpose**: User-trained and project-specific models
**Organization**:
- Training scripts and configurations
- Model performance benchmarks
- Conversion and optimization tools
- Documentation for each custom model

## 📊 Dataset Assets

### dlib ObjectCategories10 (`datasets/dlib_ObjectCategories10/`)
**Description**: 10-class object classification dataset
**Total Images**: 800+ annotated images
**Classes**: accordion, camera, chair, coffee_mug, guitar, laptop, motorbike, person, pizza, watch
**Annotations**: MATLAB format bounding boxes and labels
**Use Case**: Object classification training and evaluation

**Dataset Structure**:
```
dlib_ObjectCategories10/
├── accordion/           # 50+ images
├── camera/             # 50+ images
├── chair/              # 50+ images
├── coffee_mug/         # 50+ images
├── guitar/             # 50+ images
├── laptop/             # 50+ images
├── motorbike/          # 50+ images
├── person/             # 50+ images
├── pizza/              # 50+ images
└── watch/              # 50+ images
```

### Custom Annotations (`datasets/custom_annotations/`)
**Purpose**: Project-specific annotation data
**Formats**: JSON, XML, COCO format
**Tools**: Annotation utilities and validation scripts

## 🎬 Video Assets

### Security Camera Samples (`videos/security/`)
**Content**: Sample surveillance footage for testing security applications
**Scenarios**:
- Normal activity monitoring
- Motion detection scenarios
- Multi-person tracking
- Day/night variations

### Medical Video Sequences (`videos/medical/`)
**Content**: Simulated medical imaging sequences
**Types**:
- Ultrasound video loops
- Endoscopy footage samples
- X-ray fluoroscopy sequences
- MRI slice animations

**⚠️ Disclaimer**: All medical content is simulated/synthetic for educational purposes only.

### General Purpose (`videos/general/`)
**Content**: Various test videos for algorithm development
**Scenarios**:
- Object tracking challenges
- Scene change detection
- Motion analysis samples
- Camera shake and stabilization

## 🔧 Usage Guidelines

### Loading Assets in Code
```python
import cv2
import os

# Load test images
def load_test_image(category='basic', filename='faces01.jpg'):
    path = f'../../assets/images/{category}/{filename}'
    return cv2.imread(path)

# Load models
def load_face_cascade():
    path = '../../assets/models/face_detection/haarcascade_frontalface_default.xml'
    return cv2.CascadeClassifier(path)

# Access datasets
dataset_path = '../../assets/datasets/dlib_ObjectCategories10/'
categories = os.listdir(dataset_path)
```

### Path Conventions
- Use relative paths from notebook/script location
- Follow the established directory structure
- Check file existence before loading
- Provide fallback options for missing files

### File Naming Conventions
- Use descriptive names with consistent formatting
- Include resolution in filename when relevant
- Add quality indicators (hq, lq) when applicable
- Use standard extensions (.jpg, .png, .mp4, etc.)

## 📈 Asset Statistics

### Current Inventory
- **Total Images**: 600+ files
- **Total Models**: 27 model files
- **Dataset Images**: 328 annotated samples
- **Video Files**: 7 test sequences
- **Total Size**: ~500MB

### Quality Standards
- **Image Quality**: Minimum 240p, preferred 720p+
- **Format Standards**: JPEG for photos, PNG for graphics
- **Compression**: Balanced quality vs file size
- **Documentation**: Each major asset has usage notes

### Maintenance
- Regular quality auditing
- Dead link checking
- License compliance verification
- Storage optimization

## 🚀 Extending Assets

### Adding New Images
1. Place in appropriate category subdirectory
2. Use descriptive filenames
3. Maintain reasonable file sizes (<5MB)
4. Document any special properties

### Adding New Models
1. Include both model file and configuration
2. Provide usage examples
3. Document performance characteristics
4. Include license information

### Contributing Guidelines
1. Ensure copyright compliance
2. Maintain directory structure
3. Update this README when adding categories
4. Test assets before committing

## 📄 Licensing & Attribution

### Image Sources
- Original photography (CC0)
- Public domain collections
- Synthetic generated content
- Properly licensed stock photos

### Model Sources
- OpenCV official models (BSD License)
- dlib pre-trained models (Boost License)
- Community contributed models (Various)
- Custom trained models (Project License)

### Usage Rights
All assets in this directory are provided for:
- ✅ Educational and research use
- ✅ Non-commercial development
- ✅ Academic publication (with attribution)
- ✅ Open source project integration

Commercial usage may require additional licensing verification.

---

**Last Updated**: 2024-10-14
**Total Assets**: 600+ files
**Estimated Size**: ~500MB
**Maintenance**: Automated scripts available
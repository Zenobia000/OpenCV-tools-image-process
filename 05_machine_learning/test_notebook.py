#!/usr/bin/env python3
"""
Quick validation test for 5.1.2_WBS_object_classification notebook
Tests core functionality without full execution
"""

import cv2
import numpy as np
from pathlib import Path
import sys

def test_imports():
    """Test that all required libraries are available"""
    print("Testing imports...")
    try:
        from sklearn.svm import SVC, LinearSVC
        from sklearn.model_selection import train_test_split, GridSearchCV
        from sklearn.metrics import classification_report, confusion_matrix
        from sklearn.preprocessing import StandardScaler
        from skimage.feature import hog
        import joblib
        print("  ✅ All imports successful")
        return True
    except ImportError as e:
        print(f"  ❌ Import error: {e}")
        return False

def test_dataset_path():
    """Test that dataset path exists"""
    print("\nTesting dataset path...")
    dataset_path = Path('../assets/datasets/dlib_ObjectCategories10')

    if not dataset_path.exists():
        print(f"  ⚠️ Dataset not found at {dataset_path}")
        print("  Note: Dataset may need to be downloaded separately")
        return False

    categories = [d for d in dataset_path.iterdir() if d.is_dir()]
    print(f"  ✅ Found {len(categories)} categories")

    for cat in categories:
        image_count = len(list(cat.glob('*.jpg')))
        print(f"    - {cat.name}: {image_count} images")

    return len(categories) > 0

def test_hog_extraction():
    """Test HOG feature extraction"""
    print("\nTesting HOG feature extraction...")
    try:
        from skimage.feature import hog
        from skimage import exposure

        # Create test image
        test_img = np.zeros((128, 64), dtype=np.uint8)
        cv2.rectangle(test_img, (20, 20), (44, 108), 255, 2)
        cv2.circle(test_img, (32, 64), 15, 200, -1)

        # Extract HOG features
        features, hog_image = hog(
            test_img,
            orientations=9,
            pixels_per_cell=(8, 8),
            cells_per_block=(2, 2),
            block_norm='L2-Hys',
            visualize=True,
            feature_vector=True
        )

        print(f"  ✅ HOG feature dimension: {features.shape[0]}")
        print(f"  ✅ Feature range: [{features.min():.4f}, {features.max():.4f}]")
        return True
    except Exception as e:
        print(f"  ❌ HOG extraction error: {e}")
        return False

def test_svm_training():
    """Test basic SVM training"""
    print("\nTesting SVM training...")
    try:
        from sklearn.svm import LinearSVC
        from sklearn.datasets import make_classification

        # Create synthetic dataset
        X, y = make_classification(
            n_samples=100,
            n_features=50,
            n_classes=2,
            random_state=42
        )

        # Train SVM
        svm = LinearSVC(random_state=42, max_iter=1000)
        svm.fit(X, y)

        # Test prediction
        accuracy = svm.score(X, y)
        print(f"  ✅ SVM training successful")
        print(f"  ✅ Training accuracy: {accuracy:.4f}")
        return True
    except Exception as e:
        print(f"  ❌ SVM training error: {e}")
        return False

def test_model_save_load():
    """Test model saving and loading"""
    print("\nTesting model save/load...")
    try:
        import joblib
        from sklearn.svm import LinearSVC
        import tempfile

        # Create and train simple model
        X = np.random.randn(50, 10)
        y = np.random.randint(0, 2, 50)
        model = LinearSVC(random_state=42, max_iter=1000)
        model.fit(X, y)

        # Save and load
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
            temp_path = f.name

        joblib.dump(model, temp_path)
        loaded_model = joblib.load(temp_path)

        # Verify
        assert model.score(X, y) == loaded_model.score(X, y)

        # Cleanup
        Path(temp_path).unlink()

        print("  ✅ Model save/load successful")
        return True
    except Exception as e:
        print(f"  ❌ Model save/load error: {e}")
        return False

def main():
    """Run all tests"""
    print("="*60)
    print("5.1.2 Object Classification Notebook Validation")
    print("="*60)

    tests = [
        test_imports,
        test_dataset_path,
        test_hog_extraction,
        test_svm_training,
        test_model_save_load
    ]

    results = [test() for test in tests]

    print("\n" + "="*60)
    print("Test Summary:")
    print(f"  Passed: {sum(results)}/{len(results)}")
    print(f"  Failed: {len(results) - sum(results)}/{len(results)}")

    if all(results):
        print("\n✅ All tests passed! Notebook is ready to use.")
        return 0
    else:
        print("\n⚠️ Some tests failed. Check errors above.")
        return 1

if __name__ == '__main__':
    sys.exit(main())

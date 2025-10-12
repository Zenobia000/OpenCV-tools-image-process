"""
Pytest Configuration and Fixtures
================================

Shared fixtures and configuration for all tests.
"""

import pytest
import numpy as np
import cv2
import os
from pathlib import Path


@pytest.fixture
def sample_image_bgr():
    """Generate a simple BGR test image."""
    # Create a 100x100 BGR image with a gradient
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    for i in range(100):
        image[i, :] = [i * 2, 100, 255 - i * 2]
    return image


@pytest.fixture
def sample_image_gray():
    """Generate a simple grayscale test image."""
    # Create a 100x100 grayscale image with a gradient
    image = np.zeros((100, 100), dtype=np.uint8)
    for i in range(100):
        image[i, :] = i * 2
    return image


@pytest.fixture
def sample_image_small():
    """Generate a small 10x10 test image."""
    return np.random.randint(0, 256, (10, 10, 3), dtype=np.uint8)


@pytest.fixture
def temp_image_path(tmp_path):
    """Provide a temporary file path for testing image I/O."""
    return str(tmp_path / "test_image.jpg")


@pytest.fixture
def sample_contours():
    """Generate sample contours for testing."""
    contour1 = np.array([[[10, 10]], [[20, 10]], [[20, 20]], [[10, 20]]])
    contour2 = np.array([[[50, 50]], [[60, 50]], [[60, 60]], [[50, 60]]])
    return [contour1, contour2]


@pytest.fixture
def assets_path():
    """Get path to assets directory."""
    root = Path(__file__).parent.parent
    assets = root / "assets"
    if assets.exists():
        return str(assets)
    return None

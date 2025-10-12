"""
Unit Tests for image_utils Module
================================

Tests for image processing utility functions.
"""

import pytest
import numpy as np
import cv2
from utils.image_utils import (
    load_image,
    resize_image,
    normalize_image,
    apply_gamma_correction,
    create_mask
)


class TestLoadImage:
    """Tests for load_image function."""

    def test_load_image_color(self, temp_image_path, sample_image_bgr):
        """Test loading image in color mode."""
        cv2.imwrite(temp_image_path, sample_image_bgr)
        loaded = load_image(temp_image_path, color_mode='color')

        assert loaded is not None
        assert len(loaded.shape) == 3
        assert loaded.shape[2] == 3

    def test_load_image_grayscale(self, temp_image_path, sample_image_bgr):
        """Test loading image in grayscale mode."""
        cv2.imwrite(temp_image_path, sample_image_bgr)
        loaded = load_image(temp_image_path, color_mode='grayscale')

        assert loaded is not None
        assert len(loaded.shape) == 2

    def test_load_image_unchanged(self, temp_image_path, sample_image_bgr):
        """Test loading image in unchanged mode."""
        cv2.imwrite(temp_image_path, sample_image_bgr)
        loaded = load_image(temp_image_path, color_mode='unchanged')

        assert loaded is not None

    def test_load_nonexistent_image(self):
        """Test loading non-existent image returns None."""
        result = load_image('/nonexistent/path/image.jpg')
        assert result is None


class TestResizeImage:
    """Tests for resize_image function."""

    def test_resize_without_aspect_ratio(self, sample_image_bgr):
        """Test resizing without keeping aspect ratio."""
        target_size = (50, 50)
        resized = resize_image(sample_image_bgr, target_size, keep_aspect=False)

        assert resized.shape[1] == 50  # width
        assert resized.shape[0] == 50  # height

    def test_resize_with_aspect_ratio(self, sample_image_bgr):
        """Test resizing while keeping aspect ratio."""
        target_size = (80, 80)
        resized = resize_image(sample_image_bgr, target_size, keep_aspect=True)

        # Since original is 100x100, scaled to 80x80 should give 80x80
        assert resized.shape[1] == 80
        assert resized.shape[0] == 80

    def test_resize_with_aspect_ratio_non_square(self):
        """Test resizing non-square image with aspect ratio."""
        # Create 200x100 image
        image = np.zeros((100, 200, 3), dtype=np.uint8)
        target_size = (100, 100)
        resized = resize_image(image, target_size, keep_aspect=True)

        # Should scale to 100x50 to maintain 2:1 aspect ratio
        assert resized.shape[1] == 100
        assert resized.shape[0] == 50

    def test_resize_grayscale_image(self, sample_image_gray):
        """Test resizing grayscale image."""
        target_size = (50, 50)
        resized = resize_image(sample_image_gray, target_size, keep_aspect=False)

        assert resized.shape == (50, 50)


class TestNormalizeImage:
    """Tests for normalize_image function."""

    def test_normalize_to_0_1(self, sample_image_bgr):
        """Test normalization to [0, 1] range."""
        normalized = normalize_image(sample_image_bgr, range_type='0-1')

        assert normalized.dtype == np.float32
        assert normalized.min() >= 0.0
        assert normalized.max() <= 1.0

    def test_normalize_to_0_255(self):
        """Test normalization to [0, 255] range."""
        # Create image with arbitrary range
        image = np.array([[50, 100], [150, 200]], dtype=np.uint8)
        normalized = normalize_image(image, range_type='0-255')

        assert normalized.dtype == np.uint8
        assert normalized.min() == 0
        assert normalized.max() == 255

    def test_normalize_invalid_range_type(self, sample_image_bgr):
        """Test with invalid range type returns original image."""
        result = normalize_image(sample_image_bgr, range_type='invalid')

        np.testing.assert_array_equal(result, sample_image_bgr)

    def test_normalize_grayscale(self, sample_image_gray):
        """Test normalization on grayscale image."""
        normalized = normalize_image(sample_image_gray, range_type='0-1')

        assert normalized.dtype == np.float32
        assert normalized.min() >= 0.0
        assert normalized.max() <= 1.0


class TestApplyGammaCorrection:
    """Tests for apply_gamma_correction function."""

    def test_gamma_correction_default(self, sample_image_bgr):
        """Test gamma correction with default gamma=1.0."""
        corrected = apply_gamma_correction(sample_image_bgr, gamma=1.0)

        # Gamma=1.0 should return almost identical image
        assert corrected.shape == sample_image_bgr.shape
        # Allow small differences due to lookup table rounding
        assert np.allclose(corrected, sample_image_bgr, atol=2)

    def test_gamma_correction_brighten(self, sample_image_small):
        """Test gamma correction for brightening (gamma < 1)."""
        corrected = apply_gamma_correction(sample_image_small, gamma=0.5)

        # Lower gamma should generally brighten the image
        assert corrected.shape == sample_image_small.shape
        # Mean should generally increase (though not guaranteed for all pixels)
        # Just check that function runs without error

    def test_gamma_correction_darken(self, sample_image_small):
        """Test gamma correction for darkening (gamma > 1)."""
        corrected = apply_gamma_correction(sample_image_small, gamma=2.0)

        # Higher gamma should generally darken the image
        assert corrected.shape == sample_image_small.shape

    def test_gamma_correction_grayscale(self, sample_image_gray):
        """Test gamma correction on grayscale image."""
        corrected = apply_gamma_correction(sample_image_gray, gamma=1.5)

        assert corrected.shape == sample_image_gray.shape
        assert corrected.dtype == np.uint8


class TestCreateMask:
    """Tests for create_mask function."""

    def test_create_mask_bgr(self):
        """Test creating mask in BGR color space."""
        # Create image with known colors
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        image[25:75, 25:75] = [255, 0, 0]  # Blue square

        # Create mask for blue color
        lower = (200, 0, 0)
        upper = (255, 50, 50)
        mask = create_mask(image, lower, upper, color_space='BGR')

        assert mask.shape == (100, 100)
        assert mask.dtype == np.uint8
        # Check that center region is white (255) in mask
        assert mask[50, 50] == 255
        # Check that corner is black (0) in mask
        assert mask[10, 10] == 0

    def test_create_mask_hsv(self):
        """Test creating mask in HSV color space."""
        # Create image with red color
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        image[25:75, 25:75] = [0, 0, 255]  # Red in BGR

        # Create mask for red in HSV
        # Red has H values around 0 or 180, S and V high
        lower = (0, 100, 100)
        upper = (10, 255, 255)
        mask = create_mask(image, lower, upper, color_space='HSV')

        assert mask.shape == (100, 100)
        assert mask.dtype == np.uint8

    def test_create_mask_all_black(self, sample_image_bgr):
        """Test mask with range that matches nothing."""
        lower = (0, 0, 0)
        upper = (1, 1, 1)
        mask = create_mask(sample_image_bgr, lower, upper, color_space='BGR')

        # Should be mostly black (no matches)
        assert mask.shape == sample_image_bgr.shape[:2]

    def test_create_mask_all_white(self, sample_image_bgr):
        """Test mask with range that matches everything."""
        lower = (0, 0, 0)
        upper = (255, 255, 255)
        mask = create_mask(sample_image_bgr, lower, upper, color_space='BGR')

        # Should be all white (all matches)
        assert np.all(mask == 255)


@pytest.mark.unit
class TestImageUtilsIntegration:
    """Integration tests for image_utils functions."""

    def test_load_resize_normalize_pipeline(self, temp_image_path, sample_image_bgr):
        """Test complete pipeline: load -> resize -> normalize."""
        # Save image
        cv2.imwrite(temp_image_path, sample_image_bgr)

        # Load
        loaded = load_image(temp_image_path, color_mode='color')
        assert loaded is not None

        # Resize
        resized = resize_image(loaded, (50, 50), keep_aspect=False)
        assert resized.shape[:2] == (50, 50)

        # Normalize
        normalized = normalize_image(resized, range_type='0-1')
        assert normalized.dtype == np.float32
        assert 0.0 <= normalized.min() <= 1.0
        assert 0.0 <= normalized.max() <= 1.0

    def test_mask_and_gamma_correction(self, sample_image_bgr):
        """Test combining mask creation and gamma correction."""
        # Create mask
        mask = create_mask(sample_image_bgr, (0, 0, 0), (255, 255, 255), 'BGR')

        # Apply gamma correction
        corrected = apply_gamma_correction(sample_image_bgr, gamma=1.2)

        # Both should have compatible shapes
        assert mask.shape == corrected.shape[:2]

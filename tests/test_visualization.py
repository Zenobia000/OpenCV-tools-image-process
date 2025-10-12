"""
Unit Tests for visualization Module
==================================

Tests for visualization utility functions.
Note: Some tests use mocking to avoid displaying plots during testing.
"""

import pytest
import numpy as np
import cv2
from unittest.mock import patch, MagicMock
from utils.visualization import (
    display_image,
    display_multiple_images,
    plot_histogram,
    draw_contours_with_info,
    create_side_by_side_comparison
)


class TestDisplayImage:
    """Tests for display_image function."""

    @patch('matplotlib.pyplot.show')
    @patch('matplotlib.pyplot.figure')
    def test_display_bgr_image(self, mock_figure, mock_show, sample_image_bgr):
        """Test displaying BGR image."""
        display_image(sample_image_bgr, title="Test Image")

        # Verify plt.figure and plt.show were called
        mock_figure.assert_called_once()
        mock_show.assert_called_once()

    @patch('matplotlib.pyplot.show')
    @patch('matplotlib.pyplot.figure')
    def test_display_grayscale_image(self, mock_figure, mock_show, sample_image_gray):
        """Test displaying grayscale image."""
        display_image(sample_image_gray, title="Gray Image", cmap='gray')

        mock_figure.assert_called_once()
        mock_show.assert_called_once()

    @patch('matplotlib.pyplot.show')
    @patch('matplotlib.pyplot.figure')
    def test_display_with_custom_figsize(self, mock_figure, mock_show, sample_image_bgr):
        """Test displaying image with custom figure size."""
        display_image(sample_image_bgr, figsize=(15, 12))

        mock_figure.assert_called_once_with(figsize=(15, 12))
        mock_show.assert_called_once()


class TestDisplayMultipleImages:
    """Tests for display_multiple_images function."""

    @patch('matplotlib.pyplot.show')
    @patch('matplotlib.pyplot.subplots')
    def test_display_two_images(self, mock_subplots, mock_show, sample_image_bgr, sample_image_gray):
        """Test displaying two images side by side."""
        # Mock subplots return value
        fig = MagicMock()
        axes = [MagicMock(), MagicMock()]
        mock_subplots.return_value = (fig, axes)

        images = [sample_image_bgr, sample_image_gray]
        titles = ["BGR", "Gray"]

        display_multiple_images(images, titles, rows=1, cols=2)

        mock_subplots.assert_called_once()
        mock_show.assert_called_once()

    @patch('matplotlib.pyplot.show')
    @patch('matplotlib.pyplot.subplots')
    def test_display_single_image(self, mock_subplots, mock_show, sample_image_bgr):
        """Test displaying single image."""
        fig = MagicMock()
        axes = MagicMock()
        mock_subplots.return_value = (fig, axes)

        display_multiple_images([sample_image_bgr], ["Single"], rows=1, cols=1)

        mock_subplots.assert_called_once()
        mock_show.assert_called_once()

    @patch('matplotlib.pyplot.show')
    @patch('matplotlib.pyplot.subplots')
    def test_display_auto_cols(self, mock_subplots, mock_show, sample_image_small):
        """Test automatic column calculation."""
        fig = MagicMock()
        axes = [MagicMock(), MagicMock(), MagicMock()]
        mock_subplots.return_value = (fig, axes)

        images = [sample_image_small] * 3

        # Call with cols=None to trigger auto-calculation
        display_multiple_images(images, rows=1, cols=None)

        # Should calculate cols = 3 for 3 images in 1 row
        mock_subplots.assert_called_once_with(1, 3, figsize=(15, 10))
        mock_show.assert_called_once()

    @patch('matplotlib.pyplot.show')
    @patch('matplotlib.pyplot.subplots')
    def test_display_without_titles(self, mock_subplots, mock_show, sample_image_small):
        """Test displaying images without titles."""
        fig = MagicMock()
        axes = [MagicMock(), MagicMock()]
        mock_subplots.return_value = (fig, axes)

        images = [sample_image_small, sample_image_small]

        display_multiple_images(images, titles=None, rows=1, cols=2)

        mock_show.assert_called_once()


class TestPlotHistogram:
    """Tests for plot_histogram function."""

    @patch('matplotlib.pyplot.show')
    @patch('matplotlib.pyplot.figure')
    def test_plot_histogram_bgr(self, mock_figure, mock_show, sample_image_bgr):
        """Test plotting histogram for BGR image."""
        plot_histogram(sample_image_bgr, bins=256, title="BGR Histogram")

        mock_figure.assert_called_once()
        mock_show.assert_called_once()

    @patch('matplotlib.pyplot.show')
    @patch('matplotlib.pyplot.figure')
    def test_plot_histogram_grayscale(self, mock_figure, mock_show, sample_image_gray):
        """Test plotting histogram for grayscale image."""
        plot_histogram(sample_image_gray, bins=128, title="Gray Histogram")

        mock_figure.assert_called_once()
        mock_show.assert_called_once()

    @patch('matplotlib.pyplot.show')
    @patch('matplotlib.pyplot.figure')
    def test_plot_histogram_custom_bins(self, mock_figure, mock_show, sample_image_bgr):
        """Test plotting histogram with custom bin count."""
        plot_histogram(sample_image_bgr, bins=64)

        mock_figure.assert_called_once()
        mock_show.assert_called_once()


class TestDrawContoursWithInfo:
    """Tests for draw_contours_with_info function."""

    def test_draw_contours_basic(self, sample_image_bgr, sample_contours):
        """Test drawing contours on image."""
        result = draw_contours_with_info(sample_image_bgr, sample_contours, min_area=0)

        # Check that result has same shape as input
        assert result.shape == sample_image_bgr.shape

        # Check that result is different from original (contours drawn)
        assert not np.array_equal(result, sample_image_bgr)

    def test_draw_contours_min_area_filter(self, sample_image_bgr):
        """Test contour filtering by minimum area."""
        # Create contours with different areas
        small_contour = np.array([[[10, 10]], [[15, 10]], [[15, 15]], [[10, 15]]])
        large_contour = np.array([[[20, 20]], [[80, 20]], [[80, 80]], [[20, 80]]])
        contours = [small_contour, large_contour]

        # Set min_area to filter out small contour
        result = draw_contours_with_info(sample_image_bgr, contours, min_area=1000)

        assert result.shape == sample_image_bgr.shape

    def test_draw_contours_empty_list(self, sample_image_bgr):
        """Test drawing with empty contour list."""
        result = draw_contours_with_info(sample_image_bgr, [], min_area=100)

        # Should return copy of original image
        assert result.shape == sample_image_bgr.shape
        # Should be equal since no contours to draw
        np.testing.assert_array_equal(result, sample_image_bgr)

    def test_draw_contours_preserves_original(self, sample_image_bgr, sample_contours):
        """Test that original image is not modified."""
        original_copy = sample_image_bgr.copy()

        draw_contours_with_info(sample_image_bgr, sample_contours, min_area=0)

        # Original should remain unchanged
        np.testing.assert_array_equal(sample_image_bgr, original_copy)


class TestCreateSideBySideComparison:
    """Tests for create_side_by_side_comparison function."""

    @patch('utils.visualization.display_multiple_images')
    def test_side_by_side_basic(self, mock_display, sample_image_bgr, sample_image_gray):
        """Test basic side-by-side comparison."""
        # Convert grayscale to BGR for comparison
        gray_bgr = cv2.cvtColor(sample_image_gray, cv2.COLOR_GRAY2BGR)

        create_side_by_side_comparison(sample_image_bgr, gray_bgr)

        # Verify display_multiple_images was called correctly
        mock_display.assert_called_once()
        args, kwargs = mock_display.call_args

        # Check that two images were passed
        assert len(args[0]) == 2
        # Check default titles
        assert args[1] == ["Original", "Processed"]

    @patch('utils.visualization.display_multiple_images')
    def test_side_by_side_custom_titles(self, mock_display, sample_image_bgr):
        """Test side-by-side comparison with custom titles."""
        processed = cv2.GaussianBlur(sample_image_bgr, (5, 5), 0)

        create_side_by_side_comparison(
            sample_image_bgr,
            processed,
            title1="Before",
            title2="After"
        )

        mock_display.assert_called_once()
        args, kwargs = mock_display.call_args

        # Check custom titles
        assert args[1] == ["Before", "After"]


@pytest.mark.unit
class TestVisualizationIntegration:
    """Integration tests for visualization functions."""

    @patch('matplotlib.pyplot.show')
    @patch('matplotlib.pyplot.subplots')
    def test_contours_and_display(self, mock_subplots, mock_show, sample_image_bgr, sample_contours):
        """Test drawing contours and displaying result."""
        # Mock subplots
        fig = MagicMock()
        axes = [MagicMock(), MagicMock()]
        mock_subplots.return_value = (fig, axes)

        # Draw contours
        result = draw_contours_with_info(sample_image_bgr, sample_contours, min_area=0)

        # Display both images
        display_multiple_images(
            [sample_image_bgr, result],
            ["Original", "With Contours"],
            rows=1, cols=2
        )

        mock_show.assert_called_once()

    @patch('matplotlib.pyplot.show')
    @patch('matplotlib.pyplot.figure')
    def test_histogram_after_processing(self, mock_figure, mock_show, sample_image_bgr):
        """Test plotting histogram of processed image."""
        # Apply some processing
        gray = cv2.cvtColor(sample_image_bgr, cv2.COLOR_BGR2GRAY)

        # Plot histogram
        plot_histogram(gray, bins=256, title="Processed Histogram")

        mock_figure.assert_called_once()
        mock_show.assert_called_once()

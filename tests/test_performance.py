"""
Unit Tests for performance Module
================================

Tests for performance evaluation utility functions.
"""

import pytest
import numpy as np
import cv2
import time
from utils.performance import (
    time_function,
    benchmark_function,
    compare_algorithms,
    memory_usage,
    calculate_psnr,
    calculate_ssim,
    PerformanceProfiler
)


class TestTimeFunctionDecorator:
    """Tests for time_function decorator."""

    def test_time_function_basic(self, capsys):
        """Test that decorator measures execution time."""
        @time_function
        def sample_func():
            time.sleep(0.01)
            return 42

        result = sample_func()

        # Check function still returns correct value
        assert result == 42

        # Check that timing message was printed
        captured = capsys.readouterr()
        assert "sample_func took" in captured.out
        assert "seconds" in captured.out

    def test_time_function_with_args(self, capsys):
        """Test decorator with function arguments."""
        @time_function
        def add_numbers(a, b):
            return a + b

        result = add_numbers(5, 3)

        assert result == 8
        captured = capsys.readouterr()
        assert "add_numbers took" in captured.out


class TestBenchmarkFunction:
    """Tests for benchmark_function."""

    def test_benchmark_basic(self):
        """Test basic benchmarking functionality."""
        def simple_func(x):
            return x * 2

        stats = benchmark_function(simple_func, args=(5,), iterations=10)

        # Check all required keys exist
        assert 'mean' in stats
        assert 'std' in stats
        assert 'min' in stats
        assert 'max' in stats
        assert 'total' in stats

        # Check reasonable values
        assert stats['mean'] >= 0
        assert stats['min'] <= stats['mean'] <= stats['max']
        assert stats['total'] >= 0

    def test_benchmark_with_kwargs(self):
        """Test benchmarking with keyword arguments."""
        def func_with_kwargs(a, b=10):
            return a + b

        stats = benchmark_function(
            func_with_kwargs,
            args=(5,),
            kwargs={'b': 20},
            iterations=5
        )

        assert stats['mean'] >= 0
        assert stats['total'] >= 0

    def test_benchmark_iterations(self):
        """Test that benchmark runs specified number of iterations."""
        call_count = []

        def counting_func():
            call_count.append(1)

        benchmark_function(counting_func, iterations=15)

        # Should have been called 15 times
        assert len(call_count) == 15

    def test_benchmark_consistency(self):
        """Test that benchmark produces consistent results."""
        def consistent_func():
            time.sleep(0.001)

        stats = benchmark_function(consistent_func, iterations=5)

        # For a consistent function, std should be relatively small
        # compared to mean (though not zero due to system variance)
        assert stats['std'] < stats['mean']


class TestCompareAlgorithms:
    """Tests for compare_algorithms function."""

    def test_compare_two_algorithms(self, capsys):
        """Test comparing two algorithms."""
        def algo1(data):
            return data * 2

        def algo2(data):
            return data + data

        algorithms = {
            'multiply': algo1,
            'addition': algo2
        }

        results = compare_algorithms(algorithms, data=100, iterations=5)

        # Check both algorithms were benchmarked
        assert 'multiply' in results
        assert 'addition' in results

        # Check each has timing stats
        for name, stats in results.items():
            assert 'mean' in stats
            assert 'min' in stats
            assert 'max' in stats

        # Check output messages
        captured = capsys.readouterr()
        assert "Benchmarking multiply" in captured.out
        assert "Benchmarking addition" in captured.out

    def test_compare_single_algorithm(self):
        """Test comparing single algorithm."""
        def single_algo(data):
            return data ** 2

        results = compare_algorithms({'square': single_algo}, data=10, iterations=3)

        assert len(results) == 1
        assert 'square' in results


class TestMemoryUsageDecorator:
    """Tests for memory_usage decorator."""

    def test_memory_usage_basic(self, capsys):
        """Test memory usage monitoring."""
        @memory_usage
        def allocate_memory():
            # Allocate some memory
            data = [i for i in range(10000)]
            return len(data)

        result = allocate_memory()

        # Check function still works
        assert result == 10000

        # Check memory usage was printed
        captured = capsys.readouterr()
        assert "memory usage:" in captured.out
        assert "Current:" in captured.out
        assert "Peak:" in captured.out
        assert "MB" in captured.out

    def test_memory_usage_with_return(self):
        """Test that decorator preserves return value."""
        @memory_usage
        def return_value():
            return 42

        result = return_value()
        assert result == 42


class TestCalculatePSNR:
    """Tests for calculate_psnr function."""

    def test_psnr_identical_images(self, sample_image_bgr):
        """Test PSNR for identical images should be infinite."""
        psnr = calculate_psnr(sample_image_bgr, sample_image_bgr)

        assert psnr == float('inf')

    def test_psnr_different_images(self, sample_image_bgr):
        """Test PSNR for different images."""
        # Create slightly different image
        noisy = sample_image_bgr.copy()
        noise = np.random.randint(-10, 10, sample_image_bgr.shape, dtype=np.int16)
        noisy = np.clip(sample_image_bgr.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        psnr = calculate_psnr(sample_image_bgr, noisy)

        # PSNR should be finite and positive
        assert 0 < psnr < float('inf')

    def test_psnr_very_different_images(self):
        """Test PSNR for very different images should be low."""
        img1 = np.zeros((100, 100, 3), dtype=np.uint8)
        img2 = np.full((100, 100, 3), 255, dtype=np.uint8)

        psnr = calculate_psnr(img1, img2)

        # Should be a low value for completely different images
        assert 0 < psnr < 10

    def test_psnr_grayscale(self, sample_image_gray):
        """Test PSNR for grayscale images."""
        noisy = sample_image_gray.copy()
        noisy[::2, ::2] = np.clip(noisy[::2, ::2] + 10, 0, 255)

        psnr = calculate_psnr(sample_image_gray, noisy)

        assert 0 < psnr < float('inf')


class TestCalculateSSIM:
    """Tests for calculate_ssim function."""

    def test_ssim_identical_images(self, sample_image_bgr):
        """Test SSIM for identical images should be 1.0."""
        ssim = calculate_ssim(sample_image_bgr, sample_image_bgr)

        # Should be very close to 1.0
        assert 0.99 < ssim <= 1.0

    def test_ssim_different_images(self, sample_image_bgr):
        """Test SSIM for different images."""
        # Create slightly different image
        blurred = cv2.GaussianBlur(sample_image_bgr, (5, 5), 0)

        ssim = calculate_ssim(sample_image_bgr, blurred)

        # Should be between 0 and 1, closer to 1 for similar images
        assert 0.5 < ssim < 1.0

    def test_ssim_very_different_images(self):
        """Test SSIM for very different images should be low."""
        img1 = np.zeros((100, 100, 3), dtype=np.uint8)
        img2 = np.full((100, 100, 3), 255, dtype=np.uint8)

        ssim = calculate_ssim(img1, img2)

        # Should be low for completely different images
        assert 0 <= ssim < 0.5

    def test_ssim_grayscale(self, sample_image_gray):
        """Test SSIM for grayscale images."""
        shifted = np.roll(sample_image_gray, 5, axis=1)

        ssim = calculate_ssim(sample_image_gray, shifted)

        assert 0 <= ssim <= 1.0

    def test_ssim_custom_window_size(self, sample_image_bgr):
        """Test SSIM with custom window size."""
        ssim = calculate_ssim(sample_image_bgr, sample_image_bgr, window_size=7)

        # Should still be close to 1.0 for identical images
        assert 0.99 < ssim <= 1.0


class TestPerformanceProfiler:
    """Tests for PerformanceProfiler class."""

    def test_profiler_single_operation(self):
        """Test profiling single operation."""
        profiler = PerformanceProfiler()

        profiler.start_timer("operation1")
        time.sleep(0.01)
        profiler.end_timer("operation1")

        summary = profiler.get_summary()

        assert "operation1" in summary
        assert summary["operation1"] >= 0.01

    def test_profiler_multiple_operations(self):
        """Test profiling multiple operations."""
        profiler = PerformanceProfiler()

        profiler.start_timer("op1")
        time.sleep(0.01)
        profiler.end_timer("op1")

        profiler.start_timer("op2")
        time.sleep(0.02)
        profiler.end_timer("op2")

        summary = profiler.get_summary()

        assert "op1" in summary
        assert "op2" in summary
        assert summary["op2"] > summary["op1"]

    def test_profiler_print_summary(self, capsys):
        """Test printing profiler summary."""
        profiler = PerformanceProfiler()

        profiler.start_timer("test_op")
        time.sleep(0.01)
        profiler.end_timer("test_op")

        profiler.print_summary()

        captured = capsys.readouterr()
        assert "Performance Summary" in captured.out
        assert "test_op" in captured.out

    def test_profiler_incomplete_timing(self):
        """Test profiler with incomplete timing (no end)."""
        profiler = PerformanceProfiler()

        profiler.start_timer("incomplete")
        # Don't call end_timer

        summary = profiler.get_summary()

        # Incomplete operation should not be in summary
        assert "incomplete" not in summary

    def test_profiler_empty_summary(self):
        """Test profiler with no operations."""
        profiler = PerformanceProfiler()

        summary = profiler.get_summary()

        assert summary == {}


@pytest.mark.unit
class TestPerformanceIntegration:
    """Integration tests for performance functions."""

    def test_benchmark_image_processing(self, sample_image_bgr):
        """Test benchmarking image processing operations."""
        def gaussian_blur(img):
            return cv2.GaussianBlur(img, (5, 5), 0)

        def median_blur(img):
            return cv2.medianBlur(img, 5)

        algorithms = {
            'gaussian': gaussian_blur,
            'median': median_blur
        }

        results = compare_algorithms(algorithms, data=sample_image_bgr, iterations=5)

        assert 'gaussian' in results
        assert 'median' in results

        # Both should have reasonable timing
        assert results['gaussian']['mean'] > 0
        assert results['median']['mean'] > 0

    def test_psnr_ssim_consistency(self, sample_image_bgr):
        """Test that PSNR and SSIM give consistent results."""
        # Identical images
        psnr1 = calculate_psnr(sample_image_bgr, sample_image_bgr)
        ssim1 = calculate_ssim(sample_image_bgr, sample_image_bgr)

        assert psnr1 == float('inf')
        assert ssim1 > 0.99

        # Different images
        noisy = cv2.GaussianBlur(sample_image_bgr, (15, 15), 0)
        psnr2 = calculate_psnr(sample_image_bgr, noisy)
        ssim2 = calculate_ssim(sample_image_bgr, noisy)

        # Both metrics should show degradation
        assert psnr2 < psnr1
        assert ssim2 < ssim1

    def test_profiler_with_decorators(self, capsys):
        """Test using profiler with decorated functions."""
        profiler = PerformanceProfiler()

        @time_function
        def processing_step():
            time.sleep(0.01)
            return True

        profiler.start_timer("decorated_function")
        result = processing_step()
        profiler.end_timer("decorated_function")

        assert result is True

        # Both profiler and decorator should output timing info
        captured = capsys.readouterr()
        assert "processing_step took" in captured.out

        summary = profiler.get_summary()
        assert "decorated_function" in summary

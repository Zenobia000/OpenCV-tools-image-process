"""
Performance Evaluation Utility Functions
========================================

Functions for measuring and evaluating performance of image processing operations.
"""

import time
import numpy as np
import cv2
from typing import Callable, Any, Dict, List
from functools import wraps


def time_function(func: Callable) -> Callable:
    """
    Decorator to measure function execution time.

    Args:
        func: Function to time

    Returns:
        Wrapped function that prints execution time
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"{func.__name__} took {end_time - start_time:.4f} seconds")
        return result
    return wrapper


def benchmark_function(func: Callable, args: tuple = (), kwargs: dict = None,
                      iterations: int = 10) -> Dict[str, float]:
    """
    Benchmark a function over multiple iterations.

    Args:
        func: Function to benchmark
        args: Function arguments
        kwargs: Function keyword arguments
        iterations: Number of iterations

    Returns:
        Dictionary with timing statistics
    """
    if kwargs is None:
        kwargs = {}

    times = []
    for _ in range(iterations):
        start_time = time.time()
        func(*args, **kwargs)
        end_time = time.time()
        times.append(end_time - start_time)

    return {
        'mean': np.mean(times),
        'std': np.std(times),
        'min': np.min(times),
        'max': np.max(times),
        'total': np.sum(times)
    }


def compare_algorithms(algorithms: Dict[str, Callable], data: Any,
                      iterations: int = 10) -> Dict[str, Dict[str, float]]:
    """
    Compare performance of multiple algorithms.

    Args:
        algorithms: Dictionary of {name: function}
        data: Input data for algorithms
        iterations: Number of iterations per algorithm

    Returns:
        Performance comparison results
    """
    results = {}

    for name, func in algorithms.items():
        print(f"Benchmarking {name}...")
        stats = benchmark_function(func, args=(data,), iterations=iterations)
        results[name] = stats

    return results


def memory_usage(func: Callable) -> Callable:
    """
    Decorator to monitor memory usage of a function.

    Args:
        func: Function to monitor

    Returns:
        Wrapped function that prints memory usage
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        import tracemalloc
        tracemalloc.start()

        result = func(*args, **kwargs)

        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        print(f"{func.__name__} memory usage:")
        print(f"  Current: {current / 1024 / 1024:.2f} MB")
        print(f"  Peak: {peak / 1024 / 1024:.2f} MB")

        return result
    return wrapper


def calculate_psnr(original: np.ndarray, processed: np.ndarray) -> float:
    """
    Calculate Peak Signal-to-Noise Ratio between two images.

    Args:
        original: Original image
        processed: Processed image

    Returns:
        PSNR value in dB
    """
    mse = np.mean((original - processed) ** 2)
    if mse == 0:
        return float('inf')

    max_pixel = 255.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr


def calculate_ssim(original: np.ndarray, processed: np.ndarray,
                  window_size: int = 11) -> float:
    """
    Calculate Structural Similarity Index between two images.

    Args:
        original: Original image
        processed: Processed image
        window_size: Window size for SSIM calculation

    Returns:
        SSIM value (0-1)
    """
    # Convert to grayscale if needed
    if len(original.shape) == 3:
        original = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    if len(processed.shape) == 3:
        processed = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)

    # Constants for SSIM
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2

    # Convert to float
    original = original.astype(np.float64)
    processed = processed.astype(np.float64)

    # Calculate means
    mu1 = cv2.GaussianBlur(original, (window_size, window_size), 1.5)
    mu2 = cv2.GaussianBlur(processed, (window_size, window_size), 1.5)

    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2

    # Calculate variances and covariance
    sigma1_sq = cv2.GaussianBlur(original ** 2, (window_size, window_size), 1.5) - mu1_sq
    sigma2_sq = cv2.GaussianBlur(processed ** 2, (window_size, window_size), 1.5) - mu2_sq
    sigma12 = cv2.GaussianBlur(original * processed, (window_size, window_size), 1.5) - mu1_mu2

    # Calculate SSIM
    numerator = (2 * mu1_mu2 + C1) * (2 * sigma12 + C2)
    denominator = (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)

    ssim_map = numerator / denominator
    return np.mean(ssim_map)


class PerformanceProfiler:
    """Class for profiling image processing operations."""

    def __init__(self):
        self.measurements = {}

    def start_timer(self, operation_name: str):
        """Start timing an operation."""
        self.measurements[operation_name] = {'start': time.time()}

    def end_timer(self, operation_name: str):
        """End timing an operation."""
        if operation_name in self.measurements:
            self.measurements[operation_name]['end'] = time.time()
            duration = (self.measurements[operation_name]['end'] -
                       self.measurements[operation_name]['start'])
            self.measurements[operation_name]['duration'] = duration

    def get_summary(self) -> Dict[str, float]:
        """Get timing summary for all operations."""
        summary = {}
        for op, data in self.measurements.items():
            if 'duration' in data:
                summary[op] = data['duration']
        return summary

    def print_summary(self):
        """Print timing summary."""
        print("\nPerformance Summary:")
        print("-" * 40)
        for operation, duration in self.get_summary().items():
            print(f"{operation:25s}: {duration:.4f} seconds")
        print("-" * 40)
"""
Image Processing Utility Functions
=================================

Common image processing operations for OpenCV applications.
"""

import cv2
import numpy as np
from typing import Tuple, Optional, Union


def load_image(path: str, color_mode: str = 'color') -> np.ndarray:
    """
    Load an image from file path.

    Args:
        path: Path to image file
        color_mode: 'color', 'grayscale', or 'unchanged'

    Returns:
        Image as numpy array
    """
    mode_map = {
        'color': cv2.IMREAD_COLOR,
        'grayscale': cv2.IMREAD_GRAYSCALE,
        'unchanged': cv2.IMREAD_UNCHANGED
    }

    return cv2.imread(path, mode_map.get(color_mode, cv2.IMREAD_COLOR))


def resize_image(image: np.ndarray, size: Tuple[int, int],
                keep_aspect: bool = True) -> np.ndarray:
    """
    Resize image with optional aspect ratio preservation.

    Args:
        image: Input image
        size: Target (width, height)
        keep_aspect: Whether to maintain aspect ratio

    Returns:
        Resized image
    """
    if keep_aspect:
        h, w = image.shape[:2]
        target_w, target_h = size

        # Calculate scaling factor
        scale = min(target_w / w, target_h / h)
        new_w, new_h = int(w * scale), int(h * scale)

        return cv2.resize(image, (new_w, new_h))
    else:
        return cv2.resize(image, size)


def normalize_image(image: np.ndarray, range_type: str = '0-1') -> np.ndarray:
    """
    Normalize image pixel values.

    Args:
        image: Input image
        range_type: '0-1' or '0-255'

    Returns:
        Normalized image
    """
    if range_type == '0-1':
        return image.astype(np.float32) / 255.0
    elif range_type == '0-255':
        return cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    return image


def apply_gamma_correction(image: np.ndarray, gamma: float = 1.0) -> np.ndarray:
    """
    Apply gamma correction to image.

    Args:
        image: Input image
        gamma: Gamma value (< 1 brightens, > 1 darkens)

    Returns:
        Gamma corrected image
    """
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255
                     for i in np.arange(0, 256)]).astype("uint8")

    return cv2.LUT(image, table)


def create_mask(image: np.ndarray, lower: Tuple, upper: Tuple,
               color_space: str = 'BGR') -> np.ndarray:
    """
    Create binary mask based on color range.

    Args:
        image: Input image
        lower: Lower bound (B,G,R) or (H,S,V)
        upper: Upper bound (B,G,R) or (H,S,V)
        color_space: 'BGR' or 'HSV'

    Returns:
        Binary mask
    """
    if color_space == 'HSV':
        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    return cv2.inRange(image, lower, upper)
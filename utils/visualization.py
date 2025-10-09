"""
Visualization Utility Functions
==============================

Functions for displaying images, plots, and results.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional, Union


def display_image(image: np.ndarray, title: str = "Image",
                 cmap: str = None, figsize: Tuple[int, int] = (10, 8)) -> None:
    """
    Display image using matplotlib.

    Args:
        image: Image to display
        title: Window title
        cmap: Colormap for grayscale images
        figsize: Figure size (width, height)
    """
    plt.figure(figsize=figsize)

    if len(image.shape) == 3:
        # Convert BGR to RGB for matplotlib
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        plt.imshow(image_rgb)
    else:
        plt.imshow(image, cmap=cmap or 'gray')

    plt.title(title)
    plt.axis('off')
    plt.show()


def display_multiple_images(images: List[np.ndarray], titles: List[str] = None,
                          rows: int = 1, cols: int = None,
                          figsize: Tuple[int, int] = (15, 10)) -> None:
    """
    Display multiple images in a grid.

    Args:
        images: List of images
        titles: List of titles
        rows: Number of rows
        cols: Number of columns (auto-calculated if None)
        figsize: Figure size
    """
    n_images = len(images)
    if cols is None:
        cols = (n_images + rows - 1) // rows

    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    if n_images == 1:
        axes = [axes]
    elif rows == 1:
        axes = [axes] if cols == 1 else axes
    else:
        axes = axes.flatten()

    for i in range(n_images):
        if len(images[i].shape) == 3:
            image_rgb = cv2.cvtColor(images[i], cv2.COLOR_BGR2RGB)
            axes[i].imshow(image_rgb)
        else:
            axes[i].imshow(images[i], cmap='gray')

        if titles and i < len(titles):
            axes[i].set_title(titles[i])
        axes[i].axis('off')

    # Hide extra subplots
    for i in range(n_images, len(axes)):
        axes[i].axis('off')

    plt.tight_layout()
    plt.show()


def plot_histogram(image: np.ndarray, bins: int = 256,
                  title: str = "Histogram") -> None:
    """
    Plot image histogram.

    Args:
        image: Input image
        bins: Number of bins
        title: Plot title
    """
    plt.figure(figsize=(10, 6))

    if len(image.shape) == 3:
        colors = ('b', 'g', 'r')
        for i, color in enumerate(colors):
            hist = cv2.calcHist([image], [i], None, [bins], [0, 256])
            plt.plot(hist, color=color, label=f'{color.upper()} channel')
        plt.legend()
    else:
        hist = cv2.calcHist([image], [0], None, [bins], [0, 256])
        plt.plot(hist, color='black')

    plt.xlabel('Pixel Value')
    plt.ylabel('Frequency')
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.show()


def draw_contours_with_info(image: np.ndarray, contours: List,
                           min_area: float = 100) -> np.ndarray:
    """
    Draw contours with area information.

    Args:
        image: Input image
        contours: List of contours
        min_area: Minimum contour area to display

    Returns:
        Image with drawn contours
    """
    result = image.copy()

    for i, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if area > min_area:
            # Draw contour
            cv2.drawContours(result, [contour], -1, (0, 255, 0), 2)

            # Get bounding box
            x, y, w, h = cv2.boundingRect(contour)

            # Draw area text
            cv2.putText(result, f'Area: {int(area)}',
                       (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                       0.5, (255, 255, 255), 1)

    return result


def create_side_by_side_comparison(image1: np.ndarray, image2: np.ndarray,
                                 title1: str = "Original",
                                 title2: str = "Processed") -> None:
    """
    Display two images side by side for comparison.

    Args:
        image1: First image
        image2: Second image
        title1: Title for first image
        title2: Title for second image
    """
    display_multiple_images([image1, image2], [title1, title2],
                          rows=1, cols=2, figsize=(15, 6))
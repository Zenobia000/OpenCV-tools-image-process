#!/usr/bin/env python3
"""
7.3.2 醫學影像分析系統 - 區域分割模組

這個模組實現了專為醫學影像設計的區域分割算法，包括多種分割方法、
組織識別、量化分析等功能，適用於醫學影像的結構分析和測量。

功能特色：
- 多種分割算法 (Watershed, Region Growing, K-means, GrabCut)
- 醫學影像特定優化
- 組織類型識別
- 3D分割支援準備
- 量化測量工具
- 分割品質評估
- 互動式分割界面
- 分割結果可視化

作者: OpenCV Computer Vision Toolkit
日期: 2024-10-14
版本: 1.0

注意: 本模組僅用於教學和研究目的，不得用於實際醫療診斷。
     真實醫學影像分析需要專業醫療軟體和醫師專業判斷。
"""

import cv2
import numpy as np
import os
import sys
import time
import json
import logging
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, asdict
from enum import Enum
import math
from collections import defaultdict
from sklearn.cluster import KMeans
import scipy.ndimage as ndimage
from skimage import measure, morphology, segmentation

# 添加上級目錄到路徑
sys.path.append('../../utils')
try:
    from image_utils import load_image, resize_image
    from visualization import display_image, display_multiple_images
    from performance import time_function
except ImportError:
    print("⚠️ 無法導入工具模組，部分功能可能受限")

# 配置日誌
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SegmentationMethod(Enum):
    """分割方法枚舉"""
    WATERSHED = "watershed"
    REGION_GROWING = "region_growing"
    KMEANS = "kmeans"
    GRABCUT = "grabcut"
    OTSU = "otsu"
    ADAPTIVE_THRESHOLD = "adaptive"
    MORPHOLOGICAL = "morphological"

class TissueType(Enum):
    """組織類型枚舉"""
    BACKGROUND = 0
    SOFT_TISSUE = 1
    BONE = 2
    AIR = 3
    CONTRAST_AGENT = 4
    PATHOLOGY = 5

@dataclass
class SegmentationResult:
    """分割結果數據結構"""
    segmented_image: np.ndarray
    region_labels: np.ndarray
    region_count: int
    region_properties: List[Dict[str, Any]]
    processing_time: float
    method_used: SegmentationMethod
    parameters_used: Dict[str, Any]
    quality_score: float

@dataclass
class RegionProperties:
    """區域屬性"""
    label: int
    area: float
    centroid: Tuple[float, float]
    bbox: Tuple[int, int, int, int]
    perimeter: float
    circularity: float
    eccentricity: float
    mean_intensity: float
    std_intensity: float
    tissue_type: TissueType

class MedicalImageSegmentation:
    """醫學影像分割器"""

    def __init__(self, config_file: str = None):
        """初始化醫學影像分割器"""
        self.config = self._load_config(config_file)

        # 組織類型的強度範圍 (適用於CT影像的HU值)
        self.tissue_intensity_ranges = {
            TissueType.AIR: (-1000, -900),
            TissueType.SOFT_TISSUE: (-100, 200),
            TissueType.BONE: (200, 3000),
            TissueType.CONTRAST_AGENT: (100, 500),
            TissueType.BACKGROUND: (-1024, -900)
        }

        logger.info("醫學影像分割器初始化完成")
        logger.warning("⚠️ 僅用於教學研究，不得用於實際醫療診斷")

    def _load_config(self, config_file: str) -> Dict:
        """載入配置文件"""
        default_config = {
            "segmentation": {
                "default_method": "watershed",
                "preprocessing": True,
                "postprocessing": True,
                "min_region_size": 50,
                "max_regions": 1000
            },
            "watershed": {
                "distance_transform": True,
                "noise_removal": True,
                "sure_bg_threshold": 0.7,
                "sure_fg_threshold": 0.5,
                "kernel_size": 3
            },
            "kmeans": {
                "n_clusters": 5,
                "max_iterations": 20,
                "epsilon": 1.0,
                "attempts": 10
            },
            "region_growing": {
                "seed_points": "auto",  # auto, manual, grid
                "similarity_threshold": 10,
                "max_iterations": 1000,
                "connectivity": 8
            },
            "grabcut": {
                "iterations": 5,
                "margin": 10,
                "auto_rectangle": True
            },
            "morphological": {
                "operation": "opening",  # opening, closing, gradient
                "kernel_shape": "ellipse",  # rect, ellipse, cross
                "kernel_size": [5, 5],
                "iterations": 2
            },
            "tissue_analysis": {
                "enable_tissue_classification": True,
                "intensity_based_classification": True,
                "texture_analysis": False,
                "hu_value_analysis": False  # 僅適用於DICOM CT影像
            },
            "quality_metrics": {
                "calculate_properties": True,
                "measure_accuracy": False,  # 需要ground truth
                "connectivity_analysis": True,
                "homogeneity_analysis": True
            }
        }

        if config_file and os.path.exists(config_file):
            try:
                with open(config_file, 'r', encoding='utf-8') as f:
                    user_config = json.load(f)
                self._deep_update(default_config, user_config)
                logger.info(f"已載入配置文件: {config_file}")
            except Exception as e:
                logger.warning(f"無法載入配置文件: {e}，使用預設配置")

        return default_config

    def _deep_update(self, base_dict: Dict, update_dict: Dict):
        """深度更新字典"""
        for key, value in update_dict.items():
            if isinstance(value, dict) and key in base_dict:
                self._deep_update(base_dict[key], value)
            else:
                base_dict[key] = value

    def preprocess_for_segmentation(self, image: np.ndarray) -> np.ndarray:
        """針對分割優化的預處理"""
        if not self.config["segmentation"]["preprocessing"]:
            return image

        processed = image.copy()

        # 轉換為灰階
        if len(processed.shape) == 3:
            processed = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)

        # 降噪
        processed = cv2.medianBlur(processed, 3)

        # 增強對比度
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        processed = clahe.apply(processed)

        return processed

    def segment_watershed(self, image: np.ndarray) -> SegmentationResult:
        """Watershed分割算法"""
        start_time = time.time()
        config = self.config["watershed"]

        # 預處理
        gray = self.preprocess_for_segmentation(image)

        # 噪聲移除
        if config["noise_removal"]:
            kernel = np.ones((config["kernel_size"], config["kernel_size"]), np.uint8)
            opening = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel, iterations=2)
        else:
            opening = gray

        # 確定背景區域
        sure_bg = cv2.dilate(opening, kernel, iterations=3)

        # 距離變換
        if config["distance_transform"]:
            dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)

            # 確定前景區域
            _, sure_fg = cv2.threshold(dist_transform,
                                     config["sure_fg_threshold"] * dist_transform.max(),
                                     255, 0)
        else:
            # 使用閾值方法
            _, sure_fg = cv2.threshold(opening, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # 找到未知區域
        sure_fg = np.uint8(sure_fg)
        unknown = cv2.subtract(sure_bg, sure_fg)

        # 標記連通組件
        _, markers = cv2.connectedComponents(sure_fg)

        # 為背景添加標記
        markers = markers + 1
        markers[unknown == 255] = 0

        # 應用watershed
        if len(image.shape) == 3:
            markers = cv2.watershed(image, markers)
        else:
            # 轉換為3通道
            img_3ch = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            markers = cv2.watershed(img_3ch, markers)

        processing_time = (time.time() - start_time) * 1000

        # 創建分割結果圖像
        segmented = np.zeros_like(gray)
        segmented[markers > 1] = 255
        segmented[markers == -1] = 128  # 邊界

        # 計算區域屬性
        region_props = self._calculate_region_properties(markers, gray)

        # 計算品質分數
        quality_score = self._calculate_segmentation_quality(segmented, gray)

        return SegmentationResult(
            segmented_image=segmented,
            region_labels=markers,
            region_count=len(np.unique(markers)) - 1,  # 排除背景
            region_properties=region_props,
            processing_time=processing_time,
            method_used=SegmentationMethod.WATERSHED,
            parameters_used=config,
            quality_score=quality_score
        )

    def segment_kmeans(self, image: np.ndarray) -> SegmentationResult:
        """K-means分割算法"""
        start_time = time.time()
        config = self.config["kmeans"]

        # 預處理
        gray = self.preprocess_for_segmentation(image)

        # 重塑數據為K-means輸入格式
        pixel_values = gray.reshape((-1, 1))
        pixel_values = np.float32(pixel_values)

        # 執行K-means
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
                   config["max_iterations"], config["epsilon"])

        _, labels, centers = cv2.kmeans(
            pixel_values,
            config["n_clusters"],
            None,
            criteria,
            config["attempts"],
            cv2.KMEANS_RANDOM_CENTERS
        )

        # 重塑結果
        centers = np.uint8(centers)
        segmented = centers[labels.flatten()]
        segmented = segmented.reshape(gray.shape)

        # 創建標籤圖像
        region_labels = labels.reshape(gray.shape)

        processing_time = (time.time() - start_time) * 1000

        # 計算區域屬性
        region_props = self._calculate_region_properties(region_labels, gray)

        # 計算品質分數
        quality_score = self._calculate_segmentation_quality(segmented, gray)

        return SegmentationResult(
            segmented_image=segmented,
            region_labels=region_labels,
            region_count=config["n_clusters"],
            region_properties=region_props,
            processing_time=processing_time,
            method_used=SegmentationMethod.KMEANS,
            parameters_used=config,
            quality_score=quality_score
        )

    def segment_region_growing(self, image: np.ndarray,
                              seed_points: List[Tuple[int, int]] = None) -> SegmentationResult:
        """區域生長分割算法"""
        start_time = time.time()
        config = self.config["region_growing"]

        # 預處理
        gray = self.preprocess_for_segmentation(image)

        if seed_points is None:
            # 自動生成種子點
            seed_points = self._generate_seed_points(gray)

        # 初始化分割結果
        h, w = gray.shape
        segmented = np.zeros((h, w), dtype=np.uint8)
        visited = np.zeros((h, w), dtype=bool)

        similarity_threshold = config["similarity_threshold"]
        max_iterations = config["max_iterations"]
        connectivity = config["connectivity"]

        region_label = 1

        # 8-鄰域或4-鄰域
        if connectivity == 8:
            neighbors = [(-1, -1), (-1, 0), (-1, 1), (0, -1),
                        (0, 1), (1, -1), (1, 0), (1, 1)]
        else:  # 4-鄰域
            neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1)]

        # 對每個種子點進行區域生長
        for seed_x, seed_y in seed_points:
            if visited[seed_y, seed_x]:
                continue

            # 初始化隊列
            queue = [(seed_x, seed_y)]
            region_pixels = []
            seed_intensity = float(gray[seed_y, seed_x])

            iteration = 0
            while queue and iteration < max_iterations:
                if not queue:
                    break

                x, y = queue.pop(0)

                if visited[y, x]:
                    continue

                current_intensity = float(gray[y, x])

                # 檢查相似性
                if abs(current_intensity - seed_intensity) <= similarity_threshold:
                    visited[y, x] = True
                    segmented[y, x] = region_label
                    region_pixels.append((x, y))

                    # 添加鄰域像素到隊列
                    for dx, dy in neighbors:
                        nx, ny = x + dx, y + dy
                        if (0 <= nx < w and 0 <= ny < h and
                            not visited[ny, nx]):
                            queue.append((nx, ny))

                iteration += 1

            # 如果區域足夠大，保留此區域
            if len(region_pixels) >= self.config["segmentation"]["min_region_size"]:
                region_label += 1
            else:
                # 重置小區域
                for x, y in region_pixels:
                    segmented[y, x] = 0
                    visited[y, x] = False

        processing_time = (time.time() - start_time) * 1000

        # 計算區域屬性
        region_props = self._calculate_region_properties(segmented, gray)

        # 計算品質分數
        quality_score = self._calculate_segmentation_quality(segmented, gray)

        return SegmentationResult(
            segmented_image=segmented,
            region_labels=segmented,
            region_count=region_label - 1,
            region_properties=region_props,
            processing_time=processing_time,
            method_used=SegmentationMethod.REGION_GROWING,
            parameters_used=config,
            quality_score=quality_score
        )

    def _generate_seed_points(self, image: np.ndarray,
                            method: str = "grid") -> List[Tuple[int, int]]:
        """生成種子點"""
        h, w = image.shape
        seed_points = []

        if method == "grid":
            # 網格採樣
            step = max(h, w) // 10  # 10x10網格
            for y in range(step, h - step, step):
                for x in range(step, w - step, step):
                    seed_points.append((x, y))

        elif method == "harris_corners":
            # 使用Harris角點作為種子
            corners = cv2.goodFeaturesToTrack(image, 100, 0.01, 10)
            if corners is not None:
                for corner in corners:
                    x, y = corner.ravel()
                    seed_points.append((int(x), int(y)))

        elif method == "intensity_peaks":
            # 使用強度峰值作為種子
            # 簡化實現：使用局部最大值
            from scipy.ndimage import maximum_filter
            local_maxima = (image == maximum_filter(image, size=20))
            y_coords, x_coords = np.where(local_maxima)
            for x, y in zip(x_coords, y_coords):
                seed_points.append((int(x), int(y)))

        return seed_points

    def segment_grabcut(self, image: np.ndarray,
                       rect: Tuple[int, int, int, int] = None) -> SegmentationResult:
        """GrabCut分割算法"""
        start_time = time.time()
        config = self.config["grabcut"]

        if len(image.shape) != 3:
            # GrabCut需要彩色圖像
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        h, w = image.shape[:2]

        # 自動生成前景矩形（如果未提供）
        if rect is None or config["auto_rectangle"]:
            margin = config["margin"]
            rect = (margin, margin, w - 2*margin, h - 2*margin)

        # 初始化遮罩
        mask = np.zeros((h, w), np.uint8)

        # 前景和背景模型
        bgd_model = np.zeros((1, 65), np.float64)
        fgd_model = np.zeros((1, 65), np.float64)

        # 應用GrabCut
        cv2.grabCut(image, mask, rect, bgd_model, fgd_model,
                   config["iterations"], cv2.GC_INIT_WITH_RECT)

        # 創建最終遮罩
        mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')

        processing_time = (time.time() - start_time) * 1000

        # 應用遮罩到圖像
        segmented = image * mask2[:, :, np.newaxis]

        # 轉換為灰階用於屬性計算
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image

        # 計算區域屬性
        region_props = self._calculate_region_properties(mask2, gray)

        # 計算品質分數
        quality_score = self._calculate_segmentation_quality(mask2, gray)

        return SegmentationResult(
            segmented_image=cv2.cvtColor(segmented, cv2.COLOR_BGR2GRAY),
            region_labels=mask2,
            region_count=2,  # 前景和背景
            region_properties=region_props,
            processing_time=processing_time,
            method_used=SegmentationMethod.GRABCUT,
            parameters_used=config,
            quality_score=quality_score
        )

    def _calculate_region_properties(self, labels: np.ndarray,
                                   intensity_image: np.ndarray) -> List[Dict[str, Any]]:
        """計算區域屬性"""
        properties = []
        unique_labels = np.unique(labels)

        for label in unique_labels:
            if label == 0:  # 跳過背景
                continue

            # 創建區域遮罩
            mask = (labels == label).astype(np.uint8)

            if np.sum(mask) == 0:
                continue

            try:
                # 基本幾何屬性
                area = np.sum(mask)

                # 質心
                moments = cv2.moments(mask)
                if moments['m00'] != 0:
                    centroid_x = moments['m10'] / moments['m00']
                    centroid_y = moments['m01'] / moments['m00']
                    centroid = (centroid_x, centroid_y)
                else:
                    centroid = (0, 0)

                # 邊界框
                y_coords, x_coords = np.where(mask)
                if len(x_coords) > 0 and len(y_coords) > 0:
                    bbox = (int(np.min(x_coords)), int(np.min(y_coords)),
                           int(np.max(x_coords) - np.min(x_coords)),
                           int(np.max(y_coords) - np.min(y_coords)))
                else:
                    bbox = (0, 0, 0, 0)

                # 周長
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                perimeter = cv2.arcLength(contours[0], True) if contours else 0

                # 圓形度
                if perimeter > 0:
                    circularity = 4 * np.pi * area / (perimeter * perimeter)
                else:
                    circularity = 0

                # 強度統計
                masked_intensity = intensity_image[mask > 0]
                if len(masked_intensity) > 0:
                    mean_intensity = float(np.mean(masked_intensity))
                    std_intensity = float(np.std(masked_intensity))
                else:
                    mean_intensity = 0
                    std_intensity = 0

                # 組織類型分類
                tissue_type = self._classify_tissue_type(mean_intensity)

                properties.append({
                    'label': int(label),
                    'area': float(area),
                    'centroid': centroid,
                    'bbox': bbox,
                    'perimeter': float(perimeter),
                    'circularity': float(circularity),
                    'mean_intensity': mean_intensity,
                    'std_intensity': std_intensity,
                    'tissue_type': tissue_type.value if isinstance(tissue_type, TissueType) else tissue_type
                })

            except Exception as e:
                logger.warning(f"計算區域 {label} 屬性失敗: {e}")

        return properties

    def _classify_tissue_type(self, mean_intensity: float) -> TissueType:
        """根據強度分類組織類型"""
        if not self.config["tissue_analysis"]["intensity_based_classification"]:
            return TissueType.SOFT_TISSUE

        # 正規化強度到HU值範圍（簡化）
        # 實際應用中需要真實的DICOM數據和標定
        normalized_intensity = (mean_intensity - 127.5) * 8  # 簡化的HU值估算

        for tissue_type, (min_hu, max_hu) in self.tissue_intensity_ranges.items():
            if min_hu <= normalized_intensity <= max_hu:
                return tissue_type

        # 預設為軟組織
        return TissueType.SOFT_TISSUE

    def _calculate_segmentation_quality(self, segmented: np.ndarray,
                                      original: np.ndarray) -> float:
        """計算分割品質分數"""
        try:
            quality_score = 0.0

            # 區域連通性 (30% 權重)
            if self.config["quality_metrics"]["connectivity_analysis"]:
                num_labels, labels = cv2.connectedComponents(segmented)
                connectivity_score = 1.0 / (num_labels + 1)  # 區域越少越好
                quality_score += 0.3 * connectivity_score

            # 區域同質性 (40% 權重)
            if self.config["quality_metrics"]["homogeneity_analysis"]:
                unique_regions = np.unique(segmented)
                homogeneity_scores = []

                for region_val in unique_regions:
                    if region_val == 0:  # 跳過背景
                        continue

                    mask = segmented == region_val
                    if np.sum(mask) > 0:
                        region_intensities = original[mask]
                        if len(region_intensities) > 1:
                            # 使用變異係數衡量同質性
                            cv_coeff = np.std(region_intensities) / (np.mean(region_intensities) + 1e-6)
                            homogeneity = 1.0 / (1.0 + cv_coeff)
                            homogeneity_scores.append(homogeneity)

                if homogeneity_scores:
                    avg_homogeneity = np.mean(homogeneity_scores)
                    quality_score += 0.4 * avg_homogeneity

            # 邊界清晰度 (30% 權重)
            edges = cv2.Canny(segmented, 50, 150)
            edge_strength = np.mean(edges) / 255.0
            quality_score += 0.3 * edge_strength

            return min(1.0, quality_score)

        except Exception as e:
            logger.warning(f"品質評估計算失敗: {e}")
            return 0.5  # 預設中等品質

    def compare_segmentation_methods(self, image: np.ndarray) -> Dict[str, SegmentationResult]:
        """比較不同分割方法"""
        logger.info("開始比較不同分割方法...")

        methods_to_test = [
            ("watershed", self.segment_watershed),
            ("kmeans", self.segment_kmeans),
            ("grabcut", self.segment_grabcut)
        ]

        results = {}

        for method_name, method_func in methods_to_test:
            try:
                logger.info(f"測試 {method_name} 分割...")
                result = method_func(image)
                results[method_name] = result

                logger.info(f"  {method_name}: {result.region_count} 區域, "
                           f"{result.processing_time:.1f}ms, "
                           f"品質: {result.quality_score:.3f}")

            except Exception as e:
                logger.error(f"{method_name} 分割失敗: {e}")

        return results

    def visualize_segmentation(self, original: np.ndarray,
                             results: Dict[str, SegmentationResult]) -> np.ndarray:
        """可視化分割結果"""
        if not results:
            return original

        # 計算顯示佈局
        num_results = len(results) + 1  # 包括原始圖像
        cols = min(num_results, 3)
        rows = math.ceil(num_results / cols)

        # 創建合成圖像
        if len(original.shape) == 3:
            display_gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
        else:
            display_gray = original

        h, w = display_gray.shape
        combined = np.zeros((rows * h, cols * w, 3), dtype=np.uint8)

        # 放置原始圖像
        combined[:h, :w] = cv2.cvtColor(display_gray, cv2.COLOR_GRAY2BGR)
        cv2.putText(combined, "Original", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # 放置分割結果
        position = 1
        for method_name, result in results.items():
            row = position // cols
            col = position % cols

            y_start = row * h
            y_end = (row + 1) * h
            x_start = col * w
            x_end = (col + 1) * w

            # 創建彩色分割圖像
            colored_segmentation = self._create_colored_segmentation(result.region_labels)

            combined[y_start:y_end, x_start:x_end] = colored_segmentation

            # 添加方法名稱和統計信息
            info_text = f"{method_name}"
            cv2.putText(combined, info_text, (x_start + 10, y_start + 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            stats_text = f"Regions: {result.region_count}, Quality: {result.quality_score:.2f}"
            cv2.putText(combined, stats_text, (x_start + 10, y_start + 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            position += 1

        return combined

    def _create_colored_segmentation(self, labels: np.ndarray) -> np.ndarray:
        """創建彩色分割結果"""
        # 創建顏色映射
        unique_labels = np.unique(labels)
        colors = self._generate_colors(len(unique_labels))

        h, w = labels.shape
        colored = np.zeros((h, w, 3), dtype=np.uint8)

        for i, label in enumerate(unique_labels):
            mask = labels == label
            colored[mask] = colors[i]

        return colored

    def _generate_colors(self, num_colors: int) -> List[Tuple[int, int, int]]:
        """生成區域顏色"""
        colors = []

        # 背景使用黑色
        colors.append((0, 0, 0))

        # 為其他區域生成隨機顏色
        np.random.seed(42)  # 固定種子以獲得一致的顏色
        for i in range(num_colors - 1):
            color = tuple(np.random.randint(50, 255, 3).tolist())
            colors.append(color)

        return colors


def demo_medical_segmentation():
    """醫學影像分割演示"""
    print("🏥 醫學影像區域分割演示")
    print("=" * 50)
    print("⚠️ 注意：僅用於教學研究，不得用於實際醫療診斷")

    # 創建分割器
    segmenter = MedicalImageSegmentation()

    # 尋找測試圖像
    test_image_path = "../../assets/images/basic/faces01.jpg"

    if not os.path.exists(test_image_path):
        print("❌ 測試圖像不存在，創建模擬醫學影像")

        # 創建模擬X光影像
        demo_image = np.zeros((400, 400), dtype=np.uint8)

        # 添加不同密度的組織結構
        # 軟組織區域
        cv2.rectangle(demo_image, (50, 50), (350, 350), 120, -1)

        # 骨骼結構
        cv2.rectangle(demo_image, (100, 100), (150, 300), 220, -1)
        cv2.rectangle(demo_image, (250, 100), (300, 300), 220, -1)

        # 肺部（低密度）
        cv2.circle(demo_image, (120, 180), 40, 60, -1)
        cv2.circle(demo_image, (280, 180), 40, 60, -1)

        # 添加噪聲
        noise = np.random.normal(0, 10, demo_image.shape)
        demo_image = np.clip(demo_image.astype(np.float32) + noise, 0, 255).astype(np.uint8)

        print("✅ 已創建模擬醫學影像")

    else:
        # 載入真實圖像並轉換為醫學影像格式
        demo_image = load_image(test_image_path)
        demo_image = cv2.cvtColor(demo_image, cv2.COLOR_BGR2GRAY)
        demo_image = resize_image(demo_image, max_width=400)
        print(f"✅ 已載入測試圖像: {os.path.basename(test_image_path)}")

    print(f"🖼️  影像尺寸: {demo_image.shape}")

    # 比較不同分割方法
    comparison_results = segmenter.compare_segmentation_methods(demo_image)

    if comparison_results:
        print(f"\n📊 分割方法比較結果:")
        print("-" * 60)

        # 按品質排序
        sorted_results = sorted(comparison_results.items(),
                              key=lambda x: x[1].quality_score, reverse=True)

        for method_name, result in sorted_results:
            print(f"{method_name:12}: {result.region_count:3d} 區域, "
                  f"{result.processing_time:6.1f}ms, "
                  f"品質: {result.quality_score:.3f}")

        # 可視化比較結果
        visualization = segmenter.visualize_segmentation(demo_image, comparison_results)
        display_image(visualization, "分割方法比較", figsize=(15, 10))

        # 顯示最佳方法的詳細結果
        best_method, best_result = sorted_results[0]
        print(f"\n🏆 最佳分割方法: {best_method}")
        print(f"📈 詳細分析:")

        if best_result.region_properties:
            print(f"  區域數量: {len(best_result.region_properties)}")
            print(f"  平均區域大小: {np.mean([r['area'] for r in best_result.region_properties]):.1f} 像素")

            # 顯示前5個最大區域
            sorted_regions = sorted(best_result.region_properties,
                                  key=lambda r: r['area'], reverse=True)

            print(f"  前5大區域:")
            for i, region in enumerate(sorted_regions[:5], 1):
                tissue_name = TissueType(region['tissue_type']).name if isinstance(region['tissue_type'], int) else region['tissue_type']
                print(f"    {i}. 面積: {region['area']:6.0f}, "
                      f"強度: {region['mean_intensity']:6.1f}, "
                      f"類型: {tissue_name}")

    else:
        print("❌ 沒有成功的分割結果")

    print(f"\n📋 醫學影像分割功能特色:")
    print(f"• 多種分割算法 (Watershed, K-means, GrabCut)")
    print(f"• 組織類型自動分類")
    print(f"• 量化區域屬性分析")
    print(f"• 分割品質評估")
    print(f"• 可視化比較工具")

    print(f"\n🎯 臨床應用場景 (研究用途):")
    print(f"• 腫瘤區域分割")
    print(f"• 器官體積測量")
    print(f"• 病理組織分析")
    print(f"• 影像引導手術")
    print(f"• 放療計劃制定")

    print(f"\n⚠️ 重要提醒:")
    print(f"本分割系統僅用於教學和研究")
    print(f"實際醫療應用需要:")
    print(f"• FDA/CE認證的醫療軟體")
    print(f"• 專業醫師診斷")
    print(f"• 標準化的影像協議")
    print(f"• 質量控制和驗證")


if __name__ == "__main__":
    demo_medical_segmentation()
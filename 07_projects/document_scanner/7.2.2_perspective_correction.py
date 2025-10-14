#!/usr/bin/env python3
"""
7.2.2 智能文檔掃描器 - 透視校正模組

這個模組實現了高品質的透視變換與幾何校正功能，將傾斜或變形的文檔
圖像校正為正面平整的掃描效果。

功能特色：
- 四點透視變換校正
- 自動透視矩陣計算
- 畸變校正與幾何調整
- 多種插值方法支援
- 輸出尺寸自適應
- 品質評估與驗證
- 後處理優化
- 批量處理支援

作者: OpenCV Computer Vision Toolkit
日期: 2024-10-14
版本: 1.0
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

class PaperSize(Enum):
    """紙張尺寸枚舉"""
    A4 = (210, 297)      # mm
    A3 = (297, 420)      # mm
    A5 = (148, 210)      # mm
    LETTER = (216, 279)  # mm
    LEGAL = (216, 356)   # mm
    CUSTOM = (0, 0)      # 自定義

@dataclass
class PerspectiveTransform:
    """透視變換數據結構"""
    source_corners: List[Tuple[int, int]]
    target_corners: List[Tuple[int, int]]
    transform_matrix: np.ndarray
    inverse_matrix: np.ndarray
    output_size: Tuple[int, int]
    paper_size: PaperSize
    confidence: float

class DocumentPerspectiveCorrector:
    """文檔透視校正器"""

    def __init__(self, config_file: str = None):
        """初始化透視校正器"""
        self.config = self._load_config(config_file)

        # 標準紙張比例
        self.paper_ratios = {
            PaperSize.A4: 210 / 297,      # 0.707
            PaperSize.A3: 297 / 420,      # 0.707
            PaperSize.A5: 148 / 210,      # 0.705
            PaperSize.LETTER: 216 / 279,  # 0.774
            PaperSize.LEGAL: 216 / 356,   # 0.607
        }

        logger.info("文檔透視校正器初始化完成")

    def _load_config(self, config_file: str) -> Dict:
        """載入配置文件"""
        default_config = {
            "correction": {
                "default_paper_size": "A4",        # A4, A3, A5, LETTER, LEGAL, AUTO
                "output_dpi": 300,                  # 輸出解析度 DPI
                "interpolation_method": "cubic",    # linear, cubic, lanczos
                "border_removal": True,             # 是否移除邊框
                "border_threshold": 0.02,           # 邊框檢測閾值
                "aspect_ratio_tolerance": 0.15      # 長寬比容差
            },
            "quality_enhancement": {
                "auto_contrast": True,              # 自動對比度調整
                "noise_reduction": True,            # 降噪處理
                "sharpening": True,                 # 銳化處理
                "gamma_correction": 1.2,            # 伽馬校正值
                "brightness_adjustment": 0,         # 亮度調整 (-100 to 100)
                "contrast_adjustment": 10           # 對比度調整 (-100 to 100)
            },
            "validation": {
                "min_area_ratio": 0.1,              # 最小面積比例
                "max_area_ratio": 0.95,             # 最大面積比例
                "min_corner_distance": 50,          # 最小角點距離
                "validate_corners": True,           # 驗證角點有效性
                "corner_angle_threshold": 30        # 角點角度閾值
            },
            "output": {
                "format": "png",                    # 輸出格式 png, jpg
                "quality": 95,                      # JPEG品質 (1-100)
                "preserve_aspect_ratio": True,      # 保持長寬比
                "padding": 20                       # 邊距像素
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

    def validate_corners(self, corners: List[Tuple[int, int]],
                        image_shape: Tuple[int, int]) -> bool:
        """驗證角點的有效性"""
        if len(corners) != 4:
            logger.warning("角點數量不是4個")
            return False

        height, width = image_shape[:2]
        min_distance = self.config["validation"]["min_corner_distance"]

        # 檢查角點是否在圖像範圍內
        for x, y in corners:
            if x < 0 or x >= width or y < 0 or y >= height:
                logger.warning(f"角點 ({x}, {y}) 超出圖像範圍")
                return False

        # 檢查角點間距離
        for i in range(4):
            for j in range(i + 1, 4):
                dist = math.sqrt((corners[i][0] - corners[j][0])**2 +
                               (corners[i][1] - corners[j][1])**2)
                if dist < min_distance:
                    logger.warning(f"角點 {i} 和 {j} 距離過近: {dist:.1f}")
                    return False

        # 檢查四邊形的面積
        area = self._calculate_quadrilateral_area(corners)
        image_area = width * height
        area_ratio = area / image_area

        min_ratio = self.config["validation"]["min_area_ratio"]
        max_ratio = self.config["validation"]["max_area_ratio"]

        if not (min_ratio <= area_ratio <= max_ratio):
            logger.warning(f"四邊形面積比例不合理: {area_ratio:.3f}")
            return False

        # 檢查角度是否合理（如果啟用）
        if self.config["validation"]["validate_corners"]:
            if not self._validate_corner_angles(corners):
                return False

        return True

    def _calculate_quadrilateral_area(self, corners: List[Tuple[int, int]]) -> float:
        """計算四邊形面積（使用鞋帶公式）"""
        if len(corners) != 4:
            return 0

        # 確保角點按順序排列
        ordered_corners = self._order_corners_for_area(corners)

        # 鞋帶公式
        area = 0
        n = len(ordered_corners)
        for i in range(n):
            j = (i + 1) % n
            area += ordered_corners[i][0] * ordered_corners[j][1]
            area -= ordered_corners[j][0] * ordered_corners[i][1]

        return abs(area) / 2.0

    def _order_corners_for_area(self, corners: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        """為面積計算排序角點（按逆時針順序）"""
        # 計算質心
        cx = sum(x for x, y in corners) / 4
        cy = sum(y for x, y in corners) / 4

        # 按角度排序
        def angle_from_center(corner):
            return math.atan2(corner[1] - cy, corner[0] - cx)

        return sorted(corners, key=angle_from_center)

    def _validate_corner_angles(self, corners: List[Tuple[int, int]]) -> bool:
        """驗證角點角度是否合理"""
        threshold = self.config["validation"]["corner_angle_threshold"]

        # 計算四個內角
        angles = []
        ordered_corners = self._order_corners_for_area(corners)

        for i in range(4):
            p1 = np.array(ordered_corners[(i - 1) % 4])
            p2 = np.array(ordered_corners[i])
            p3 = np.array(ordered_corners[(i + 1) % 4])

            # 計算兩個向量
            v1 = p1 - p2
            v2 = p3 - p2

            # 計算角度
            cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
            cos_angle = np.clip(cos_angle, -1.0, 1.0)
            angle = np.arccos(cos_angle) * 180 / np.pi
            angles.append(angle)

        # 檢查角度是否接近直角
        valid_angles = sum(1 for angle in angles if abs(angle - 90) <= threshold)

        if valid_angles < 3:  # 至少3個角接近直角
            logger.warning(f"角度不合理: {angles}")
            return False

        return True

    def determine_paper_size(self, corners: List[Tuple[int, int]]) -> PaperSize:
        """自動判斷紙張尺寸"""
        if len(corners) != 4:
            return PaperSize.A4  # 預設

        # 計算四邊形的寬度和高度
        width = max(abs(corners[1][0] - corners[0][0]),
                   abs(corners[2][0] - corners[3][0]))
        height = max(abs(corners[3][1] - corners[0][1]),
                    abs(corners[2][1] - corners[1][1]))

        if width == 0 or height == 0:
            return PaperSize.A4

        # 計算長寬比
        aspect_ratio = min(width, height) / max(width, height)
        tolerance = self.config["correction"]["aspect_ratio_tolerance"]

        # 與標準紙張比例比較
        best_match = PaperSize.A4
        min_diff = float('inf')

        for paper_size, standard_ratio in self.paper_ratios.items():
            diff = abs(aspect_ratio - standard_ratio)
            if diff < min_diff and diff <= tolerance:
                min_diff = diff
                best_match = paper_size

        logger.info(f"檢測到紙張類型: {best_match.name} (比例: {aspect_ratio:.3f})")
        return best_match

    def calculate_output_size(self, paper_size: PaperSize,
                            corners: List[Tuple[int, int]] = None) -> Tuple[int, int]:
        """計算輸出圖像尺寸"""
        dpi = self.config["correction"]["output_dpi"]

        if paper_size == PaperSize.CUSTOM and corners:
            # 基於角點計算尺寸
            width = max(abs(corners[1][0] - corners[0][0]),
                       abs(corners[2][0] - corners[3][0]))
            height = max(abs(corners[3][1] - corners[0][1]),
                        abs(corners[2][1] - corners[1][1]))
            return (int(width), int(height))

        elif paper_size in self.paper_ratios:
            # 基於標準紙張尺寸
            paper_width_mm, paper_height_mm = paper_size.value

            # 轉換為像素（DPI轉換）
            width_pixels = int(paper_width_mm * dpi / 25.4)
            height_pixels = int(paper_height_mm * dpi / 25.4)

            return (width_pixels, height_pixels)

        else:
            # 預設 A4 尺寸
            return (int(210 * dpi / 25.4), int(297 * dpi / 25.4))

    def create_perspective_transform(self, corners: List[Tuple[int, int]],
                                   target_size: Tuple[int, int] = None,
                                   paper_size: PaperSize = None) -> PerspectiveTransform:
        """創建透視變換"""
        if not self.validate_corners(corners, (2000, 2000)):  # 假設最大圖像尺寸
            raise ValueError("無效的角點")

        # 自動判斷紙張尺寸
        if paper_size is None:
            paper_size = self.determine_paper_size(corners)

        # 計算輸出尺寸
        if target_size is None:
            target_size = self.calculate_output_size(paper_size, corners)

        # 排序角點：左上、右上、右下、左下
        ordered_corners = self._order_corners(corners)

        # 定義目標角點（標準矩形）
        padding = self.config["output"]["padding"]
        target_corners = [
            (padding, padding),                           # 左上
            (target_size[0] - padding, padding),         # 右上
            (target_size[0] - padding, target_size[1] - padding), # 右下
            (padding, target_size[1] - padding)          # 左下
        ]

        # 計算透視變換矩陣
        src_points = np.array(ordered_corners, dtype=np.float32)
        dst_points = np.array(target_corners, dtype=np.float32)

        transform_matrix = cv2.getPerspectiveTransform(src_points, dst_points)
        inverse_matrix = cv2.getPerspectiveTransform(dst_points, src_points)

        # 計算變換置信度
        confidence = self._calculate_transform_confidence(ordered_corners, target_corners)

        return PerspectiveTransform(
            source_corners=ordered_corners,
            target_corners=target_corners,
            transform_matrix=transform_matrix,
            inverse_matrix=inverse_matrix,
            output_size=target_size,
            paper_size=paper_size,
            confidence=confidence
        )

    def _order_corners(self, corners: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        """排序角點為左上、右上、右下、左下"""
        if len(corners) != 4:
            return corners

        # 轉換為numpy陣列
        pts = np.array(corners, dtype=np.float32)

        # 按x+y的和排序，找出左上和右下
        sum_pts = pts.sum(axis=1)
        top_left = pts[np.argmin(sum_pts)]
        bottom_right = pts[np.argmax(sum_pts)]

        # 按x-y的差排序，找出右上和左下
        diff_pts = np.diff(pts, axis=1).flatten()
        top_right = pts[np.argmin(diff_pts)]
        bottom_left = pts[np.argmax(diff_pts)]

        return [
            tuple(top_left.astype(int)),
            tuple(top_right.astype(int)),
            tuple(bottom_right.astype(int)),
            tuple(bottom_left.astype(int))
        ]

    def _calculate_transform_confidence(self, source_corners: List[Tuple[int, int]],
                                      target_corners: List[Tuple[int, int]]) -> float:
        """計算變換置信度"""
        confidence = 1.0

        # 檢查源角點的矩形度
        rectangularity = self._measure_rectangularity(source_corners)
        confidence *= rectangularity

        # 檢查面積變化合理性
        src_area = self._calculate_quadrilateral_area(source_corners)
        dst_area = self._calculate_quadrilateral_area(target_corners)

        if src_area > 0:
            area_ratio = min(src_area, dst_area) / max(src_area, dst_area)
            confidence *= area_ratio

        return confidence

    def _measure_rectangularity(self, corners: List[Tuple[int, int]]) -> float:
        """測量四邊形的矩形度 (0-1)"""
        if len(corners) != 4:
            return 0.0

        # 計算四個內角
        angles = []
        ordered_corners = self._order_corners_for_area(corners)

        for i in range(4):
            p1 = np.array(ordered_corners[(i - 1) % 4])
            p2 = np.array(ordered_corners[i])
            p3 = np.array(ordered_corners[(i + 1) % 4])

            v1 = p1 - p2
            v2 = p3 - p2

            cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
            cos_angle = np.clip(cos_angle, -1.0, 1.0)
            angle = np.arccos(cos_angle) * 180 / np.pi
            angles.append(angle)

        # 計算角度與90度的偏差
        angle_score = 0.0
        for angle in angles:
            deviation = abs(angle - 90) / 90
            angle_score += max(0, 1 - deviation)

        return angle_score / 4.0  # 平均分數

    def apply_perspective_transform(self, image: np.ndarray,
                                  transform: PerspectiveTransform) -> np.ndarray:
        """應用透視變換"""
        interpolation_map = {
            'linear': cv2.INTER_LINEAR,
            'cubic': cv2.INTER_CUBIC,
            'lanczos': cv2.INTER_LANCZOS4
        }

        interpolation = interpolation_map.get(
            self.config["correction"]["interpolation_method"],
            cv2.INTER_CUBIC
        )

        # 應用透視變換
        corrected = cv2.warpPerspective(
            image,
            transform.transform_matrix,
            transform.output_size,
            flags=interpolation,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(255, 255, 255)  # 白色邊框
        )

        return corrected

    def enhance_quality(self, image: np.ndarray) -> np.ndarray:
        """增強圖像品質"""
        enhanced = image.copy()
        quality_config = self.config["quality_enhancement"]

        # 轉換為LAB色彩空間進行處理
        if len(enhanced.shape) == 3:
            lab = cv2.cvtColor(enhanced, cv2.COLOR_BGR2LAB)
            l_channel = lab[:, :, 0]
        else:
            l_channel = enhanced.copy()

        # 自動對比度調整 (CLAHE)
        if quality_config["auto_contrast"]:
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            l_channel = clahe.apply(l_channel)

        # 噪聲減少
        if quality_config["noise_reduction"]:
            l_channel = cv2.medianBlur(l_channel, 3)

        # 銳化處理
        if quality_config["sharpening"]:
            kernel = np.array([[-1, -1, -1],
                              [-1,  9, -1],
                              [-1, -1, -1]])
            l_channel = cv2.filter2D(l_channel, -1, kernel)

        # 伽馬校正
        gamma = quality_config["gamma_correction"]
        if gamma != 1.0:
            lookup_table = np.array([((i / 255.0) ** (1.0 / gamma)) * 255
                                   for i in np.arange(0, 256)]).astype("uint8")
            l_channel = cv2.LUT(l_channel, lookup_table)

        # 亮度和對比度調整
        brightness = quality_config["brightness_adjustment"]
        contrast = quality_config["contrast_adjustment"]

        if brightness != 0 or contrast != 0:
            alpha = 1.0 + contrast / 100.0  # 對比度
            beta = brightness  # 亮度
            l_channel = cv2.convertScaleAbs(l_channel, alpha=alpha, beta=beta)

        # 重組圖像
        if len(enhanced.shape) == 3:
            lab[:, :, 0] = l_channel
            enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        else:
            enhanced = l_channel

        return enhanced

    def remove_border(self, image: np.ndarray) -> np.ndarray:
        """移除圖像邊框"""
        if not self.config["correction"]["border_removal"]:
            return image

        threshold_ratio = self.config["correction"]["border_threshold"]
        h, w = image.shape[:2]

        # 計算邊框檢測範圍
        border_size = int(min(h, w) * threshold_ratio)

        # 檢測上下邊框
        top_border = 0
        bottom_border = h

        for y in range(border_size):
            if len(image.shape) == 3:
                mean_val = np.mean(image[y, :])
            else:
                mean_val = np.mean(image[y, :])

            if mean_val > 240:  # 白色邊框
                top_border = y + 1
            else:
                break

        for y in range(h - 1, h - border_size - 1, -1):
            if len(image.shape) == 3:
                mean_val = np.mean(image[y, :])
            else:
                mean_val = np.mean(image[y, :])

            if mean_val > 240:
                bottom_border = y
            else:
                break

        # 檢測左右邊框
        left_border = 0
        right_border = w

        for x in range(border_size):
            if len(image.shape) == 3:
                mean_val = np.mean(image[:, x])
            else:
                mean_val = np.mean(image[:, x])

            if mean_val > 240:
                left_border = x + 1
            else:
                break

        for x in range(w - 1, w - border_size - 1, -1):
            if len(image.shape) == 3:
                mean_val = np.mean(image[:, x])
            else:
                mean_val = np.mean(image[:, x])

            if mean_val > 240:
                right_border = x
            else:
                break

        # 裁剪圖像
        if (top_border < bottom_border - 10 and
            left_border < right_border - 10):
            cropped = image[top_border:bottom_border, left_border:right_border]
            logger.info(f"移除邊框: 上{top_border}, 下{h-bottom_border}, "
                       f"左{left_border}, 右{w-right_border}")
            return cropped

        return image

    def correct_document(self, image: np.ndarray,
                        corners: List[Tuple[int, int]],
                        paper_size: PaperSize = None,
                        target_size: Tuple[int, int] = None) -> Dict[str, Any]:
        """完整的文檔校正流程"""
        results = {
            'success': False,
            'corrected_image': None,
            'transform': None,
            'processing_steps': {},
            'quality_metrics': {}
        }

        start_time = time.time()

        try:
            logger.info("開始文檔透視校正...")

            # 步驟1: 創建透視變換
            step_start = time.time()
            transform = self.create_perspective_transform(corners, target_size, paper_size)
            results['processing_steps']['transform_creation'] = (time.time() - step_start) * 1000
            results['transform'] = {
                'paper_size': transform.paper_size.name,
                'output_size': transform.output_size,
                'confidence': transform.confidence
            }

            # 步驟2: 應用透視變換
            step_start = time.time()
            corrected = self.apply_perspective_transform(image, transform)
            results['processing_steps']['perspective_transform'] = (time.time() - step_start) * 1000

            # 步驟3: 移除邊框
            step_start = time.time()
            corrected = self.remove_border(corrected)
            results['processing_steps']['border_removal'] = (time.time() - step_start) * 1000

            # 步驟4: 品質增強
            step_start = time.time()
            corrected = self.enhance_quality(corrected)
            results['processing_steps']['quality_enhancement'] = (time.time() - step_start) * 1000

            results['corrected_image'] = corrected
            results['success'] = True

            # 計算品質指標
            results['quality_metrics'] = self._calculate_quality_metrics(
                image, corrected, transform.confidence
            )

            total_time = (time.time() - start_time) * 1000
            results['total_time'] = total_time

            logger.info(f"文檔校正完成，耗時 {total_time:.1f}ms")
            logger.info(f"輸出尺寸: {corrected.shape[1]}x{corrected.shape[0]}")
            logger.info(f"變換置信度: {transform.confidence:.3f}")

        except Exception as e:
            logger.error(f"文檔校正失敗: {e}")
            results['error'] = str(e)

        return results

    def _calculate_quality_metrics(self, original: np.ndarray,
                                 corrected: np.ndarray, confidence: float) -> Dict[str, float]:
        """計算品質指標"""
        metrics = {
            'transform_confidence': confidence,
            'output_resolution': corrected.shape[1] * corrected.shape[0]
        }

        try:
            # 計算銳度 (Laplacian方差)
            if len(corrected.shape) == 3:
                gray_corrected = cv2.cvtColor(corrected, cv2.COLOR_BGR2GRAY)
            else:
                gray_corrected = corrected

            laplacian = cv2.Laplacian(gray_corrected, cv2.CV_64F)
            metrics['sharpness'] = laplacian.var()

            # 計算對比度 (標準差)
            metrics['contrast'] = np.std(gray_corrected)

            # 計算亮度 (平均值)
            metrics['brightness'] = np.mean(gray_corrected)

        except Exception as e:
            logger.warning(f"品質指標計算失敗: {e}")

        return metrics


def demo_perspective_correction():
    """透視校正演示"""
    print("📐 智能文檔透視校正演示")
    print("=" * 50)

    # 創建校正器
    corrector = DocumentPerspectiveCorrector()

    # 測試圖像
    test_image_path = "../../assets/images/basic/faces01.jpg"

    if not os.path.exists(test_image_path):
        print("❌ 測試圖像不存在，使用模擬數據演示")

        # 創建模擬測試圖像
        demo_image = np.ones((600, 800, 3), dtype=np.uint8) * 240

        # 添加一些內容
        cv2.rectangle(demo_image, (100, 100), (700, 500), (0, 0, 0), 2)
        cv2.putText(demo_image, "Document Content", (200, 300),
                   cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 3)

        # 模擬文檔角點（稍微傾斜）
        corners = [(120, 80), (680, 120), (660, 520), (80, 480)]

    else:
        # 載入真實圖像
        demo_image = load_image(test_image_path)
        demo_image = resize_image(demo_image, max_width=800)

        # 模擬檢測到的角點
        h, w = demo_image.shape[:2]
        corners = [
            (int(w * 0.1), int(h * 0.1)),     # 左上
            (int(w * 0.9), int(h * 0.15)),    # 右上
            (int(w * 0.85), int(h * 0.9)),    # 右下
            (int(w * 0.05), int(h * 0.85))    # 左下
        ]

    print(f"🖼️  測試圖像尺寸: {demo_image.shape}")
    print(f"📍 模擬角點: {corners}")

    try:
        # 執行透視校正
        results = corrector.correct_document(
            demo_image,
            corners,
            paper_size=PaperSize.A4
        )

        if results['success']:
            print("✅ 透視校正成功！")

            # 顯示結果統計
            print(f"📊 處理結果:")
            print(f"  總耗時: {results['total_time']:.1f}ms")
            print(f"  紙張類型: {results['transform']['paper_size']}")
            print(f"  輸出尺寸: {results['transform']['output_size']}")
            print(f"  變換置信度: {results['transform']['confidence']:.3f}")

            # 顯示品質指標
            print(f"📈 品質指標:")
            for metric, value in results['quality_metrics'].items():
                if isinstance(value, float):
                    print(f"  {metric}: {value:.2f}")
                else:
                    print(f"  {metric}: {value}")

            # 顯示處理步驟耗時
            print(f"⏱️  處理步驟:")
            for step, time_ms in results['processing_steps'].items():
                print(f"  {step}: {time_ms:.1f}ms")

            # 繪製角點在原圖上
            original_with_corners = demo_image.copy()
            for i, corner in enumerate(corners):
                cv2.circle(original_with_corners, corner, 10, (0, 255, 0), -1)
                cv2.putText(original_with_corners, str(i+1),
                           (corner[0]+15, corner[1]), cv2.FONT_HERSHEY_SIMPLEX,
                           0.8, (255, 0, 0), 2)

            # 連接角點形成四邊形
            corners_array = np.array(corners, np.int32)
            cv2.polylines(original_with_corners, [corners_array], True, (0, 0, 255), 3)

            # 可視化結果
            images = [original_with_corners, results['corrected_image']]
            titles = [
                f"原始圖像 + 角點\n{demo_image.shape[1]}x{demo_image.shape[0]}",
                f"校正結果\n{results['corrected_image'].shape[1]}x{results['corrected_image'].shape[0]}"
            ]

            display_multiple_images(images, titles, figsize=(15, 8))

        else:
            print("❌ 透視校正失敗")
            if 'error' in results:
                print(f"錯誤信息: {results['error']}")

    except Exception as e:
        print(f"❌ 演示過程中發生錯誤: {e}")


if __name__ == "__main__":
    demo_perspective_correction()
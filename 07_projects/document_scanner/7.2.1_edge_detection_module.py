#!/usr/bin/env python3
"""
7.2.1 智能文檔掃描器 - 邊緣檢測模組

這個模組實現了專為文檔掃描優化的邊緣檢測算法，包括文檔邊界檢測、
角點提取、輪廓分析等功能。

功能特色：
- 多種邊緣檢測方法 (Canny, Sobel, Laplacian)
- 自適應閾值處理
- 文檔邊界智能檢測
- 角點檢測與優化
- 輪廓過濾與排序
- 多邊形近似與簡化
- 透視變換準備
- 質量評估與驗證

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
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
from collections import deque
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

@dataclass
class DocumentContour:
    """文檔輪廓數據結構"""
    contour: np.ndarray
    area: float
    perimeter: float
    aspect_ratio: float
    extent: float
    convexity: float
    corners: List[Tuple[int, int]]
    bounding_rect: Tuple[int, int, int, int]
    confidence: float
    is_rectangular: bool

    def to_dict(self):
        """轉換為字典格式 (排除numpy數組)"""
        return {
            'area': self.area,
            'perimeter': self.perimeter,
            'aspect_ratio': self.aspect_ratio,
            'extent': self.extent,
            'convexity': self.convexity,
            'corners': self.corners,
            'bounding_rect': self.bounding_rect,
            'confidence': self.confidence,
            'is_rectangular': self.is_rectangular
        }

class DocumentEdgeDetector:
    """文檔邊緣檢測器"""

    def __init__(self, config_file: str = None):
        """初始化文檔邊緣檢測器"""
        self.config = self._load_config(config_file)
        self.preprocessing_methods = {
            'gaussian': self._gaussian_preprocessing,
            'bilateral': self._bilateral_preprocessing,
            'median': self._median_preprocessing,
            'combined': self._combined_preprocessing
        }

        self.edge_detection_methods = {
            'canny': self._canny_detection,
            'sobel': self._sobel_detection,
            'laplacian': self._laplacian_detection,
            'adaptive': self._adaptive_detection
        }

        logger.info("文檔邊緣檢測器初始化完成")

    def _load_config(self, config_file: str) -> Dict:
        """載入配置文件"""
        default_config = {
            "preprocessing": {
                "method": "combined",  # gaussian, bilateral, median, combined
                "gaussian_kernel": 5,
                "gaussian_sigma": 1.0,
                "bilateral_d": 9,
                "bilateral_sigma_color": 75,
                "bilateral_sigma_space": 75,
                "median_kernel": 5,
                "resize_width": 800
            },
            "edge_detection": {
                "method": "adaptive",  # canny, sobel, laplacian, adaptive
                "canny_low": 50,
                "canny_high": 150,
                "canny_aperture": 3,
                "sobel_kernel": 3,
                "laplacian_kernel": 3,
                "adaptive_block_size": 11,
                "adaptive_c": 2
            },
            "contour_filtering": {
                "min_area_ratio": 0.01,    # 相對於圖像面積的最小比例
                "max_area_ratio": 0.95,    # 相對於圖像面積的最大比例
                "min_aspect_ratio": 0.2,   # 最小長寬比
                "max_aspect_ratio": 5.0,   # 最大長寬比
                "min_extent": 0.3,         # 最小填充率
                "min_convexity": 0.7,      # 最小凸性
                "min_perimeter": 100       # 最小周長
            },
            "corner_detection": {
                "approximation_epsilon": 0.02,  # Douglas-Peucker近似精度
                "min_corners": 4,                # 最少角點數
                "max_corners": 8,                # 最多角點數
                "corner_threshold": 0.01,        # 角點檢測閾值
                "corner_quality": 0.3,           # 角點質量參數
                "corner_min_distance": 10        # 角點最小距離
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

    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """預處理圖像"""
        method = self.config["preprocessing"]["method"]

        if method in self.preprocessing_methods:
            return self.preprocessing_methods[method](image)
        else:
            logger.warning(f"未知預處理方法: {method}, 使用高斯預處理")
            return self._gaussian_preprocessing(image)

    def _gaussian_preprocessing(self, image: np.ndarray) -> np.ndarray:
        """高斯模糊預處理"""
        kernel_size = self.config["preprocessing"]["gaussian_kernel"]
        sigma = self.config["preprocessing"]["gaussian_sigma"]

        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        blurred = cv2.GaussianBlur(gray, (kernel_size, kernel_size), sigma)
        return blurred

    def _bilateral_preprocessing(self, image: np.ndarray) -> np.ndarray:
        """雙邊濾波預處理"""
        config = self.config["preprocessing"]

        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        filtered = cv2.bilateralFilter(
            gray, config["bilateral_d"],
            config["bilateral_sigma_color"],
            config["bilateral_sigma_space"]
        )
        return filtered

    def _median_preprocessing(self, image: np.ndarray) -> np.ndarray:
        """中值濾波預處理"""
        kernel_size = self.config["preprocessing"]["median_kernel"]

        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        filtered = cv2.medianBlur(gray, kernel_size)
        return filtered

    def _combined_preprocessing(self, image: np.ndarray) -> np.ndarray:
        """組合預處理方法"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        # 首先使用高斯模糊
        gaussian = cv2.GaussianBlur(gray, (5, 5), 1.0)

        # 然後使用雙邊濾波
        bilateral = cv2.bilateralFilter(gaussian, 9, 75, 75)

        return bilateral

    def detect_edges(self, image: np.ndarray) -> np.ndarray:
        """檢測邊緣"""
        method = self.config["edge_detection"]["method"]

        if method in self.edge_detection_methods:
            return self.edge_detection_methods[method](image)
        else:
            logger.warning(f"未知邊緣檢測方法: {method}, 使用Canny")
            return self._canny_detection(image)

    def _canny_detection(self, image: np.ndarray) -> np.ndarray:
        """Canny邊緣檢測"""
        config = self.config["edge_detection"]

        edges = cv2.Canny(
            image,
            config["canny_low"],
            config["canny_high"],
            apertureSize=config["canny_aperture"]
        )

        return edges

    def _sobel_detection(self, image: np.ndarray) -> np.ndarray:
        """Sobel邊緣檢測"""
        kernel_size = self.config["edge_detection"]["sobel_kernel"]

        # 計算x和y方向的梯度
        sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=kernel_size)
        sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=kernel_size)

        # 計算梯度幅值
        sobel_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
        sobel_magnitude = np.uint8(sobel_magnitude)

        # 閾值化
        _, edges = cv2.threshold(sobel_magnitude, 50, 255, cv2.THRESH_BINARY)

        return edges

    def _laplacian_detection(self, image: np.ndarray) -> np.ndarray:
        """Laplacian邊緣檢測"""
        kernel_size = self.config["edge_detection"]["laplacian_kernel"]

        # 計算Laplacian
        laplacian = cv2.Laplacian(image, cv2.CV_64F, ksize=kernel_size)
        laplacian = np.uint8(np.absolute(laplacian))

        # 閾值化
        _, edges = cv2.threshold(laplacian, 50, 255, cv2.THRESH_BINARY)

        return edges

    def _adaptive_detection(self, image: np.ndarray) -> np.ndarray:
        """自適應邊緣檢測"""
        config = self.config["edge_detection"]

        # 首先使用自適應閾值
        adaptive = cv2.adaptiveThreshold(
            image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, config["adaptive_block_size"], config["adaptive_c"]
        )

        # 然後使用Canny邊緣檢測
        edges = cv2.Canny(image, config["canny_low"], config["canny_high"])

        # 結合兩種結果
        combined = cv2.bitwise_or(adaptive, edges)

        return combined

    def find_document_contours(self, edges: np.ndarray, original_shape: Tuple[int, int]) -> List[DocumentContour]:
        """查找文檔輪廓"""
        # 形態學操作改善邊緣連接
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

        # 查找輪廓
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return []

        # 計算圖像面積用於過濾
        image_area = original_shape[0] * original_shape[1]

        document_contours = []

        for contour in contours:
            doc_contour = self._analyze_contour(contour, image_area)
            if doc_contour and self._is_valid_document_contour(doc_contour):
                document_contours.append(doc_contour)

        # 按置信度排序
        document_contours.sort(key=lambda x: x.confidence, reverse=True)

        return document_contours

    def _analyze_contour(self, contour: np.ndarray, image_area: float) -> Optional[DocumentContour]:
        """分析單個輪廓"""
        try:
            # 基本幾何屬性
            area = cv2.contourArea(contour)
            if area < 100:  # 過濾過小的輪廓
                return None

            perimeter = cv2.arcLength(contour, True)
            if perimeter < 50:  # 過濾過小的周長
                return None

            # 邊界矩形
            x, y, w, h = cv2.boundingRect(contour)
            bounding_rect = (x, y, w, h)

            # 幾何特徵
            aspect_ratio = float(w) / h if h > 0 else 0
            extent = area / (w * h) if (w * h) > 0 else 0

            # 凸包和凸性
            hull = cv2.convexHull(contour)
            hull_area = cv2.contourArea(hull)
            convexity = area / hull_area if hull_area > 0 else 0

            # 多邊形近似
            epsilon = self.config["corner_detection"]["approximation_epsilon"] * perimeter
            approx = cv2.approxPolyDP(contour, epsilon, True)

            # 提取角點
            corners = [(int(point[0][0]), int(point[0][1])) for point in approx]

            # 判斷是否為矩形
            is_rectangular = self._is_rectangular_shape(approx, aspect_ratio)

            # 計算置信度
            confidence = self._calculate_confidence(area, image_area, aspect_ratio,
                                                  extent, convexity, len(corners))

            return DocumentContour(
                contour=contour,
                area=area,
                perimeter=perimeter,
                aspect_ratio=aspect_ratio,
                extent=extent,
                convexity=convexity,
                corners=corners,
                bounding_rect=bounding_rect,
                confidence=confidence,
                is_rectangular=is_rectangular
            )

        except Exception as e:
            logger.warning(f"輪廓分析失敗: {e}")
            return None

    def _is_rectangular_shape(self, approx: np.ndarray, aspect_ratio: float) -> bool:
        """判斷是否為矩形狀"""
        # 檢查角點數量
        if len(approx) != 4:
            return False

        # 檢查長寬比是否合理
        if aspect_ratio < 0.2 or aspect_ratio > 5.0:
            return False

        # 檢查角度（應該接近90度）
        angles = []
        for i in range(4):
            p1 = approx[i][0]
            p2 = approx[(i + 1) % 4][0]
            p3 = approx[(i + 2) % 4][0]

            # 計算兩個向量
            v1 = p1 - p2
            v2 = p3 - p2

            # 計算夾角
            cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
            cos_angle = np.clip(cos_angle, -1.0, 1.0)
            angle = np.arccos(cos_angle) * 180 / np.pi
            angles.append(angle)

        # 檢查是否所有角度都接近90度
        angle_threshold = 25  # 容許25度的偏差
        near_right_angles = sum(1 for angle in angles
                               if abs(angle - 90) < angle_threshold)

        return near_right_angles >= 3  # 至少3個角接近直角

    def _calculate_confidence(self, area: float, image_area: float,
                            aspect_ratio: float, extent: float,
                            convexity: float, corner_count: int) -> float:
        """計算置信度分數"""
        confidence = 0.0

        # 面積佔比 (30% 權重)
        area_ratio = area / image_area
        if 0.1 <= area_ratio <= 0.8:
            confidence += 0.3 * (1.0 - abs(area_ratio - 0.4) / 0.4)

        # 長寬比 (25% 權重)
        if 0.5 <= aspect_ratio <= 2.0:
            ideal_ratio = math.sqrt(2)  # A4紙張比例
            ratio_score = 1.0 - abs(aspect_ratio - ideal_ratio) / ideal_ratio
            confidence += 0.25 * ratio_score

        # 填充率 (20% 權重)
        if extent >= 0.7:
            confidence += 0.2 * extent

        # 凸性 (15% 權重)
        if convexity >= 0.8:
            confidence += 0.15 * convexity

        # 角點數量 (10% 權重)
        if corner_count == 4:
            confidence += 0.1
        elif corner_count in [3, 5, 6]:
            confidence += 0.05

        return min(1.0, confidence)

    def _is_valid_document_contour(self, doc_contour: DocumentContour) -> bool:
        """檢查是否為有效的文檔輪廓"""
        config = self.config["contour_filtering"]

        # 檢查各項條件
        conditions = [
            doc_contour.aspect_ratio >= config["min_aspect_ratio"],
            doc_contour.aspect_ratio <= config["max_aspect_ratio"],
            doc_contour.extent >= config["min_extent"],
            doc_contour.convexity >= config["min_convexity"],
            doc_contour.perimeter >= config["min_perimeter"],
            len(doc_contour.corners) >= config["min_corners"] if config["min_corners"] else True,
            len(doc_contour.corners) <= config["max_corners"] if config["max_corners"] else True,
        ]

        # 至少滿足大部分條件
        return sum(conditions) >= len(conditions) * 0.7

    def enhance_corners(self, image: np.ndarray, corners: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        """增強角點檢測精度"""
        if len(corners) != 4:
            return corners

        enhanced_corners = []
        corner_config = self.config["corner_detection"]

        # 在每個角點周圍的小區域內進行精確定位
        search_radius = 20

        for corner_x, corner_y in corners:
            # 定義搜索區域
            x1 = max(0, corner_x - search_radius)
            y1 = max(0, corner_y - search_radius)
            x2 = min(image.shape[1], corner_x + search_radius)
            y2 = min(image.shape[0], corner_y + search_radius)

            roi = image[y1:y2, x1:x2]

            if roi.size == 0:
                enhanced_corners.append((corner_x, corner_y))
                continue

            # 使用Harris角點檢測
            try:
                corners_harris = cv2.goodFeaturesToTrack(
                    roi, 1, corner_config["corner_quality"],
                    corner_config["corner_min_distance"]
                )

                if corners_harris is not None and len(corners_harris) > 0:
                    # 轉換回原始圖像座標
                    refined_x = int(corners_harris[0][0][0] + x1)
                    refined_y = int(corners_harris[0][0][1] + y1)
                    enhanced_corners.append((refined_x, refined_y))
                else:
                    enhanced_corners.append((corner_x, corner_y))

            except Exception as e:
                logger.warning(f"角點增強失敗: {e}")
                enhanced_corners.append((corner_x, corner_y))

        return enhanced_corners

    def order_corners(self, corners: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        """按照左上、右上、右下、左下的順序排列角點"""
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

        ordered_corners = [
            tuple(top_left.astype(int)),
            tuple(top_right.astype(int)),
            tuple(bottom_right.astype(int)),
            tuple(bottom_left.astype(int))
        ]

        return ordered_corners

    def visualize_results(self, original_image: np.ndarray,
                         preprocessed: np.ndarray, edges: np.ndarray,
                         document_contours: List[DocumentContour]) -> np.ndarray:
        """可視化檢測結果"""
        # 創建結果圖像
        result = original_image.copy()

        # 繪製所有檢測到的輪廓
        for i, doc_contour in enumerate(document_contours[:3]):  # 最多顯示3個
            color = [(0, 255, 0), (255, 0, 0), (0, 0, 255)][i]

            # 繪製輪廓
            cv2.drawContours(result, [doc_contour.contour], -1, color, 2)

            # 繪製角點
            for corner in doc_contour.corners:
                cv2.circle(result, corner, 8, color, -1)
                cv2.circle(result, corner, 12, (255, 255, 255), 2)

            # 添加標籤
            x, y, w, h = doc_contour.bounding_rect
            label = f"Doc{i+1}: {doc_contour.confidence:.2f}"
            cv2.putText(result, label, (x, y-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        return result

    def process_document(self, image: np.ndarray, save_intermediate: bool = False) -> Dict[str, Any]:
        """處理文檔圖像的完整流程"""
        results = {
            'original_shape': image.shape,
            'processing_steps': {},
            'document_contours': [],
            'best_document': None,
            'success': False
        }

        start_time = time.time()

        # 步驟1: 預處理
        logger.info("開始文檔邊緣檢測處理...")

        step_start = time.time()
        preprocessed = self.preprocess_image(image)
        results['processing_steps']['preprocessing'] = (time.time() - step_start) * 1000

        # 步驟2: 邊緣檢測
        step_start = time.time()
        edges = self.detect_edges(preprocessed)
        results['processing_steps']['edge_detection'] = (time.time() - step_start) * 1000

        # 步驟3: 查找文檔輪廓
        step_start = time.time()
        document_contours = self.find_document_contours(edges, image.shape[:2])
        results['processing_steps']['contour_analysis'] = (time.time() - step_start) * 1000

        # 儲存中間結果
        if save_intermediate:
            results['intermediate'] = {
                'preprocessed': preprocessed,
                'edges': edges
            }

        # 步驟4: 角點優化
        if document_contours:
            step_start = time.time()
            best_contour = document_contours[0]

            if len(best_contour.corners) == 4:
                # 增強角點精度
                enhanced_corners = self.enhance_corners(preprocessed, best_contour.corners)
                # 排序角點
                ordered_corners = self.order_corners(enhanced_corners)

                # 更新最佳文檔
                best_contour.corners = ordered_corners
                results['best_document'] = best_contour.to_dict()
                results['success'] = True

            results['processing_steps']['corner_optimization'] = (time.time() - step_start) * 1000

        # 轉換輪廓為可序列化格式
        results['document_contours'] = [contour.to_dict() for contour in document_contours]

        # 計算總處理時間
        results['total_time'] = (time.time() - start_time) * 1000

        # 可視化結果
        results['visualization'] = self.visualize_results(
            image, preprocessed, edges, document_contours
        )

        logger.info(f"文檔邊緣檢測完成，耗時 {results['total_time']:.1f}ms")
        logger.info(f"找到 {len(document_contours)} 個可能的文檔輪廓")

        if results['success']:
            logger.info(f"最佳文檔置信度: {document_contours[0].confidence:.3f}")

        return results


def demo_document_edge_detection():
    """文檔邊緣檢測演示"""
    print("📄 智能文檔邊緣檢測演示")
    print("=" * 50)

    # 創建檢測器
    detector = DocumentEdgeDetector()

    # 測試圖像路徑
    test_images = [
        "../../assets/images/basic/faces01.jpg",  # 臨時用其他圖片測試
        "../../assets/images/basic/face03.jpg"
    ]

    for image_path in test_images:
        if os.path.exists(image_path):
            print(f"\n🖼️  處理圖像: {os.path.basename(image_path)}")

            # 載入圖像
            image = load_image(image_path)
            if image is None:
                print("❌ 無法載入圖像")
                continue

            # 調整圖像大小
            original_width = image.shape[1]
            target_width = detector.config["preprocessing"]["resize_width"]

            if original_width > target_width:
                image = resize_image(image, max_width=target_width)
                print(f"🔄 圖像已調整大小到 {image.shape[1]}x{image.shape[0]}")

            # 處理文檔
            results = detector.process_document(image, save_intermediate=True)

            # 顯示結果統計
            print(f"📊 處理結果:")
            print(f"  總耗時: {results['total_time']:.1f}ms")
            print(f"  檢測成功: {'✅' if results['success'] else '❌'}")
            print(f"  找到輪廓數: {len(results['document_contours'])}")

            if results['success']:
                best_doc = results['best_document']
                print(f"  最佳文檔置信度: {best_doc['confidence']:.3f}")
                print(f"  角點數量: {len(best_doc['corners'])}")
                print(f"  是否矩形: {'是' if best_doc['is_rectangular'] else '否'}")

            # 顯示處理步驟時間
            print(f"📈 處理步驟耗時:")
            for step, time_ms in results['processing_steps'].items():
                print(f"  {step}: {time_ms:.1f}ms")

            # 可視化結果
            if 'intermediate' in results:
                images = [
                    image,
                    results['intermediate']['preprocessed'],
                    results['intermediate']['edges'],
                    results['visualization']
                ]
                titles = [
                    "原始圖像",
                    "預處理結果",
                    "邊緣檢測",
                    f"檢測結果 ({len(results['document_contours'])} 個輪廓)"
                ]

                display_multiple_images(images, titles, figsize=(16, 12))

            break
    else:
        print("❌ 未找到測試圖像")


if __name__ == "__main__":
    demo_document_edge_detection()
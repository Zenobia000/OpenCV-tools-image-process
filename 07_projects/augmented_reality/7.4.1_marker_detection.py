#!/usr/bin/env python3
"""
7.4.1 擴增實境系統 - 標記檢測模組

這個模組實現了擴增實境中的標記檢測與追蹤功能，支援ArUco標記、
自定義標記、以及基於特徵的標記檢測。

功能特色：
- ArUco標記檢測與識別
- 自定義標記模板匹配
- 標記姿態估計 (位置與旋轉)
- 3D座標系統建立
- 實時追蹤與穩定
- 多標記同時檢測
- 相機標定整合
- 虛擬物體投影準備

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

class MarkerType(Enum):
    """標記類型枚舉"""
    ARUCO = "ArUco"                    # ArUco標記
    CUSTOM_TEMPLATE = "CustomTemplate" # 自定義模板
    QR_CODE = "QRCode"                 # QR碼
    FEATURE_BASED = "FeatureBased"     # 基於特徵的標記

@dataclass
class MarkerInfo:
    """標記信息數據結構"""
    marker_id: int
    marker_type: MarkerType
    corners: np.ndarray              # 4個角點座標
    center: Tuple[float, float]      # 中心點
    rotation_vector: np.ndarray      # 旋轉向量 (rvec)
    translation_vector: np.ndarray   # 平移向量 (tvec)
    pose_matrix: np.ndarray          # 4x4姿態矩陣
    confidence: float                # 檢測置信度
    size: float                      # 標記大小 (像素)

@dataclass
class CameraParameters:
    """相機參數"""
    camera_matrix: np.ndarray        # 內參矩陣
    distortion_coeffs: np.ndarray    # 畸變係數
    image_size: Tuple[int, int]      # 圖像尺寸
    is_calibrated: bool = False

class ARMarkerDetector:
    """擴增實境標記檢測器"""

    def __init__(self, config_file: str = None):
        """初始化AR標記檢測器"""
        self.config = self._load_config(config_file)
        self.camera_params = None
        self.aruco_dict = None
        self.aruco_params = None
        self.custom_templates = {}
        self.detection_history = {}  # 用於追蹤穩定性

        # 初始化ArUco檢測器
        self._initialize_aruco_detector()

        # 載入自定義模板
        self._load_custom_templates()

        logger.info("AR標記檢測器初始化完成")

    def _load_config(self, config_file: str) -> Dict:
        """載入配置文件"""
        default_config = {
            "aruco": {
                "dictionary": "DICT_6X6_250",      # ArUco字典類型
                "marker_size": 0.1,                # 標記實際大小 (米)
                "adaptive_thresh_win_size": 7,      # 自適應閾值窗口大小
                "adaptive_thresh_constant": 7,      # 自適應閾值常數
                "min_marker_perimeter": 80,         # 最小標記周長
                "max_marker_perimeter": 4000,       # 最大標記周長
                "accuracy_rate": 0.6,               # 準確率閾值
                "corner_refinement": True,          # 角點精細化
                "error_correction_rate": 0.6        # 錯誤校正率
            },
            "custom_template": {
                "template_dir": "templates",        # 模板目錄
                "match_threshold": 0.8,             # 匹配閾值
                "scale_range": [0.5, 2.0],          # 縮放範圍
                "rotation_range": [-30, 30]         # 旋轉角度範圍
            },
            "tracking": {
                "enable_tracking": True,            # 啟用追蹤
                "tracking_smoothing": 0.7,          # 追蹤平滑係數
                "max_tracking_frames": 10,          # 最大追蹤幀數
                "position_threshold": 50,           # 位置變化閾值
                "angle_threshold": 30               # 角度變化閾值
            },
            "pose_estimation": {
                "enable_pose": True,                # 啟用姿態估計
                "coordinate_system": "opencv",      # 座標系統 opencv/opengl
                "axis_length": 0.05,                # 座標軸長度
                "use_pnp_ransac": True              # 使用RANSAC PnP
            },
            "camera": {
                "auto_calibration": False,          # 自動標定
                "calibration_file": "camera_calibration.json",
                "default_focal_length": 800,       # 預設焦距
                "default_center": [320, 240]       # 預設主點
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

    def _initialize_aruco_detector(self):
        """初始化ArUco檢測器"""
        try:
            # 獲取ArUco字典
            dict_name = self.config["aruco"]["dictionary"]
            if hasattr(cv2.aruco, dict_name):
                self.aruco_dict = cv2.aruco.Dictionary_get(getattr(cv2.aruco, dict_name))
            else:
                # 嘗試新版本OpenCV的方法
                self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)

            # 創建檢測器參數
            if hasattr(cv2.aruco, 'DetectorParameters_create'):
                self.aruco_params = cv2.aruco.DetectorParameters_create()
            else:
                # 新版本OpenCV
                self.aruco_params = cv2.aruco.DetectorParameters()

            # 設置參數
            config = self.config["aruco"]
            self.aruco_params.adaptiveThreshWinSizeMin = config["adaptive_thresh_win_size"]
            self.aruco_params.adaptiveThreshWinSizeMax = config["adaptive_thresh_win_size"] * 3
            self.aruco_params.adaptiveThreshConstant = config["adaptive_thresh_constant"]
            self.aruco_params.minMarkerPerimeterRate = config["min_marker_perimeter"] / 1000.0
            self.aruco_params.maxMarkerPerimeterRate = config["max_marker_perimeter"] / 1000.0

            if config["corner_refinement"]:
                self.aruco_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX

            logger.info("ArUco檢測器初始化成功")

        except Exception as e:
            logger.error(f"ArUco檢測器初始化失敗: {e}")
            self.aruco_dict = None
            self.aruco_params = None

    def _load_custom_templates(self):
        """載入自定義標記模板"""
        template_dir = self.config["custom_template"]["template_dir"]

        if os.path.exists(template_dir):
            for filename in os.listdir(template_dir):
                if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                    template_path = os.path.join(template_dir, filename)
                    template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)

                    if template is not None:
                        template_id = os.path.splitext(filename)[0]
                        self.custom_templates[template_id] = template
                        logger.info(f"載入自定義模板: {template_id}")
        else:
            logger.info(f"自定義模板目錄不存在: {template_dir}")

    def load_camera_calibration(self, calibration_file: str = None) -> bool:
        """載入相機標定參數"""
        if calibration_file is None:
            calibration_file = self.config["camera"]["calibration_file"]

        try:
            if os.path.exists(calibration_file):
                with open(calibration_file, 'r') as f:
                    calib_data = json.load(f)

                self.camera_params = CameraParameters(
                    camera_matrix=np.array(calib_data["camera_matrix"]),
                    distortion_coeffs=np.array(calib_data["distortion_coeffs"]),
                    image_size=tuple(calib_data["image_size"]),
                    is_calibrated=True
                )

                logger.info("相機標定參數載入成功")
                return True

        except Exception as e:
            logger.warning(f"相機標定參數載入失敗: {e}")

        # 使用預設參數
        config = self.config["camera"]
        self.camera_params = CameraParameters(
            camera_matrix=np.array([[config["default_focal_length"], 0, config["default_center"][0]],
                                   [0, config["default_focal_length"], config["default_center"][1]],
                                   [0, 0, 1]], dtype=np.float32),
            distortion_coeffs=np.zeros(5, dtype=np.float32),
            image_size=(640, 480),
            is_calibrated=False
        )

        logger.info("使用預設相機參數")
        return False

    def detect_aruco_markers(self, image: np.ndarray) -> List[MarkerInfo]:
        """檢測ArUco標記"""
        if self.aruco_dict is None or self.aruco_params is None:
            return []

        try:
            # 轉換為灰階
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image

            # 檢測標記
            if hasattr(cv2.aruco, 'detectMarkers'):
                corners, ids, _ = cv2.aruco.detectMarkers(
                    gray, self.aruco_dict, parameters=self.aruco_params
                )
            else:
                # 新版本OpenCV
                detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.aruco_params)
                corners, ids, _ = detector.detectMarkers(gray)

            markers = []

            if ids is not None and len(ids) > 0:
                for i, marker_id in enumerate(ids.flatten()):
                    corner_points = corners[i][0]

                    # 計算中心點
                    center = np.mean(corner_points, axis=0)

                    # 計算標記大小
                    size = np.linalg.norm(corner_points[1] - corner_points[0])

                    # 姿態估計
                    rvec, tvec, pose_matrix = self._estimate_pose(corner_points)

                    marker_info = MarkerInfo(
                        marker_id=int(marker_id),
                        marker_type=MarkerType.ARUCO,
                        corners=corner_points,
                        center=(float(center[0]), float(center[1])),
                        rotation_vector=rvec,
                        translation_vector=tvec,
                        pose_matrix=pose_matrix,
                        confidence=1.0,  # ArUco檢測通常很可靠
                        size=float(size)
                    )

                    markers.append(marker_info)

            return markers

        except Exception as e:
            logger.error(f"ArUco標記檢測失敗: {e}")
            return []

    def detect_custom_template_markers(self, image: np.ndarray) -> List[MarkerInfo]:
        """檢測自定義模板標記"""
        if not self.custom_templates:
            return []

        markers = []

        try:
            # 轉換為灰階
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image

            config = self.config["custom_template"]
            threshold = config["match_threshold"]

            for template_id, template in self.custom_templates.items():
                # 多尺度模板匹配
                best_match = None
                best_confidence = 0

                scale_range = config["scale_range"]
                for scale in np.arange(scale_range[0], scale_range[1], 0.1):
                    # 縮放模板
                    scaled_template = cv2.resize(
                        template,
                        (int(template.shape[1] * scale), int(template.shape[0] * scale))
                    )

                    if (scaled_template.shape[0] > gray.shape[0] or
                        scaled_template.shape[1] > gray.shape[1]):
                        continue

                    # 模板匹配
                    result = cv2.matchTemplate(gray, scaled_template, cv2.TM_CCOEFF_NORMED)
                    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

                    if max_val > best_confidence:
                        best_confidence = max_val
                        best_match = {
                            'location': max_loc,
                            'scale': scale,
                            'template_size': scaled_template.shape
                        }

                # 如果找到足夠好的匹配
                if best_match and best_confidence >= threshold:
                    x, y = best_match['location']
                    w, h = best_match['template_size']

                    # 構建角點
                    corners = np.array([
                        [x, y],
                        [x + w, y],
                        [x + w, y + h],
                        [x, y + h]
                    ], dtype=np.float32)

                    # 計算中心點
                    center = (x + w/2, y + h/2)

                    # 姿態估計
                    rvec, tvec, pose_matrix = self._estimate_pose(corners)

                    marker_info = MarkerInfo(
                        marker_id=hash(template_id) % 1000,  # 簡單的ID映射
                        marker_type=MarkerType.CUSTOM_TEMPLATE,
                        corners=corners,
                        center=center,
                        rotation_vector=rvec,
                        translation_vector=tvec,
                        pose_matrix=pose_matrix,
                        confidence=float(best_confidence),
                        size=float(max(w, h))
                    )

                    markers.append(marker_info)

        except Exception as e:
            logger.error(f"自定義模板檢測失敗: {e}")

        return markers

    def _estimate_pose(self, corners: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """估計標記姿態"""
        if self.camera_params is None:
            self.load_camera_calibration()

        try:
            # 定義3D物件點（假設標記在XY平面，Z=0）
            marker_size = self.config["aruco"]["marker_size"]
            object_points = np.array([
                [-marker_size/2, -marker_size/2, 0],
                [ marker_size/2, -marker_size/2, 0],
                [ marker_size/2,  marker_size/2, 0],
                [-marker_size/2,  marker_size/2, 0]
            ], dtype=np.float32)

            # 圖像點
            image_points = corners.astype(np.float32)

            # 使用PnP求解姿態
            if self.config["pose_estimation"]["use_pnp_ransac"]:
                success, rvec, tvec, _ = cv2.solvePnPRansac(
                    object_points, image_points,
                    self.camera_params.camera_matrix,
                    self.camera_params.distortion_coeffs
                )
            else:
                success, rvec, tvec = cv2.solvePnP(
                    object_points, image_points,
                    self.camera_params.camera_matrix,
                    self.camera_params.distortion_coeffs
                )

            if success:
                # 構建4x4姿態矩陣
                rotation_matrix, _ = cv2.Rodrigues(rvec)
                pose_matrix = np.eye(4)
                pose_matrix[:3, :3] = rotation_matrix
                pose_matrix[:3, 3] = tvec.flatten()

                return rvec, tvec, pose_matrix
            else:
                # 返回單位矩陣作為預設值
                return (np.zeros(3), np.zeros(3), np.eye(4))

        except Exception as e:
            logger.warning(f"姿態估計失敗: {e}")
            return (np.zeros(3), np.zeros(3), np.eye(4))

    def apply_tracking_smoothing(self, current_markers: List[MarkerInfo]) -> List[MarkerInfo]:
        """應用追蹤平滑"""
        if not self.config["tracking"]["enable_tracking"]:
            return current_markers

        smoothing_factor = self.config["tracking"]["tracking_smoothing"]
        smoothed_markers = []

        for current_marker in current_markers:
            marker_id = current_marker.marker_id

            if marker_id in self.detection_history:
                # 獲取歷史數據
                history = self.detection_history[marker_id]
                last_marker = history[-1]

                # 計算位置變化
                pos_diff = np.linalg.norm(
                    np.array(current_marker.center) - np.array(last_marker.center)
                )

                # 如果變化不大，應用平滑
                if pos_diff < self.config["tracking"]["position_threshold"]:
                    # 位置平滑
                    smoothed_center = (
                        smoothing_factor * np.array(last_marker.center) +
                        (1 - smoothing_factor) * np.array(current_marker.center)
                    )

                    # 姿態平滑（簡化版）
                    smoothed_rvec = (
                        smoothing_factor * last_marker.rotation_vector +
                        (1 - smoothing_factor) * current_marker.rotation_vector
                    )

                    smoothed_tvec = (
                        smoothing_factor * last_marker.translation_vector +
                        (1 - smoothing_factor) * current_marker.translation_vector
                    )

                    # 更新標記信息
                    current_marker.center = tuple(smoothed_center)
                    current_marker.rotation_vector = smoothed_rvec
                    current_marker.translation_vector = smoothed_tvec

                # 更新歷史記錄
                history.append(current_marker)
                if len(history) > self.config["tracking"]["max_tracking_frames"]:
                    history.pop(0)
            else:
                # 第一次檢測到此標記
                self.detection_history[marker_id] = [current_marker]

            smoothed_markers.append(current_marker)

        return smoothed_markers

    def detect_markers(self, image: np.ndarray) -> List[MarkerInfo]:
        """檢測所有類型的標記"""
        all_markers = []

        # 檢測ArUco標記
        aruco_markers = self.detect_aruco_markers(image)
        all_markers.extend(aruco_markers)

        # 檢測自定義模板標記
        template_markers = self.detect_custom_template_markers(image)
        all_markers.extend(template_markers)

        # 應用追蹤平滑
        smoothed_markers = self.apply_tracking_smoothing(all_markers)

        return smoothed_markers

    def draw_markers(self, image: np.ndarray, markers: List[MarkerInfo]) -> np.ndarray:
        """繪製檢測到的標記"""
        result = image.copy()

        for marker in markers:
            # 繪製標記邊框
            corners = marker.corners.astype(int)

            # 不同類型使用不同顏色
            if marker.marker_type == MarkerType.ARUCO:
                color = (0, 255, 0)  # 綠色
            elif marker.marker_type == MarkerType.CUSTOM_TEMPLATE:
                color = (255, 0, 0)  # 藍色
            else:
                color = (0, 0, 255)  # 紅色

            # 繪製邊框
            cv2.polylines(result, [corners], True, color, 2)

            # 繪製中心點
            center = tuple(map(int, marker.center))
            cv2.circle(result, center, 5, color, -1)

            # 顯示標記ID
            cv2.putText(result, f"ID:{marker.marker_id}",
                       (center[0] + 10, center[1]),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            # 顯示置信度
            if marker.confidence < 1.0:
                cv2.putText(result, f"Conf:{marker.confidence:.2f}",
                           (center[0] + 10, center[1] + 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

            # 繪製3D座標軸（如果啟用姿態估計）
            if (self.config["pose_estimation"]["enable_pose"] and
                self.camera_params is not None):
                self._draw_3d_axis(result, marker)

        return result

    def _draw_3d_axis(self, image: np.ndarray, marker: MarkerInfo):
        """繪製3D座標軸"""
        try:
            axis_length = self.config["pose_estimation"]["axis_length"]

            # 定義座標軸端點
            axis_points = np.array([
                [0, 0, 0],           # 原點
                [axis_length, 0, 0], # X軸
                [0, axis_length, 0], # Y軸
                [0, 0, -axis_length] # Z軸
            ], dtype=np.float32)

            # 投影到圖像平面
            projected_points, _ = cv2.projectPoints(
                axis_points,
                marker.rotation_vector,
                marker.translation_vector,
                self.camera_params.camera_matrix,
                self.camera_params.distortion_coeffs
            )

            # 轉換為整數座標
            points = projected_points.reshape(-1, 2).astype(int)
            origin = tuple(points[0])

            # 繪製座標軸
            # X軸 - 紅色
            cv2.line(image, origin, tuple(points[1]), (0, 0, 255), 3)
            # Y軸 - 綠色
            cv2.line(image, origin, tuple(points[2]), (0, 255, 0), 3)
            # Z軸 - 藍色
            cv2.line(image, origin, tuple(points[3]), (255, 0, 0), 3)

        except Exception as e:
            logger.warning(f"3D座標軸繪製失敗: {e}")


def create_sample_aruco_markers():
    """創建範例ArUco標記"""
    try:
        # 獲取ArUco字典
        aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_250)

        # 創建幾個標記
        marker_ids = [0, 1, 2, 3, 4]
        marker_size = 200

        print("📄 創建範例ArUco標記...")

        for marker_id in marker_ids:
            # 生成標記
            marker_image = cv2.aruco.drawMarker(aruco_dict, marker_id, marker_size)

            # 儲存標記
            filename = f"aruco_marker_{marker_id}.png"
            cv2.imwrite(filename, marker_image)
            print(f"  創建標記: {filename}")

        print("✅ ArUco標記創建完成！")
        print("📋 使用方法:")
        print("  1. 列印這些標記圖片")
        print("  2. 將標記放在攝像頭前")
        print("  3. 運行檢測演示")

    except Exception as e:
        print(f"❌ ArUco標記創建失敗: {e}")


def demo_ar_marker_detection():
    """AR標記檢測演示"""
    print("🎯 擴增實境標記檢測演示")
    print("=" * 50)

    # 創建檢測器
    detector = ARMarkerDetector()

    # 創建範例標記
    create_sample_aruco_markers()

    # 嘗試載入測試圖像
    test_image_path = "../../assets/images/basic/faces01.jpg"

    if os.path.exists(test_image_path):
        print(f"\n🖼️  測試靜態圖像: {os.path.basename(test_image_path)}")

        # 載入圖像
        test_image = load_image(test_image_path)
        test_image = resize_image(test_image, max_width=800)

        # 檢測標記
        markers = detector.detect_markers(test_image)

        print(f"📊 檢測結果: 找到 {len(markers)} 個標記")

        # 繪製結果
        result_image = detector.draw_markers(test_image, markers)

        # 顯示結果
        display_multiple_images(
            [test_image, result_image],
            ["原始圖像", f"檢測結果 ({len(markers)} 標記)"],
            figsize=(12, 6)
        )

    print("\n🎥 攝像頭實時檢測演示")
    print("操作說明:")
    print("  按 'q' 或 ESC 退出")
    print("  按 's' 儲存當前幀")
    print("  按 'c' 創建新的ArUco標記")
    print("  將列印的ArUco標記放在攝像頭前進行檢測")

    # 嘗試開啟攝像頭
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ 無法開啟攝像頭")
        return

    frame_count = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("❌ 無法讀取攝像頭幀")
                break

            frame_count += 1

            # 檢測標記
            start_time = time.time()
            markers = detector.detect_markers(frame)
            detection_time = (time.time() - start_time) * 1000

            # 繪製結果
            result_frame = detector.draw_markers(frame, markers)

            # 添加信息顯示
            info_text = [
                f"幀數: {frame_count}",
                f"檢測時間: {detection_time:.1f}ms",
                f"標記數量: {len(markers)}"
            ]

            for i, text in enumerate(info_text):
                cv2.putText(result_frame, text, (10, 30 + i * 25),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            # 顯示檢測到的標記信息
            y_offset = 120
            for marker in markers:
                info = f"ID:{marker.marker_id} Type:{marker.marker_type.value}"
                cv2.putText(result_frame, info, (10, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                y_offset += 20

            # 顯示結果
            cv2.imshow('AR Marker Detection', result_frame)

            # 按鍵處理
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:  # 'q' 或 ESC
                break
            elif key == ord('s'):  # 儲存截圖
                timestamp = time.strftime('%Y%m%d_%H%M%S')
                filename = f"ar_detection_{timestamp}.jpg"
                cv2.imwrite(filename, result_frame)
                print(f"📸 已儲存截圖: {filename}")
            elif key == ord('c'):  # 創建新標記
                print("🔄 創建新的ArUco標記...")
                create_sample_aruco_markers()

    except KeyboardInterrupt:
        print("\n⚠️ 接收到中斷信號")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("👋 AR標記檢測演示結束")

    # 顯示統計信息
    print(f"\n📊 會話統計:")
    print(f"  總處理幀數: {frame_count}")
    print(f"  標記檢測歷史: {len(detector.detection_history)} 個標記")

    print("\n🎯 下一步可以嘗試:")
    print("• 列印創建的ArUco標記並進行檢測")
    print("• 添加自定義標記模板")
    print("• 整合相機標定以獲得更準確的姿態估計")
    print("• 基於檢測結果添加虛擬物體渲染")


if __name__ == "__main__":
    demo_ar_marker_detection()
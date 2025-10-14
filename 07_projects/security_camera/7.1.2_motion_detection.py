#!/usr/bin/env python3
"""
7.1.2 智能安全監控系統 - 進階動作檢測模組

這個模組實現了進階的動作檢測算法，包括多種背景建模方法、
智能區域監控、行為分析等功能。

功能特色：
- 多種背景減除算法 (MOG2, GMM, KNN)
- 智能區域劃分監控
- 行為模式分析
- 異常行為檢測
- 人員計數與追蹤
- 滯留檢測
- 入侵檢測
- 可視化監控界面

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
import threading
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
from collections import deque, defaultdict
import math

# 添加上級目錄到路徑
sys.path.append('../../utils')
try:
    from image_utils import load_image, resize_image
    from visualization import display_image
    from performance import time_function
except ImportError:
    print("⚠️ 無法導入工具模組，部分功能可能受限")

# 配置日誌
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class MotionObject:
    """動作物體數據結構"""
    id: int
    center: Tuple[int, int]
    bbox: Tuple[int, int, int, int]  # (x, y, w, h)
    area: float
    confidence: float
    track_history: List[Tuple[int, int]]
    first_seen: float
    last_seen: float
    velocity: Tuple[float, float]  # (vx, vy)
    is_person: bool = False

    def age(self) -> float:
        """物體存在時間"""
        return self.last_seen - self.first_seen

    def speed(self) -> float:
        """計算速度大小"""
        return math.sqrt(self.velocity[0]**2 + self.velocity[1]**2)

@dataclass
class MonitoringZone:
    """監控區域數據結構"""
    name: str
    polygon: List[Tuple[int, int]]
    zone_type: str  # 'restricted', 'entrance', 'exit', 'loitering'
    alert_on_entry: bool = True
    alert_on_exit: bool = False
    max_loitering_time: float = 30.0  # 秒

    def contains_point(self, point: Tuple[int, int]) -> bool:
        """檢查點是否在區域內"""
        return cv2.pointPolygonTest(np.array(self.polygon), point, False) >= 0

class AdvancedMotionDetector:
    """進階動作檢測器"""

    def __init__(self, config_file: str = None):
        """初始化進階動作檢測器"""
        self.config = self._load_config(config_file)
        self.background_methods = {}
        self.current_method = "MOG2"
        self.frame_count = 0

        # 物體追蹤
        self.tracked_objects = {}
        self.next_object_id = 1
        self.max_track_distance = 100
        self.max_track_age = 30  # 幀數

        # 監控區域
        self.monitoring_zones = []
        self.zone_events = deque(maxlen=1000)

        # 統計數據
        self.person_count = 0
        self.total_entries = 0
        self.total_exits = 0
        self.loitering_alerts = 0

        # 初始化背景減除器
        self._initialize_background_subtractors()

        # 載入監控區域
        self._load_monitoring_zones()

        logger.info("進階動作檢測器初始化完成")

    def _load_config(self, config_file: str) -> Dict:
        """載入配置文件"""
        default_config = {
            "background_subtraction": {
                "method": "MOG2",  # MOG2, KNN, GMG
                "history": 500,
                "var_threshold": 16,
                "detect_shadows": True,
                "learning_rate": -1
            },
            "motion_filtering": {
                "min_area": 500,
                "max_area": 50000,
                "min_width": 20,
                "min_height": 20,
                "morphology_kernel_size": 3,
                "gaussian_blur_kernel": 5
            },
            "tracking": {
                "max_track_distance": 100,
                "max_track_age": 30,
                "min_track_length": 5
            },
            "zones": {
                "config_file": "monitoring_zones.json",
                "show_zones": True,
                "zone_colors": {
                    "restricted": [0, 0, 255],    # 紅色
                    "entrance": [0, 255, 0],      # 綠色
                    "exit": [255, 0, 0],          # 藍色
                    "loitering": [0, 255, 255]    # 黃色
                }
            },
            "alerts": {
                "loitering_threshold": 30.0,      # 秒
                "intrusion_alert": True,
                "counting_alert": True,
                "speed_threshold": 5.0            # 像素/幀
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

    def _initialize_background_subtractors(self):
        """初始化背景減除器"""
        bg_config = self.config["background_subtraction"]

        # MOG2
        self.background_methods["MOG2"] = cv2.createBackgroundSubtractorMOG2(
            history=bg_config["history"],
            varThreshold=bg_config["var_threshold"],
            detectShadows=bg_config["detect_shadows"]
        )

        # KNN
        self.background_methods["KNN"] = cv2.createBackgroundSubtractorKNN(
            history=bg_config["history"],
            dist2Threshold=400.0,
            detectShadows=bg_config["detect_shadows"]
        )

        logger.info(f"初始化了 {len(self.background_methods)} 種背景減除方法")

    def _load_monitoring_zones(self):
        """載入監控區域配置"""
        zone_file = self.config["zones"]["config_file"]

        # 創建範例監控區域配置
        if not os.path.exists(zone_file):
            self._create_sample_zones_config(zone_file)

        try:
            with open(zone_file, 'r', encoding='utf-8') as f:
                zones_data = json.load(f)

            for zone_data in zones_data:
                zone = MonitoringZone(
                    name=zone_data["name"],
                    polygon=zone_data["polygon"],
                    zone_type=zone_data["zone_type"],
                    alert_on_entry=zone_data.get("alert_on_entry", True),
                    alert_on_exit=zone_data.get("alert_on_exit", False),
                    max_loitering_time=zone_data.get("max_loitering_time", 30.0)
                )
                self.monitoring_zones.append(zone)

            logger.info(f"載入了 {len(self.monitoring_zones)} 個監控區域")

        except Exception as e:
            logger.warning(f"無法載入監控區域配置: {e}")

    def _create_sample_zones_config(self, zone_file: str):
        """創建範例監控區域配置"""
        sample_zones = [
            {
                "name": "入口區域",
                "zone_type": "entrance",
                "polygon": [[100, 100], [300, 100], [300, 200], [100, 200]],
                "alert_on_entry": True,
                "alert_on_exit": False
            },
            {
                "name": "限制區域",
                "zone_type": "restricted",
                "polygon": [[400, 150], [600, 150], [600, 300], [400, 300]],
                "alert_on_entry": True,
                "alert_on_exit": True
            },
            {
                "name": "滯留監控",
                "zone_type": "loitering",
                "polygon": [[200, 300], [400, 300], [400, 400], [200, 400]],
                "max_loitering_time": 15.0,
                "alert_on_entry": False
            }
        ]

        with open(zone_file, 'w', encoding='utf-8') as f:
            json.dump(sample_zones, f, ensure_ascii=False, indent=2)

        logger.info(f"已創建範例監控區域配置: {zone_file}")

    def detect_motion(self, frame: np.ndarray) -> Tuple[np.ndarray, List[Dict]]:
        """檢測動作並返回物體信息"""
        # 獲取當前背景減除器
        bg_subtractor = self.background_methods[self.current_method]

        # 前處理
        processed_frame = self._preprocess_frame(frame)

        # 背景減除
        learning_rate = self.config["background_subtraction"]["learning_rate"]
        fg_mask = bg_subtractor.apply(processed_frame, learningRate=learning_rate)

        # 後處理遮罩
        cleaned_mask = self._postprocess_mask(fg_mask)

        # 查找運動物體
        motion_objects = self._find_motion_objects(cleaned_mask, frame)

        # 更新物體追蹤
        self._update_tracking(motion_objects)

        # 檢查監控區域
        zone_events = self._check_monitoring_zones()

        return cleaned_mask, motion_objects

    def _preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """前處理幀"""
        # 高斯模糊減少噪聲
        kernel_size = self.config["motion_filtering"]["gaussian_blur_kernel"]
        if kernel_size > 1:
            frame = cv2.GaussianBlur(frame, (kernel_size, kernel_size), 0)

        return frame

    def _postprocess_mask(self, mask: np.ndarray) -> np.ndarray:
        """後處理遮罩"""
        # 移除陰影（如果檢測到）
        mask[mask == 127] = 0  # 127是陰影像素值

        # 形態學操作
        kernel_size = self.config["motion_filtering"]["morphology_kernel_size"]
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                         (kernel_size, kernel_size))

        # 開運算去除噪聲
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        # 閉運算填充孔洞
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        return mask

    def _find_motion_objects(self, mask: np.ndarray, frame: np.ndarray) -> List[Dict]:
        """查找運動物體"""
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        motion_config = self.config["motion_filtering"]
        min_area = motion_config["min_area"]
        max_area = motion_config["max_area"]
        min_width = motion_config["min_width"]
        min_height = motion_config["min_height"]

        objects = []

        for contour in contours:
            area = cv2.contourArea(contour)

            if area < min_area or area > max_area:
                continue

            x, y, w, h = cv2.boundingRect(contour)

            if w < min_width or h < min_height:
                continue

            # 計算中心點
            center = (int(x + w/2), int(y + h/2))

            # 計算物體特徵
            aspect_ratio = float(w) / h
            extent = area / (w * h)

            # 估算是否為人
            is_person = self._estimate_person(w, h, area, aspect_ratio)

            obj_info = {
                "center": center,
                "bbox": (x, y, w, h),
                "area": area,
                "confidence": min(1.0, area / min_area),
                "aspect_ratio": aspect_ratio,
                "extent": extent,
                "is_person": is_person
            }

            objects.append(obj_info)

        return objects

    def _estimate_person(self, width: int, height: int, area: float, aspect_ratio: float) -> bool:
        """估算是否為人"""
        # 簡單的人形檢測規則
        person_height_range = (40, 200)
        person_width_range = (20, 100)
        person_aspect_ratio_range = (0.3, 0.8)
        person_area_range = (800, 20000)

        height_ok = person_height_range[0] <= height <= person_height_range[1]
        width_ok = person_width_range[0] <= width <= person_width_range[1]
        ratio_ok = person_aspect_ratio_range[0] <= aspect_ratio <= person_aspect_ratio_range[1]
        area_ok = person_area_range[0] <= area <= person_area_range[1]

        # 如果大部分條件滿足，則認為是人
        conditions_met = sum([height_ok, width_ok, ratio_ok, area_ok])
        return conditions_met >= 3

    def _update_tracking(self, detected_objects: List[Dict]):
        """更新物體追蹤"""
        current_time = time.time()

        # 匹配檢測到的物體與現有軌跡
        matched_objects = {}
        unmatched_detections = detected_objects.copy()

        for obj_id, tracked_obj in self.tracked_objects.items():
            best_match = None
            min_distance = float('inf')

            for i, detection in enumerate(unmatched_detections):
                # 計算距離
                distance = math.sqrt(
                    (detection["center"][0] - tracked_obj.center[0])**2 +
                    (detection["center"][1] - tracked_obj.center[1])**2
                )

                if distance < min_distance and distance < self.max_track_distance:
                    min_distance = distance
                    best_match = i

            if best_match is not None:
                detection = unmatched_detections.pop(best_match)

                # 更新追蹤物體
                old_center = tracked_obj.center
                new_center = detection["center"]

                # 計算速度
                dt = current_time - tracked_obj.last_seen
                if dt > 0:
                    vx = (new_center[0] - old_center[0]) / dt
                    vy = (new_center[1] - old_center[1]) / dt
                    tracked_obj.velocity = (vx, vy)

                # 更新屬性
                tracked_obj.center = new_center
                tracked_obj.bbox = detection["bbox"]
                tracked_obj.area = detection["area"]
                tracked_obj.confidence = detection["confidence"]
                tracked_obj.last_seen = current_time
                tracked_obj.is_person = detection["is_person"]

                # 更新軌跡歷史
                tracked_obj.track_history.append(new_center)
                if len(tracked_obj.track_history) > 50:  # 限制歷史長度
                    tracked_obj.track_history.pop(0)

                matched_objects[obj_id] = tracked_obj

        # 創建新的追蹤物體
        for detection in unmatched_detections:
            new_obj = MotionObject(
                id=self.next_object_id,
                center=detection["center"],
                bbox=detection["bbox"],
                area=detection["area"],
                confidence=detection["confidence"],
                track_history=[detection["center"]],
                first_seen=current_time,
                last_seen=current_time,
                velocity=(0, 0),
                is_person=detection["is_person"]
            )

            matched_objects[self.next_object_id] = new_obj
            self.next_object_id += 1

        # 移除過期的追蹤物體
        self.tracked_objects = {}
        for obj_id, obj in matched_objects.items():
            if (current_time - obj.last_seen) < self.max_track_age:
                self.tracked_objects[obj_id] = obj

        # 更新人員計數
        self.person_count = sum(1 for obj in self.tracked_objects.values() if obj.is_person)

    def _check_monitoring_zones(self) -> List[Dict]:
        """檢查監控區域事件"""
        events = []
        current_time = time.time()

        for zone in self.monitoring_zones:
            for obj_id, obj in self.tracked_objects.items():
                is_in_zone = zone.contains_point(obj.center)

                # 檢查入侵
                if is_in_zone and zone.zone_type == "restricted":
                    if zone.alert_on_entry:
                        event = {
                            "type": "intrusion",
                            "zone": zone.name,
                            "object_id": obj_id,
                            "timestamp": current_time,
                            "is_person": obj.is_person
                        }
                        events.append(event)
                        logger.warning(f"入侵警報: 物體 {obj_id} 進入限制區域 '{zone.name}'")

                # 檢查滯留
                elif is_in_zone and zone.zone_type == "loitering":
                    loiter_time = current_time - obj.first_seen
                    if loiter_time > zone.max_loitering_time:
                        event = {
                            "type": "loitering",
                            "zone": zone.name,
                            "object_id": obj_id,
                            "timestamp": current_time,
                            "loiter_time": loiter_time,
                            "is_person": obj.is_person
                        }
                        events.append(event)
                        logger.warning(f"滯留警報: 物體 {obj_id} 在 '{zone.name}' 滯留 {loiter_time:.1f} 秒")

                # 檢查入口/出口
                elif zone.zone_type in ["entrance", "exit"]:
                    # 這裡需要更複雜的邏輯來檢測穿越
                    pass

        self.zone_events.extend(events)
        return events

    def draw_results(self, frame: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """繪製檢測結果"""
        result_frame = frame.copy()

        # 繪製監控區域
        if self.config["zones"]["show_zones"]:
            for zone in self.monitoring_zones:
                color = self.config["zones"]["zone_colors"].get(
                    zone.zone_type, [255, 255, 255]
                )

                # 繪製區域多邊形
                pts = np.array(zone.polygon, np.int32)
                cv2.polylines(result_frame, [pts], True, color, 2)

                # 添加區域標籤
                label_pos = (zone.polygon[0][0], zone.polygon[0][1] - 10)
                cv2.putText(result_frame, zone.name, label_pos,
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # 繪製追蹤物體
        for obj_id, obj in self.tracked_objects.items():
            x, y, w, h = obj.bbox
            center = obj.center

            # 選擇顏色（人/非人）
            color = (0, 255, 0) if obj.is_person else (255, 0, 0)

            # 繪製邊界框
            cv2.rectangle(result_frame, (x, y), (x+w, y+h), color, 2)

            # 繪製中心點
            cv2.circle(result_frame, center, 3, color, -1)

            # 繪製軌跡
            if len(obj.track_history) > 1:
                pts = np.array(obj.track_history, np.int32)
                cv2.polylines(result_frame, [pts], False, color, 1)

            # 添加標籤
            label = f"ID:{obj_id}"
            if obj.is_person:
                label += " (人)"

            label_pos = (x, y - 10)
            cv2.putText(result_frame, label, label_pos,
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

            # 顯示速度
            speed = obj.speed()
            if speed > 0.1:
                speed_label = f"速度: {speed:.1f}"
                cv2.putText(result_frame, speed_label, (x, y + h + 15),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

        # 添加統計信息
        stats_text = [
            f"背景方法: {self.current_method}",
            f"追蹤物體: {len(self.tracked_objects)}",
            f"人員數量: {self.person_count}",
            f"幀數: {self.frame_count}"
        ]

        for i, text in enumerate(stats_text):
            y_pos = 30 + i * 25
            cv2.putText(result_frame, text, (10, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        return result_frame

    def switch_background_method(self, method: str):
        """切換背景減除方法"""
        if method in self.background_methods:
            self.current_method = method
            logger.info(f"切換到背景減除方法: {method}")
        else:
            logger.warning(f"未知的背景減除方法: {method}")

    def get_statistics(self) -> Dict:
        """獲取統計信息"""
        return {
            "frame_count": self.frame_count,
            "current_method": self.current_method,
            "tracked_objects": len(self.tracked_objects),
            "person_count": self.person_count,
            "total_entries": self.total_entries,
            "total_exits": self.total_exits,
            "zone_events": len(self.zone_events),
            "monitoring_zones": len(self.monitoring_zones)
        }

    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """處理單個幀"""
        self.frame_count += 1

        # 檢測動作
        mask, objects = self.detect_motion(frame)

        # 繪製結果
        result_frame = self.draw_results(frame, mask)

        return result_frame, mask


def demo_motion_detection():
    """動作檢測演示"""
    print("🎬 進階動作檢測演示")
    print("=" * 50)

    # 創建檢測器
    detector = AdvancedMotionDetector()

    # 嘗試開啟攝像頭
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ 無法開啟攝像頭，請檢查攝像頭連接")
        return

    print("✅ 攝像頭已開啟")
    print("\n操作說明:")
    print("  按 'q' 或 ESC 退出")
    print("  按 '1' 切換到 MOG2 背景減除")
    print("  按 '2' 切換到 KNN 背景減除")
    print("  按 's' 顯示統計信息")
    print("  按 'r' 重置背景模型")
    print("\n開始檢測...")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("❌ 無法讀取攝像頭幀")
                break

            # 處理幀
            result_frame, mask = detector.process_frame(frame)

            # 顯示結果
            cv2.imshow('Advanced Motion Detection', result_frame)
            cv2.imshow('Motion Mask', mask)

            # 按鍵處理
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q') or key == 27:  # 'q' 或 ESC
                break
            elif key == ord('1'):
                detector.switch_background_method("MOG2")
            elif key == ord('2'):
                detector.switch_background_method("KNN")
            elif key == ord('s'):
                stats = detector.get_statistics()
                print(f"\n📊 系統統計: {stats}")
            elif key == ord('r'):
                # 重新初始化背景減除器
                detector._initialize_background_subtractors()
                print("🔄 背景模型已重置")

    except KeyboardInterrupt:
        print("\n⚠️ 接收到中斷信號")

    finally:
        cap.release()
        cv2.destroyAllWindows()

        # 顯示最終統計
        final_stats = detector.get_statistics()
        print(f"\n📊 最終統計: {final_stats}")
        print("👋 動作檢測演示結束")


if __name__ == "__main__":
    demo_motion_detection()
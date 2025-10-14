#!/usr/bin/env python3
"""
7.1.1 智能安全監控系統 - 實時檢測模組

這個模組實現了基於OpenCV的智能監控系統，包含人臉檢測、動作偵測、
異常行為分析等功能。適合用於辦公室、家庭或商業場所的安全監控。

功能特色：
- 多種檢測方法 (Haar, DNN, HOG)
- 實時人臉識別與追蹤
- 動作檢測與行為分析
- 自動告警系統
- 錄影與截圖功能
- 可配置的檢測參數
- 性能監控與統計

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
from collections import deque
import argparse

# 添加上級目錄到路徑以使用工具模組
sys.path.append('../../utils')
try:
    from image_utils import load_image, resize_image
    from visualization import display_image
    from performance import time_function
except ImportError:
    print("⚠️ 無法導入工具模組，部分功能可能受限")

# 配置日誌
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('security_monitor.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class DetectionEvent:
    """檢測事件數據結構"""
    timestamp: str
    event_type: str  # 'face', 'motion', 'intrusion'
    confidence: float
    bbox: Tuple[int, int, int, int]  # (x, y, w, h)
    frame_id: int
    additional_data: Dict[str, Any] = None

    def to_dict(self):
        """轉換為字典格式"""
        return asdict(self)


class SecurityCameraDetector:
    """智能安全監控檢測器"""

    def __init__(self, config_file: str = None):
        """
        初始化安全監控檢測器

        Args:
            config_file: 配置文件路徑
        """
        self.config = self._load_config(config_file)
        self.frame_count = 0
        self.detection_history = deque(maxlen=1000)
        self.background_subtractor = None
        self.face_cascade = None
        self.face_dnn = None
        self.is_running = False
        self.alert_active = False
        self.last_motion_time = None

        # 性能監控
        self.fps_counter = deque(maxlen=30)
        self.processing_times = deque(maxlen=100)

        # 初始化檢測器
        self._initialize_detectors()

        logger.info("安全監控系統初始化完成")

    def _load_config(self, config_file: str) -> Dict:
        """載入配置文件"""
        default_config = {
            "detection": {
                "face_detection_method": "haar",  # haar, dnn, hog
                "motion_detection": True,
                "intrusion_detection": True,
                "face_recognition": False,
                "min_face_size": (30, 30),
                "face_confidence_threshold": 0.5,
                "motion_threshold": 25,
                "motion_area_threshold": 500
            },
            "alert": {
                "enable_alerts": True,
                "alert_cooldown": 5,  # 秒
                "save_alert_images": True,
                "alert_image_dir": "alerts",
                "max_alert_images": 100
            },
            "recording": {
                "enable_recording": False,
                "record_on_detection": True,
                "max_recording_minutes": 30,
                "recording_dir": "recordings"
            },
            "display": {
                "show_fps": True,
                "show_detection_boxes": True,
                "show_motion_areas": True,
                "window_size": (800, 600)
            },
            "performance": {
                "max_fps": 30,
                "frame_skip": 1,
                "resize_factor": 1.0
            }
        }

        if config_file and os.path.exists(config_file):
            try:
                with open(config_file, 'r', encoding='utf-8') as f:
                    user_config = json.load(f)
                # 深度合併配置
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

    def _initialize_detectors(self):
        """初始化各種檢測器"""
        # 初始化背景減除器用於動作檢測
        if self.config["detection"]["motion_detection"]:
            self.background_subtractor = cv2.createBackgroundSubtractorMOG2(
                detectShadows=True,
                varThreshold=16,
                history=500
            )
            logger.info("動作檢測器初始化完成")

        # 初始化人臉檢測器
        face_method = self.config["detection"]["face_detection_method"]

        if face_method == "haar":
            try:
                self.face_cascade = cv2.CascadeClassifier(
                    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
                )
                if self.face_cascade.empty():
                    raise Exception("Haar分類器載入失敗")
                logger.info("Haar人臉檢測器初始化完成")
            except Exception as e:
                logger.error(f"Haar人臉檢測器初始化失敗: {e}")

        elif face_method == "dnn":
            try:
                # 嘗試載入DNN模型
                model_path = "../../assets/models/opencv_face_detector_uint8.pb"
                config_path = "../../assets/models/opencv_face_detector.pbtxt"

                if os.path.exists(model_path) and os.path.exists(config_path):
                    self.face_dnn = cv2.dnn.readNetFromTensorflow(model_path, config_path)
                    logger.info("DNN人臉檢測器初始化完成")
                else:
                    logger.warning("DNN模型文件不存在，回退到Haar方法")
                    self.config["detection"]["face_detection_method"] = "haar"
                    self._initialize_detectors()
                    return
            except Exception as e:
                logger.error(f"DNN人臉檢測器初始化失敗: {e}")

        # 創建警報目錄
        alert_dir = self.config["alert"]["alert_image_dir"]
        if not os.path.exists(alert_dir):
            os.makedirs(alert_dir)

        # 創建錄影目錄
        if self.config["recording"]["enable_recording"]:
            recording_dir = self.config["recording"]["recording_dir"]
            if not os.path.exists(recording_dir):
                os.makedirs(recording_dir)

    def detect_faces_haar(self, frame: np.ndarray) -> List[Tuple[int, int, int, int, float]]:
        """使用Haar分類器檢測人臉"""
        if self.face_cascade is None:
            return []

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=self.config["detection"]["min_face_size"]
        )

        # 轉換為統一格式 (x, y, w, h, confidence)
        return [(x, y, w, h, 1.0) for (x, y, w, h) in faces]

    def detect_faces_dnn(self, frame: np.ndarray) -> List[Tuple[int, int, int, int, float]]:
        """使用DNN檢測人臉"""
        if self.face_dnn is None:
            return []

        h, w = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(
            frame, 1.0, (300, 300), [104, 117, 123], False, False
        )

        self.face_dnn.setInput(blob)
        detections = self.face_dnn.forward()

        faces = []
        confidence_threshold = self.config["detection"]["face_confidence_threshold"]

        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]

            if confidence > confidence_threshold:
                x1 = int(detections[0, 0, i, 3] * w)
                y1 = int(detections[0, 0, i, 4] * h)
                x2 = int(detections[0, 0, i, 5] * w)
                y2 = int(detections[0, 0, i, 6] * h)

                faces.append((x1, y1, x2-x1, y2-y1, confidence))

        return faces

    def detect_motion(self, frame: np.ndarray) -> List[Tuple[int, int, int, int, float]]:
        """檢測動作"""
        if self.background_subtractor is None:
            return []

        # 應用背景減除
        fg_mask = self.background_subtractor.apply(frame)

        # 形態學操作去除噪聲
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)

        # 查找輪廓
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        motion_areas = []
        min_area = self.config["detection"]["motion_area_threshold"]

        for contour in contours:
            area = cv2.contourArea(contour)
            if area > min_area:
                x, y, w, h = cv2.boundingRect(contour)
                # 計算動作強度（基於面積）
                confidence = min(1.0, area / (min_area * 5))
                motion_areas.append((x, y, w, h, confidence))

        return motion_areas

    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, List[DetectionEvent]]:
        """處理單個幀"""
        start_time = time.time()

        # 調整幀大小以提高性能
        resize_factor = self.config["performance"]["resize_factor"]
        if resize_factor != 1.0:
            new_width = int(frame.shape[1] * resize_factor)
            new_height = int(frame.shape[0] * resize_factor)
            frame = cv2.resize(frame, (new_width, new_height))

        processed_frame = frame.copy()
        events = []
        current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        # 人臉檢測
        face_method = self.config["detection"]["face_detection_method"]
        if face_method == "haar":
            faces = self.detect_faces_haar(frame)
        elif face_method == "dnn":
            faces = self.detect_faces_dnn(frame)
        else:
            faces = []

        # 處理人臉檢測結果
        for x, y, w, h, conf in faces:
            if self.config["display"]["show_detection_boxes"]:
                cv2.rectangle(processed_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(processed_frame, f'Face: {conf:.2f}', (x, y-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            # 創建檢測事件
            event = DetectionEvent(
                timestamp=current_time,
                event_type='face',
                confidence=conf,
                bbox=(x, y, w, h),
                frame_id=self.frame_count
            )
            events.append(event)

        # 動作檢測
        if self.config["detection"]["motion_detection"]:
            motions = self.detect_motion(frame)

            for x, y, w, h, conf in motions:
                if self.config["display"]["show_motion_areas"]:
                    cv2.rectangle(processed_frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
                    cv2.putText(processed_frame, f'Motion: {conf:.2f}', (x, y-10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

                # 創建動作事件
                event = DetectionEvent(
                    timestamp=current_time,
                    event_type='motion',
                    confidence=conf,
                    bbox=(x, y, w, h),
                    frame_id=self.frame_count
                )
                events.append(event)

            # 更新最後動作時間
            if motions:
                self.last_motion_time = time.time()

        # 添加FPS顯示
        if self.config["display"]["show_fps"]:
            fps = self.get_current_fps()
            cv2.putText(processed_frame, f'FPS: {fps:.1f}', (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # 添加檢測統計
        face_count = len(faces)
        motion_count = len(motions) if self.config["detection"]["motion_detection"] else 0
        cv2.putText(processed_frame, f'Faces: {face_count} | Motion: {motion_count}',
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # 記錄處理時間
        processing_time = (time.time() - start_time) * 1000
        self.processing_times.append(processing_time)

        # 更新檢測歷史
        self.detection_history.extend(events)

        # 檢查是否需要觸發警報
        if events and self.config["alert"]["enable_alerts"]:
            self._handle_alerts(processed_frame, events)

        self.frame_count += 1
        return processed_frame, events

    def _handle_alerts(self, frame: np.ndarray, events: List[DetectionEvent]):
        """處理警報邏輯"""
        current_time = time.time()
        cooldown = self.config["alert"]["alert_cooldown"]

        # 檢查是否在冷卻期內
        if self.alert_active and hasattr(self, 'last_alert_time'):
            if current_time - self.last_alert_time < cooldown:
                return

        # 判斷是否需要觸發警報
        should_alert = False
        alert_reason = []

        for event in events:
            if event.event_type == 'face' and event.confidence > 0.8:
                should_alert = True
                alert_reason.append(f"高信心度人臉檢測 ({event.confidence:.2f})")
            elif event.event_type == 'motion' and event.confidence > 0.7:
                should_alert = True
                alert_reason.append(f"顯著動作檢測 ({event.confidence:.2f})")

        if should_alert:
            self.alert_active = True
            self.last_alert_time = current_time

            alert_message = f"安全警報: {', '.join(alert_reason)}"
            logger.warning(alert_message)

            # 儲存警報圖像
            if self.config["alert"]["save_alert_images"]:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                alert_filename = f"alert_{timestamp}.jpg"
                alert_path = os.path.join(
                    self.config["alert"]["alert_image_dir"],
                    alert_filename
                )
                cv2.imwrite(alert_path, frame)
                logger.info(f"警報圖像已儲存: {alert_path}")

                # 清理舊的警報圖像
                self._cleanup_alert_images()

            # 在畫面上顯示警報
            cv2.rectangle(frame, (10, 10), (400, 100), (0, 0, 255), -1)
            cv2.putText(frame, "SECURITY ALERT!", (20, 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(frame, alert_message[:30], (20, 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    def _cleanup_alert_images(self):
        """清理舊的警報圖像"""
        alert_dir = self.config["alert"]["alert_image_dir"]
        max_images = self.config["alert"]["max_alert_images"]

        try:
            # 獲取所有警報圖像文件
            alert_files = [f for f in os.listdir(alert_dir)
                          if f.startswith('alert_') and f.endswith('.jpg')]

            if len(alert_files) > max_images:
                # 按修改時間排序，刪除最舊的文件
                alert_files.sort(key=lambda x: os.path.getmtime(
                    os.path.join(alert_dir, x)
                ))

                files_to_delete = alert_files[:len(alert_files) - max_images]
                for file_to_delete in files_to_delete:
                    os.remove(os.path.join(alert_dir, file_to_delete))

                logger.info(f"已清理 {len(files_to_delete)} 個舊警報圖像")

        except Exception as e:
            logger.error(f"清理警報圖像時發生錯誤: {e}")

    def get_current_fps(self) -> float:
        """獲取當前FPS"""
        if len(self.fps_counter) < 2:
            return 0.0

        time_diff = self.fps_counter[-1] - self.fps_counter[0]
        if time_diff == 0:
            return 0.0

        return (len(self.fps_counter) - 1) / time_diff

    def get_statistics(self) -> Dict:
        """獲取系統統計信息"""
        current_time = time.time()

        # 計算檢測統計
        face_detections = sum(1 for event in self.detection_history
                             if event.event_type == 'face')
        motion_detections = sum(1 for event in self.detection_history
                               if event.event_type == 'motion')

        # 最後動作時間
        last_motion_ago = (current_time - self.last_motion_time
                          if self.last_motion_time else None)

        return {
            'frame_count': self.frame_count,
            'current_fps': self.get_current_fps(),
            'avg_processing_time': np.mean(self.processing_times) if self.processing_times else 0,
            'total_face_detections': face_detections,
            'total_motion_detections': motion_detections,
            'total_events': len(self.detection_history),
            'last_motion_seconds_ago': last_motion_ago,
            'alert_active': self.alert_active
        }

    def run_camera_monitor(self, camera_id: int = 0, display: bool = True):
        """運行攝像頭監控"""
        logger.info(f"開始攝像頭監控 (ID: {camera_id})")

        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            logger.error(f"無法開啟攝像頭 {camera_id}")
            return

        # 設置攝像頭參數
        window_width, window_height = self.config["display"]["window_size"]
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, window_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, window_height)

        self.is_running = True

        try:
            while self.is_running:
                ret, frame = cap.read()
                if not ret:
                    logger.warning("無法讀取攝像頭幀")
                    break

                # 更新FPS計數器
                self.fps_counter.append(time.time())

                # 處理幀
                processed_frame, events = self.process_frame(frame)

                # 顯示結果
                if display:
                    cv2.imshow('Security Monitor', processed_frame)

                    # 按鍵處理
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q') or key == 27:  # 'q' 或 ESC
                        break
                    elif key == ord('s'):  # 's' 儲存截圖
                        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                        screenshot_path = f"screenshot_{timestamp}.jpg"
                        cv2.imwrite(screenshot_path, processed_frame)
                        logger.info(f"截圖已儲存: {screenshot_path}")
                    elif key == ord(' '):  # 空白鍵顯示統計
                        stats = self.get_statistics()
                        logger.info(f"系統統計: {stats}")

                # FPS限制
                max_fps = self.config["performance"]["max_fps"]
                if max_fps > 0:
                    time.sleep(1.0 / max_fps)

        except KeyboardInterrupt:
            logger.info("接收到中斷信號，停止監控")
        finally:
            self.is_running = False
            cap.release()
            if display:
                cv2.destroyAllWindows()

            # 顯示最終統計
            final_stats = self.get_statistics()
            logger.info("監控會話結束")
            logger.info(f"最終統計: {final_stats}")

    def run_video_monitor(self, video_path: str, output_path: str = None):
        """運行影片文件監控"""
        logger.info(f"開始影片監控: {video_path}")

        if not os.path.exists(video_path):
            logger.error(f"影片文件不存在: {video_path}")
            return

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error(f"無法開啟影片文件: {video_path}")
            return

        # 獲取影片屬性
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        logger.info(f"影片屬性: {frame_width}x{frame_height}, {fps}fps, {total_frames} 幀")

        # 設置輸出影片編寫器
        out = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps,
                                (frame_width, frame_height))

        try:
            frame_idx = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # 處理幀
                processed_frame, events = self.process_frame(frame)

                # 寫入輸出影片
                if out:
                    out.write(processed_frame)

                # 顯示進度
                if frame_idx % 30 == 0:  # 每30幀顯示一次進度
                    progress = (frame_idx / total_frames) * 100
                    logger.info(f"處理進度: {progress:.1f}% ({frame_idx}/{total_frames})")

                frame_idx += 1

        except Exception as e:
            logger.error(f"處理影片時發生錯誤: {e}")
        finally:
            cap.release()
            if out:
                out.release()
                logger.info(f"輸出影片已儲存: {output_path}")

            # 顯示最終統計
            final_stats = self.get_statistics()
            logger.info("影片處理完成")
            logger.info(f"最終統計: {final_stats}")


def create_default_config():
    """創建預設配置文件"""
    config = {
        "detection": {
            "face_detection_method": "haar",
            "motion_detection": True,
            "intrusion_detection": True,
            "face_recognition": False,
            "min_face_size": [30, 30],
            "face_confidence_threshold": 0.5,
            "motion_threshold": 25,
            "motion_area_threshold": 500
        },
        "alert": {
            "enable_alerts": True,
            "alert_cooldown": 5,
            "save_alert_images": True,
            "alert_image_dir": "alerts",
            "max_alert_images": 100
        },
        "recording": {
            "enable_recording": False,
            "record_on_detection": True,
            "max_recording_minutes": 30,
            "recording_dir": "recordings"
        },
        "display": {
            "show_fps": True,
            "show_detection_boxes": True,
            "show_motion_areas": True,
            "window_size": [800, 600]
        },
        "performance": {
            "max_fps": 30,
            "frame_skip": 1,
            "resize_factor": 1.0
        }
    }

    with open('security_config.json', 'w', encoding='utf-8') as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

    print("✅ 預設配置文件已創建: security_config.json")


def main():
    """主函數"""
    parser = argparse.ArgumentParser(description='智能安全監控系統')
    parser.add_argument('--config', '-c', help='配置文件路徑')
    parser.add_argument('--camera', '-cam', type=int, default=0,
                       help='攝像頭ID (預設: 0)')
    parser.add_argument('--video', '-v', help='影片文件路徑')
    parser.add_argument('--output', '-o', help='輸出影片路徑')
    parser.add_argument('--no-display', action='store_true',
                       help='不顯示視窗 (僅後台運行)')
    parser.add_argument('--create-config', action='store_true',
                       help='創建預設配置文件')

    args = parser.parse_args()

    if args.create_config:
        create_default_config()
        return

    # 初始化監控系統
    try:
        detector = SecurityCameraDetector(args.config)

        if args.video:
            # 影片監控模式
            detector.run_video_monitor(args.video, args.output)
        else:
            # 攝像頭監控模式
            detector.run_camera_monitor(args.camera, not args.no_display)

    except Exception as e:
        logger.error(f"系統錯誤: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
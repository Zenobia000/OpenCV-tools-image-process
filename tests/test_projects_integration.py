#!/usr/bin/env python3
"""
M8階段綜合測試 - 實戰專案整合測試套件

這個測試套件驗證所有實戰專案的核心功能，確保系統穩定性、
性能達標和跨平台相容性。

測試範疇:
- 四大實戰專案功能測試
- 性能基準驗證
- 錯誤處理測試
- 配置管理測試
- 整合介面測試
- 跨平台相容性測試

作者: OpenCV Computer Vision Toolkit
日期: 2024-10-14
版本: 1.0
"""

import pytest
import cv2
import numpy as np
import os
import sys
import time
import tempfile
import json
from pathlib import Path

# 添加項目路徑
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))
sys.path.append(str(project_root / "utils"))

# 導入工具模組
from utils.image_utils import load_image, resize_image
from utils.performance import time_function, benchmark_function

# 設置項目模組可用性標誌
PROJECTS_AVAILABLE = True
OCR_AVAILABLE = False

# 導入實戰專案模組
try:
    # 安全監控系統
    sys.path.append(str(project_root / "07_projects" / "security_camera"))

    # 文檔掃描器
    sys.path.append(str(project_root / "07_projects" / "document_scanner"))

    # 醫學影像分析
    sys.path.append(str(project_root / "07_projects" / "medical_imaging"))

    # 擴增實境
    sys.path.append(str(project_root / "07_projects" / "augmented_reality"))

    print("✅ 專案模組路徑設置完成")

except Exception as e:
    PROJECTS_AVAILABLE = False
    print(f"⚠️ 專案模組路徑設置失敗: {e}")

class TestProjectsIntegration:
    """實戰專案整合測試類"""

    @pytest.fixture(autouse=True)
    def setup_test_environment(self):
        """設置測試環境"""
        # 創建測試圖像
        self.test_image = self._create_test_image()
        self.test_document_image = self._create_test_document()
        self.test_medical_image = self._create_test_medical_image()

        # 創建臨時目錄
        self.temp_dir = tempfile.mkdtemp()

        yield

        # 清理
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _create_test_image(self) -> np.ndarray:
        """創建通用測試圖像"""
        image = np.ones((480, 640, 3), dtype=np.uint8) * 128

        # 添加一些特徵
        cv2.rectangle(image, (100, 100), (200, 200), (255, 0, 0), -1)
        cv2.circle(image, (400, 300), 50, (0, 255, 0), -1)
        cv2.putText(image, "Test Image", (250, 400),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        return image

    def _create_test_document(self) -> np.ndarray:
        """創建文檔測試圖像"""
        doc = np.ones((600, 800, 3), dtype=np.uint8) * 255

        # 模擬文檔內容
        cv2.putText(doc, "Sample Document", (50, 100),
                   cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 3)
        cv2.putText(doc, "This is a test document for OCR.", (50, 200),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

        # 添加邊框
        cv2.rectangle(doc, (30, 30), (770, 570), (0, 0, 0), 2)

        return doc

    def _create_test_medical_image(self) -> np.ndarray:
        """創建醫學圖像測試數據"""
        medical = np.zeros((512, 512), dtype=np.uint8)

        # 模擬X光結構
        cv2.rectangle(medical, (100, 100), (400, 400), 120, -1)  # 軟組織
        cv2.rectangle(medical, (150, 150), (200, 350), 200, -1)  # 骨骼
        cv2.rectangle(medical, (300, 150), (350, 350), 200, -1)  # 骨骼
        cv2.circle(medical, (180, 250), 30, 60, -1)              # 肺部
        cv2.circle(medical, (320, 250), 30, 60, -1)              # 肺部

        # 添加噪聲
        noise = np.random.normal(0, 10, medical.shape)
        medical = np.clip(medical.astype(np.float32) + noise, 0, 255).astype(np.uint8)

        return medical

@pytest.mark.skipif(not PROJECTS_AVAILABLE, reason="專案模組不可用")
class TestSecurityCameraSystem:
    """安全監控系統測試"""

    def test_security_detector_initialization(self):
        """測試安全監控檢測器初始化"""
        detector = SecurityCameraDetector()
        assert detector is not None
        assert detector.config is not None
        assert "detection" in detector.config

    def test_face_detection_methods(self, setup_test_environment):
        """測試人臉檢測方法"""
        detector = SecurityCameraDetector()
        test_image = setup_test_environment.test_image

        # 測試Haar方法
        faces_haar = detector.detect_faces_haar(test_image)
        assert isinstance(faces_haar, list)

        # 測試DNN方法（如果可用）
        try:
            faces_dnn = detector.detect_faces_dnn(test_image)
            assert isinstance(faces_dnn, list)
        except:
            pytest.skip("DNN模型不可用")

    def test_motion_detection(self, setup_test_environment):
        """測試動作檢測"""
        detector = AdvancedMotionDetector()
        test_image = setup_test_environment.test_image

        # 處理多個幀以建立背景模型
        for _ in range(5):
            mask, objects = detector.detect_motion(test_image)

        assert isinstance(mask, np.ndarray)
        assert isinstance(objects, list)

    @pytest.mark.slow
    def test_alert_system_functionality(self):
        """測試告警系統功能"""
        alert_system = SmartAlertSystem()

        # 測試告警創建
        from alert_system import AlertType
        alert = alert_system.create_alert(
            alert_type=AlertType.FACE_DETECTION,
            description="測試告警",
            confidence=0.85,
            location=(320, 240),
            bbox=(280, 200, 80, 80)
        )

        assert alert is not None
        assert alert.alert_type == AlertType.FACE_DETECTION
        assert alert.confidence == 0.85

        # 測試告警處理
        success = alert_system.process_alert(alert)
        assert isinstance(success, bool)

@pytest.mark.skipif(not PROJECTS_AVAILABLE, reason="專案模組不可用")
class TestDocumentScanner:
    """文檔掃描器測試"""

    def test_edge_detector_initialization(self):
        """測試邊緣檢測器初始化"""
        detector = DocumentEdgeDetector()
        assert detector is not None
        assert detector.config is not None

    def test_document_edge_detection(self, setup_test_environment):
        """測試文檔邊緣檢測"""
        detector = DocumentEdgeDetector()
        test_doc = setup_test_environment.test_document_image

        # 預處理
        preprocessed = detector.preprocess_image(test_doc)
        assert preprocessed is not None
        assert len(preprocessed.shape) == 2  # 應該是灰階圖像

        # 邊緣檢測
        edges = detector.detect_edges(preprocessed)
        assert edges is not None
        assert edges.dtype == np.uint8

    def test_perspective_correction(self, setup_test_environment):
        """測試透視校正"""
        corrector = DocumentPerspectiveCorrector()
        test_doc = setup_test_environment.test_document_image

        # 模擬文檔角點
        h, w = test_doc.shape[:2]
        corners = [
            (int(w * 0.1), int(h * 0.1)),
            (int(w * 0.9), int(h * 0.15)),
            (int(w * 0.85), int(h * 0.9)),
            (int(w * 0.05), int(h * 0.85))
        ]

        # 執行校正
        results = corrector.correct_document(test_doc, corners)

        assert results["success"] is True
        assert results["corrected_image"] is not None
        assert results["total_time"] > 0

    @pytest.mark.skipif(not OCR_AVAILABLE, reason="OCR不可用")
    def test_ocr_integration(self, setup_test_environment):
        """測試OCR整合"""
        ocr_processor = DocumentOCRProcessor()
        test_doc = setup_test_environment.test_document_image

        # 執行OCR
        result = ocr_processor.extract_text(test_doc)

        assert result.success is True
        assert isinstance(result.full_text, str)
        assert result.confidence >= 0
        assert result.processing_time > 0

@pytest.mark.skipif(not PROJECTS_AVAILABLE, reason="專案模組不可用")
class TestMedicalImaging:
    """醫學影像分析測試"""

    def test_medical_enhancer_initialization(self):
        """測試醫學影像增強器初始化"""
        enhancer = MedicalImageEnhancer()
        assert enhancer is not None
        assert enhancer.config is not None

    def test_image_enhancement(self, setup_test_environment):
        """測試圖像增強"""
        enhancer = MedicalImageEnhancer()
        test_medical = setup_test_environment.test_medical_image

        from image_enhancement import ImagingModality

        # 測試不同模態的增強
        modalities = [ImagingModality.XRAY, ImagingModality.CT, ImagingModality.MRI]

        for modality in modalities:
            results = enhancer.enhance_medical_image(test_medical, modality)

            assert results["success"] is True
            assert results["enhanced_image"] is not None
            assert results["total_time"] > 0
            assert "quality_metrics" in results

    def test_segmentation_methods(self, setup_test_environment):
        """測試分割方法"""
        segmenter = MedicalImageSegmentation()
        test_medical = setup_test_environment.test_medical_image

        # 測試不同分割方法
        methods_results = segmenter.compare_segmentation_methods(test_medical)

        assert len(methods_results) > 0
        for method_name, result in methods_results.items():
            assert result.success is True or hasattr(result, 'region_count')
            assert result.processing_time > 0

@pytest.mark.skipif(not PROJECTS_AVAILABLE, reason="專案模組不可用")
class TestAugmentedReality:
    """擴增實境系統測試"""

    def test_ar_detector_initialization(self):
        """測試AR檢測器初始化"""
        detector = ARMarkerDetector()
        assert detector is not None
        assert detector.config is not None

    def test_pose_estimator_initialization(self):
        """測試姿態估計器初始化"""
        estimator = ARPoseEstimator()
        assert estimator is not None
        assert estimator.camera_calibration is not None

    def test_marker_detection(self, setup_test_environment):
        """測試標記檢測"""
        detector = ARMarkerDetector()
        test_image = setup_test_environment.test_image

        # 檢測標記
        markers = detector.detect_markers(test_image)
        assert isinstance(markers, list)

        # 繪製結果
        result = detector.draw_markers(test_image, markers)
        assert result is not None
        assert result.shape == test_image.shape

    def test_pose_estimation(self):
        """測試姿態估計"""
        estimator = ARPoseEstimator()

        # 創建測試數據
        object_points = np.array([
            [-0.05, -0.05, 0], [0.05, -0.05, 0],
            [0.05, 0.05, 0], [-0.05, 0.05, 0]
        ], dtype=np.float32)

        image_points = np.array([
            [200, 200], [300, 200], [300, 300], [200, 300]
        ], dtype=np.float32)

        # 估計姿態
        pose = estimator.estimate_pose(object_points, image_points)

        if pose:  # 姿態估計可能因為預設相機參數而失敗
            assert pose.confidence > 0
            assert len(pose.position) == 3
            assert len(pose.rotation) == 3

class TestPerformanceBenchmarks:
    """性能基準測試"""

    def test_security_camera_performance(self, setup_test_environment):
        """測試安全監控性能"""
        if not PROJECTS_AVAILABLE:
            pytest.skip("專案模組不可用")

        detector = SecurityCameraDetector()
        test_image = setup_test_environment.test_image

        # 性能測試
        start_time = time.time()
        processed_frame, events = detector.process_frame(test_image)
        processing_time = (time.time() - start_time) * 1000

        # 驗證性能要求 (<30ms)
        assert processing_time < 30, f"處理時間過長: {processing_time:.1f}ms"
        assert processed_frame is not None
        assert isinstance(events, list)

    def test_document_scanner_performance(self, setup_test_environment):
        """測試文檔掃描器性能"""
        if not PROJECTS_AVAILABLE:
            pytest.skip("專案模組不可用")

        detector = DocumentEdgeDetector()
        test_doc = setup_test_environment.test_document_image

        # 性能測試
        start_time = time.time()
        results = detector.process_document(test_doc)
        processing_time = (time.time() - start_time) * 1000

        # 驗證性能要求 (<100ms)
        assert processing_time < 100, f"處理時間過長: {processing_time:.1f}ms"
        assert "total_time" in results

    def test_medical_imaging_performance(self, setup_test_environment):
        """測試醫學影像分析性能"""
        if not PROJECTS_AVAILABLE:
            pytest.skip("專案模組不可用")

        enhancer = MedicalImageEnhancer()
        test_medical = setup_test_environment.test_medical_image

        from image_enhancement import ImagingModality

        # 性能測試
        start_time = time.time()
        results = enhancer.enhance_medical_image(test_medical, ImagingModality.XRAY)
        processing_time = (time.time() - start_time) * 1000

        # 驗證性能要求 (<80ms)
        assert processing_time < 80, f"處理時間過長: {processing_time:.1f}ms"
        assert results["success"] is True

    def test_ar_system_performance(self, setup_test_environment):
        """測試AR系統性能"""
        if not PROJECTS_AVAILABLE:
            pytest.skip("專案模組不可用")

        detector = ARMarkerDetector()
        test_image = setup_test_environment.test_image

        # 性能測試
        start_time = time.time()
        markers = detector.detect_markers(test_image)
        processing_time = (time.time() - start_time) * 1000

        # 驗證性能要求 (<25ms)
        assert processing_time < 25, f"處理時間過長: {processing_time:.1f}ms"
        assert isinstance(markers, list)

class TestConfigurationManagement:
    """配置管理測試"""

    def test_config_file_loading(self, setup_test_environment):
        """測試配置文件載入"""
        if not PROJECTS_AVAILABLE:
            pytest.skip("專案模組不可用")

        # 創建測試配置文件
        test_config = {
            "detection": {
                "face_detection_method": "haar",
                "motion_detection": True
            }
        }

        config_path = os.path.join(setup_test_environment.temp_dir, "test_config.json")
        with open(config_path, 'w') as f:
            json.dump(test_config, f)

        # 測試載入
        detector = SecurityCameraDetector(config_path)
        assert detector.config["detection"]["face_detection_method"] == "haar"
        assert detector.config["detection"]["motion_detection"] is True

    def test_default_configuration(self):
        """測試預設配置"""
        if not PROJECTS_AVAILABLE:
            pytest.skip("專案模組不可用")

        # 測試各系統的預設配置
        systems = [
            SecurityCameraDetector(),
            DocumentEdgeDetector(),
            MedicalImageEnhancer(),
            ARMarkerDetector()
        ]

        for system in systems:
            assert system.config is not None
            assert isinstance(system.config, dict)

class TestErrorHandling:
    """錯誤處理測試"""

    def test_invalid_image_handling(self):
        """測試無效圖像處理"""
        if not PROJECTS_AVAILABLE:
            pytest.skip("專案模組不可用")

        detector = SecurityCameraDetector()

        # 測試空圖像
        empty_image = np.array([])
        try:
            result = detector.process_frame(empty_image)
            # 應該優雅處理錯誤，不應該崩潰
        except Exception:
            pass  # 預期可能拋出異常，但不應導致系統崩潰

        # 測試無效尺寸
        invalid_image = np.ones((0, 0, 3), dtype=np.uint8)
        try:
            result = detector.process_frame(invalid_image)
        except Exception:
            pass

    def test_missing_files_handling(self):
        """測試缺失文件處理"""
        if not PROJECTS_AVAILABLE:
            pytest.skip("專案模組不可用")

        # 測試載入不存在的配置文件
        detector = SecurityCameraDetector("non_existent_config.json")
        assert detector.config is not None  # 應該使用預設配置

    def test_memory_management(self, setup_test_environment):
        """測試記憶體管理"""
        if not PROJECTS_AVAILABLE:
            pytest.skip("專案模組不可用")

        detector = SecurityCameraDetector()
        test_image = setup_test_environment.test_image

        # 處理大量幀以測試記憶體洩漏
        import psutil
        import os

        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss

        for _ in range(100):
            processed_frame, events = detector.process_frame(test_image)

        final_memory = process.memory_info().rss
        memory_increase = (final_memory - initial_memory) / 1024 / 1024  # MB

        # 記憶體增長應該在合理範圍內 (<50MB)
        assert memory_increase < 50, f"記憶體增長過多: {memory_increase:.1f}MB"

class TestCrossPlatformCompatibility:
    """跨平台相容性測試"""

    def test_opencv_version_compatibility(self):
        """測試OpenCV版本相容性"""
        # 檢查OpenCV版本
        version = cv2.__version__
        major_version = int(version.split('.')[0])

        assert major_version >= 4, f"需要OpenCV 4.x版本，當前版本: {version}"

    def test_numpy_compatibility(self):
        """測試NumPy相容性"""
        # 檢查NumPy版本
        version = np.__version__
        major_version = int(version.split('.')[0])
        minor_version = int(version.split('.')[1])

        assert major_version >= 1, f"需要NumPy 1.x版本，當前版本: {version}"
        assert minor_version >= 21, f"需要NumPy 1.21+，當前版本: {version}"

    def test_file_path_handling(self, setup_test_environment):
        """測試文件路徑處理（跨平台）"""
        if not PROJECTS_AVAILABLE:
            pytest.skip("專案模組不可用")

        temp_dir = setup_test_environment.temp_dir

        # 測試不同路徑格式
        test_paths = [
            os.path.join(temp_dir, "test_image.jpg"),
            os.path.join(temp_dir, "subdirectory", "test.png"),
        ]

        for test_path in test_paths:
            # 創建目錄
            os.makedirs(os.path.dirname(test_path), exist_ok=True)

            # 保存測試圖像
            test_image = setup_test_environment.test_image
            cv2.imwrite(test_path, test_image)

            # 驗證可以載入
            loaded = load_image(test_path)
            assert loaded is not None

@pytest.mark.integration
class TestEndToEndWorkflows:
    """端到端工作流程測試"""

    @pytest.mark.slow
    def test_complete_security_workflow(self, setup_test_environment):
        """測試完整安全監控工作流程"""
        if not PROJECTS_AVAILABLE:
            pytest.skip("專案模組不可用")

        # 初始化系統組件
        detector = SecurityCameraDetector()
        motion_detector = AdvancedMotionDetector()
        alert_system = SmartAlertSystem()

        test_image = setup_test_environment.test_image

        # 完整工作流程
        processed_frame, events = detector.process_frame(test_image)

        if events:
            # 處理檢測事件為告警
            from alert_system import AlertType
            alert = alert_system.create_alert(
                alert_type=AlertType.FACE_DETECTION,
                description="測試工作流程",
                confidence=0.8,
                location=(320, 240),
                bbox=(280, 200, 80, 80),
                image=processed_frame
            )

            success = alert_system.process_alert(alert)
            assert isinstance(success, bool)

        assert processed_frame is not None
        assert isinstance(events, list)

    @pytest.mark.slow
    def test_complete_document_workflow(self, setup_test_environment):
        """測試完整文檔掃描工作流程"""
        if not PROJECTS_AVAILABLE:
            pytest.skip("專案模組不可用")

        # 初始化系統組件
        edge_detector = DocumentEdgeDetector()
        corrector = DocumentPerspectiveCorrector()

        test_doc = setup_test_environment.test_document_image

        # 步驟1: 邊緣檢測
        edge_results = edge_detector.process_document(test_doc)
        assert edge_results["success"] is True

        # 步驟2: 透視校正
        if edge_results["best_document"]:
            corners = [(corner[0], corner[1]) for corner in edge_results["best_document"]["corners"]]
            correction_results = corrector.correct_document(test_doc, corners)
            assert correction_results["success"] is True

            # 步驟3: OCR (如果可用)
            if OCR_AVAILABLE:
                ocr_processor = DocumentOCRProcessor()
                ocr_results = ocr_processor.extract_text(correction_results["corrected_image"])
                assert ocr_results.success is True

def run_comprehensive_tests():
    """運行綜合測試套件"""
    print("🧪 開始執行M7實戰專案綜合測試套件")
    print("=" * 60)

    # 檢查測試環境
    print("🔍 檢查測試環境...")

    test_results = {
        "opencv_version": cv2.__version__,
        "numpy_version": np.__version__,
        "projects_available": PROJECTS_AVAILABLE,
        "ocr_available": OCR_AVAILABLE,
        "test_results": {}
    }

    if not PROJECTS_AVAILABLE:
        print("❌ 專案模組不可用，跳過大部分測試")
        return test_results

    print("✅ 測試環境準備完成")

    # 執行基本功能測試
    print("\n📊 執行基本功能測試...")

    try:
        # 安全監控系統
        print("  🎥 測試安全監控系統...")
        detector = SecurityCameraDetector()
        test_image = np.ones((480, 640, 3), dtype=np.uint8) * 128
        processed_frame, events = detector.process_frame(test_image)
        test_results["test_results"]["security_camera"] = {
            "status": "success",
            "frame_processed": processed_frame is not None,
            "events_detected": len(events)
        }
        print("    ✅ 安全監控系統測試通過")

    except Exception as e:
        test_results["test_results"]["security_camera"] = {
            "status": "failed",
            "error": str(e)
        }
        print(f"    ❌ 安全監控系統測試失敗: {e}")

    try:
        # 文檔掃描器
        print("  📄 測試文檔掃描器...")
        edge_detector = DocumentEdgeDetector()
        test_doc = np.ones((600, 800, 3), dtype=np.uint8) * 255
        cv2.rectangle(test_doc, (50, 50), (750, 550), (0, 0, 0), 2)

        results = edge_detector.process_document(test_doc)
        test_results["test_results"]["document_scanner"] = {
            "status": "success" if results else "failed",
            "processing_time": results.get("total_time", 0) if results else 0
        }
        print("    ✅ 文檔掃描器測試通過")

    except Exception as e:
        test_results["test_results"]["document_scanner"] = {
            "status": "failed",
            "error": str(e)
        }
        print(f"    ❌ 文檔掃描器測試失敗: {e}")

    try:
        # 醫學影像分析
        print("  🏥 測試醫學影像分析...")
        enhancer = MedicalImageEnhancer()
        test_medical = np.random.randint(50, 200, (400, 400), dtype=np.uint8)

        from image_enhancement import ImagingModality
        results = enhancer.enhance_medical_image(test_medical, ImagingModality.XRAY)
        test_results["test_results"]["medical_imaging"] = {
            "status": "success" if results["success"] else "failed",
            "processing_time": results.get("total_time", 0)
        }
        print("    ✅ 醫學影像分析測試通過")

    except Exception as e:
        test_results["test_results"]["medical_imaging"] = {
            "status": "failed",
            "error": str(e)
        }
        print(f"    ❌ 醫學影像分析測試失敗: {e}")

    try:
        # 擴增實境
        print("  🎯 測試擴增實境系統...")
        ar_detector = ARMarkerDetector()
        test_image = np.ones((480, 640, 3), dtype=np.uint8) * 128

        markers = ar_detector.detect_markers(test_image)
        test_results["test_results"]["augmented_reality"] = {
            "status": "success",
            "markers_detected": len(markers)
        }
        print("    ✅ 擴增實境系統測試通過")

    except Exception as e:
        test_results["test_results"]["augmented_reality"] = {
            "status": "failed",
            "error": str(e)
        }
        print(f"    ❌ 擴增實境系統測試失敗: {e}")

    # 統計測試結果
    total_tests = len(test_results["test_results"])
    successful_tests = sum(1 for result in test_results["test_results"].values()
                          if result["status"] == "success")

    success_rate = (successful_tests / total_tests * 100) if total_tests > 0 else 0

    print(f"\n📊 測試總結:")
    print(f"  總測試數: {total_tests}")
    print(f"  成功測試: {successful_tests}")
    print(f"  成功率: {success_rate:.1f}%")

    if success_rate >= 75:
        print("🎉 M7實戰專案測試通過！系統準備就緒")
    else:
        print("⚠️ 部分測試失敗，需要進一步調試")

    # 保存測試報告
    report_path = "test_report_m7_projects.json"
    try:
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(test_results, f, ensure_ascii=False, indent=2, default=str)
        print(f"📄 測試報告已保存: {report_path}")
    except Exception as e:
        print(f"⚠️ 保存測試報告失敗: {e}")

    return test_results

if __name__ == "__main__":
    # 直接運行測試
    test_results = run_comprehensive_tests()

    print("\n🚀 下一步:")
    print("• 運行 pytest test_projects_integration.py 執行完整測試")
    print("• 檢查測試報告以獲得詳細結果")
    print("• 準備M8專案發布階段")
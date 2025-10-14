#!/usr/bin/env python3
"""
7.3.1 醫學影像分析系統 - 影像增強模組

這個模組實現了專為醫學影像優化的增強算法，包括對比度增強、
降噪處理、銳化、直方圖均化等功能，適用於X光、CT、MRI等醫學影像。

功能特色：
- 多種對比度增強方法 (CLAHE, HE, Gamma)
- 專業降噪算法 (Bilateral, NLM, Gaussian)
- 自適應銳化處理
- 直方圖分析與均化
- 醫學影像特定優化
- 量化品質評估
- 批量處理支援
- DICOM相容性準備

作者: OpenCV Computer Vision Toolkit
日期: 2024-10-14
版本: 1.0

注意: 本模組僅用於教學和研究目的，不得用於實際醫療診斷。
     真實醫學影像處理需要專業醫療軟體和醫師專業判斷。
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

class ImagingModality(Enum):
    """醫學影像模態枚舉"""
    XRAY = "X-Ray"           # X光影像
    CT = "CT"                # 電腦斷層
    MRI = "MRI"              # 磁共振
    ULTRASOUND = "US"        # 超音波
    MAMMOGRAPHY = "MG"       # 乳房攝影
    GENERAL = "General"      # 一般醫學影像

@dataclass
class EnhancementParameters:
    """影像增強參數"""
    contrast_method: str = "clahe"      # 對比度增強方法
    noise_reduction: str = "bilateral"  # 降噪方法
    sharpening: bool = True            # 是否銳化
    gamma_correction: float = 1.0      # 伽馬校正
    brightness_adjustment: int = 0      # 亮度調整
    contrast_adjustment: int = 0        # 對比度調整

@dataclass
class QualityMetrics:
    """影像品質指標"""
    contrast: float = 0.0              # 對比度
    sharpness: float = 0.0             # 銳度
    noise_level: float = 0.0           # 噪聲水準
    brightness: float = 0.0            # 亮度
    entropy: float = 0.0               # 熵值
    snr: float = 0.0                   # 信噪比

class MedicalImageEnhancer:
    """醫學影像增強器"""

    def __init__(self, config_file: str = None):
        """初始化醫學影像增強器"""
        self.config = self._load_config(config_file)

        # 不同醫學影像模態的預設參數
        self.modality_presets = {
            ImagingModality.XRAY: EnhancementParameters(
                contrast_method="clahe",
                noise_reduction="bilateral",
                sharpening=True,
                gamma_correction=1.2,
                brightness_adjustment=10,
                contrast_adjustment=15
            ),
            ImagingModality.CT: EnhancementParameters(
                contrast_method="histogram_equalization",
                noise_reduction="gaussian",
                sharpening=False,
                gamma_correction=1.0,
                brightness_adjustment=0,
                contrast_adjustment=20
            ),
            ImagingModality.MRI: EnhancementParameters(
                contrast_method="adaptive_histogram",
                noise_reduction="nlm",
                sharpening=True,
                gamma_correction=0.9,
                brightness_adjustment=5,
                contrast_adjustment=10
            ),
            ImagingModality.ULTRASOUND: EnhancementParameters(
                contrast_method="clahe",
                noise_reduction="bilateral",
                sharpening=True,
                gamma_correction=1.1,
                brightness_adjustment=0,
                contrast_adjustment=25
            ),
            ImagingModality.MAMMOGRAPHY: EnhancementParameters(
                contrast_method="clahe",
                noise_reduction="nlm",
                sharpening=True,
                gamma_correction=1.3,
                brightness_adjustment=5,
                contrast_adjustment=20
            ),
            ImagingModality.GENERAL: EnhancementParameters()
        }

        logger.info("醫學影像增強器初始化完成")
        logger.warning("⚠️ 僅用於教學研究，不得用於實際醫療診斷")

    def _load_config(self, config_file: str) -> Dict:
        """載入配置文件"""
        default_config = {
            "enhancement": {
                "default_modality": "GENERAL",
                "preserve_dynamic_range": True,
                "output_bit_depth": 16,  # 醫學影像通常需要更高位深
                "roi_based_enhancement": False
            },
            "contrast_enhancement": {
                "clahe_clip_limit": 3.0,
                "clahe_tile_grid_size": [8, 8],
                "adaptive_histogram_window": 64,
                "gamma_range": [0.5, 2.0]
            },
            "noise_reduction": {
                "bilateral_d": 9,
                "bilateral_sigma_color": 75,
                "bilateral_sigma_space": 75,
                "gaussian_kernel_size": 5,
                "gaussian_sigma": 1.0,
                "nlm_h": 10,
                "nlm_template_window": 7,
                "nlm_search_window": 21
            },
            "sharpening": {
                "unsharp_mask_amount": 1.5,
                "unsharp_mask_radius": 1.0,
                "laplacian_kernel_size": 3,
                "adaptive_sharpening": True
            },
            "quality_assessment": {
                "calculate_metrics": True,
                "reference_based": False,
                "save_metrics": True
            },
            "safety": {
                "preserve_original": True,
                "limit_enhancement_range": True,
                "prevent_over_enhancement": True
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

    def enhance_contrast_clahe(self, image: np.ndarray) -> np.ndarray:
        """CLAHE (對比度限制自適應直方圖均化)"""
        config = self.config["contrast_enhancement"]

        # 轉換為適當的數據類型
        if image.dtype != np.uint8:
            if image.max() <= 1.0:
                image_8bit = (image * 255).astype(np.uint8)
            else:
                image_8bit = cv2.convertScaleAbs(image)
        else:
            image_8bit = image.copy()

        # 創建CLAHE物件
        clahe = cv2.createCLAHE(
            clipLimit=config["clahe_clip_limit"],
            tileGridSize=tuple(config["clahe_tile_grid_size"])
        )

        if len(image_8bit.shape) == 3:
            # 彩色影像：在Lab色彩空間中處理L通道
            lab = cv2.cvtColor(image_8bit, cv2.COLOR_BGR2LAB)
            lab[:, :, 0] = clahe.apply(lab[:, :, 0])
            enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        else:
            # 灰階影像
            enhanced = clahe.apply(image_8bit)

        return enhanced

    def enhance_contrast_histogram_equalization(self, image: np.ndarray) -> np.ndarray:
        """傳統直方圖均化"""
        if image.dtype != np.uint8:
            if image.max() <= 1.0:
                image_8bit = (image * 255).astype(np.uint8)
            else:
                image_8bit = cv2.convertScaleAbs(image)
        else:
            image_8bit = image.copy()

        if len(image_8bit.shape) == 3:
            # 彩色影像
            lab = cv2.cvtColor(image_8bit, cv2.COLOR_BGR2LAB)
            lab[:, :, 0] = cv2.equalizeHist(lab[:, :, 0])
            enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        else:
            # 灰階影像
            enhanced = cv2.equalizeHist(image_8bit)

        return enhanced

    def enhance_contrast_adaptive_histogram(self, image: np.ndarray) -> np.ndarray:
        """自適應直方圖增強"""
        if image.dtype != np.uint8:
            if image.max() <= 1.0:
                image_8bit = (image * 255).astype(np.uint8)
            else:
                image_8bit = cv2.convertScaleAbs(image)
        else:
            image_8bit = image.copy()

        window_size = self.config["contrast_enhancement"]["adaptive_histogram_window"]

        if len(image_8bit.shape) == 3:
            lab = cv2.cvtColor(image_8bit, cv2.COLOR_BGR2LAB)
            l_channel = lab[:, :, 0]
        else:
            l_channel = image_8bit.copy()

        # 實現自適應直方圖增強
        h, w = l_channel.shape
        enhanced = np.zeros_like(l_channel)

        for i in range(0, h, window_size // 2):
            for j in range(0, w, window_size // 2):
                # 定義窗口
                y1 = max(0, i - window_size // 2)
                y2 = min(h, i + window_size // 2)
                x1 = max(0, j - window_size // 2)
                x2 = min(w, j + window_size // 2)

                # 提取窗口區域
                window = l_channel[y1:y2, x1:x2]

                if window.size > 0:
                    # 計算局部直方圖均化
                    equalized = cv2.equalizeHist(window)

                    # 混合原始和均化的結果
                    alpha = 0.7
                    blended = cv2.addWeighted(window, 1 - alpha, equalized, alpha, 0)

                    # 將結果放回對應位置
                    enhanced[y1:y2, x1:x2] = blended

        if len(image_8bit.shape) == 3:
            lab[:, :, 0] = enhanced
            result = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        else:
            result = enhanced

        return result

    def apply_gamma_correction(self, image: np.ndarray, gamma: float) -> np.ndarray:
        """伽馬校正"""
        if gamma <= 0:
            gamma = 1.0

        # 限制伽馬值範圍以防止過度增強
        gamma_range = self.config["contrast_enhancement"]["gamma_range"]
        gamma = np.clip(gamma, gamma_range[0], gamma_range[1])

        # 建立查找表
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255
                         for i in np.arange(0, 256)]).astype("uint8")

        # 應用伽馬校正
        if image.dtype != np.uint8:
            if image.max() <= 1.0:
                image_8bit = (image * 255).astype(np.uint8)
            else:
                image_8bit = cv2.convertScaleAbs(image)
        else:
            image_8bit = image.copy()

        corrected = cv2.LUT(image_8bit, table)
        return corrected

    def reduce_noise_bilateral(self, image: np.ndarray) -> np.ndarray:
        """雙邊濾波降噪"""
        config = self.config["noise_reduction"]

        if image.dtype != np.uint8:
            if image.max() <= 1.0:
                image_8bit = (image * 255).astype(np.uint8)
            else:
                image_8bit = cv2.convertScaleAbs(image)
        else:
            image_8bit = image.copy()

        denoised = cv2.bilateralFilter(
            image_8bit,
            config["bilateral_d"],
            config["bilateral_sigma_color"],
            config["bilateral_sigma_space"]
        )

        return denoised

    def reduce_noise_gaussian(self, image: np.ndarray) -> np.ndarray:
        """高斯濾波降噪"""
        config = self.config["noise_reduction"]

        kernel_size = config["gaussian_kernel_size"]
        sigma = config["gaussian_sigma"]

        denoised = cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)
        return denoised

    def reduce_noise_nlm(self, image: np.ndarray) -> np.ndarray:
        """非局部均值降噪 (NLM)"""
        config = self.config["noise_reduction"]

        if image.dtype != np.uint8:
            if image.max() <= 1.0:
                image_8bit = (image * 255).astype(np.uint8)
            else:
                image_8bit = cv2.convertScaleAbs(image)
        else:
            image_8bit = image.copy()

        if len(image_8bit.shape) == 3:
            denoised = cv2.fastNlMeansDenoisingColored(
                image_8bit,
                None,
                config["nlm_h"],
                config["nlm_h"],
                config["nlm_template_window"],
                config["nlm_search_window"]
            )
        else:
            denoised = cv2.fastNlMeansDenoising(
                image_8bit,
                None,
                config["nlm_h"],
                config["nlm_template_window"],
                config["nlm_search_window"]
            )

        return denoised

    def apply_sharpening(self, image: np.ndarray, adaptive: bool = True) -> np.ndarray:
        """銳化處理"""
        config = self.config["sharpening"]

        if image.dtype != np.uint8:
            if image.max() <= 1.0:
                working_image = (image * 255).astype(np.uint8)
            else:
                working_image = cv2.convertScaleAbs(image)
        else:
            working_image = image.copy()

        if adaptive and config["adaptive_sharpening"]:
            # 自適應銳化：根據局部對比度調整銳化強度
            sharpened = self._adaptive_unsharp_mask(working_image)
        else:
            # 傳統Unsharp Mask銳化
            sharpened = self._unsharp_mask(working_image)

        return sharpened

    def _unsharp_mask(self, image: np.ndarray) -> np.ndarray:
        """Unsharp Mask銳化"""
        config = self.config["sharpening"]

        # 高斯模糊
        radius = config["unsharp_mask_radius"]
        kernel_size = int(2 * math.ceil(2 * radius) + 1)
        blurred = cv2.GaussianBlur(image, (kernel_size, kernel_size), radius)

        # 計算差值
        mask = cv2.subtract(image, blurred)

        # 應用銳化
        amount = config["unsharp_mask_amount"]
        sharpened = cv2.addWeighted(image, 1.0, mask, amount, 0)

        return sharpened

    def _adaptive_unsharp_mask(self, image: np.ndarray) -> np.ndarray:
        """自適應Unsharp Mask銳化"""
        # 計算局部標準差作為對比度指標
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        # 計算局部標準差
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
        local_mean = cv2.morphologyEx(gray.astype(np.float32), cv2.MORPH_CLOSE, kernel)
        local_variance = cv2.morphologyEx((gray.astype(np.float32) - local_mean) ** 2,
                                         cv2.MORPH_CLOSE, kernel)
        local_std = np.sqrt(local_variance)

        # 正規化標準差到0-1範圍作為銳化強度
        std_normalized = cv2.normalize(local_std, None, 0, 1, cv2.NORM_MINMAX, cv2.CV_32F)

        # 應用基礎Unsharp Mask
        base_sharpened = self._unsharp_mask(image)

        # 根據局部對比度混合原始和銳化圖像
        if len(image.shape) == 3:
            std_normalized = cv2.cvtColor(std_normalized, cv2.COLOR_GRAY2BGR)

        # 擴展std_normalized維度以匹配圖像
        std_normalized = np.repeat(std_normalized[:, :, np.newaxis], image.shape[2] if len(image.shape) == 3 else 1, axis=2)

        adaptive_sharpened = cv2.addWeighted(
            image.astype(np.float32), 1.0 - std_normalized,
            base_sharpened.astype(np.float32), std_normalized, 0
        )

        return np.clip(adaptive_sharpened, 0, 255).astype(np.uint8)

    def calculate_quality_metrics(self, image: np.ndarray,
                                reference: np.ndarray = None) -> QualityMetrics:
        """計算影像品質指標"""
        metrics = QualityMetrics()

        try:
            # 轉換為灰階以便計算
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image.copy()

            # 對比度 (標準差)
            metrics.contrast = float(np.std(gray))

            # 銳度 (Laplacian方差)
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            metrics.sharpness = float(laplacian.var())

            # 亮度 (平均值)
            metrics.brightness = float(np.mean(gray))

            # 熵值 (信息量)
            hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
            hist_normalized = hist / hist.sum()
            hist_normalized = hist_normalized[hist_normalized > 0]  # 避免log(0)
            metrics.entropy = float(-np.sum(hist_normalized * np.log2(hist_normalized)))

            # 估計噪聲水準 (使用Laplacian的標準差)
            noise_estimate = cv2.Laplacian(gray, cv2.CV_64F)
            metrics.noise_level = float(np.std(noise_estimate))

            # 信噪比估算
            signal_power = np.mean(gray ** 2)
            noise_power = metrics.noise_level ** 2
            if noise_power > 0:
                metrics.snr = float(10 * np.log10(signal_power / noise_power))
            else:
                metrics.snr = float('inf')

        except Exception as e:
            logger.warning(f"品質指標計算部分失敗: {e}")

        return metrics

    def enhance_medical_image(self, image: np.ndarray,
                            modality: ImagingModality = ImagingModality.GENERAL,
                            custom_params: EnhancementParameters = None) -> Dict[str, Any]:
        """完整的醫學影像增強流程"""
        results = {
            'original_image': image.copy(),
            'enhanced_image': None,
            'processing_steps': {},
            'quality_metrics': {
                'original': None,
                'enhanced': None
            },
            'parameters_used': None,
            'success': False
        }

        start_time = time.time()

        try:
            logger.info(f"開始增強 {modality.value} 影像...")

            # 選擇增強參數
            if custom_params:
                params = custom_params
            else:
                params = self.modality_presets.get(modality, self.modality_presets[ImagingModality.GENERAL])

            results['parameters_used'] = asdict(params)

            # 計算原始影像品質指標
            if self.config["quality_assessment"]["calculate_metrics"]:
                step_start = time.time()
                results['quality_metrics']['original'] = asdict(
                    self.calculate_quality_metrics(image)
                )
                results['processing_steps']['original_metrics'] = (time.time() - step_start) * 1000

            # 開始處理
            enhanced = image.copy()

            # 步驟1: 對比度增強
            step_start = time.time()
            if params.contrast_method == "clahe":
                enhanced = self.enhance_contrast_clahe(enhanced)
            elif params.contrast_method == "histogram_equalization":
                enhanced = self.enhance_contrast_histogram_equalization(enhanced)
            elif params.contrast_method == "adaptive_histogram":
                enhanced = self.enhance_contrast_adaptive_histogram(enhanced)

            results['processing_steps']['contrast_enhancement'] = (time.time() - step_start) * 1000

            # 步驟2: 亮度和對比度調整
            if params.brightness_adjustment != 0 or params.contrast_adjustment != 0:
                step_start = time.time()
                alpha = 1.0 + params.contrast_adjustment / 100.0
                beta = params.brightness_adjustment
                enhanced = cv2.convertScaleAbs(enhanced, alpha=alpha, beta=beta)
                results['processing_steps']['brightness_contrast'] = (time.time() - step_start) * 1000

            # 步驟3: 伽馬校正
            if params.gamma_correction != 1.0:
                step_start = time.time()
                enhanced = self.apply_gamma_correction(enhanced, params.gamma_correction)
                results['processing_steps']['gamma_correction'] = (time.time() - step_start) * 1000

            # 步驟4: 降噪
            step_start = time.time()
            if params.noise_reduction == "bilateral":
                enhanced = self.reduce_noise_bilateral(enhanced)
            elif params.noise_reduction == "gaussian":
                enhanced = self.reduce_noise_gaussian(enhanced)
            elif params.noise_reduction == "nlm":
                enhanced = self.reduce_noise_nlm(enhanced)

            results['processing_steps']['noise_reduction'] = (time.time() - step_start) * 1000

            # 步驟5: 銳化
            if params.sharpening:
                step_start = time.time()
                enhanced = self.apply_sharpening(enhanced)
                results['processing_steps']['sharpening'] = (time.time() - step_start) * 1000

            results['enhanced_image'] = enhanced

            # 計算增強後的品質指標
            if self.config["quality_assessment"]["calculate_metrics"]:
                step_start = time.time()
                results['quality_metrics']['enhanced'] = asdict(
                    self.calculate_quality_metrics(enhanced, image)
                )
                results['processing_steps']['enhanced_metrics'] = (time.time() - step_start) * 1000

            total_time = (time.time() - start_time) * 1000
            results['total_time'] = total_time
            results['success'] = True

            logger.info(f"醫學影像增強完成，耗時 {total_time:.1f}ms")

        except Exception as e:
            logger.error(f"醫學影像增強失敗: {e}")
            results['error'] = str(e)

        return results


def demo_medical_image_enhancement():
    """醫學影像增強演示"""
    print("🏥 醫學影像增強系統演示")
    print("=" * 50)
    print("⚠️ 注意：僅用於教學研究，不得用於實際醫療診斷")

    # 創建增強器
    enhancer = MedicalImageEnhancer()

    # 測試不同的影像模態
    modalities_to_test = [
        ImagingModality.XRAY,
        ImagingModality.CT,
        ImagingModality.ULTRASOUND
    ]

    # 尋找測試圖像
    test_image_path = "../../assets/images/basic/faces01.jpg"

    if not os.path.exists(test_image_path):
        print("❌ 測試圖像不存在，創建模擬醫學影像")

        # 創建模擬X光影像
        demo_image = np.random.randint(50, 200, (512, 512), dtype=np.uint8)

        # 添加一些結構（模擬骨骼）
        cv2.rectangle(demo_image, (100, 100), (400, 400), 255, -1)
        cv2.circle(demo_image, (250, 250), 80, 100, -1)

        # 添加噪聲
        noise = np.random.normal(0, 20, demo_image.shape)
        demo_image = np.clip(demo_image.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    else:
        # 載入真實圖像並轉換為灰階（模擬醫學影像）
        demo_image = load_image(test_image_path)
        demo_image = cv2.cvtColor(demo_image, cv2.COLOR_BGR2GRAY)
        demo_image = resize_image(demo_image, max_width=512)

    print(f"🖼️  測試影像尺寸: {demo_image.shape}")

    # 測試不同醫學影像模態的增強
    for modality in modalities_to_test:
        print(f"\n🔬 測試 {modality.value} 影像增強:")

        try:
            # 執行增強
            results = enhancer.enhance_medical_image(demo_image, modality)

            if results['success']:
                print("✅ 增強成功！")

                # 顯示處理時間
                print(f"⏱️  總處理時間: {results['total_time']:.1f}ms")

                # 顯示處理步驟
                print("📊 處理步驟耗時:")
                for step, time_ms in results['processing_steps'].items():
                    if not step.endswith('_metrics'):
                        print(f"  {step}: {time_ms:.1f}ms")

                # 顯示品質指標比較
                if results['quality_metrics']['original'] and results['quality_metrics']['enhanced']:
                    original_metrics = results['quality_metrics']['original']
                    enhanced_metrics = results['quality_metrics']['enhanced']

                    print("📈 品質指標比較:")
                    metrics_to_show = ['contrast', 'sharpness', 'brightness', 'entropy']

                    for metric in metrics_to_show:
                        orig_val = original_metrics[metric]
                        enh_val = enhanced_metrics[metric]
                        improvement = ((enh_val - orig_val) / orig_val * 100) if orig_val != 0 else 0

                        print(f"  {metric:>10}: {orig_val:6.2f} → {enh_val:6.2f} "
                              f"({improvement:+5.1f}%)")

                # 可視化結果
                images = [results['original_image'], results['enhanced_image']]
                titles = [
                    f"原始 {modality.value} 影像",
                    f"增強後 {modality.value} 影像"
                ]

                display_multiple_images(images, titles, figsize=(12, 6))

            else:
                print("❌ 增強失敗")
                if 'error' in results:
                    print(f"錯誤: {results['error']}")

        except Exception as e:
            print(f"❌ 演示過程中發生錯誤: {e}")

    print("\n📋 增強器功能總結:")
    print("• 支援多種醫學影像模態 (X-Ray, CT, MRI, 超音波等)")
    print("• CLAHE 對比度限制自適應直方圖均化")
    print("• 多種降噪算法 (雙邊濾波, 高斯濾波, 非局部均值)")
    print("• 自適應銳化處理")
    print("• 量化品質評估")
    print("• 專為醫學影像優化的參數預設")

    print("\n⚠️ 重要提醒:")
    print("本系統僅用於教學和研究目的")
    print("實際醫療應用需要:")
    print("• 專業醫療軟體認證")
    print("• 醫師專業判斷")
    print("• 符合醫療標準的品質控制")
    print("• DICOM 標準相容性")


if __name__ == "__main__":
    demo_medical_image_enhancement()
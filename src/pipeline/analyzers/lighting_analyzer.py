from .base_analyzer import BaseAnalyzer
from .base_analyzer import InputType
import cv2
import numpy as np
import os
import torch
from typing import Dict, List, Any, Optional, Tuple
import torch.nn.functional as F
import json
import logging
import time
from ..utils.result_types import (
    AnalysisResult,
    ImageAnalysisResult,
    VideoAnalysisResult,
    AnalyzerResult
)
from ..utils.model_config import ModelConfig
from ..utils.feature_extractors import FeatureExtractor
from abc import ABC, abstractmethod

class LightingAnalyzer(BaseAnalyzer):
    """光照分析器 - 分析室内光照条件"""
    
    _instances = {}
    
    def __new__(cls, model_config: ModelConfig):
        config_id = id(model_config)
        if config_id not in cls._instances:
            cls._instances[config_id] = super(LightingAnalyzer, cls).__new__(cls)
        return cls._instances[config_id]

    def __init__(self, model_config: ModelConfig):
        """初始化光照分析器"""
        super().__init__(model_config)  # 调用父类初始化
        try:
            # 获取配置
            config = self.model_config.get_analyzer_config('lighting')
            
            # 设置阈值
            self.thresholds = config.get('thresholds', {
                'brightness': 0.15,
                'uniformity': 0.4,
                'contrast': 0.1
            })
            
            # 设置理想范围
            self.ideal_ranges = {
                'brightness': [0.4, 0.7],  # 亮度范围
                'uniformity': [0.6, 0.9],  # 均匀度范围
                'contrast': [0.2, 0.5]     # 对比度范围
            }
            
            # 初始化特征提取器
            self.feature_extractor = FeatureExtractor(model_config)
            
            self._initialized = True  # 设置初始化完成标志
            self.logger.info("光照分析器初始化完成")
            
        except Exception as e:
            self.logger.error(f"光照分析器初始化失败: {str(e)}")
            raise

    @abstractmethod
    def _analyze(self, tensor: InputType) -> AnalyzerResult:
        """实际分析逻辑"""

    def analyze(self, frame: np.ndarray) -> Dict:
        """分析图像的光照情况"""
        try:
            self.logger.debug("开始光照分析...")
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            brightness = np.mean(gray) / 255.0
            std_dev = np.std(gray) / 255.0
            uniformity = 1.0 - std_dev
            
            # 扩展分析
            color_temp = self._estimate_color_temperature(frame)
            light_sources = self._detect_light_sources(frame)
            natural_light = self._calculate_natural_light_intensity(frame, self._detect_windows(frame))
            
            result = {
                'type': 'lighting',
                'confidence': brightness,
                'features': {
                    'basic_metrics': {
                        'brightness': float(brightness),
                        'uniformity': float(uniformity),
                        'contrast': self._calculate_contrast(frame)
                    },
                    'quality': {
                        'score': self._calculate_quality_score(brightness, uniformity, self._calculate_contrast(frame)),
                        'color_temperature': float(color_temp)
                    },
                    'light_sources': {
                        'natural_light_ratio': float(natural_light),
                        'sources': light_sources
                    }
                },
                'metadata': {'timestamp': time.time()}
            }
            
            self.logger.info(f"[光照分析器] 输出: \n"
                            f"- 基础指标:\n"
                            f"  • 整体亮度: {brightness:.2f}\n"
                            f"  • 光照均匀度: {uniformity:.2f}\n"
                            f"  • 对比度: {result['features']['basic_metrics']['contrast']:.2f}\n"
                            f"- 光照质量:\n"
                            f"  • 质量评分: {result['features']['quality']['score']:.2f}\n"
                            f"  • 色温: {color_temp:.0f}K\n"
                            f"- 光源分析:\n"
                            f"  • 自然光比例: {natural_light:.2f}\n"
                            f"  • 主光源数量: {len(light_sources)}")
            return result
            
        except Exception as e:
            self.logger.error(f"光照分析失败: {str(e)}")
            return {
                'type': 'lighting',
                'confidence': 0.0,
                'features': {},
                'metadata': {'error': str(e)}
            }

    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息"""
        return {
            'name': 'lighting_analyzer',
            'type': 'LightingAnalyzer',
            'version': '1.0.0',
            'metrics': ['brightness', 'contrast', 'uniformity']
        }
        
    def validate_input(self, frame: np.ndarray) -> bool:
        """验证输入是否有效"""
        try:
            if frame is None:
                return False
            if not isinstance(frame, np.ndarray):
                return False
            if len(frame.shape) != 3:
                return False
            if frame.shape[2] != 3:
                return False
            if frame.dtype != np.uint8:
                return False
            return True
        except Exception:
            return False
            
    def _create_model(self) -> torch.nn.Module:
        """创建模型"""
        # 光照分析暂时不使用深度学习模型
        return None
        
    def _split_image(self, img: np.ndarray, rows: int, cols: int) -> List[np.ndarray]:
        """将图像分割成块"""
        blocks = []
        h, w = img.shape
        block_h, block_w = h // rows, w // cols
        
        for i in range(rows):
            for j in range(cols):
                block = img[i*block_h:(i+1)*block_h, j*block_w:(j+1)*block_w]
                blocks.append(block)
            
        return blocks
        
    def _analyze_brightness(self, frame: np.ndarray) -> float:
        """分析整体亮度
        Args:
            frame: 预处理后的图像
        Returns:
            亮度值(0-1)
        """
        try:
            # 转换为灰度图
            if len(frame.shape) == 3:
                gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            else:
                gray = frame
                
            # 计算平均亮度
            return np.mean(gray) / 255.0
            
        except Exception as e:
            self.logger.error(f"亮度分析失败: {str(e)}")
            return 0.0
            
    def _analyze_uniformity(self, frame: np.ndarray) -> float:
        """分析光照均匀性
        Args:
            frame: 预处理后的图像
        Returns:
            均匀性值(0-1)
        """
        try:
            # 转换为灰度图
            if len(frame.shape) == 3:
                gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            else:
                gray = frame
                
            # 计算标准差
            std = np.std(gray)
            return 1.0 - (std / 128.0)  # 标准差越小越均匀
            
        except Exception as e:
            self.logger.error(f"均匀性分析失败: {str(e)}")
            return 0.0
            
    def _calculate_contrast(self, frame: np.ndarray) -> float:
        """计算图像对比度"""
        try:
            # 确保输入是BGR格式
            if len(frame.shape) == 2:
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            elif frame.shape[2] == 4:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
            
            # 转换为灰度图
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # 使用Sobel算子计算梯度
            sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            
            # 计算梯度幅值
            gradient = np.sqrt(sobelx**2 + sobely**2)
            
            # 归一化对比度
            contrast = np.mean(gradient) / 255.0
            
            return float(contrast)
            
        except Exception as e:
            self.logger.error(f"对比度计算失败: {str(e)}")
            return 0.0
            
    def _detect_light_sources(self, frame: np.ndarray) -> list:
        """检测光源
        Args:
            frame: 预处理后的图像
        Returns:
            光源列表
        """
        try:
            # 这里应该实现实际的光源检测逻辑
            # 现在返回模拟数据
            return [
                {'type': 'window', 'position': [100, 100], 'intensity': 0.8},
                {'type': 'lamp', 'position': [300, 200], 'intensity': 0.6}
            ]
            
        except Exception as e:
            self.logger.error(f"光源检测失败: {str(e)}")
            return []
            
    def _calculate_quality_score(self, brightness: float, uniformity: float, contrast: float) -> float:
        """计算光照质量分数
        Args:
            brightness: 亮度值
            uniformity: 均匀性值
            contrast: 对比度值
        Returns:
            质量分数(0-1)
        """
        try:
            # 加权平均
            weights = {
                'brightness': 0.4,
                'uniformity': 0.3,
                'contrast': 0.3
            }
            
            score = (
                weights['brightness'] * brightness +
                weights['uniformity'] * uniformity +
                weights['contrast'] * contrast
            )
            
            return min(max(score, 0.0), 1.0)  # 限制在0-1范围内
            
        except Exception as e:
            self.logger.error(f"质量分数计算失败: {str(e)}")
            return 0.0

    def _analyze_basic_lighting(self, frame: np.ndarray) -> Dict:
        """分析基础光照信息"""
        try:
            # 转换到HSV空间
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            
            # 添加数据有效性检查
            if hsv.size == 0:
                return {
                    'brightness': 0,
                    'brightness_std': 0,
                    'contrast': 0,
                    'uniformity': 0
                }
            
            # 计算亮度统计
            brightness = np.mean(hsv[:,:,2])
            brightness_std = np.std(hsv[:,:,2])
            
            # 避免除零
            uniformity = 1 - (brightness_std / (brightness + 1e-6))
            
            return {
                'brightness': float(brightness),
                'brightness_std': float(brightness_std),
                'contrast': self._calculate_contrast(frame),
                'uniformity': float(uniformity)
            }
            
        except Exception as e:
            print(f"基础光照分析失败: {str(e)}")
            return None
        
    def _analyze_natural_light(self, frame: np.ndarray, depth_map: np.ndarray) -> Dict:
        """分析自然光"""
        # 检测窗户区域
        windows = self._detect_windows(frame)
        
        # 计算色温
        color_temp = self._calculate_color_temperature(frame)
        
        # 计算自然光强度
        intensity = self._calculate_natural_light_intensity(frame, windows)
        
        return {
            'windows': windows,
            'color_temperature': color_temp,
            'intensity': intensity
        }
        
    def _analyze_shadows(self, frame: np.ndarray, depth_map: np.ndarray) -> Dict:
        """分析阴影"""
        try:
            # 转换到灰度图
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # 使用深度信息辅助阴影检测
            if depth_map is not None:
                # 结合深度不连续区域
                depth_edges = cv2.Canny(
                    (depth_map * 255).astype(np.uint8), 
                    50, 150
                )
                
                # 自适应阈值分割
                thresh = cv2.adaptiveThreshold(
                    gray,
                    255,
                    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                    cv2.THRESH_BINARY,
                    11,
                    2
                )
                
                # 结合深度边缘和亮度阈值
                shadow_mask = (thresh == 0) & (depth_edges > 0)
                
                # 添加数据有效性检查
                if np.sum(shadow_mask) > 0:
                    shadow_ratio = np.sum(shadow_mask) / shadow_mask.size
                    shadow_intensity = np.mean(gray[shadow_mask])
                else:
                    shadow_ratio = 0.0
                    shadow_intensity = 0.0
                
                return {
                    'shadow_ratio': float(shadow_ratio),
                    'shadow_intensity': float(shadow_intensity),
                    'shadow_map': shadow_mask.astype(np.uint8)
                }
            
            return {
                'shadow_ratio': 0.0,
                'shadow_intensity': 0.0,
                'shadow_map': np.zeros_like(gray, dtype=np.uint8)
            }
            
        except Exception as e:
            print(f"阴影分析失败: {str(e)}")
            return None
        
    def _estimate_depth(self, frame: np.ndarray) -> np.ndarray:
        """估计场景深度"""
        try:
            # 预处理图像到正确的尺寸
            input_size = (384, 384)  # 使用模型期望的输入尺寸
            frame_resized = cv2.resize(frame, input_size)
            tensor = self.preprocess(frame_resized)
            
            # 推理
            with torch.no_grad():
                depth = self.depth_model(tensor)
            
            # 调整输出尺寸以匹配输入图像
            depth = F.interpolate(depth, size=frame.shape[:2], mode='bilinear', align_corners=False)
            return depth.squeeze().cpu().numpy()
            
        except Exception as e:
            print(f"深度估计失败: {str(e)}")
            return None
            
    def _detect_windows(self, frame: np.ndarray) -> List[Dict]:
        """检测窗户区域"""
        try:
            # 转换到灰度图
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # 边缘检测
            edges = cv2.Canny(gray, 50, 150)
            
            # 寻找轮廓
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            windows = []
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area > 1000:  # 过滤小区域
                    x, y, w, h = cv2.boundingRect(cnt)
                    aspect_ratio = w / h
                    
                    # 判断是否可能是窗户
                    if 0.5 < aspect_ratio < 2.0:
                        windows.append({
                            'position': (x, y),
                            'size': (w, h),
                            'area': area
                        })
            
            return windows
            
        except Exception as e:
            print(f"窗户检测失败: {str(e)}")
            return []

    def _calculate_color_temperature(self, frame: np.ndarray) -> float:
        """计算色温"""
        try:
            # 转换到RGB空间
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # 计算平均RGB值
            avg_color = np.mean(rgb, axis=(0,1))
            r, g, b = avg_color
            
            # 使用McCamy's公式计算色温
            if b == 0:
                b = 1e-6
            x = (r - g) / (g - b)
            y = (2 * r - g - b) / (g - b)
            
            # McCamy's approximation
            n = (x - 0.3320) / (0.1858 - y)
            CCT = 449 * n**3 + 3525 * n**2 + 6823.3 * n + 5520.33
            
            return float(CCT)
            
        except Exception as e:
            self.logger.error(f"色温估算失败: {str(e)}")
            return 6500  # 返回标准日光色温

    def _calculate_natural_light_intensity(self, frame: np.ndarray, windows: List[Dict]) -> float:
        """计算自然光强度"""
        try:
            if not windows:
                return 0.0
            
            # 转换到HSV空间
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            
            # 计算窗户区域的亮度
            total_brightness = 0
            total_area = 0
            
            for window in windows:
                x, y = window['position']
                w, h = window['size']
                window_region = hsv[y:y+h, x:x+w, 2]  # 提取V通道
                
                # 计算加权亮度
                brightness = np.mean(window_region)
                area = window['area']
                
                total_brightness += brightness * area
                total_area += area
            
            # 归一化强度分数
            if total_area > 0:
                intensity = total_brightness / (total_area * 255)
                return float(intensity)
            return 0.0
            
        except Exception as e:
            self.log_error(f"自然光强度计算出错: {str(e)}")
            return 0.0

    def preprocess(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """预处理输入图像"""
        try:
            if not isinstance(frame, np.ndarray):
                return None
            
            # 基础图像处理
            processed = cv2.resize(frame, (224, 224))  # 使用固定大小
            processed = processed.astype(np.float32) / 255.0
            
            return processed
            
        except Exception as e:
            self.logger.error(f"预处理失败: {str(e)}")
            return None

    def _score_in_range(self, value: float, ideal_range: Tuple[float, float]) -> float:
        """计算值在理想范围内的得分"""
        min_val, max_val = ideal_range
        if value < min_val:
            return 1.0 - (min_val - value) / min_val
        elif value > max_val:
            return 1.0 - (value - max_val) / (1.0 - max_val)
        else:
            return 1.0

    def _evaluate_lighting_quality(self, brightness, contrast, uniformity, 
                                 overexposed, underexposed, ideal_ranges):
        """评估光照质量"""
        try:
            issues = []
            
            # 检查亮度
            if brightness < ideal_ranges['brightness'][0]:
                issues.append("光线不足")
            elif brightness > ideal_ranges['brightness'][1]:
                issues.append("光线过强")
            
            # 检查均匀度
            if uniformity < ideal_ranges['uniformity'][0]:
                issues.append("光照不均匀")
            elif uniformity > ideal_ranges['uniformity'][1]:
                issues.append("缺乏层次感")
            
            # 检查对比度
            if contrast < ideal_ranges['contrast'][0]:
                issues.append("对比度不足")
            elif contrast > ideal_ranges['contrast'][1]:
                issues.append("对比度过高")
            
            # 检查曝光问题
            if overexposed > 5:  # 过曝区域超过5%
                issues.append("存在过曝")
            if underexposed > 10:  # 欠曝区域超过10%
                issues.append("存在暗部")
            
            # 返回质量评价
            if not issues:
                return "良好"
            elif len(issues) <= 2:
                return "需改进: " + ", ".join(issues)
            else:
                return "较差: " + ", ".join(issues[:3])
            
        except Exception as e:
            self.logger.error(f"光照质量评估失败: {str(e)}")
            return "评估失败"

    def _load_model(self, config: Dict) -> None:
        """光照分析不需要加载模型"""
        pass  # 光照分析使用传统方法，不需要加载模型 

    def _get_empty_result(self) -> Dict[str, Any]:
        """返回空结果"""
        return {
            'type': 'lighting_analysis',
            'confidence': 0.0,
            'features': {
                'brightness': 0.0,
                'uniformity': 0.0,
                'contrast': 0.0,
                'overexposed_ratio': 0.0,
                'underexposed_ratio': 0.0,
                'quality': '未知'
            }
        } 

    @property
    def confidence_threshold(self) -> float:
        """获取置信度阈值"""
        config = self.model_config.get_analyzer_config('lighting')
        threshold = config.get('confidence_threshold', 0.4)  # 光照分析默认阈值
        self._confidence_threshold = threshold  # 更新基类中的值
        return self._confidence_threshold

    @confidence_threshold.setter
    def confidence_threshold(self, value: float):
        """设置置信度阈值"""
        self._confidence_threshold = value

    def _analyze(self, tensor: InputType) -> AnalyzerResult:
        frame = tensor.cpu().numpy().squeeze().transpose(1,2,0)
        result = self.analyze(frame)  # 调用原有逻辑
        return AnalyzerResult(**result)

    def _estimate_color_temperature(self, frame: np.ndarray) -> float:
        """估算场景色温"""
        try:
            # 转换到RGB空间
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # 计算平均RGB值
            avg_color = np.mean(rgb, axis=(0,1))
            r, g, b = avg_color
            
            # 使用McCamy's公式计算色温
            if b == 0:
                b = 1e-6
            x = (r - g) / (g - b)
            y = (2 * r - g - b) / (g - b)
            
            # McCamy's approximation
            n = (x - 0.3320) / (0.1858 - y)
            CCT = 449 * n**3 + 3525 * n**2 + 6823.3 * n + 5520.33
            
            return float(CCT)
            
        except Exception as e:
            self.logger.error(f"色温估算失败: {str(e)}")
            return 6500  # 返回标准日光色温
from abc import ABC, abstractmethod
import logging
import numpy as np
from typing import Dict, Any, Optional
from ..utils import AnalysisType
import cv2
import torch
from ..utils.model_config import ModelConfig

class BaseProcessor(ABC):
    """处理器基类"""
    
    def __init__(self, model_config: ModelConfig):
        """初始化基础处理器
        Args:
            model_config: 模型配置
        """
        self.logger = logging.getLogger(f"pipeline.{self.__class__.__name__}")
        self.model_config = model_config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
    def _validate_input(self, input_data: Any) -> bool:
        """验证输入数据"""
        return input_data is not None
        
    def _preprocess(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """基础预处理"""
        try:
            if not isinstance(frame, np.ndarray):
                self.logger.error("输入必须是numpy数组")
                return None
                
            # 复制帧以避免修改原始数据
            frame = frame.copy()
            
            # 验证帧格式
            if len(frame.shape) != 3 or frame.shape[2] != 3:
                self.logger.error(f"无效的帧格式: {frame.shape}")
                return None
                
            # 确保是BGR格式和uint8类型
            if frame.dtype != np.uint8:
                frame = (frame * 255).astype(np.uint8)
                
            return frame
            
        except Exception as e:
            self.logger.error(f"预处理失败: {str(e)}")
            return None
        
    def _postprocess(self, result: Dict) -> Dict:
        """后处理结果"""
        return result
        
    @abstractmethod
    def process(self, input_data):
        """处理输入数据"""
        pass
        
    def _comprehensive_analysis(self, frame: np.ndarray) -> Dict:
        """全面分析"""
        result = {}
        
        # 场景分析
        scene_result = self.scene_analyzer.analyze(frame)
        if scene_result:
            result['scene'] = scene_result
            
        # 家具检测
        furniture_result = self.furniture_detector.analyze(frame)
        if furniture_result:
            result['furniture'] = furniture_result
            
        # 光照分析
        lighting_result = self.lighting_analyzer.analyze(frame)
        if lighting_result:
            result['lighting'] = lighting_result
            
        # 风格分析
        style_result = self.style_analyzer.analyze(frame)
        if style_result:
            result['style'] = style_result
            
        # 颜色分析
        color_result = self.color_analyzer.analyze(frame)
        if color_result:
            result['color'] = color_result
            
        return result
        
    def _quick_analysis(self, frame: np.ndarray) -> Dict:
        """快速分析"""
        result = {}
        
        # 只进行场景和家具分析
        scene_result = self.scene_analyzer.analyze(frame)
        if scene_result:
            result['scene'] = scene_result
            
        furniture_result = self.furniture_detector.analyze(frame)
        if furniture_result:
            result['furniture'] = furniture_result
            
        return result

    def preprocess_image(self, frame: np.ndarray) -> np.ndarray:
        """预处理输入图像
        
        只进行基础的图像预处理，不涉及 tensor 转换
        """
        try:
            # 调整图像大小
            frame = cv2.resize(frame, (224, 224))  # 使用标准输入尺寸
            
            # 转换为RGB
            if len(frame.shape) == 2:
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
            elif frame.shape[2] == 4:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2RGB)
            elif frame.shape[2] == 3:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # 标准化
            frame = frame.astype(np.float32) / 255.0
            frame = (frame - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
            
            return frame
        
        except Exception as e:
            self.logger.error(f"图像预处理失败: {str(e)}")
            return None 

    @abstractmethod
    def preprocess(self, input_data):
        """预处理输入数据"""
        pass 
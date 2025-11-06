from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Union, TypeVar, Generic, List
import torch
import numpy as np
import cv2
from ..utils.model_config import ModelConfig
import os
import timm
import torch.nn as nn
import logging
import traceback
import time
from ..utils.result_types import AnalysisResult, AnalyzerResult
from ..utils.feature_extractors import FeatureExtractor

InputType = TypeVar('InputType', np.ndarray, torch.Tensor)

class BaseAnalyzer(Generic[InputType], ABC):
    """分析器基类 - 定义所有分析器的通用接口和功能"""
    
    def __new__(cls, *args, **kwargs):
        # 使用单例模式，确保每个分析器只被初始化一次
        if not hasattr(cls, '_instance'):
            cls._instance = super().__new__(cls)
            # 在这里初始化实例属性
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, model_config: ModelConfig):
        """初始化基础分析器
        Args:
            model_config: 模型配置对象
        """
        # 如果已经初始化过，直接返回
        if getattr(self, '_initialized', False):
            return
            
        self.model_config = model_config
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # 验证配置
        if not hasattr(self, 'analyzer_type'):
            self.analyzer_type = self.__class__.__name__.lower().replace('analyzer', '')
        
        # 获取配置
        try:
            self.config = self.model_config.get_analyzer_config(self.analyzer_type)
            if not self.config:
                raise ValueError(f"找不到 {self.analyzer_type} 分析器的配置")
        except Exception as e:
            self.logger.error(f"获取 {self.analyzer_type} 分析器配置失败: {str(e)}")
            raise
        
        # 初始化基础属性
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self._model_loaded = False
        self._confidence_threshold = 0.15
        self.high_confidence_threshold = 0.3
        self.feature_extractor = FeatureExtractor()
        self.type = self.__class__.__name__.lower()
        
    @abstractmethod
    def analyze(self, frame: np.ndarray) -> Dict:
        """分析单帧"""
        pass

    def log_debug(self, msg: str):
        """调试日志"""
        self.logger.debug(msg)
    
    def log_info(self, msg: str):
        """信息日志"""
        self.logger.info(msg)
    
    def log_warning(self, msg: str):
        """警告日志"""
        self.logger.warning(msg)
    
    def log_error(self, msg: str):
        """错误日志"""
        self.logger.error(msg)
    
    def validate_model_path(self, model_name: str) -> bool:
        """验证模型文件是否存在"""
        path = self.get_model_path(model_name)
        self.log_debug(f"正在验证模型路径: {path}")
        if path is None:
            return False
        
        exists = os.path.exists(path)
        self.log_debug(f"模型文件存在: {exists}")
        return exists

    def _load_model(self, config: Dict) -> Optional[torch.nn.Module]:
        """加载模型
        Args:
            config: 模型配置
        Returns:
            加载的模型或None
        """
        try:
            if not config:
                self.logger.error("模型配置为空")
                return None
                
            model = self._create_model()
            if model is not None:
                self._model_loaded = True
                self.logger.info(f"{self.__class__.__name__}模型加载成功")
                return model
            return None
            
        except Exception as e:
            self.logger.error(f"模型加载失败: {str(e)}")
            return None
    
    def analyze(self, frame: np.ndarray) -> Dict[str, Any]:
        """基础分析方法
        Args:
            frame: 输入图像帧
        Returns:
            分析结果字典，包含:
            {
                'type': str,          # 分析类型
                'confidence': float,   # 置信度 0-1
                'features': dict,      # 特征字典
                'metadata': dict      # 元数据
            }
        """
        try:
            if not isinstance(frame, np.ndarray):
                self.logger.error("输入必须是numpy数组")
                return self._get_empty_result()
                
            # 预处理
            processed = self.preprocess(frame)
            if processed is None:
                return self._get_empty_result()
                
            # 执行分析
            result = self._analyze(processed)
            if result is None:
                return self._get_empty_result()
                
            return result
            
        except Exception as e:
            self.logger.error(f"分析失败: {str(e)}")
            return self._get_empty_result()

    @abstractmethod
    def _analyze(self, frame: np.ndarray) -> AnalyzerResult:
        """内部分析实现"""
        raise NotImplementedError("子类必须实现_analyze方法")
        
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
            return True
        except Exception:
            return False
            
    def preprocess(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """基础预处理"""
        try:
            if frame is None:
                return None
                
            # 保持原始比例
            frame = cv2.resize(frame, (640, 480))
            if len(frame.shape) == 2:
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            elif frame.shape[2] == 4:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
                
            return frame
            
        except Exception as e:
            self.logger.error(f"预处理失败: {str(e)}")
            return None
        
    def _normalize(self, tensor: torch.Tensor) -> torch.Tensor:
        """标准化张量"""
        mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)
        return (tensor - mean) / std
        
    @property
    @abstractmethod
    def confidence_threshold(self) -> float:
        """获取置信度阈值"""
        return self._confidence_threshold
        
    @confidence_threshold.setter
    def confidence_threshold(self, value: float):
        """设置置信度阈值"""
        self._confidence_threshold = value
    
    def load_model(self, model_name: str) -> None:
        """加载模型"""
        try:
            if self._model_loaded:  # 避免重复加载
                return
            
            config = self.model_config.get_model_config(model_name)
            if config is None:
                raise ValueError(f"找不到模型配置: {model_name}")
            
            self._load_model(config)
            self._model_loaded = True
            
        except Exception as e:
            self.logger.error(f"模型加载失败: {str(e)}")
            raise
    
    @abstractmethod
    def _create_model(self) -> Optional[torch.nn.Module]:
        """创建模型"""
        pass
    
    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息"""
        pass
    
    def _print_analysis_result(self, result: Dict, analyzer_name: str) -> None:
        """打印分析结果 - 仅在调试模式下输出"""
        if result is None:
            self.log_warning(f"{analyzer_name}未能得出结果")
            return
            
        if self.logger.getEffectiveLevel() <= logging.DEBUG:
            self.log_debug(f"\n{analyzer_name}分析结果如下:")
            self._format_result(result, prefix="  ")
    
    def _format_result(self, data: Any, prefix: str = "") -> None:
        """格式化结果输出"""
        if isinstance(data, dict):
            for key, value in data.items():
                if isinstance(value, dict):
                    self.log_debug(f"{prefix}{key}:")
                    self._format_result(value, prefix + "  ")
                elif isinstance(value, float):
                    self.log_debug(f"{prefix}{key}: {value:.4f}")
                else:
                    self.log_debug(f"{prefix}{key}: {value}")
        elif isinstance(data, (list, tuple)):
            for item in data:
                self._format_result(item, prefix) 

    def set_log_level(self, level: int):
        """设置日志级别"""
        self.logger.setLevel(level) 

    def _log_analysis_start(self):
        """记录分析开始"""
        self.logger.info(f"开始{self.__class__.__name__}分析")
        self.start_time = time.time()
        
    def _log_analysis_result(self, result: Dict):
        """记录分析结果"""
        duration = time.time() - self.start_time
        self.logger.info(f"{self.__class__.__name__}分析完成, 耗时: {duration:.3f}秒")
        self.logger.debug(f"分析结果: {result}")
        
        # 记录关键指标
        if result:
            if 'confidence' in result:
                self.logger.info(f"置信度: {result['confidence']:.3f}")
            if 'count' in result:
                self.logger.info(f"检测到对象数量: {result['count']}")
                
    def _log_error(self, error: Exception, context: str = ""):
        """记录错误信息"""
        self.logger.error(f"{self.__class__.__name__} {context}错误: {str(error)}")
        self.logger.debug(f"错误详情: {traceback.format_exc()}")
        
    def _log_analysis_error(self, error: Exception):
        """记录分析错误 (兼容旧方法)"""
        self._log_error(error, "分析")
        
    def _log_warning(self, message: str):
        """记录警告信息"""
        self.logger.warning(f"{self.__class__.__name__}: {message}")
        
    def _log_model_info(self, model_info: Dict):
        """记录模型信息"""
        self.logger.info(f"模型信息:")
        for key, value in model_info.items():
            self.logger.info(f"- {key}: {value}")
            
    def _log_preprocessing(self, frame: np.ndarray):
        """记录预处理信息"""
        self.logger.debug(f"预处理输入尺寸: {frame.shape}")
        self.logger.debug(f"预处理输入类型: {frame.dtype}")
        
    def _log_postprocessing(self, output: Any):
        """记录后处理信息"""
        self.logger.debug(f"后处理输出类型: {type(output)}")
        if isinstance(output, dict):
            self.logger.debug(f"后处理输出键: {list(output.keys())}") 

    def _get_quality_level(self, *metrics: float) -> str:
        """根据指标评估质量等级"""
        avg_score = sum(metrics) / len(metrics)
        if avg_score > 0.8:
            return "优秀"
        elif avg_score > 0.6:
            return "良好"
        elif avg_score > 0.4:
            return "一般"
        else:
            return "需改进" 

    def _get_empty_result(self) -> Dict[str, Any]:
        """返回空结果"""
        return {
            'type': self.__class__.__name__.lower(),
            'confidence': 0.0,
            'features': {},
            'metadata': {
                'error': True,
                'timestamp': time.time()
            }
        } 

    def _preprocess(self, frame: np.ndarray) -> np.ndarray:
        """标准化预处理流程"""
        # 统一尺寸
        frame = cv2.resize(frame, (640, 480))
        # 颜色空间转换
        if len(frame.shape) == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        elif frame.shape[2] == 4:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        return frame

    def analyze_video(self, frames: List[np.ndarray]) -> Dict:
        """分析视频帧序列"""
        try:
            frame_results = []
            for frame in frames:
                result = self.analyze(frame)
                if result:
                    frame_results.append(result)
                    
            return {
                'type': 'video_analysis',
                'analyzer': self.__class__.__name__.lower(),
                'results': {
                    'frame_results': frame_results,
                    'metadata': {
                        'frame_count': len(frame_results),
                        'timestamp': time.time()
                    }
                }
            }
            
        except Exception as e:
            self.logger.error(f"视频分析失败: {str(e)}")
            return None
        
    def get_model_path(self, model_name: str) -> Optional[str]:
        """获取模型文件路径"""
        config = self.model_config.get_model_config(model_name)
        if config:
            return config.get('path')
        return None

    def _preprocess_numpy(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """预处理numpy数组"""
        try:
            if not isinstance(frame, np.ndarray):
                return None
                
            # 基础图像处理
            processed = cv2.resize(frame, (224, 224))
            processed = processed.astype(np.float32) / 255.0
            
            return processed
            
        except Exception as e:
            self.logger.error(f"预处理失败: {str(e)}")
            return None
        
    def _preprocess_tensor(self, tensor: torch.Tensor) -> Optional[torch.Tensor]:
        """预处理张量"""
        try:
            if not isinstance(tensor, torch.Tensor):
                return None
                
            # 基础图像处理
            processed = torch.nn.functional.interpolate(tensor, size=(224, 224), mode='bilinear', align_corners=False)
            processed = processed.float() / 255.0
            
            return processed
            
        except Exception as e:
            self.logger.error(f"预处理失败: {str(e)}")
            return None 

    def _to_tensor(self, frame: np.ndarray) -> torch.Tensor:
        """转换图像为张量
        Args:
            frame: 输入图像(BGR格式)
        Returns:
            标准化的图像张量
        """
        try:
            # 1. BGR转RGB
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # 2. 转换为float32并归一化
            rgb = rgb.astype(np.float32) / 255.0
            
            # 3. 标准化
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            normalized = (rgb - mean) / std
            
            # 4. 转换为张量
            tensor = torch.from_numpy(normalized)
            
            # 5. 调整维度顺序
            tensor = tensor.permute(2, 0, 1)
            
            # 6. 添加batch维度
            tensor = tensor.unsqueeze(0)
            
            # 7. 移动到正确的设备
            tensor = tensor.to(self.device)
            
            return tensor
            
        except Exception as e:
            self.logger.error(f"张量转换失败: {str(e)}")
            return None

def validate_result(func):
    """结果验证装饰器"""
    def wrapper(self, *args, **kwargs):
        result = func(self, *args, **kwargs)
        if result and not isinstance(result, AnalyzerResult):
            self.logger.error(f"无效结果类型: {type(result)}")
            return None
        return result
    return wrapper 
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Union
import numpy as np
import logging
import time
import cv2
from concurrent.futures import ThreadPoolExecutor
import hashlib
from ..validators.data_validator import DataValidator
from ..analyzers.base_analyzer import BaseAnalyzer

class BaseStrategy(ABC):
    """策略基类 - 决定要使用的分析器组合"""
    
    def __init__(self, analyzers: Dict[str, BaseAnalyzer]):
        """初始化基础策略
        Args:
            analyzers: 分析器字典
        """
        self.logger = logging.getLogger("pipeline.BaseStrategy")
        self.analyzers = analyzers
        self.selected_analyzers = []  # 由子类决定具体分析器
        self.data_validator = DataValidator()
        
    @abstractmethod
    def get_analyzer_names(self) -> List[str]:
        """获取需要使用的分析器名称列表"""
        return list(self.analyzers.keys())

    def execute(self, input_data: Union[str, np.ndarray], analyzer_type: Optional[str] = None) -> Dict:
        """执行分析策略
        Args:
            input_data: 输入数据
            analyzer_type: 分析器类型（可选）
        Returns:
            分析结果
        """
        raise NotImplementedError("子类必须实现execute方法")
        
    @abstractmethod
    def process(self, frames: List[np.ndarray]) -> Dict:
        """处理帧序列（用于视频分析）"""
        pass
        
    def _analyze(self, frame: np.ndarray) -> Dict:
        """分析单帧"""
        try:
            results = {}
            for name, analyzer in self.analyzers.items():
                result = analyzer.analyze(frame)
                if result:
                    results[name] = result
            return results
        except Exception as e:
            self.logger.error(f"分析失败: {str(e)}")
            return {}

# 为了向后兼容，保留AnalysisStrategy
AnalysisStrategy = BaseStrategy 
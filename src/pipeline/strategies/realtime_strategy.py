from typing import Dict, Optional, List, Any, Union
import numpy as np
import logging
import time
from .base_strategy import BaseStrategy
from ..analyzers.base_analyzer import BaseAnalyzer
from ..utils.result_processor import ProcessedResult

class RealtimeStrategy(BaseStrategy):
    """实时分析策略 - 快速轻量级的实时分析"""
    _instance = None
    _initialized = False
    
    def __new__(cls, analyzers=None):
        if cls._instance is None:
            cls._instance = super(RealtimeStrategy, cls).__new__(cls)
        return cls._instance
    
    def __init__(self, analyzers=None):
        if self._initialized:
            return
            
        super().__init__(analyzers)
        self.logger = logging.getLogger("pipeline.RealtimeStrategy")
        self.logger.propagate = False
        
        # 只启用场景和家具分析器
        self.enabled_analyzers = {
            'scene': self.analyzers.get('scene'),
            'furniture': self.analyzers.get('furniture')
        }
        
        # 设置实时处理参数
        self.skip_frames = 2  # 每隔几帧分析一次
        self.frame_count = 0
        self.last_results = {}  # 缓存上一次的结果
        
        self._initialized = True
    
    def process(self, frame: np.ndarray) -> Dict:
        """实现基类要求的process方法"""
        return self.execute(frame)
        
    def execute(self, input_data: Union[str, np.ndarray], analyzer_type: Optional[str] = None) -> Dict:
        """执行实时分析
        Args:
            input_data: 输入数据
            analyzer_type: 不使用此参数，保持接口一致性
        Returns:
            实时分析结果
        """
        try:
            self.logger.debug("开始执行实时分析...")
            results = {}
            
            # 只使用轻量级分析器
            light_analyzers = ['scene', 'furniture']
            for name in light_analyzers:
                analyzer = self.analyzers.get(name)
                if analyzer:
                    self.logger.debug(f"执行 {name} 分析...")
                    results[name] = analyzer.analyze(input_data)
                    self.logger.debug(f"{name} 分析完成")
                    
            return results
            
        except Exception as e:
            self.logger.error(f"实时分析执行失败: {str(e)}")
            return {}
            
    def get_analyzer_names(self) -> List[str]:
        """获取需要使用的分析器名称列表
        Returns:
            分析器名称列表
        """
        # 实时分析只使用轻量级分析器
        return ['scene', 'color']  # 优先使用计算量小的分析器
        
    def _quick_analysis(self, frame: np.ndarray) -> Optional[Dict]:
        """快速分析 - 只进行优先级高的分析
        Args:
            frame: 输入帧
        Returns:
            快速分析结果
        """
        try:
            result = {}
            
            # 只分析优先项目
            for analyzer_type in self.priority_analyzers:
                if analyzer_type in self.analyzers:
                    analyzer_result = self.analyzers[analyzer_type].analyze(frame)
                    if analyzer_result:
                        result[analyzer_type] = analyzer_result
                        
            return result
            
        except Exception as e:
            self.logger.error(f"快速分析失败: {str(e)}")
            return None
            
    def _full_analysis(self, frame: np.ndarray) -> Optional[Dict]:
        """完整分析 - 进行所有分析
        Args:
            frame: 输入帧
        Returns:
            完整分析结果
        """
        try:
            result = {}
            
            # 分析所有项目
            for analyzer_type, analyzer in self.analyzers.items():
                analyzer_result = analyzer.analyze(frame)
                if analyzer_result:
                    result[analyzer_type] = analyzer_result
                    
            return result
            
        except Exception as e:
            self.logger.error(f"完整分析失败: {str(e)}")
            return None
            
    def _merge_results(self, quick_result: Dict) -> Dict:
        """合并快速分析和上一次完整分析的结果
        Args:
            quick_result: 快速分析结果
        Returns:
            合并后的结果
        """
        try:
            merged = {
                'timestamp': time.time(),
                'frame_count': self.frame_count
            }
            
            # 1. 添加快速分析结果
            for key, value in quick_result.items():
                merged[key] = value
                
            # 2. 添加上次完整分析的其他结果
            if self.last_full_result:
                for key, value in self.last_full_result.items():
                    if key not in merged:
                        merged[key] = value
                        
            return merged
            
        except Exception as e:
            self.logger.error(f"结果合并失败: {str(e)}")
            return quick_result
            
    def update_priority(self, new_priority: List[str]):
        """更新分析优先级
        Args:
            new_priority: 新的优先级列表
        """
        try:
            # 验证新优先级列表
            valid_analyzers = set(self.analyzers.keys())
            if not all(p in valid_analyzers for p in new_priority):
                raise ValueError("无效的分析器类型")
                
            self.priority_analyzers = new_priority
            
        except Exception as e:
            self.logger.error(f"优先级更新失败: {str(e)}")
            
    def get_performance_stats(self) -> Dict:
        """获取性能统计信息"""
        return {
            'frame_count': self.frame_count,
            'skip_frames': self.skip_frames,
            'priority_analyzers': self.priority_analyzers
        } 
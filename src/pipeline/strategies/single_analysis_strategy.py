from typing import Dict, Any, Optional, List, Union
import numpy as np
import logging
from .base_strategy import BaseStrategy
from ..analyzers.base_analyzer import BaseAnalyzer
from ..validators.data_validator import DataValidator
import time
import cv2

class SingleAnalysisStrategy(BaseStrategy):
    """单项分析策略"""
    
    def __init__(self, analyzers: Dict[str, BaseAnalyzer], analyzer_type: str):
        """初始化单项分析策略
        Args:
            analyzers: 分析器字典
            analyzer_type: 要使用的分析器类型
        """
        self.analyzers = analyzers
        self.set_analyzer_type(analyzer_type)  # 使用设置方法确保类型有效
        self.logger = logging.getLogger(__name__)
        
    def set_analyzer_type(self, analyzer_type: str):
        """设置要使用的分析器类型"""
        if analyzer_type not in self.analyzers:
            raise ValueError(f"不支持的分析器类型: {analyzer_type}")
        self.analyzer_type = analyzer_type
        
    def execute(self, input_data: Union[str, np.ndarray], analyzer_type: Optional[str] = None) -> Dict:
        """执行单项分析
        Args:
            input_data: 输入数据
            analyzer_type: 要使用的分析器类型
        Returns:
            单个分析器的分析结果
        """
        try:
            if analyzer_type:
                self.set_analyzer_type(analyzer_type)  # 如果提供了类型，更新类型

            if not self.analyzer_type:
                self.logger.error("分析器类型未指定")
                raise ValueError("单项分析必须指定分析器类型")

            analyzer = self.analyzers.get(self.analyzer_type)
            if not analyzer:
                raise ValueError(f"未找到分析器: {self.analyzer_type}")
                
            self.logger.debug(f"执行 {self.analyzer_type} 分析...")
            result = analyzer.analyze(input_data)
            self.logger.debug(f"{self.analyzer_type} 分析完成")
            
            return {self.analyzer_type: result}
            
        except Exception as e:
            self.logger.error(f"单项分析执行失败: {str(e)}")
            return {}
    
    def get_analyzer_names(self) -> List[str]:
        return [self.analyzer_type] if self.analyzer_type in self.analyzers else []
    
    def get_analyzers(self) -> Dict:
        """获取分析器字典
        Returns:
            包含单个分析器的字典
        """
        return {self.analyzer_type: self.analyzers[self.analyzer_type]}
    
    def analyze(self, frame: np.ndarray) -> Dict:
        """执行单项分析"""
        if not self.analyzer_type:
            self.logger.error("分析器类型未指定")
            raise ValueError("单项分析必须指定分析器类型")

        # 执行分析逻辑
        try:
            # 根据分析器类型选择相应的分析器
            if self.analyzer_type == 'scene':
                return self.scene_analyzer.analyze(frame)
            elif self.analyzer_type == 'furniture':
                return self.furniture_analyzer.analyze(frame)
            # 其他分析器...
        except Exception as e:
            self.logger.error(f"单项分析执行失败: {str(e)}")
            return {}
    
    def integrate_results(self, results: List[Dict]) -> Dict:
        """整合多帧分析结果"""
        if not results:
            return {}
        
        # 提取所有帧中当前分析器的结果
        analyzer_results = [
            result.get(self.analyzer_type, {})
            for result in results
        ]
        
        # 返回整合的结果，使用正确的键名格式
        return {
            'analysis': {
                # 使用下划线格式的键名
                self.analyzer_type.replace('analyzer', '_analyzer'): self._merge_analyzer_results(analyzer_results)
            }
        }
    
    def _merge_analyzer_results(self, results: List[Dict]) -> Dict:
        """合并同一分析器的多帧结果"""
        if not results:
            return {}
            
        # 如果只有一帧，直接返回
        if len(results) == 1:
            return results[0]
            
        # 合并多帧结果
        merged = {}
        for key in results[0].keys():
            if key == 'type':
                merged[key] = results[0][key]
            elif key == 'confidence':
                merged[key] = np.mean([r.get(key, 0) for r in results])
            elif key == 'features':
                merged[key] = self._merge_features([r.get(key, {}) for r in results])
            else:
                merged[key] = results[-1][key]  # 使用最后一帧的值
                
        return merged
    
    def _merge_features(self, features_list: List[Dict]) -> Dict:
        """合并多帧的特征结果"""
        if not features_list:
            return {}
        
        merged = {}
        # 保留最后一帧的场景类型和置信度
        merged['scene_type'] = features_list[-1].get('scene_type', '未知')
        
        # 合并空间特征
        spatial_features = {
            'aspect_ratio': np.mean([f.get('scene_features', {}).get('scene_features', {}).get('spatial', {}).get('aspect_ratio', 0.0) for f in features_list]),
            'area': np.mean([f.get('scene_features', {}).get('scene_features', {}).get('spatial', {}).get('area', 0.0) for f in features_list]),
            'symmetry': np.mean([f.get('scene_features', {}).get('scene_features', {}).get('spatial', {}).get('symmetry', 0.0) for f in features_list]),
            'texture_complexity': np.mean([f.get('scene_features', {}).get('scene_features', {}).get('spatial', {}).get('texture_complexity', 0.0) for f in features_list]),
            'lighting_quality': np.mean([f.get('scene_features', {}).get('scene_features', {}).get('spatial', {}).get('lighting_quality', 0.0) for f in features_list])
        }
        
        # 合并视觉特征
        visual_features = {
            'natural_light': np.mean([f.get('scene_features', {}).get('scene_features', {}).get('visual', {}).get('natural_light', 0.0) for f in features_list]),
            'ceiling_height': np.mean([f.get('scene_features', {}).get('scene_features', {}).get('visual', {}).get('ceiling_height', 0.0) for f in features_list]),
            'wall_visibility': np.mean([f.get('scene_features', {}).get('scene_features', {}).get('visual', {}).get('wall_visibility', 0.0) for f in features_list])
        }
        
        # 构建完整的特征结构
        merged['scene_features'] = {
            'scene_type': merged['scene_type'],
            'scene_confidence': np.mean([f.get('scene_features', {}).get('scene_confidence', 0.0) for f in features_list]),
            'scene_features': {
                'spatial': spatial_features,
                'visual': visual_features
            }
        }
        
        return merged
    
    def _preprocess_frame(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """预处理输入帧
        Args:
            frame: 原始输入帧
        Returns:
            处理后的帧或None（处理失败时）
        """
        try:
            # 确保帧格式正确
            if len(frame.shape) != 3:
                self.logger.error(f"无效的帧维度: {frame.shape}")
                return None
                
            # 统一尺寸
            target_size = (640, 480)  # 可以从配置中读取
            if frame.shape[:2] != target_size:
                frame = cv2.resize(frame, target_size)
                
            # 确保类型正确
            if frame.dtype != np.uint8:
                frame = frame.astype(np.uint8)
                
            return frame
            
        except Exception as e:
            self.logger.error(f"帧预处理失败: {str(e)}")
            return None

    def process(self, frames: List[np.ndarray]) -> List[Dict]:
        """处理帧"""
        try:
            results = []
            for frame in frames:
                # 分析帧
                result = self.analyzers[self.analyzer_type].analyze(frame)
                if result:
                    # 分析器已经返回了正确的格式，直接添加到结果列表
                    results.append(result)

            return results
            
        except Exception as e:
            self.logger.error(f"处理失败: {str(e)}")
            return []

    def _get_analyzer(self) -> BaseAnalyzer:
        """获取当前分析器"""
        return self.analyzers[self.analyzer_type]
        
    def get_results(self) -> Dict:
        """获取分析结果"""
        return self.result_processor.get_results() 
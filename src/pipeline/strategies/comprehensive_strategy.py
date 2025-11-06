from .base_strategy import BaseStrategy
from typing import Dict, Any, Optional, List, Union
import numpy as np
import logging
import time
import cv2
import hashlib
import torch
from ..validators.data_validator import DataValidator
from collections import Counter
from ..utils.result_processor import ResultProcessor
from ..utils.suggestion_generator import SuggestionGenerator
from ..analyzers.base_analyzer import BaseAnalyzer
from ..analyzers.scene_analyzer import SceneAnalyzer
from ..analyzers.furniture_detector import FurnitureDetector
from ..analyzers.lighting_analyzer import LightingAnalyzer
from ..analyzers.style_analyzer import StyleAnalyzer
from ..analyzers.color_analyzer import ColorAnalyzer
from concurrent.futures import ThreadPoolExecutor

class ComprehensiveStrategy(BaseStrategy):
    """综合分析策略 - 使用所有分析器进行分析"""
    
    def __init__(self, analyzers: Dict[str, BaseAnalyzer]):
        """初始化综合分析策略
        Args:
            analyzers: 分析器字典，可选
        """
        super().__init__(analyzers)
        self.logger = logging.getLogger(__name__)
        
        # 定义分析器执行顺序
        self.execution_order = [
            'scene',
            'furniture',
            'lighting',
            'color',
            'style'
        ]
        
        # 只使用已初始化的分析器
        self.execution_order = [name for name in self.execution_order if name in analyzers]
        if not self.execution_order:
            raise ValueError("没有可用的分析器")
        
    def process(self, frames: List[np.ndarray]) -> Dict:
        try:
            frame_results = []
            for frame in frames:
                results = {}
                for name, analyzer in self.analyzers.items():
                    result = analyzer.analyze(frame)
                    if result:
                        results[name] = result
                if results:
                    frame_results.append(results)
                    
            return {
                'type': 'comprehensive_analysis',
                'results': frame_results,
                'metadata': {
                    'frame_count': len(frame_results)
                }
            }
            
        except Exception as e:
            self.logger.error(f"处理失败: {str(e)}")
            return {}
            
    def _get_empty_result(self) -> Dict[str, Any]:
        """返回空结果"""
        return {
            'type': 'comprehensive_analysis',
            'results': {},
            'metadata': {
                'error': True,
                'timestamp': time.time(),
                'strategy': self.__class__.__name__
            }
        }

    def _aggregate_results(self, results: List[Dict]) -> Dict:
        """聚合单个分析器的结果"""
        if not results:
            return {}
            
        # 获取第一个结果的类型和置信度
        result_type = results[0].get('type', 'unknown')
        confidence = np.mean([r.get('confidence', 0.0) for r in results])
        
        # 聚合特征
        features = {}
        for result in results:
            result_features = result.get('features', {})
            for key, value in result_features.items():
                if key not in features:
                    features[key] = []
                features[key].append(value)
        
        # 计算特征的平均值或众数
        aggregated_features = {}
        for key, values in features.items():
            if isinstance(values[0], (int, float)):
                aggregated_features[key] = float(np.mean(values))
            elif isinstance(values[0], str):
                aggregated_features[key] = Counter(values).most_common(1)[0][0]
            elif isinstance(values[0], dict):
                # 递归聚合嵌套字典
                aggregated_features[key] = self._aggregate_dict_values(values)
        
        return {
            'type': result_type,
            'confidence': float(confidence),
            'features': aggregated_features
        }
    
    def _aggregate_dict_values(self, dict_list: List[Dict]) -> Dict:
        """聚合字典值"""
        if not dict_list:
            return {}
            
        result = {}
        keys = dict_list[0].keys()
        
        for key in keys:
            values = [d.get(key) for d in dict_list if key in d]
            if not values:
                continue
                
            if isinstance(values[0], (int, float)):
                result[key] = float(np.mean(values))
            elif isinstance(values[0], str):
                result[key] = Counter(values).most_common(1)[0][0]
            elif isinstance(values[0], dict):
                result[key] = self._aggregate_dict_values(values)
                
        return result

    def _get_cache_key(self, frame: np.ndarray, prefix: str = '') -> str:
        """生成缓存键"""
        frame_key = hashlib.md5(frame[::20,::20].tobytes()).hexdigest()
        return f"{prefix}_{frame_key}" if prefix else frame_key

    def get_analysis_config(self) -> Dict:
        """获取完整分析配置"""
        return {
            'scene': True,
            'furniture': True,
            'lighting': True,
            'style': True,
            'color': True
        }

    def _prepare_input(self, frame: np.ndarray) -> torch.Tensor:
        """准备输入数据"""
        try:
            tensor = torch.from_numpy(frame).float().permute(2, 0, 1).unsqueeze(0)
            return tensor.to(self.device)
        except Exception as e:
            self.logger.error(f"输入准备失败: {str(e)}")
            return None

    def _execute_analyzer(self, name: str, tensor: torch.Tensor, frame: np.ndarray) -> Optional[Dict]:
        """执行单个分析器"""
        try:
            analyzer = self.analyzers.get(name)
            if analyzer is None:
                return None
                
            # 根据分析器类型选择输入
            if name in ['scene', 'style']:
                return analyzer.analyze(tensor)
            else:
                numpy_frame = tensor.cpu().numpy()[0].transpose(1, 2, 0)
                return analyzer.analyze(numpy_frame)
                
        except Exception as e:
            self.logger.error(f"{name} 分析失败: {str(e)}")
            return None

    def _get_metadata(self) -> Dict:
        """获取元数据"""
        return {
            'strategy': self.__class__.__name__,
            'timestamp': time.time(),
            'device': str(self.device),
            'analyzers': list(self.analyzers.keys())
        }

    def _analyze_scenes(self, scene_results: List[Dict]) -> Dict:
        """分析场景结果"""
        try:
            if not scene_results:
                return {
                    'type': 'scene_analysis',
                    'confidence': 0.0,
                    'features': {'scene_type': '未识别'}
                }
            
            # 添加置信度阈值
            CONFIDENCE_THRESHOLD = 0.3
            
            # 过滤低置信度结果
            valid_results = [
                r for r in scene_results 
                if isinstance(r, dict) and r.get('confidence', 0) > CONFIDENCE_THRESHOLD
            ]
            
            if not valid_results:
                return {
                    'type': 'scene_analysis',
                    'confidence': 0.0,
                    'features': {'scene_type': '未识别'}
                }
            
            # 从features中获取场景类型
            scene_types = []
            for result in valid_results:
                features = result.get('features', {})
                if isinstance(features, dict):
                    scene_type = features.get('scene_type', '未知')
                    scene_types.append(scene_type)
            
            if not scene_types:
                return {
                    'type': 'scene_analysis',
                    'confidence': 0.0,
                    'features': {'scene_type': '未识别'}
                }
            
            most_common = Counter(scene_types).most_common(1)[0]
            scene_type = most_common[0]
            confidence = most_common[1] / len(scene_types)
            
            return {
                'type': 'scene_analysis',
                'confidence': confidence,
                'features': {
                    'scene_type': scene_type,
                    'space_features': {}
                }
            }
            
        except Exception as e:
            self.logger.error(f"场景分析失败: {str(e)}")
            return {
                'type': 'scene_analysis',
                'confidence': 0.0,
                'features': {'scene_type': '未识别'}
            }

    def _analyze_styles(self, style_results: List[Dict]) -> Dict:
        """分析风格结果"""
        try:
            if not style_results:
                return {'style_type': '未知'}
            
            # 统计风格类型
            style_types = [result.get('style_type', '未知') for result in style_results]
            most_common = Counter(style_types).most_common(1)[0]
            style_type = most_common[0]
            confidence = most_common[1] / len(style_results)
            
            # 计算平均特征值
            features = {
                'color_tone': np.mean([r.get('color_tone', 0) for r in style_results]),
                'texture': np.mean([r.get('texture', 0) for r in style_results]),
                'pattern': np.mean([r.get('pattern', 0) for r in style_results])
            }
            
            return {
                'style_type': style_type,
                'confidence': confidence,
                'features': features
            }
            
        except Exception as e:
            self.logger.error(f"风格分析失败: {str(e)}")
            return {'style_type': '未知'}

    def _analyze_colors(self, color_results: List[Dict]) -> Dict:
        """分析颜色结果"""
        try:
            if not color_results:
                return {'main_colors': {}}
            
            # 合并所有颜色结果
            all_colors = {}
            for result in color_results:
                colors = result.get('main_colors', {})
                for color, ratio in colors.items():
                    if color not in all_colors:
                        all_colors[color] = []
                    all_colors[color].append(ratio)
            
            # 计算平均比例
            avg_colors = {
                color: float(np.mean(ratios))
                for color, ratios in all_colors.items()
            }
            
            return {
                'main_colors': avg_colors,
                'avg_hue': np.mean([r.get('avg_hue', 0) for r in color_results]),
                'avg_saturation': np.mean([r.get('avg_saturation', 0) for r in color_results]),
                'avg_value': np.mean([r.get('avg_value', 0) for r in color_results])
            }
            
        except Exception as e:
            self.logger.error(f"颜色分析失败: {str(e)}")
            return {'main_colors': {}}

    def _analyze_lighting(self, lighting_results: List[Dict]) -> Dict:
        """分析光照结果"""
        try:
            if not lighting_results:
                return {'brightness': 0}
            
            # 计算平均光照指标
            brightness = np.mean([r.get('features', {}).get('brightness', 0) for r in lighting_results])
            uniformity = np.mean([r.get('features', {}).get('uniformity', 0) for r in lighting_results])
            contrast = np.mean([r.get('features', {}).get('contrast', 0) for r in lighting_results])
            
            return {
                'type': 'lighting_analysis',
                'confidence': 1.0,
                'features': {
                    'brightness': float(brightness),
                    'uniformity': float(uniformity),
                    'contrast': float(contrast)
                }
            }
            
        except Exception as e:
            self.logger.error(f"光照分析失败: {str(e)}")
            return {'brightness': 0}

    def _generate_summary(self, scene, style, color, lighting) -> Dict:
        """生成分析总结"""
        try:
            return {
                'scene_type': scene.get('features', {}).get('scene_type', '未知'),
                'style_type': style.get('features', {}).get('style_type', '未知'),
                'main_color': max(color.get('main_colors', {}).items(), key=lambda x: x[1])[0] if color.get('main_colors') else '未知',
                'brightness': f"{lighting.get('features', {}).get('brightness', 0):.2f}"
            }
        except Exception as e:
            self.logger.error(f"总结生成失败: {str(e)}")
            return {
                'scene_type': '未知',
                'style_type': '未知',
                'main_color': '未知',
                'brightness': '0.00'
            }

    def _extract_results(self, frame_results: List[Dict], result_type: str) -> List[Dict]:
        """提取特定类型的分析结果"""
        try:
            results = []
            for frame in frame_results:
                # 从 frame['results'] 中获取结果
                if frame.get('type') == 'frame_analysis' and 'results' in frame:
                    result = frame['results'].get(result_type)
                    if result:
                        results.append(result)
                    
            self.logger.debug(f"提取到 {len(results)} 个 {result_type} 结果")
            return results
            
        except Exception as e:
            self.logger.error(f"结果提取失败: {str(e)}")
            return []

    def _calculate_confidence(self, results: List[Dict]) -> float:
        """计算置信度"""
        if not results:
            return 0.0
        return float(np.mean([r.get('confidence', 0.0) for r in results]))

    def _get_majority_scene_type(self, scene_results: List[Dict]) -> str:
        """获取主要场景类型"""
        if not scene_results:
            return '未知'
        scene_types = [r.get('features', {}).get('scene_type', '未知') for r in scene_results]
        return Counter(scene_types).most_common(1)[0][0]

    def _aggregate_lighting_features(self, lighting_results: List[Dict]) -> Dict:
        """聚合光照特征"""
        if not lighting_results:
            return {'brightness': 0.0, 'uniformity': 0.0, 'contrast': 0.0}
        
        features = {}
        for key in ['brightness', 'uniformity', 'contrast']:
            values = [r.get('features', {}).get(key, 0.0) for r in lighting_results]
            # 归一化到 [0,1] 范围
            if key == 'brightness':
                values = [min(1.0, v * 100) for v in values]  # 亮度值放大100倍
            elif key == 'contrast':
                values = [min(1.0, v * 200) for v in values]  # 对比度值放大200倍
            features[key] = float(np.mean(values))
        
        return features

    def _aggregate_style_features(self, style_results: List[Dict]) -> Dict:
        """聚合风格特征"""
        if not style_results:
            return {'style_type': '未知', 'color_tone': 0.0, 'texture': 0.0, 'pattern': 0.0}
        
        features = {}
        for key in ['color_tone', 'texture', 'pattern']:
            values = [r.get('features', {}).get(key, 0.0) for r in style_results]
            features[key] = float(np.mean(values))
        
        # 获取主要风格类型
        style_types = [r.get('features', {}).get('style_type', '未知') for r in style_results]
        features['style_type'] = Counter(style_types).most_common(1)[0][0]
        return features

    def _aggregate_color_features(self, color_results: List[Dict]) -> Dict:
        """聚合颜色特征"""
        if not color_results:
            return {'main_colors': {}, 'avg_saturation': 0.0}
        
        # 合并所有颜色结果
        all_colors = {}
        for result in color_results:
            colors = result.get('features', {}).get('main_colors', {})
            for color, ratio in colors.items():
                if color not in all_colors:
                    all_colors[color] = []
                all_colors[color].append(ratio)
        
        # 计算平均比例
        main_colors = {
            color: float(np.mean(ratios))
            for color, ratios in all_colors.items()
        }
        
        # 计算平均饱和度
        avg_saturation = np.mean([
            r.get('features', {}).get('avg_saturation', 0.0) 
            for r in color_results
        ])
        
        return {
            'main_colors': main_colors,
            'avg_saturation': float(avg_saturation)
        }

    def _aggregate_space_features(self, scene_results: List[Dict]) -> Dict:
        """聚合空间特征"""
        if not scene_results:
            return {}
        
        features = {}
        # 面积和比例
        areas = [r.get('features', {}).get('details', {}).get('space_features', {}).get('area', 0) for r in scene_results]
        ratios = [r.get('features', {}).get('details', {}).get('space_features', {}).get('aspect_ratio', 1.0) for r in scene_results]
        
        features['avg_area'] = float(np.mean(areas))
        features['avg_aspect_ratio'] = float(np.mean(ratios))
        
        return features

    def _analyze_spatial_layout(self, scene_results: List[Dict]) -> Dict:
        """分析空间布局"""
        if not scene_results:
            return {}
        
        layouts = []
        for result in scene_results:
            layout = result.get('features', {}).get('details', {}).get('spatial_layout', {})
            if layout:
                layouts.append(layout)
        
        if not layouts:
            return {}
        
        # 统计最常见的布局特征
        layout_features = {}
        for key in ['symmetry', 'openness', 'complexity']:
            values = [layout.get(key, 'unknown') for layout in layouts]
            if values:
                layout_features[key] = Counter(values).most_common(1)[0][0]
            
        return layout_features

    def _get_top_colors(self, main_colors: Dict[str, float], top_n: int = 3) -> List[Dict[str, Any]]:
        """获取主要颜色"""
        if not main_colors:
            return []
        
        sorted_colors = sorted(main_colors.items(), key=lambda x: x[1], reverse=True)
        return [
            {'color': color, 'ratio': ratio}
            for color, ratio in sorted_colors[:top_n]
        ]

    def _evaluate_lighting_quality(self, lighting_features: Dict) -> Dict[str, Any]:
        """评估光照质量"""
        brightness = lighting_features.get('brightness', 0)
        uniformity = lighting_features.get('uniformity', 0)
        contrast = lighting_features.get('contrast', 0)
        
        # 计算综合评分
        score = (brightness * 0.4 + uniformity * 0.4 + contrast * 0.2) * 10
        
        return {
            'score': float(score),
            'brightness_level': 'low' if brightness < 0.3 else 'medium' if brightness < 0.7 else 'high',
            'uniformity_level': 'poor' if uniformity < 0.5 else 'good' if uniformity < 0.8 else 'excellent',
            'contrast_level': 'low' if contrast < 0.2 else 'medium' if contrast < 0.5 else 'high'
        }

    def _calculate_overall_score(self, analysis_results: Dict) -> float:
        """计算总体评分"""
        try:
            # 各部分权重
            weights = {
                'lighting': 0.3,
                'style': 0.3,
                'color': 0.2,
                'space': 0.2
            }
            
            scores = {
                'lighting': float(analysis_results['lighting']['features'].get('brightness', 0)) * 10,
                'style': float(analysis_results['style']['confidence']),
                'color': float(analysis_results['color']['features'].get('avg_saturation', 0)) / 255 * 10,
                'space': float(analysis_results['scene']['confidence']) * 10
            }
            
            overall_score = sum(score * weights[key] for key, score in scores.items())
            return round(float(overall_score), 2)
            
        except Exception as e:
            self.logger.error(f"总体评分计算失败: {str(e)}")
            return 0.0

    def get_analyzer_names(self) -> List[str]:
        """获取分析器名称列表"""
        return list(self.analyzers.keys())

    def execute(self, frame: np.ndarray) -> Dict:
        """执行综合分析"""
        try:
            self.logger.debug("开始执行综合分析...")
            results = {}
            
            # 按顺序执行分析器
            for analyzer_name in self.execution_order:
                self.logger.debug(f"执行 {analyzer_name} 分析...")
                analyzer = self.analyzers.get(analyzer_name)
                if analyzer:
                    result = analyzer.analyze(frame)
                    self.logger.debug(f"{analyzer_name} 分析结果: {result}")
                    results[analyzer_name] = result
                    
            self.logger.debug("综合分析完成")
            return results
            
        except Exception as e:
            self.logger.error(f"综合分析失败: {str(e)}")
            return {}
        
    def analyze_video(self, frames: List[np.ndarray]) -> Dict:
        """分析视频帧序列"""
        try:
            self.logger.debug(f"开始综合分析视频, 共 {len(frames)} 帧")
            frame_results = []
            
            for i, frame in enumerate(frames):
                self.logger.debug(f"分析第 {i+1}/{len(frames)} 帧")
                frame_result = {}
                
                for analyzer_name in self.execution_order:
                    analyzer = self.analyzers.get(analyzer_name)
                    self.logger.debug(f"使用分析器 {analyzer_name}: {analyzer is not None}")
                    if analyzer:
                        try:
                            result = analyzer.analyze(frame)
                            self.logger.debug(f"{analyzer_name} 分析结果: {result}")
                            if isinstance(result, dict) and result.get('confidence') is not None:
                                frame_result[analyzer_name] = result
                        except Exception as e:
                            self.logger.error(f"{analyzer_name} 分析失败: {str(e)}")
                            
                if frame_result:
                    frame_results.append(frame_result)
                    
            return {
                'type': 'comprehensive_analysis',
                'analyzer': 'comprehensive',
                'results': {
                    'frame_results': frame_results,
                    'metadata': {
                        'frame_count': len(frame_results),
                        'total_frames': len(frames),
                        'timestamp': time.time()
                    }
                }
            }
            
        except Exception as e:
            self.logger.error(f"视频分析失败: {str(e)}")
            return None 

    def select_analyzers(self):
        return ['scene', 'furniture', 'lighting', 'color', 'style']

    def analyze_frames(self, frames):
        with ThreadPoolExecutor() as executor:
            results = list(executor.map(self.analyze_single_frame, frames))
        return results

class SingleAnalysisStrategy(BaseStrategy):
    def __init__(self, analyzers: Dict, analyzer_name: str):
        super().__init__(analyzers)
        self.target_analyzer = analyzer_name
        
    def select_analyzers(self):
        return [self.target_analyzer] 
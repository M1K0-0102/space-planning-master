from typing import Dict, List, Any, Optional, Tuple, Union
import logging
from .result_types import (
    AnalysisResult,
    ImageAnalysisResult,
    VideoAnalysisResult,
    SceneResult,
    FurnitureResult,
    LightingResult,
    StyleResult,
    ColorResult,
    RawAnalysisResult,
    ProcessedResult,
    AnalyzerResult,
    FrameResult
)
from .suggestion_generator import SuggestionGenerator
from .suggestion_formatter import SuggestionFormatter
import numpy as np
from collections import Counter, defaultdict
from .result_collector import ResultCollector, CollectedResults
import time
from dataclasses import dataclass, field
import copy
import json

class ResultProcessor:
    """结果处理器 - 处理和格式化分析结果"""
    
    def __init__(self):
        self.logger = logging.getLogger("pipeline.ResultProcessor")
        self.suggestion_generator = SuggestionGenerator()
    
    def process(self, collected_results: CollectedResults) -> ProcessedResult:
        """处理收集的结果"""
        try:
            self.logger.info("开始处理分析结果...")
            
            # 1. 聚合结果
            aggregated = {}
            for analyzer_type, results in collected_results.analyzer_results.items():
                self.logger.debug(f"聚合 {analyzer_type} 结果")
                # 确保结果是列表
                if isinstance(results, list):
                    aggregated[analyzer_type] = self._aggregate_results(results)
                else:
                    aggregated[analyzer_type] = self._process_single_result(results)
            
            # 2. 验证结果
            self.logger.debug("验证分析结果")
            if not self._validate_results(aggregated):
                raise ValueError("结果验证失败")
            
            # 3. 返回处理后的结果
            processed = ProcessedResult(
                success=True,
                results=aggregated
            )
            
            self.logger.info("结果处理完成")
            return processed
            
        except Exception as e:
            self.logger.error(f"结果处理失败: {str(e)}")
            return ProcessedResult.error_result(str(e))

    def _process_single_result(self, result: Any) -> Dict:
        """处理单个分析器的结果"""
        try:
            # 处理字典类型结果
            if isinstance(result, dict):
                return {
                    'confidence': result.get('confidence', 0.0),
                    'features': result.get('features', {}),
                    'metadata': result.get('metadata', {})
                }
            # 处理对象类型结果
            else:
                return {
                    'confidence': getattr(result, 'confidence', 0.0),
                    'features': getattr(result, 'features', {}),
                    'metadata': getattr(result, 'metadata', {})
                }
        except Exception as e:
            self.logger.error(f"处理结果失败: {str(e)}")
            return {}

    def process_image_result(self, result: ImageAnalysisResult) -> Optional[Dict[str, Any]]:
        """处理图像分析结果"""
        try:
            if not result:
                return None
            
            # 直接处理图像分析结果
            processed_result = {
                'scene': self._process_scene_result(result.scene),
                'furniture': self._process_furniture_result(result.furniture),
                'lighting': self._process_lighting_result(result.lighting),
                'style': self._process_style_result(result.style),
                'color': self._process_color_result(result.color),
                'suggestions': result.suggestions,
                'metadata': {
                    'timestamp': result.timestamp.isoformat()
                }
            }
            
            return processed_result
            
        except Exception as e:
            self.logger.error(f"图像结果处理失败: {str(e)}")
            return None

    def process_video_result(self, result: VideoAnalysisResult) -> Optional[Dict[str, Any]]:
        """处理视频分析结果"""
        try:
            if not result:
                return None
            
            # 处理每一帧的结果
            processed_frames = []
            for frame_result in result.frames:
                processed_frame = self.process_image_result(frame_result)
                if processed_frame:
                    processed_frames.append(processed_frame)
            
            # 聚合视频结果
            aggregated_result = self._aggregate_video_results(processed_frames)
            
            return {
                'frames': processed_frames,
                'summary': aggregated_result['summary'],
                'suggestions': result.suggestions,
                'metadata': {
                    **result.metadata,
                    'frame_count': len(processed_frames),
                    'timestamp': result.timestamp.isoformat()
                }
            }
            
        except Exception as e:
            self.logger.error(f"视频结果处理失败: {str(e)}")
            return None

    def _aggregate_video_results(self, processed_frames: List[Dict[str, Any]]) -> Dict:
        """聚合视频分析结果"""
        try:
            self.logger.debug(f"开始聚合 {len(processed_frames)} 帧结果")
            total_confidence = 0
            frame_count = len(processed_frames)

            for frame in processed_frames:
                total_confidence += frame.get('confidence', 0)

            avg_confidence = total_confidence / frame_count if frame_count > 0 else 0
            
            return {
                'summary': {
                    'average_confidence': avg_confidence
                }
            }
            
        except Exception as e:
            self.logger.error(f"视频结果聚合失败: {str(e)}")
            return {}

    def _process_analyzer_result(self, result: Dict, analyzer_name: str) -> AnalyzerResult:
        """处理单个分析器的结果"""
        try:
            return AnalyzerResult(
                type=analyzer_name,
                confidence=result.get('confidence', 0.0),
                features=result.get('features', {}),
                metadata=result.get('metadata', {})
            )
        except Exception as e:
            self.logger.error(f"处理{analyzer_name}结果失败: {str(e)}")
            return self._get_empty_analyzer_result(analyzer_name)

    def _get_empty_analyzer_result(self, analyzer_name: str) -> AnalyzerResult:
        """获取空的分析器结果"""
        return AnalyzerResult(
            type=analyzer_name,
            confidence=0.0,
            features={},
            metadata={}
        )
        
    def _process_scene_result(self, result: Optional[SceneResult]) -> Dict:
        """处理场景分析结果"""
        if not result:
            return {}
            
        return {
            'type': result.type,
            'scene_type': result.scene_type,
            'confidence': result.confidence,
            'features': {
                'spatial': result.spatial_features,
                'lighting': result.lighting_features,
                'texture': result.texture_features
            }
        }
        
    def _process_furniture_result(self, result: Optional[FurnitureResult]) -> Dict:
        """处理家具分析结果"""
        if not result:
            return {}
        
        # 处理家具检测结果
        processed_detections = {}
        for furniture_type, stats in result.detections.items():
            confidence = stats.get('confidence', 0.0)
            count = stats.get('count', 0.0)
            
            # 根据置信度调整数量
            if confidence > 0.3:  # 只有置信度超过30%才考虑
                # 对数量进行取整处理
                adjusted_count = round(count)  # 四舍五入到整数
                if adjusted_count > 0:  # 只保留有效数量
                    processed_detections[furniture_type] = {
                        'count': adjusted_count,
                        'confidence': confidence
                    }
        
        return {
            'type': result.type,
            'confidence': result.confidence,
            'detections': processed_detections,
            'layout': result.layout,
            'count': sum(item['count'] for item in processed_detections.values())
        }
        
    def _process_lighting_result(self, result: Optional[LightingResult]) -> Dict:
        """处理光照分析结果"""
        if not result:
            return {}
            
        return {
            'type': result.type,
            'confidence': result.confidence,
            'brightness': result.brightness,
            'uniformity': result.uniformity,
            'contrast': result.contrast,
            'light_sources': result.light_sources,
            'quality_score': result.quality_score
        }
        
    def _process_style_result(self, result: Optional[StyleResult]) -> Dict:
        """处理风格分析结果"""
        if not result:
            return {}
            
        return {
            'type': result.type,
            'confidence': result.confidence,
            'main_style': result.main_style,
            'style_probabilities': result.style_probabilities,
            'style_features': result.style_features
        }
        
    def _process_color_result(self, result: Optional[ColorResult]) -> Dict:
        """处理颜色分析结果"""
        if not result:
            return {}
            
        return {
            'type': result.type,
            'confidence': result.confidence,
            'dominant_colors': result.dominant_colors,
            'color_scheme': result.color_scheme,
            'harmony_score': result.harmony_score
        }
        
    def _get_empty_formatted_result(self) -> Dict:
        """返回格式化的空结果"""
        return {
            'analysis': {
                'scene': {},
                'furniture': {},
                'lighting': {},
                'style': {},
                'color': {}
            },
            'suggestions': [],
            'metadata': {
                'timestamp': None,
                'version': '1.0.0'
            }
        }

    def _analyze_scene_results(self, scene_result: Dict) -> Dict:
        """分析场景结果"""
        try:
            return {
                'type': 'scene_analysis',
                'confidence': scene_result.get('confidence', 0.0),
                'features': {
                    'scene_type': scene_result.get('scene_type', '未知'),
                    'space_features': scene_result.get('space_features', {})
                }
            }
        except Exception as e:
            self.logger.error(f"场景结果分析失败: {str(e)}")
            return self._get_empty_result('scene_analysis')
            
    def _analyze_style_results(self, style_result: Dict) -> Dict:
        """分析风格结果"""
        try:
            return {
                'type': 'style_analysis',
                'confidence': style_result.get('confidence', 0.0),
                'features': {
                    'style_type': style_result.get('style_type', '未知'),
                    'features': style_result.get('features', {})
                }
            }
        except Exception as e:
            self.logger.error(f"风格结果分析失败: {str(e)}")
            return self._get_empty_result('style_analysis')
            
    def _analyze_color_results(self, color_result: Dict) -> Dict:
        """分析颜色结果"""
        try:
            return {
                'type': 'color_analysis',
                'confidence': 1.0,  # 颜色分析不涉及置信度
                'features': {
                    'main_colors': color_result.get('main_colors', {}),
                    'avg_hue': color_result.get('avg_hue', 0),
                    'avg_saturation': color_result.get('avg_saturation', 0),
                    'avg_value': color_result.get('avg_value', 0)
                }
            }
        except Exception as e:
            self.logger.error(f"颜色结果分析失败: {str(e)}")
            return self._get_empty_result('color_analysis')
            
    def _analyze_lighting_results(self, lighting_result: Dict) -> Dict:
        """分析光照结果"""
        try:
            if not lighting_result:
                return self._get_empty_result('lighting_analysis')
            
            # 获取特征，注意层级结构
            features = lighting_result.get('features', {})
            if not features:  # 如果features为空，尝试直接从结果中获取
                features = {
                    'brightness': lighting_result.get('brightness', 0.0),
                    'uniformity': lighting_result.get('uniformity', 0.0),
                    'contrast': lighting_result.get('contrast', 0.0)
                }
            
            return {
                'type': 'lighting_analysis',
                'confidence': lighting_result.get('confidence', 1.0),
                'features': {
                    'brightness': float(features.get('brightness', 0.0)),
                    'uniformity': float(features.get('uniformity', 0.0)),
                    'contrast': float(features.get('contrast', 0.0))
                }
            }
            
        except Exception as e:
            self.logger.error(f"光照结果分析失败: {str(e)}")
            return self._get_empty_result('lighting_analysis')
            
    def _get_empty_result(self, analysis_type: str) -> Dict:
        """获取空结果模板"""
        return {
            'type': analysis_type,
            'confidence': 0.0,
            'features': {}
        }

    def _process_scene(self, scene_result: RawAnalysisResult) -> RawAnalysisResult:
        """处理场景分析结果"""
        if not scene_result:
            return RawAnalysisResult(
                type='scene_analysis',
                confidence=0.0,
                features={'scene_type': '未知'}
            )
        return scene_result

    def _process_lighting(self, lighting_result: RawAnalysisResult) -> RawAnalysisResult:
        """处理光照分析结果"""
        if not lighting_result:
            return RawAnalysisResult(
                type='lighting_analysis',
                confidence=0.0,
                features={'brightness': 0.0, 'uniformity': 0.0, 'contrast': 0.0}
            )
        return lighting_result

    def _process_style(self, style_result: RawAnalysisResult) -> RawAnalysisResult:
        """处理风格分析结果"""
        if not style_result:
            return RawAnalysisResult(
                type='style_analysis',
                confidence=0.0,
                features={'style_type': '未知'}
            )
        return style_result

    def _process_color(self, color_result: RawAnalysisResult) -> RawAnalysisResult:
        """处理颜色分析结果"""
        if not color_result:
            return RawAnalysisResult(
                type='color_analysis',
                confidence=0.0,
                features={'main_colors': {}}
            )
        return color_result

    def _integrate_results(self, frame_results: List[Dict]) -> Dict:
        """整合所有帧的结果"""
        try:
            # 类型检查
            if not isinstance(frame_results, list):
                self.logger.error(f"期望列表输入，实际收到: {type(frame_results)}")
                return self._get_empty_results()
            
            # 按分析器类型分组
            analyzer_results = defaultdict(list)
            for frame_result in frame_results:
                # 确保每个结果是字典
                if not isinstance(frame_result, dict):
                    self.logger.error(f"期望字典结果，实际收到: {type(frame_result)}")
                    continue
                
                self.logger.debug(f"处理帧结果: {frame_result}")
                
                for analyzer_name, result in frame_result.items():
                    analyzer_results[analyzer_name].append(result)
            
            # 整合每个分析器的结果
            integrated = {}
            for analyzer_name, results in analyzer_results.items():
                if analyzer_name == 'scene_analyzer':
                    integrated[analyzer_name] = self._integrate_scene_results(results)
                elif analyzer_name == 'lighting_analyzer':
                    integrated[analyzer_name] = self._integrate_lighting_results(results)
                elif analyzer_name == 'style_analyzer':
                    integrated[analyzer_name] = self._integrate_style_results(results)
                elif analyzer_name == 'color_analyzer':
                    integrated[analyzer_name] = self._integrate_color_results(results)
                elif analyzer_name == 'furniture_detector':
                    integrated[analyzer_name] = self._integrate_furniture_results(results)
                
            return integrated
            
        except Exception as e:
            self.logger.error(f"结果整合失败: {str(e)}")
            return self._get_empty_results()

    def _integrate_scene_results(self, results: List[Dict]) -> Dict:
        """整合场景分析结果"""
        try:
            self.logger.debug(f"开始整合场景结果，输入: {results}")
            
            # 使用 result_collector 收集结果
            scene_result = self.result_collector.collect_scene_results(results)
            self.logger.debug(f"收集器返回结果: {scene_result}")
            
            # 确保特征数据完整性
            if 'features' in scene_result:
                features = scene_result['features']
                self.logger.debug(f"特征数据: {features}")
                
                if 'scene_features' in features:
                    scene_features = features['scene_features']
                    self.logger.debug(f"场景特征: {scene_features}")
                    
                    if 'scene_features' not in scene_features:
                        self.logger.debug("添加缺失的场景特征结构")
                        scene_features['scene_features'] = {
                            'spatial': scene_features.get('spatial', {}),
                            'visual': scene_features.get('visual', {})
                        }
            
            self.logger.debug(f"最终整合结果: {scene_result}")
            return scene_result
            
        except Exception as e:
            self.logger.error(f"场景结果整合失败: {str(e)}")
            return {}

    def _integrate_lighting_results(self, results: List[Dict]) -> Dict:
        """整合光照分析结果"""
        try:
            # 计算平均值
            brightness = np.mean([r.get('features', {}).get('brightness', 0) for r in results])
            uniformity = np.mean([r.get('features', {}).get('uniformity', 0) for r in results])
            contrast = np.mean([r.get('features', {}).get('contrast', 0) for r in results])
            
            return {
                'type': 'lighting_analysis',
                'confidence': 1.0,
                'features': {
                    'brightness': brightness,
                    'uniformity': uniformity,
                    'contrast': contrast,
                    'quality': self._evaluate_lighting_quality(brightness, uniformity, contrast)
                }
            }
        except Exception as e:
            self.logger.error(f"光照结果整合失败: {str(e)}")
            return {}

    def _integrate_style_results(self, results: List[Dict]) -> Dict:
        """整合风格分析结果"""
        try:
            # 选择最频繁的风格
            styles = [r.get('features', {}).get('style_type') for r in results]
            most_common = Counter(styles).most_common(1)
            style_type = most_common[0][0] if most_common else '未知'
            
            return {
                'type': 'style_analysis',
                'confidence': 1.0,
                'features': {
                    'style_type': style_type,
                    'elements': results[0].get('features', {}).get('elements', {})
                }
            }
        except Exception as e:
            self.logger.error(f"风格结果整合失败: {str(e)}")
            return {}

    def _integrate_color_results(self, results: List[Dict]) -> Dict:
        """整合颜色分析结果"""
        try:
            # 合并所有颜色结果
            all_colors = {}
            for result in results:
                colors = result.get('features', {}).get('main_colors', {})
                for color, ratio in colors.items():
                    all_colors[color] = all_colors.get(color, 0) + ratio
            
            # 计算平均比例
            if all_colors:
                total = sum(all_colors.values())
                normalized_colors = {k: v/total for k, v in all_colors.items()}
            else:
                normalized_colors = {}
            
            return {
                'type': 'color_analysis',
                'confidence': 1.0,
                'features': {
                    'main_colors': normalized_colors,
                    'avg_saturation': np.mean([
                        r.get('features', {}).get('avg_saturation', 0) 
                        for r in results
                    ])
                }
            }
        except Exception as e:
            self.logger.error(f"颜色结果整合失败: {str(e)}")
            return {}

    def _integrate_furniture_results(self, results: List[Dict]) -> Dict:
        """整合家具分析结果"""
        try:
            # 计算平均置信度
            avg_confidence = sum(r.get('confidence', 0) for r in results) / len(results)
            return {'avg_confidence': avg_confidence}
        except Exception as e:
            self.logger.error(f"家具结果整合失败: {str(e)}")
            return {}

    def _calculate_distance(self, p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
        """计算两点之间的欧氏距离"""
        return ((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)**0.5

    def _evaluate_lighting_quality(self, brightness: float, uniformity: float, contrast: float) -> str:
        """评估光照质量"""
        # 定义理想范围
        IDEAL_BRIGHTNESS = (0.4, 0.7)  # 理想亮度范围
        IDEAL_UNIFORMITY = (0.6, 0.9)  # 理想均匀性范围
        IDEAL_CONTRAST = (0.2, 0.5)    # 理想对比度范围
        
        # 计算各指标的得分
        brightness_score = self._score_in_range(brightness, IDEAL_BRIGHTNESS)
        uniformity_score = self._score_in_range(uniformity, IDEAL_UNIFORMITY)
        contrast_score = self._score_in_range(contrast, IDEAL_CONTRAST)
        
        # 计算总分
        total_score = (brightness_score + uniformity_score + contrast_score) / 3
        
        # 评估等级
        if total_score > 0.8:
            return "优秀"
        elif total_score > 0.6:
            return "良好"
        elif total_score > 0.4:
            return "一般"
        else:
            return "需改进"

    def _score_in_range(self, value: float, ideal_range: Tuple[float, float]) -> float:
        """计算值在理想范围内的得分"""
        min_val, max_val = ideal_range
        if value < min_val:
            return 1.0 - (min_val - value) / min_val
        elif value > max_val:
            return 1.0 - (value - max_val) / (1.0 - max_val)
        else:
            return 1.0

    def _get_analyzer_key(self, analyzer_type: str) -> str:
        """统一分析器类型命名"""
        analyzer_map = {
            'scene': 'scene_analyzer',
            'lighting': 'lighting_analyzer',
            'style': 'style_analyzer',
            'color': 'color_analyzer',
            'furniture': 'furniture_detector'
        }
        return analyzer_map.get(analyzer_type, analyzer_type)

    def _merge_results(self, results: List[AnalyzerResult]) -> AnalyzerResult:
        """统一的结果合并方法"""
        if not results:
            raise ValueError("No results to merge")
            
        # 获取最高置信度结果
        best_result = max(results, key=lambda x: x.confidence)
        
        return AnalyzerResult(
            type=best_result.type,
            confidence=best_result.confidence,
            features=best_result.features,
            metadata=self._create_metadata(results)
        )

    def _create_metadata(self, results: List) -> Dict:
        """统一的元数据生成"""
        return {
            'total_frames': len(results),
            'processed_frames': len(results),
            'timestamp': time.time()
        }

    def _create_error_result(self, error_msg: str) -> ProcessedResult:
        """统一的错误结果生成"""
        return ProcessedResult(
            type='error',
            analysis_results={},
            suggestions=[],
            metadata={'error': error_msg}
        )

    def _get_empty_results(self) -> Dict:
        """返回空结果结构"""
        return {
            'scene_analyzer': {
                'type': 'scene_analysis',
                'confidence': 0.0,
                'features': {},
                'metadata': {}
            },
            'lighting_analyzer': {
                'type': 'lighting_analysis',
                'confidence': 0.0,
                'features': {},
                'metadata': {}
            },
            # ... 其他分析器的空结果
        }

    def _merge_frame_results(self, frame_results: List[Dict]) -> Dict:
        """合并所有帧的分析结果"""
        try:
            # 按分析器类型分组
            merged = defaultdict(list)
            for result in frame_results:
                for analyzer_name, analyzer_result in result.items():
                    merged[analyzer_name].append(analyzer_result)
                
            # 统计每个分析器的结果
            final_results = {}
            for analyzer_name, results in merged.items():
                final_results[analyzer_name] = self._summarize_analyzer_results(results)
            
            return final_results
        except Exception as e:
            self.logger.error(f"结果合并失败: {str(e)}")
            return {}

    def _summarize_analyzer_results(self, analyzer_results: List[Dict]) -> Dict:
        """汇总分析器的结果"""
        if not analyzer_results:
            return {}
        
        # 获取最高置信度的结果
        best_result = max(analyzer_results, key=lambda x: x.get('confidence', 0))
        features = best_result.get('features', {})
        
        return {
            'type': best_result.get('type', ''),
            'confidence': best_result.get('confidence', 0),
            'features': {
                'brightness': features.get('brightness', 0),
                'contrast': features.get('contrast', 0),
                'uniformity': features.get('uniformity', 0),
                'quality': features.get('quality', '光照质量良好')
            },
            'metadata': {
                'frame_count': len(analyzer_results),
                'timestamp': time.time()
            }
        }

    def _generate_summary(self, analysis_results: Dict, suggestions: List[str]) -> str:
        """生成报告摘要"""
        summary = ["=== 空间分析报告 ===\n"]
        summary.append(f"分析时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        # 添加场景分析结果
        if 'scene_analyzer' in analysis_results:
            scene_result = analysis_results['scene_analyzer']
            features = scene_result.get('features', {})
            
            self.logger.debug(f"处理场景分析结果: {scene_result}")
            
            summary.append("\n场景分析结果:")
            summary.append(f"- 主要场景类型: {features.get('scene_type', '未知')}")
            summary.append(f"- 空间面积: {features.get('area', 0.0):.2f}㎡")
            
            scene_stats = features.get('scene_stats', {})
            summary.append("\n场景类型统计:")
            for scene_type, stats in scene_stats.items():
                if scene_type != '未知':
                    avg_conf = stats['total_confidence'] / stats['count']
                    summary.append(f"  - {scene_type}: {avg_conf:.2%}")
        
        # 添加家具分析结果
        if 'furnituredetector' in analysis_results:
            furniture_result = analysis_results['furnituredetector']
            features = furniture_result['features']
            
            summary.append("\n家具分析结果:")
            summary.append(f"- 检测到的家具类型: {len(features['furniture_types'])}种")
            for ftype in features['furniture_types']:
                summary.append(f"  - {ftype}")
            
            summary.append(f"\n空间布局分析:")
            layout = features['layout']
            summary.append(f"- 空间密度: {layout['density']:.2f}")
            summary.append(f"- 布局评分: {layout['layout_score']:.2f}")
        
        # 添加建议
        if suggestions:
            summary.append("\n改进建议:")
            for suggestion in suggestions:
                summary.append(f"- {suggestion}")
        
        return '\n'.join(summary)

    def _process_furniture_results(self, frame_results: List) -> AnalyzerResult:
        """处理家具检测器的结果"""
        furniture_stats = {}
        # ... (处理家具检测结果的代码)
        
    def _process_scene_results(self, frame_results: List) -> AnalyzerResult:
        """处理场景分析器的结果"""
        scene_stats = {}
        # ... (处理场景分析结果的代码)
        
    def _process_lighting_results(self, frame_results: List) -> AnalyzerResult:
        """处理光照分析器的结果"""
        lighting_stats = {}
        # ... (处理光照分析结果的代码)
        
    def _process_style_results(self, frame_results: List) -> AnalyzerResult:
        """处理风格分析器的结果"""
        style_stats = {}
        # ... (处理风格分析结果的代码)

    def _merge_analyzer_results(self, frame_results: List[Union[AnalyzerResult, Dict]], analyzer_name: str) -> AnalyzerResult:
        """合并单个分析器的多帧结果"""
        try:
            # 转换字典结果为AnalyzerResult
            converted = []
            for r in frame_results:
                if isinstance(r, Dict):
                    converted.append(AnalyzerResult(
                        result_type=analyzer_name,
                        confidence=r.get('confidence', 0.0),
                        features=r.get('features', {}),
                        metadata=r.get('metadata', {})
                    ))
                else:
                    converted.append(r)
            
            # 合并逻辑...
            return AnalyzerResult(
                result_type=analyzer_name,
                confidence=avg_confidence,
                features=merged_features,
                metadata=metadata
            )
        except Exception as e:
            self.logger.error(f"合并{analyzer_name}结果失败: {str(e)}")
            return AnalyzerResult(
                result_type=analyzer_name,
                confidence=0.0,
                features={},
                metadata={'error': str(e)}
            )

    def _merge_furniture_results(self, results):
        if not results:
            return {}
        avg_confidence = sum(r.get('confidence', 0) for r in results) / len(results)
        return {'avg_confidence': avg_confidence} 

    def process_frame_results(self, frame_results: List[Dict]) -> Dict:
        self.logger.debug(f"原始帧结果结构: {json.dumps(frame_results, indent=2)}")
        try:
            # 确保输入有效性
            if not isinstance(frame_results, dict):
                self.logger.error(f"无效的帧结果类型: {type(frame_results)}")
                return {}
            
            # 深度复制防止修改原始数据
            frame_results = copy.deepcopy(frame_results)
            processed = {}
            
            # 处理场景分析结果
            scene_result = frame_results.get('scene') or {}
            if scene_result:  # 仅处理有效结果
                processed['scene'] = {
                    'type': scene_result.get('type'),
                    'scene_type': scene_result.get('scene_type'),
                    'confidence': float(scene_result.get('confidence', 0))
                }
                
            # 处理家具检测结果
            furniture_result = frame_results.get('furniture') or {}
            if furniture_result:
                processed['furniture'] = {
                    'type': 'furniture',
                    'detected_items': [
                        {
                            'type': item.get('type', 'unknown'),
                            'confidence': float(item.get('confidence', 0.0)),
                            'box': list(item.get('box', []))
                        }
                        for item in furniture_result.get('detected_items') or []
                    ]
                }
                
            # 处理光照分析结果
            lighting_result = frame_results.get('lighting') or {}
            if lighting_result:
                processed['lighting'] = {
                    'type': 'lighting',
                    'overall_brightness': float(lighting_result.get('overall_brightness', 0.0)),
                    'uniformity': float(lighting_result.get('uniformity', 0.0))
                }
                
            # 处理风格分析结果
            style_result = frame_results.get('style') or {}
            if style_result:
                processed['style'] = {
                    'type': 'style',
                    'style_type': style_result.get('style_type', 'unknown'),
                    'confidence': float(style_result.get('confidence', 0.0)),
                    'consistency_score': float(style_result.get('consistency_score', 0.0))
                }
                
            self.logger.debug(f"处理后的帧结果: {json.dumps(processed, indent=2)}")
            return processed
            
        except Exception as e:
            self.logger.error(f"处理帧结果失败: {str(e)}")
            return {}

    def process_results(self, results: Union[List[AnalyzerResult], Dict[str, AnalyzerResult]]) -> ProcessedResult:
        try:
            # 判断是否为视频分析结果(列表类型)
            if isinstance(results, list):
                return self._process_video_results(results)
            # 图片分析结果(字典类型)
            elif isinstance(results, dict):
                return self._process_image_results(results)
            else:
                raise ValueError("不支持的结果类型")
                
        except Exception as e:
            self.logger.error(f"处理结果失败: {str(e)}")
            return ProcessedResult.error_result(str(e))

    def _process_image_results(self, results: Dict[str, AnalyzerResult]) -> ProcessedResult:
        """处理单张图片的分析结果"""
        try:
            processed_results = {}
            
            # 直接处理各个分析器的结果
            for analyzer_type, result in results.items():
                if not result:
                    continue
                    
                processed_results[analyzer_type] = {
                    'confidence': getattr(result, 'confidence', 0.0),
                    'features': getattr(result, 'features', {}),
                    'metadata': getattr(result, 'metadata', {})
                }
            
            if not processed_results:
                return ProcessedResult.error_result("没有有效的分析结果")
                
            return ProcessedResult(
                success=True,
                results=processed_results
            )
            
        except Exception as e:
            self.logger.error(f"图片结果处理失败: {str(e)}")
            return ProcessedResult.error_result(str(e))
            
    def _process_video_results(self, results: List[Dict]) -> ProcessedResult:
        """处理视频的分析结果"""
        try:
            processed_results = {}
            
            # 按分析器类型分组
            analyzer_groups = {}
            for result in results:  # 每个结果已经包含了 type 字段
                analyzer_type = result.get('type')
                if analyzer_type not in analyzer_groups:
                    analyzer_groups[analyzer_type] = []
                analyzer_groups[analyzer_type].append(result)
            
            # 处理每种类型的结果
            for analyzer_type, group_results in analyzer_groups.items():
                if not group_results:
                    continue
                    
                # 计算平均置信度
                confidences = [r.get('confidence', 0.0) for r in group_results]
                avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
                
                # 合并特征
                merged_features = self._merge_features_by_type(analyzer_type, group_results)
                
                # 使用最后一帧的元数据
                metadata = group_results[-1].get('metadata', {})
                
                processed_results[analyzer_type] = {
                    'confidence': avg_confidence,
                    'features': merged_features,
                    'metadata': metadata
                }
            
            if not processed_results:
                return ProcessedResult.error_result("没有有效的分析结果")
                
            return ProcessedResult(
                success=True,
                results=processed_results
            )
            
        except Exception as e:
            self.logger.error(f"视频结果处理失败: {str(e)}")
            return ProcessedResult.error_result(str(e))

    def _merge_features_by_type(self, analyzer_type: str, results: List[Any]) -> Dict:
        """根据分析器类型合并特征"""
        if not results:
            return {}
        
        # 获取特征
        def get_features(result):
            if isinstance(result, dict):
                return result.get('features', {})
            return getattr(result, 'features', {})
        
        if analyzer_type == 'style':
            return self._merge_style_features([get_features(r) for r in results])
        elif analyzer_type == 'color':
            return self._merge_color_features([get_features(r) for r in results])
        elif analyzer_type == 'scene':
            return self._merge_scene_features([get_features(r) for r in results])
        elif analyzer_type == 'lighting':
            return self._merge_lighting_features([get_features(r) for r in results])
        else:
            return get_features(results[-1])

    def _merge_style_features(self, results: List[Dict]) -> Dict:
        """合并风格特征"""
        features = results[-1].get('features', {}).copy()
        
        # 计算主要风格的平均置信度
        primary_styles = [r.get('features', {}).get('primary_style', {}) for r in results]
        if primary_styles:
            avg_confidence = sum(s.get('confidence', 0.0) for s in primary_styles) / len(primary_styles)
            features['primary_style']['confidence'] = avg_confidence
        
        return features

    def _merge_color_features(self, results: List[Dict]) -> Dict:
        """合并颜色特征"""
        features = {}
        
        # 合并主要颜色
        all_colors = {}
        for result in results:
            colors = result.get('features', {}).get('main_colors', [])
            for color in colors:
                rgb = tuple(color.get('rgb', [0, 0, 0]))
                if rgb not in all_colors:
                    all_colors[rgb] = []
                all_colors[rgb].append(color.get('percentage', 0.0))
        
        # 计算平均百分比
        main_colors = [
            {
                'rgb': list(rgb),
                'percentage': sum(percentages) / len(percentages)
            }
            for rgb, percentages in all_colors.items()
        ]
        
        features['main_colors'] = main_colors
        
        # 计算平均饱和度和和谐度
        features['avg_saturation'] = sum(r.get('features', {}).get('avg_saturation', 0.0) 
                                       for r in results) / len(results)
        features['harmony_score'] = sum(r.get('features', {}).get('harmony_score', 0.0) 
                                      for r in results) / len(results)
        
        return features

    def _merge_scene_features(self, results: List[Dict]) -> Dict:
        """合并场景特征"""
        features = results[-1].get('features', {}).copy()
        
        # 合并空间特征
        if 'spatial_features' in features:
            spatial = features['spatial_features']
            for key in spatial:
                values = [r.get('features', {}).get('spatial_features', {}).get(key, 0) 
                         for r in results]
                spatial[key] = sum(values) / len(values) if values else 0
        
        return features

    def _merge_lighting_features(self, results: List[Dict]) -> Dict:
        """合并光照特征"""
        features = {}
        
        # 合并基础指标
        basic_metrics = {
            'brightness': 0.0,
            'uniformity': 0.0,
            'contrast': 0.0
        }
        
        for result in results:
            metrics = result.get('features', {}).get('basic_metrics', {})
            for key in basic_metrics:
                basic_metrics[key] += metrics.get(key, 0.0)
        
        # 计算平均值
        for key in basic_metrics:
            basic_metrics[key] /= len(results)
        
        features['basic_metrics'] = basic_metrics
        
        # 使用最后一帧的其他特征
        last_features = results[-1].get('features', {})
        features['quality'] = last_features.get('quality', {})
        features['light_sources'] = last_features.get('light_sources', {})
        
        return features

    def _aggregate_results(self, analyzer_results: List[Any]) -> Dict:
        """聚合单个分析器的结果"""
        try:
            if not analyzer_results:
                return {}
            
            # 计算平均置信度
            confidences = []
            for r in analyzer_results:
                if isinstance(r, dict):
                    confidences.append(r.get('confidence', 0.0))
                else:
                    confidences.append(getattr(r, 'confidence', 0.0))
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
            
            # 获取分析器类型
            first_result = analyzer_results[0]
            analyzer_type = (first_result.get('type') if isinstance(first_result, dict) 
                            else getattr(first_result, 'type', ''))
            
            # 合并特征
            merged_features = self._merge_features_by_type(analyzer_type, analyzer_results)
            
            # 使用最后一帧的元数据
            last_result = analyzer_results[-1]
            metadata = (last_result.get('metadata', {}) if isinstance(last_result, dict)
                       else getattr(last_result, 'metadata', {}))
            
            return {
                'confidence': avg_confidence,
                'features': merged_features,
                'metadata': metadata
            }
            
        except Exception as e:
            self.logger.error(f"结果聚合失败: {str(e)}")
            return {}

    def _validate_results(self, results: Dict) -> bool:
        """验证处理后的结果"""
        try:
            if not results:
                self.logger.warning("结果为空")
                return False
                
            for analyzer_type, result in results.items():
                # 检查必要字段
                if not isinstance(result, dict):
                    self.logger.warning(f"{analyzer_type} 结果格式错误")
                    return False
                    
                if 'confidence' not in result:
                    self.logger.warning(f"{analyzer_type} 结果缺少置信度")
                    return False
                    
                if 'features' not in result:
                    self.logger.warning(f"{analyzer_type} 结果缺少特征")
                    return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"结果验证失败: {str(e)}")
            return False

    def _format_furniture_section(self, furniture_result: Dict) -> str:
        """格式化家具分析结果"""
        section = "\n二、家具布置分析\n"
        section += "-" * 40 + "\n"
        
        if not furniture_result:
            return section + "未获取家具分析结果\n"
        
        features = furniture_result.get('features', {})
        furniture_types = features.get('furniture_types', {})
        layout = features.get('layout', {})
        
        # 1. 家具清单
        section += "1. 主要家具清单\n"
        detected_furniture = [
            (ftype, stats) 
            for ftype, stats in furniture_types.items() 
            if isinstance(stats, dict) and stats.get('avg_confidence', 0) > 0.15
        ]
        
        if detected_furniture:
            for ftype, stats in detected_furniture:
                count = int(round(stats.get('count', 0)))  # 确保数量为整数
                conf = stats.get('avg_confidence', 0) * 100
                if count > 0:  # 只显示数量大于0的家具
                    section += f"   • 检测到{ftype}{count}件，识别置信度{conf:.0f}%\n"
        else:
            section += "   未检测到明显的家具\n"
        
        # 2. 布局评估
        section += "\n2. 空间布局评估\n"
        layout_score = layout.get('layout_score', 0)
        density = layout.get('density', 0)
        
        # 布局评分描述
        layout_desc = "非常合理" if layout_score > 0.8 else "比较合理" if layout_score > 0.6 else "一般" if layout_score > 0.4 else "需要改善"
        section += f"   • 整体布局{layout_desc}，评分为{layout_score*100:.0f}分\n"
        
        # 空间利用描述
        density_desc = "充分" if density > 0.8 else "适中" if density > 0.4 else "较低"
        section += f"   • 空间利用率{density_desc}，为{density*100:.0f}%\n"
        
        return section

@dataclass
class ProcessedResult:
    """处理后的结果"""
    success: bool = False
    results: Dict = field(default_factory=dict)
    error_message: Optional[str] = None
    
    @classmethod
    def error_result(cls, message: str) -> 'ProcessedResult':
        return cls(success=False, error_message=message) 
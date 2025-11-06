from typing import Dict, List, Any, Optional, Union
import logging
import numpy as np
from collections import Counter, defaultdict
from .result_types import (
    AnalyzerResult,
    SceneResult,
    FurnitureResult, 
    LightingResult,
    ColorResult,
    StyleResult,
    CollectedResults,
    ImageAnalysisResult,
    VideoAnalysisResult,
    RealtimeAnalysisResult
)
from .suggestion_generator import SuggestionGenerator
from ..validators.data_validator import DataValidator
import time
from datetime import datetime
import cv2
from threading import Lock
import queue
import threading
import json

class ResultCollector:
    """结果收集器 - 收集和管理分析结果"""
    
    def __init__(self):
        self.logger = logging.getLogger("pipeline.ResultCollector")
        self.lock = Lock()
        self.results = []  # 改为列表存储
        self.queue = queue.Queue()
        self.processing_thread = threading.Thread(target=self._start_processing_thread)
        self.processing_thread.daemon = True
        self.processing_thread.start()
        
    def collect(self, results: Dict) -> CollectedResults:
        """收集单帧结果"""
        try:
            self.logger.info("=== 开始收集分析结果 ===")
            # 记录原始结果，不使用 JSON 序列化
            self.logger.debug("收到原始结果:")
            for analyzer_type, result in results.items():
                self.logger.debug(f"{analyzer_type}: {result}")
            
            collected_results = {}
            for analyzer_type, result in results.items():
                if not isinstance(result, dict):
                    self.logger.warning(f"跳过非字典结果: {analyzer_type}")
                    continue
                    
                # 确保结果格式正确
                if 'type' not in result or 'features' not in result:
                    self.logger.warning(f"结果格式不正确: {analyzer_type}")
                    self.logger.debug(f"结果内容: {result}")
                    continue
                    
                # 转换为对应的结果类型
                try:
                    if analyzer_type == 'scene':
                        collected_results[analyzer_type] = SceneResult(
                            analyzer_type=analyzer_type,
                            confidence=result.get('confidence', 0.0),
                            features=result.get('features', {}),
                            metadata=result.get('metadata', {})
                        )
                        self.logger.info(f"场景分析结果: 置信度={collected_results[analyzer_type].confidence:.2f}")
                    elif analyzer_type == 'furniture':
                        collected_results[analyzer_type] = FurnitureResult(
                            analyzer_type=analyzer_type,
                            confidence=result.get('confidence', 0.0),
                            features=result.get('features', {}),
                            metadata=result.get('metadata', {})
                        )
                    elif analyzer_type == 'lighting':
                        collected_results[analyzer_type] = LightingResult(
                            analyzer_type=analyzer_type,
                            confidence=result.get('confidence', 0.0),
                            features=result.get('features', {}),
                            metadata=result.get('metadata', {})
                        )
                    elif analyzer_type == 'style':
                        collected_results[analyzer_type] = StyleResult(
                            analyzer_type=analyzer_type,
                            confidence=result.get('confidence', 0.0),
                            features=result.get('features', {}),
                            metadata=result.get('metadata', {})
                        )
                    elif analyzer_type == 'color':
                        collected_results[analyzer_type] = ColorResult(
                            analyzer_type=analyzer_type,
                            confidence=result.get('confidence', 0.0),
                            features=result.get('features', {}),
                            metadata=result.get('metadata', {})
                        )
                except Exception as e:
                    self.logger.error(f"结果转换失败 {analyzer_type}: {str(e)}")
                    continue
            
            # 记录收集到的结果，使用 to_dict() 方法
            self.logger.info("收集到的结果:")
            for analyzer_type, result in collected_results.items():
                self.logger.info(f"{analyzer_type}:")
                self.logger.info(f"- 置信度: {result.confidence:.2f}")
                self.logger.info(f"- 特征: {result.features}")
            
            self.logger.info("=== 结果收集完成 ===")
            
            return CollectedResults(
                analyzer_results=collected_results,
                metadata={
                    'timestamp': time.time(),
                    'frame_count': 1
                }
            )
            
        except Exception as e:
            self.logger.error(f"结果收集失败: {str(e)}")
            return CollectedResults(
                analyzer_results={},
                metadata={'error': str(e)}
            )
        
    def _start_processing_thread(self):
        while True:
            try:
                result = self.queue.get()
                if result is None:
                    self.logger.debug("收到空结果，跳过处理")
                    continue
                with self.lock:
                    for analyzer_name, data in result.items():
                        if data is None:
                            self.logger.warning(f"{analyzer_name} 数据为空")
                            continue
                        self.results.append(data)
            except Exception as e:
                self.logger.error(f"处理失败: {str(e)}")

    def get_collected_results(self) -> CollectedResults:
        """获取所有收集的结果"""
        try:
            self.logger.info(f"[结果收集器] 完成收集，共 {len(self.results)} 个结果")
            
            # 按分析器类型分组结果
            grouped_results = defaultdict(list)
            for result in self.results:
                if isinstance(result, AnalyzerResult):
                    analyzer_type = result.analyzer_type
                    self.logger.info(f"[结果收集器] - {analyzer_type}: 置信度={result.confidence:.2f}")
                    grouped_results[analyzer_type].append(result)
                else:
                    self.logger.warning(f"跳过非 AnalyzerResult 类型的结果: {type(result)}")

            return CollectedResults(
                analyzer_results=dict(grouped_results),
                metadata={
                    'total_frames': len(self.results),
                    'timestamp': datetime.now().isoformat()
                }
            )
            
        except Exception as e:
            self.logger.error(f"获取结果失败: {str(e)}")
            return CollectedResults(
                analyzer_results={},
                metadata={'error': str(e)}
            )

    def _collect_scene_results(self, frame_results: List[Dict]) -> Dict:
        """收集场景分析结果"""
        scene_stats = defaultdict(lambda: {'count': 0, 'total_confidence': 0.0, 'area': 0.0})
        
        for frame in frame_results:
            # 直接使用帧结果，因为它已经是场景分析结果
            scene_result = frame
            features = scene_result.get('features', {})
            scene_type = features.get('scene_type', '未知')
            confidence = scene_result.get('confidence', 0.0)
            area = features.get('scene_features', {}).get('spatial', {}).get('area', 0.0)
            
            stats = scene_stats[scene_type]
            stats['count'] += 1
            stats['total_confidence'] += confidence
            stats['area'] = max(stats['area'], area)
        
        # 找出主要场景类型
        main_scene = max(scene_stats.items(), key=lambda x: x[1]['total_confidence'])
        
        return {
            'type': 'scene_analysis',
            'confidence': main_scene[1]['total_confidence'] / main_scene[1]['count'],
            'features': {
                'scene_type': main_scene[0],
                'area': main_scene[1]['area'],
                'scene_stats': dict(scene_stats)
            }
        }

    def _collect_furniture_results(self, frame_results: List[Dict]) -> Dict:
        """收集家具检测结果"""
        furniture_stats = defaultdict(lambda: {'count': 0, 'detections': []})
        
        for frame in frame_results:
            furniture_result = frame.get('furnituredetector', {})
            detections = furniture_result.get('detections', [])
            
            for det in detections:
                ftype = det.get('type')
                if ftype:
                    stats = furniture_stats[ftype]
                    stats['count'] += 1
                    stats['detections'].append(det)
        
        # 筛选稳定检测的家具
        stable_furniture = {}
        final_detections = []
        
        for ftype, stats in furniture_stats.items():
            if stats['count'] >= len(frame_results) * 0.2:  # 至少在20%的帧中出现
                stable_furniture[ftype] = stats['count']
                # 选择置信度最高的检测结果
                best_detection = max(stats['detections'], key=lambda x: x.get('confidence', 0))
                final_detections.append(best_detection)
        
        return {
            'type': 'furniture_analysis',
            'confidence': 1.0 if final_detections else 0.0,
            'features': {
                'furniture_types': stable_furniture,
                'detections': final_detections,
                'total_count': len(final_detections)
            }
        }

    def _preprocess_frame(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """预处理帧"""
        try:
            # 1. 基础检查
            if frame is None or not isinstance(frame, np.ndarray):
                return None
            
            # 2. 调整大小
            target_size = (640, 480)
            if frame.shape[:2] != target_size:
                frame = cv2.resize(frame, target_size)
            
            # 3. 颜色空间转换
            if len(frame.shape) == 2:
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            elif frame.shape[2] == 4:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
            
            # 4. 数据类型转换
            if frame.dtype != np.uint8:
                frame = (frame * 255).astype(np.uint8)
            
            return frame
            
        except Exception as e:
            self.logger.error(f"帧预处理失败: {str(e)}")
            return None
        
    def collect_video_results(self, frame_results: List[Dict]) -> CollectedResults:
        """收集视频分析结果"""
        try:
            self.logger.info(f"开始收集视频结果，共 {len(frame_results)} 帧")
            
            # 按帧收集结果
            for frame_idx, frame_result in enumerate(frame_results):
                self.logger.info(f"\n处理第 {frame_idx+1} 帧结果:")
                # 先检查和记录原始结果
                self.logger.debug("原始帧结果:")
                for analyzer_type, result in frame_result.items():
                    self.logger.debug(f"{analyzer_type}: {result}")
                
                # 收集结果
                collected = self.collect(frame_result)
                if collected and collected.analyzer_results:
                    for analyzer_type, result in collected.analyzer_results.items():
                        if isinstance(result, AnalyzerResult):
                            self.results.append(result)
                            self.logger.info(f"收集到 {analyzer_type} 结果:")
                            self.logger.info(f"- 置信度: {result.confidence:.2f}")
                            self.logger.info(f"- 特征: {result.features}")
            
            # 获取最终结果
            final_results = self.get_collected_results()
            if final_results:
                self.logger.info("\n最终收集的结果:")
                # 使用 to_dict() 方法转换结果
                result_dict = {
                    'analyzer_results': {},
                    'metadata': {
                        'total_frames': len(frame_results),
                        'timestamp': datetime.now().isoformat()
                    }
                }
                
                # 按分析器类型分组结果
                grouped_results = defaultdict(list)
                for result in self.results:
                    grouped_results[result.analyzer_type].append(result)
                
                # 转换每个分析器的结果
                for analyzer_type, results in grouped_results.items():
                    result_dict['analyzer_results'][analyzer_type] = [
                        r.to_dict() for r in results
                    ]
                
                self.logger.info(json.dumps(result_dict, indent=2))
                
                return CollectedResults(
                    analyzer_results=dict(grouped_results),
                    metadata=result_dict['metadata']
                )
            
            return CollectedResults(
                analyzer_results={},
                metadata={'error': '没有收集到有效结果'}
            )
            
        except Exception as e:
            self.logger.error(f"视频结果收集失败: {str(e)}")
            return CollectedResults(
                analyzer_results={},
                metadata={'error': str(e)}
            )

    def _standardize_result(self, result: Dict, analyzer_type: str) -> Dict:
        """标准化分析结果格式"""
        if not isinstance(result, dict):
            return {
                'type': analyzer_type,
                'confidence': 0.0,
                'features': {},
                'metadata': {'error': 'Invalid result format'}
            }
        
        # 确保基本字段存在
        standardized = {
            'type': result.get('type', analyzer_type),
            'confidence': result.get('confidence', 0.0),
            'features': result.get('features', {}),
            'metadata': result.get('metadata', {})
        }
        
        # 处理特殊情况
        if analyzer_type == 'furniture' and 'detected_items' in result:
            standardized['features'].update({
                'detected_items': result['detected_items'],
                'count': len(result['detected_items'])
            })
        
        return standardized

    def _filter_log_data(self, result: Dict) -> Dict:
        """过滤掉不需要记录到日志的大型数据"""
        if not isinstance(result, dict):
            return result
        
        filtered = result.copy()
        # 移除大型数组数据
        if 'features' in filtered:
            features = filtered['features']
            if isinstance(features, dict):
                if 'image' in features:
                    features['image'] = '<image_data>'
                if 'size' in features:
                    features['size'] = str(features['size'])
        return filtered

    def _is_duplicate_result(self, prev_result: Dict, curr_result: Dict) -> bool:
        """检查两个结果是否完全相同"""
        if not isinstance(prev_result, dict) or not isinstance(curr_result, dict):
            return False
        
        # 根据不同分析器类型判断
        result_type = curr_result.get('type')
        if result_type == 'scene':
            return (prev_result.get('scene_type') == curr_result.get('scene_type') and
                    abs(prev_result.get('confidence', 0) - curr_result.get('confidence', 0)) < 1e-6)
                
        elif result_type == 'furniture':
            # 家具检测比较数量和类型
            prev_items = sorted([item['type'] for item in prev_result.get('detected_items', [])])
            curr_items = sorted([item['type'] for item in curr_result.get('detected_items', [])])
            return prev_items == curr_items
        
        elif result_type == 'lighting':
            # 光照分析比较亮度值
            return (abs(prev_result.get('overall_brightness', 0) - curr_result.get('overall_brightness', 0)) < 1e-6 and
                    abs(prev_result.get('uniformity', 0) - curr_result.get('uniformity', 0)) < 1e-6)
        
        elif result_type == 'color':
            # 颜色分析比较主要颜色的名称和比例
            prev_colors = [(c['name'], round(c['percentage'], 3)) for c in prev_result.get('main_colors', [])]
            curr_colors = [(c['name'], round(c['percentage'], 3)) for c in curr_result.get('main_colors', [])]
            return prev_colors == curr_colors
                
        elif result_type == 'style':
            # 风格分析比较类型和置信度
            return (prev_result.get('style_type') == curr_result.get('style_type') and
                    abs(prev_result.get('confidence', 0) - curr_result.get('confidence', 0)) < 1e-6)
                
        return False

    def _validate_result(self, result: Dict) -> bool:
        """验证结果格式"""
        if result is None:
            return False
        
        # 对于 AnalyzerResult 对象，需要检查是否有实际内容
        if str(result) == 'AnalyzerResult()':
            self.logger.warning("收到空的 AnalyzerResult")
            return False
        
        if not isinstance(result, dict):
            return False
        
        # 检查必要字段
        if 'type' not in result:
            return False
        
        # 根据不同类型检查
        result_type = result.get('type')
        if result_type == 'furniture':
            # 家具检测需要有检测结果
            items = result.get('detected_items', [])
            return len(items) > 0 and all('type' in item and 'confidence' in item for item in items)
        
        elif result_type == 'color':
            # 颜色分析需要有主要颜色
            return bool(result.get('main_colors'))
        
        elif result_type == 'scene':
            # 场景分析需要有场景类型和置信度
            return 'scene_type' in result and 'confidence' in result
        
        elif result_type == 'style':
            # 风格分析需要有风格类型和置信度
            return 'style_type' in result and 'confidence' in result
        
        elif result_type == 'lighting':
            # 光照分析需要有亮度值
            return 'overall_brightness' in result or 'uniformity' in result
        
        return True
            
    def _summarize_scene_results(self, scene_results: List[Dict]) -> Dict:
        """汇总场景分析结果"""
        try:
            if not scene_results:
                return {'error': '无场景分析结果'}
                
            # 统计场景类型
            scene_types = [result['scene_type'] for result in scene_results if 'scene_type' in result]
            main_scene = Counter(scene_types).most_common(1)[0] if scene_types else ('未知', 0)
            
            # 计算平均置信度
            confidences = [result['confidence'] for result in scene_results if 'confidence' in result]
            avg_confidence = np.mean(confidences) if confidences else 0
            
            return {
                'main_scene': main_scene[0],
                'confidence': float(avg_confidence),
                'scene_distribution': dict(Counter(scene_types))
            }
            
        except Exception as e:
            self.logger.error(f"场景结果汇总失败: {str(e)}")
            return {'error': str(e)}
            
    def _summarize_furniture_results(self, frame_results: List[Dict]) -> Dict:
        """汇总家具分析结果"""
        try:
            furniture_stats = {
                'detections': [],
                'counts': {},
                'density': 0
            }
            
            for result in frame_results:
                if result and 'furniture' in result:
                    furniture = result['furniture']
                    # 统计检测结果
                    for detection in furniture.get('detections', []):
                        furniture_stats['detections'].append(detection)
                        cls = detection['class']
                        furniture_stats['counts'][cls] = furniture_stats['counts'].get(cls, 0) + 1
                        
            total_frames = len(frame_results)
            if total_frames > 0:
                furniture_stats['density'] = len(furniture_stats['detections']) / total_frames
                
            return furniture_stats
            
        except Exception as e:
            self.logger.error(f"家具结果汇总失败: {str(e)}")
            return {'error': str(e)}
            
    def _summarize_lighting_results(self, frame_results: List[Dict]) -> Dict:
        """汇总光照分析结果"""
        try:
            lighting_stats = {
                'brightness_values': [],
                'contrast_values': [],
                'quality_counts': {}
            }
            
            for result in frame_results:
                if result and 'lighting' in result:
                    lighting = result['lighting']
                    lighting_stats['brightness_values'].append(lighting.get('brightness', 0))
                    lighting_stats['contrast_values'].append(lighting.get('contrast', 0))
                    quality = lighting.get('quality', 'unknown')
                    lighting_stats['quality_counts'][quality] = lighting_stats['quality_counts'].get(quality, 0) + 1
                    
            # 计算平均值
            lighting_stats['average_brightness'] = np.mean(lighting_stats['brightness_values'])
            lighting_stats['average_contrast'] = np.mean(lighting_stats['contrast_values'])
            
            return lighting_stats
            
        except Exception as e:
            self.logger.error(f"光照结果汇总失败: {str(e)}")
            return {'error': str(e)}
            
    def _summarize_style_results(self, frame_results: List[Dict]) -> Dict:
        """汇总风格分析结果"""
        try:
            style_stats = {
                'style_counts': {},
                'feature_stats': {}
            }
            
            for result in frame_results:
                if result and 'style' in result:
                    style = result['style']
                    style_type = style.get('style_type', 'unknown')
                    style_stats['style_counts'][style_type] = style_stats['style_counts'].get(style_type, 0) + 1
                    
                    # 统计特征
                    for feature, value in style.get('features', {}).items():
                        if feature not in style_stats['feature_stats']:
                            style_stats['feature_stats'][feature] = []
                        style_stats['feature_stats'][feature].append(value)
                        
            # 找出主要风格（处理空结果的情况）
            if style_stats['style_counts']:
                main_style = max(style_stats['style_counts'].items(), key=lambda x: x[1])
                style_stats['main_style'] = main_style[0]
            else:
                style_stats['main_style'] = 'unknown'
                
            return style_stats
            
        except Exception as e:
            self.logger.error(f"风格结果汇总失败: {str(e)}")
            return {'main_style': 'unknown', 'style_counts': {}}
            
    def _summarize_color_results(self, frame_results: List[Dict]) -> Dict:
        """汇总颜色分析结果"""
        try:
            color_stats = {
                'dominant_colors': {},
                'color_schemes': {},
                'saturation_values': [],
                'brightness_values': []
            }
            
            for result in frame_results:
                if result and 'color' in result:
                    color = result['color']
                    
                    # 统计主色
                    for c in color.get('dominant_colors', []):
                        color_stats['dominant_colors'][str(c)] = color_stats['dominant_colors'].get(str(c), 0) + 1
                        
                    # 统计配色方案
                    scheme = color.get('color_scheme', {}).get('main_scheme', 'unknown')
                    color_stats['color_schemes'][scheme] = color_stats['color_schemes'].get(scheme, 0) + 1
                    
                    # 记录饱和度和明度
                    if 'metrics' in color:
                        metrics = color['metrics']
                        color_stats['saturation_values'].append(metrics.get('average_saturation', 0))
                        color_stats['brightness_values'].append(metrics.get('average_brightness', 0))
                        
            return color_stats
            
        except Exception as e:
            self.logger.error(f"颜色结果汇总失败: {str(e)}")
            return {'error': str(e)}

    def _process_analysis_results(self, raw_results: Dict) -> Dict:
        """处理和整理分析结果"""
        processed = {}
        
        try:
            # 场景分析
            if 'scene' in raw_results:
                scene = raw_results['scene']
                processed['scene'] = {
                    'type': scene.get('scene_type', '未知'),
                    'confidence': scene.get('confidence', 0),
                    'message': scene.get('message', '')
                }
            
            # 家具检测
            if 'furniture' in raw_results:
                furniture = raw_results['furniture']
                processed['furniture'] = {
                    'count': furniture.get('count', 0),
                    'items': furniture.get('boxes', [])
                }
            
            # 光照分析
            if 'lighting' in raw_results:
                lighting = raw_results['lighting']
                processed['lighting'] = {
                    'quality': lighting.get('quality', '未知'),
                    'metrics': {
                        'brightness': lighting.get('brightness', 0),
                        'contrast': lighting.get('contrast', 0),
                        'uniformity': lighting.get('uniformity', 0)
                    }
                }
            
            # 风格分析
            if 'style' in raw_results:
                style = raw_results['style']
                processed['style'] = {
                    'type': style.get('style_type', '未知'),
                    'confidence': style.get('confidence', 0),
                    'features': style.get('features', {})
                }
            
            # 颜色分析
            if 'color' in raw_results:
                color = raw_results['color']
                processed['color'] = {
                    'dominant_colors': color.get('dominant_colors', [])[:3],  # 只保留前3个主色
                    'harmony_score': color.get('stats', {}).get('harmony_score', 0)
                }
            
            return processed
            
        except Exception as e:
            self.logger.error(f"结果处理失败: {str(e)}")
            return {}

    def _format_summary(self, processed: Dict) -> str:
        """格式化分析总结，生成用户友好的报告"""
        summary = []
        
        # 标题
        summary.append("=== 室内空间分析报告 ===\n")
        
        # 场景分析
        if 'scene' in processed:
            scene = processed['scene']
            if scene['confidence'] > 0.3:
                summary.append(f"这是一个{scene['type']}空间。")
            else:
                summary.append("空间类型识别度较低。")
        
        # 家具分析
        if 'furniture' in processed:
            furniture = processed['furniture']
            summary.append("\n【家具布局】")
            if furniture['count'] > 0:
                summary.append(f"空间内共检测到{furniture['count']}件家具：")
                for item in furniture['boxes']:
                    summary.append(f"- {item['class']}")
        
        # 光照分析
        if 'lighting' in processed:
            lighting = processed['lighting']
            metrics = lighting['metrics']
            summary.append("\n【光照环境】")
            summary.append(f"• 整体质量：{lighting['quality']}")
            summary.append(f"• 亮度水平：{self._format_brightness(metrics['brightness'])}")
            summary.append(f"• 光照均匀度：{self._format_uniformity(metrics['uniformity'])}")
        
        # 风格分析
        if 'style' in processed:
            style = processed['style']
            summary.append("\n【设计风格】")
            if style['confidence'] > 0.3:
                summary.append(f"空间呈现{style['type']}特征，主要表现在：")
                features = style['features']
                for feature, value in features.items():
                    summary.append(f"• {self._format_feature(feature)}: {value}")
        
        # 颜色分析
        if 'color' in processed:
            color = processed['color']
            summary.append("\n【配色方案】")
            if color['dominant_colors']:
                summary.append("主要色系：")
                for c in color['dominant_colors'][:3]:
                    rgb = c['rgb']
                    pct = c['percentage']
                    color_name = self._get_color_name(rgb)
                    summary.append(f"• {color_name}（占比 {pct:.0%}）")
        
        return "\n".join(summary)

    def _format_brightness(self, value: float) -> str:
        """格式化亮度值"""
        if value < 0.3:
            return "偏暗"
        elif value > 0.7:
            return "明亮"
        return "适中"

    def _format_uniformity(self, value: float) -> str:
        """格式化均匀度值"""
        if value < 0.5:
            return "不均匀"
        return "均匀"

    def _format_feature(self, feature: str) -> str:
        """格式化风格特征名称"""
        feature_map = {
            'color_tone': '色调',
            'texture': '质感',
            'lines': '线条',
            'symmetry': '对称性'
        }
        return feature_map.get(feature, feature)

    def _get_color_name(self, rgb: List[int]) -> str:
        """将RGB值转换为友好的颜色名称"""
        # 这里可以实现一个简单的颜色映射
        color_map = {
            (255, 0, 0): "红色",
            (0, 255, 0): "绿色",
            (0, 0, 255): "蓝色",
            # 可以添加更多颜色映射
        }
        
        # 找到最接近的颜色
        min_distance = float('inf')
        color_name = "未知色"
        
        for rgb_key, name in color_map.items():
            distance = sum((a - b) ** 2 for a, b in zip(rgb, rgb_key))
            if distance < min_distance:
                min_distance = distance
                color_name = name
        
        return color_name

    def _format_furniture_result(self, furniture: Dict) -> str:
        """格式化家具检测结果"""
        lines = []
        if furniture['count'] > 0:
            lines.append(f"\n检测到 {furniture['count']} 个家具:")
            for item in furniture['boxes']:
                conf = item['confidence'] * 100  # 转换为百分比
                lines.append(f"- {item['class']}: {conf:.1f}% 置信度")
                box = item['box']
                lines.append(f"  位置: ({box[0]}, {box[1]}) -> ({box[2]}, {box[3]})")
        else:
            lines.append("\n未检测到家具")
        return "\n".join(lines)

    def _calculate_weights(self, results: List[Dict]) -> List[float]:
        """计算每个结果的权重"""
        weights = []
        for result in results:
            # 基础权重 = 置信度
            weight = result.get('confidence', 0)
            
            # 特征完整性权重
            features = result.get('features', {})
            completeness = len(features) / 10  # 假设完整特征有10个
            weight *= (0.7 + 0.3 * completeness)  # 70%置信度 + 30%完整性
            
            # 时间衰减权重 (最新的结果权重更高)
            time_weight = 1.0  # 可以基于时间戳计算
            weight *= time_weight
            
            weights.append(weight)
        
        # 归一化权重
        total = sum(weights) or 1
        return [w/total for w in weights]

    def _calculate_weighted_confidence(self, results: List[Dict], weights: List[float]) -> float:
        """计算加权后的置信度"""
        if not results or not weights:
            return 0.0
        
        weighted_confidence = sum(r.get('confidence', 0) * w for r, w in zip(results, weights))
        return weighted_confidence

    def _calculate_quality_score(self, results: List[Dict]) -> float:
        """计算结果质量分数"""
        if not results:
            return 0.0
        
        scores = []
        for result in results:
            # 基础分数 = 置信度
            score = result.get('confidence', 0)
            
            # 特征完整性得分
            features = result.get('features', {})
            completeness = len(features) / 10  # 假设完整特征有10个
            score *= (0.7 + 0.3 * completeness)
            
            # 一致性得分
            consistency = self._calculate_consistency(result, results)
            score *= (0.8 + 0.2 * consistency)
            
            scores.append(score)
        
        return sum(scores) / len(scores)

    def _calculate_consistency(self, result: Dict, all_results: List[Dict]) -> float:
        """计算结果与其他结果的一致性"""
        consistency = 0.0
        count = 0
        
        for other in all_results:
            if other is result:
                continue
            
            # 计算特征相似度
            similarity = self._calculate_similarity(
                result.get('features', {}),
                other.get('features', {})
            )
            consistency += similarity
            count += 1
        
        return consistency / count if count > 0 else 0.0

    def _calculate_similarity(self, features1: Dict, features2: Dict) -> float:
        """计算两组特征的相似度"""
        if not features1 or not features2:
            return 0.0
        
        common_keys = set(features1.keys()) & set(features2.keys())
        if not common_keys:
            return 0.0
        
        similarities = []
        for key in common_keys:
            val1 = features1[key]
            val2 = features2[key]
            
            if isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
                # 数值类型：计算相对差异
                max_val = max(abs(val1), abs(val2))
                if max_val == 0:
                    similarities.append(1.0)
                else:
                    similarities.append(1.0 - abs(val1 - val2) / max_val)
            elif isinstance(val1, str) and isinstance(val2, str):
                # 字符串类型：完全匹配
                similarities.append(1.0 if val1 == val2 else 0.0)
            elif isinstance(val1, dict) and isinstance(val2, dict):
                # 递归计算子特征的相似度
                similarities.append(self._calculate_similarity(val1, val2))
        
        return sum(similarities) / len(similarities)

    def collect_image_result(self, results: Dict[str, Any]) -> Optional[CollectedResults]:
        """收集图像分析结果"""
        try:
            # 修改返回类型为 CollectedResults
            collected_results = self.collect(results)
            return collected_results
        except Exception as e:
            self.logger.error(f"图像结果收集失败: {str(e)}")
            return None
            
    def collect_video_result(self, results: List[Dict[str, Any]]) -> Optional[VideoAnalysisResult]:
        """收集视频分析结果"""
        try:
            # 验证结果
            if not self.validator.validate_video_result(results):
                return None
                
            # 收集帧结果
            frames = []
            for frame_result in results:
                image_result = self.collect_image_result(frame_result)
                if image_result:
                    frames.append(image_result)
                    
            # 创建结果对象
            return VideoAnalysisResult(
                frames=frames,
                summary=self._generate_summary(frames),
                metadata={
                    'frame_count': len(frames),
                    'timestamp': datetime.now()
                }
            )
            
        except Exception as e:
            self.logger.error(f"视频结果收集失败: {str(e)}")
            return None
            
    def _generate_summary(self, frames: List[ImageAnalysisResult]) -> Dict:
        """生成视频分析总结"""
        try:
            return {
                'frame_count': len(frames),
                'duration': len(frames) / 30.0,  # 假设30fps
                'timestamp': datetime.now()
            }
        except Exception as e:
            self.logger.error(f"总结生成失败: {str(e)}")
            return {}

    def _merge_features(self, results_list: List[Dict]) -> Dict:
        """合并特征"""
        try:
            if not results_list:
                return {}
            
            merged = {}
            for result in results_list:
                features = result.get('features', {})
                for key, value in features.items():
                    if key not in merged:
                        merged[key] = []
                    merged[key].append(value)
            
            # 统计或平均每个特征
            summary = {}
            for key, values in merged.items():
                if isinstance(values[0], (int, float)):
                    summary[key] = sum(values) / len(values)
                elif isinstance(values[0], str):
                    # 取最常见的值
                    from collections import Counter
                    counter = Counter(values)
                    summary[key] = counter.most_common(1)[0][0]
                elif isinstance(values[0], dict):
                    summary[key] = self._merge_features(values)
            
            return summary
            
        except Exception as e:
            self.logger.error(f"特征合并失败: {str(e)}")
            return {}

    def _merge_dict_features(self, dict_features: List[Dict], weights: List[float]) -> Dict:
        """合并字典类型的特征"""
        if not dict_features:
            return {}
        
        # 使用第一个字典的结构作为基础
        merged = dict_features[0].copy()
        
        # 对每个键进行加权合并
        for key in merged:
            values = []
            key_weights = []
            
            for d, weight in zip(dict_features, weights):
                if key in d:
                    values.append(d[key])
                    key_weights.append(weight)
            
            if not values:
                continue
            
            # 根据值类型选择合并策略
            if isinstance(values[0], (int, float)):
                # 数值类型：加权平均
                merged[key] = sum(v * w for v, w in zip(values, key_weights))
            elif isinstance(values[0], str):
                # 字符串类型：选择权重最高的
                value_weights = defaultdict(float)
                for value, weight in zip(values, key_weights):
                    value_weights[value] += weight
                merged[key] = max(value_weights.items(), key=lambda x: x[1])[0]
            elif isinstance(values[0], dict):
                # 字典类型：递归合并
                if sum(key_weights) > 0:
                    normalized_weights = [w/sum(key_weights) for w in key_weights]
                    merged[key] = self._merge_dict_features(values, normalized_weights)
                else:
                    merged[key] = {}
            elif isinstance(values[0], (list, tuple)):
                # 列表类型：保留最高权重的
                max_weight_idx = key_weights.index(max(key_weights))
                merged[key] = values[max_weight_idx]
            
        return merged

    def _nms_merge(self, detections: List[Dict], iou_threshold: float = 0.5) -> List[Dict]:
        """使用非极大值抑制(NMS)合并重叠的检测框
        
        Args:
            detections: 检测结果列表，每个检测包含 'box' 和 'confidence' 键
            iou_threshold: IoU阈值，高于此值的框会被合并
            
        Returns:
            合并后的检测列表
        """
        if not detections:
            return []
        
        # 按置信度排序
        detections = sorted(detections, key=lambda x: x.get('confidence', 0), reverse=True)
        
        # 存储保留的检测结果
        kept_detections = []
        
        while detections:
            # 取置信度最高的检测框
            best = detections.pop(0)
            kept_detections.append(best)
            
            # 计算其他框与当前最佳框的IoU
            i = 0
            while i < len(detections):
                if self._calculate_iou(best['box'], detections[i]['box']) > iou_threshold:
                    # 如果是同类物体，合并检测结果
                    if best.get('class') == detections[i].get('class'):
                        best['confidence'] = max(best['confidence'], detections[i]['confidence'])
                        # 可以选择更新框的位置为两个框的平均值
                        best['box'] = self._average_boxes(best['box'], detections[i]['box'])
                    detections.pop(i)
                else:
                    i += 1
                
        return kept_detections

    def _calculate_iou(self, box1: List[float], box2: List[float]) -> float:
        """计算两个框的IoU(交并比)
        
        Args:
            box1: [x1, y1, x2, y2] 格式的框坐标
            box2: [x1, y1, x2, y2] 格式的框坐标
            
        Returns:
            IoU值
        """
        # 计算交集区域
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        # 如果没有交集
        if x2 < x1 or y2 < y1:
            return 0.0
        
        intersection = (x2 - x1) * (y2 - y1)
        
        # 计算两个框的面积
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        
        # 计算并集面积
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0

    def _average_boxes(self, box1: List[float], box2: List[float]) -> List[float]:
        """计算两个框的平均位置
        
        Args:
            box1: [x1, y1, x2, y2] 格式的框坐标
            box2: [x1, y1, x2, y2] 格式的框坐标
            
        Returns:
            平均后的框坐标
        """
        return [
            (box1[0] + box2[0]) / 2,  # x1
            (box1[1] + box2[1]) / 2,  # y1
            (box1[2] + box2[2]) / 2,  # x2
            (box1[3] + box2[3]) / 2   # y2
        ] 

    def collect_scene_results(self, results: List[Dict]) -> Dict:
        """收集场景分析结果"""
        scene_type = "其他"
        max_confidence = 0.0
        scene_features = {}
        
        self.logger.debug(f"开始收集场景结果，共 {len(results)} 条")
        
        for result in results:
            if result.get('type') == 'scene_analysis':
                features = result.get('features', {})
                confidence = result.get('confidence', 0.0)
                
                self.logger.debug(f"处理场景结果: confidence={confidence}")
                self.logger.debug(f"特征数据: {features}")
                
                if confidence > max_confidence:
                    max_confidence = confidence
                    scene_type = features.get('scene_type', '其他')
                    scene_features = features.get('scene_features', {})
                    
                    self.logger.debug(f"更新最佳结果: type={scene_type}")
                    self.logger.debug(f"场景特征: {scene_features}")
        
        result = {
            'type': 'scene_analysis',
            'confidence': max_confidence,
            'features': {
                'scene_type': scene_type,
                'scene_features': scene_features
            }
        }
        
        self.logger.debug(f"最终收集结果: {result}")
        return result 

    def _convert_to_analyzer_result(self, frame_result: Dict, analyzer_name: str) -> AnalyzerResult:
        """将帧结果转换为分析器结果"""
        try:
            # 获取分析器结果
            analyzer_result = frame_result.get(analyzer_name, {})
            
            return AnalyzerResult(
                type=analyzer_name,
                confidence=analyzer_result.get('confidence', 0.0),
                features=analyzer_result.get('features', {}),
                metadata=analyzer_result.get('metadata', {})
            )
        except Exception as e:
            self.logger.error(f"结果转换失败: {str(e)}")
            return AnalyzerResult(
                type=analyzer_name,
                confidence=0.0,
                features={},
                metadata={}
            ) 

    def _aggregate_scene_results(self, frame_results: List[Dict]) -> Dict:
        """汇总场景分析结果"""
        scene_stats = defaultdict(lambda: {'count': 0, 'confidence': 0.0})
        
        for frame in frame_results:
            if 'scene' in frame:
                scene_result = frame['scene']
                scene_type = scene_result.get('scene_type', 'unknown')
                confidence = scene_result.get('confidence', 0.0)
                
                stats = scene_stats[scene_type]
                stats['count'] += 1
                stats['confidence'] += confidence
        
        # 找出最常见的场景类型
        main_scene = max(scene_stats.items(), key=lambda x: x[1]['count'])
        
        return {
            'scene_type': main_scene[0],
            'confidence': main_scene[1]['confidence'] / main_scene[1]['count'],
            'scene_stats': dict(scene_stats)
        }

    def _aggregate_furniture_results(self, frame_results: List[Dict]) -> Dict:
        """汇总家具检测结果"""
        furniture_stats = defaultdict(lambda: {'count': 0, 'confidence': 0.0})
        
        for frame in frame_results:
            if 'furniture' in frame:
                for item in frame['furniture'].get('detected_items', []):
                    furniture_type = item.get('type', 'unknown')
                    confidence = item.get('confidence', 0.0)
                    
                    stats = furniture_stats[furniture_type]
                    stats['count'] += 1
                    stats['confidence'] += confidence
        
        return {
            'detected_items': [
                {
                    'type': ftype,
                    'count': stats['count'],
                    'avg_confidence': stats['confidence'] / stats['count']
                }
                for ftype, stats in furniture_stats.items()
            ]
        }

    def _aggregate_lighting_results(self, frame_results: List[Dict]) -> Dict:
        """汇总光照分析结果"""
        total_brightness = 0.0
        total_uniformity = 0.0
        valid_frames = 0
        
        for frame in frame_results:
            if 'lighting' in frame and isinstance(frame['lighting'], dict):
                brightness = frame['lighting'].get('overall_brightness', 0.0)
                uniformity = frame['lighting'].get('uniformity', 0.0)
                
                total_brightness += brightness
                total_uniformity += uniformity
                valid_frames += 1
        
        return {
            'overall_brightness': total_brightness / valid_frames if valid_frames > 0 else 0.0,
            'uniformity': total_uniformity / valid_frames if valid_frames > 0 else 0.0
        }

    def _aggregate_color_results(self, frame_results: List[Dict]) -> Dict:
        """汇总颜色分析结果"""
        color_stats = defaultdict(lambda: {'count': 0, 'percentage': 0.0})
        total_harmony = 0.0
        valid_frames = 0
        
        for frame in frame_results:
            if 'color' in frame:
                color_result = frame['color']
                
                # 统计主色
                for color in color_result.get('main_colors', []):
                    color_name = color.get('name', 'unknown')
                    percentage = color.get('percentage', 0.0)
                    
                    stats = color_stats[color_name]
                    stats['count'] += 1
                    stats['percentage'] += percentage
                
                # 统计和谐度
                harmony = color_result.get('harmony_score', 0.0)
                if harmony > 0:
                    total_harmony += harmony
                    valid_frames += 1
        
        return {
            'main_colors': [
                {
                    'name': color_name,
                    'avg_percentage': stats['percentage'] / stats['count'],
                    'frequency': stats['count']
                }
                for color_name, stats in color_stats.items()
            ],
            'color_scheme': max(
                (frame['color'].get('color_scheme', '') 
                 for frame in frame_results if 'color' in frame),
                default=''
            ),
            'avg_harmony_score': total_harmony / valid_frames if valid_frames > 0 else 0.0
        }

    def _aggregate_style_results(self, frame_results: List[Dict]) -> Dict:
        """汇总风格分析结果"""
        style_stats = defaultdict(lambda: {'count': 0, 'confidence': 0.0})
        total_consistency = 0.0
        valid_frames = 0
        
        for frame in frame_results:
            if 'style' in frame:
                style_result = frame['style']
                style_type = style_result.get('style_type', 'unknown')
                confidence = style_result.get('confidence', 0.0)
                consistency = style_result.get('consistency_score', 0.0)
                
                stats = style_stats[style_type]
                stats['count'] += 1
                stats['confidence'] += confidence
                
                if consistency > 0:
                    total_consistency += consistency
                    valid_frames += 1
        
        # 找出最主要的风格
        main_style = max(style_stats.items(), key=lambda x: x[1]['count'])
        
        return {
            'style_type': main_style[0],
            'confidence': main_style[1]['confidence'] / main_style[1]['count'],
            'consistency_score': total_consistency / valid_frames if valid_frames > 0 else 0.0,
            'style_stats': dict(style_stats)
        } 

    def collect_result(self, analyzer_type: str, result: AnalyzerResult):
        """收集单个分析器的结果"""
        self.logger.info(f"[结果收集器] 收到{analyzer_type}分析器结果")
        self.results.append(result)
        self.logger.info(f"[结果收集器] 当前已收集 {len(self.results)}/{self.expected_count} 个结果")

    def get_collected_results(self) -> List[AnalyzerResult]:
        """获取所有收集的结果"""
        self.logger.info(f"[结果收集器] 完成收集，共 {len(self.results)} 个结果")
        for result in self.results:
            self.logger.info(f"[结果收集器] - {result.analyzer_type}: 置信度={result.confidence:.2f}")
        return self.results 
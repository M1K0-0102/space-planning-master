from typing import Dict, List, Optional
import cv2
import numpy as np
from .visualizer import Visualizer
import logging
from collections import defaultdict
from src.pipeline.utils.result_types import AnalyzerResult

class VisualizationCoordinator:
    """可视化协调器 - 协调不同类型的可视化"""
    
    def __init__(self):
        self.logger = logging.getLogger("pipeline.VisualizationCoordinator")
        self.visualizer = Visualizer()
        
    def process_result(self, frame: np.ndarray, result: Dict, mode: str = "image") -> Dict:
        """处理分析结果的可视化
        Args:
            frame: 输入图像
            result: 分析结果
            mode: 输入模式 ("image"/"video"/"realtime")
        """
        try:
            if frame is None or result is None:
                return result
                
            if result['status'] == 'success' and 'results' in result:
                results = result['results']
                
                if mode == "realtime":
                    # 实时模式才进行可视化渲染
                    vis_frame = self.visualizer.render_realtime_results(frame, results)
                    results['visualization'] = vis_frame
                else:
                    # 图片/视频模式只生成文字建议
                    results['suggestions'] = self._generate_text_suggestions(results)
                    
            return result
            
        except Exception as e:
            print(f"结果处理失败: {str(e)}")
            return result
            
    def _generate_text_suggestions(self, results: Dict) -> List[str]:
        """生成文字建议"""
        suggestions = []
        
        # 场景建议
        if 'scene' in results:
            scene_type = results['scene'].get('scene_type')
            if scene_type:
                suggestions.append(f"场景类型: {scene_type}")
                
        # 家具建议
        if 'furniture' in results:
            furniture = results['furniture']
            if 'suggestions' in furniture:
                suggestions.extend(furniture['suggestions'])
                
        # 光照建议
        if 'lighting' in results:
            lighting = results['lighting']
            if 'suggestions' in lighting:
                suggestions.extend(lighting['suggestions'])
                
        return suggestions 

    def visualize_video_result(self, result: Dict) -> Dict:
        """可视化视频分析结果"""
        try:
            visualization = {}
            
            # 场景分析可视化
            if 'scene_analysis' in result:
                visualization['scene'] = self._visualize_scene_analysis(result['scene_analysis'])
                
            # 家具分析可视化
            if 'furniture_analysis' in result:
                visualization['furniture'] = self._visualize_furniture_analysis(result['furniture_analysis'])
                
            # 光照分析可视化
            if 'lighting_analysis' in result:
                visualization['lighting'] = self._visualize_lighting_analysis(result['lighting_analysis'])
                
            # 风格分析可视化
            if 'style_analysis' in result:
                visualization['style'] = self._visualize_style_analysis(result['style_analysis'])
                
            # 颜色分析可视化
            if 'color_analysis' in result:
                visualization['color'] = self._visualize_color_analysis(result['color_analysis'])
                
            return visualization
            
        except Exception as e:
            self.logger.error(f"视频结果可视化失败: {str(e)}")
            return {} 

    def visualize_realtime(self, frame: np.ndarray, result: Dict) -> np.ndarray:
        """实时可视化
        Args:
            frame: 原始帧
            result: 实时分析结果
        Returns:
            可视化后的帧
        """
        try:
            return self.visualizer.visualize(frame, result)
            
        except Exception as e:
            self.logger.error(f"实时可视化失败: {str(e)}")
            return frame

    def generate(self, frame: np.ndarray, result: Dict) -> Optional[Dict]:
        """生成可视化结果
        Args:
            frame: 原始帧
            result: 分析结果
        Returns:
            可视化结果
        """
        try:
            if frame is None or result is None:
                return None
                
            # 1. 基础可视化
            vis_frame = self.visualizer.visualize(frame, result)
            
            # 2. 生成可视化结果
            visualization = {
                'frame': vis_frame,
                'type': 'image',
                'metadata': {
                    'width': vis_frame.shape[1],
                    'height': vis_frame.shape[0],
                    'channels': vis_frame.shape[2]
                }
            }
            
            return visualization
            
        except Exception as e:
            self.logger.error(f"可视化生成失败: {str(e)}")
            return None
            
    def _visualize_furniture(self, frame: np.ndarray, furniture_result: Dict) -> np.ndarray:
        """家具检测可视化"""
        vis_frame = frame.copy()
        
        # 绘制检测框和标签
        for box in furniture_result.get('boxes', []):
            x1, y1, x2, y2 = box['box']
            label = box['class']
            conf = box['confidence']
            
            cv2.rectangle(vis_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(vis_frame, f"{label} {conf:.2f}", 
                       (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                       
        return vis_frame 

    def coordinate_suggestions(self, results: Dict) -> Dict:
        """协调可视化建议
        Args:
            results: 分析结果
        Returns:
            整合后的可视化建议
        """
        try:
            # 1. 获取各类型的可视化建议
            visual_suggestions = self.visualizer.generate_visual_suggestions(results)
            
            # 2. 整合建议
            coordinated_suggestions = {
                'visual': visual_suggestions,
                'interactive': self._generate_interactive_elements(results),
                'layout': self._generate_layout_suggestions(results)
            }
            
            return coordinated_suggestions
            
        except Exception as e:
            self.logger.error(f"可视化建议协调失败: {str(e)}")
            return {} 

    def generate_video_report(self, analysis_results: List[Dict]) -> Dict:
        """生成视频分析的可视化报告"""
        try:
            # 实现视频报告生成逻辑
            return {
                'heatmaps': self._generate_heatmaps(analysis_results),
                'timeline': self._generate_timeline(analysis_results)
            }
        except Exception as e:
            self.logger.error(f"视频报告生成失败: {str(e)}")
            return {} 

    def _generate_heatmaps(self, analysis_results: List[AnalyzerResult]) -> Dict:
        """生成热力图"""
        try:
            heatmap_data = defaultdict(float)
            for result in analysis_results:
                if result.type == 'furniture':
                    for detection in result.features.get('detections', []):
                        x_center = (detection['box'][0] + detection['box'][2]) / 2
                        y_center = (detection['box'][1] + detection['box'][3]) / 2
                        heatmap_data[(int(x_center), int(y_center))] += 1
            return self.visualizer.generate_heatmap(heatmap_data)
        except Exception as e:
            self.logger.error(f"热力图生成失败: {str(e)}")
            return {}

    def _generate_timeline(self, analysis_results: List[AnalyzerResult]) -> List:
        """生成时间线数据"""
        timeline = []
        for idx, result in enumerate(analysis_results):
            timeline.append({
                'frame': idx,
                'timestamp': result.metadata.get('timestamp', 0),
                'key_events': self._detect_key_events(result)
            })
        return timeline 
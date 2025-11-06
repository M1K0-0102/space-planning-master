from typing import Dict, List, Optional
import cv2
import numpy as np
import logging
from src.pipeline.utils.result_types import AnalyzerResult

class Visualizer:
    """可视化器 - 负责渲染分析结果"""
    
    def __init__(self):
        self.logger = logging.getLogger("pipeline.Visualizer")
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = 0.6
        self.text_color = (0, 255, 0)  # BGR格式
        self.box_color = (0, 255, 0)
        self.line_thickness = 2
        
    def visualize(self, frame: np.ndarray, result: Dict) -> np.ndarray:
        """可视化分析结果
        Args:
            frame: 原始帧
            result: 分析结果
        Returns:
            可视化后的帧
        """
        try:
            if frame is None or result is None:
                return frame
                
            vis_frame = frame.copy()
            
            # 1. 家具检测可视化
            if 'furniture' in result:
                vis_frame = self._visualize_furniture(vis_frame, result['furniture'])
                
            # 2. 光照分析可视化
            if 'lighting' in result:
                vis_frame = self._visualize_lighting(vis_frame, result['lighting'])
                
            # 3. 场景分析可视化
            if 'scene' in result:
                vis_frame = self._visualize_scene(vis_frame, result['scene'])
                
            # 4. 风格分析可视化
            if 'style' in result:
                vis_frame = self._visualize_style(vis_frame, result['style'])
                
            # 5. 添加性能信息
            if 'metadata' in result:
                vis_frame = self._add_performance_info(vis_frame, result['metadata'])
                
            return vis_frame
            
        except Exception as e:
            self.logger.error(f"可视化失败: {str(e)}")
            return frame
            
    def _visualize_furniture(self, frame: np.ndarray, furniture_result: Dict) -> np.ndarray:
        """家具检测可视化"""
        try:
            for detection in furniture_result.get('detections', []):
                # 绘制边界框
                box = detection.get('bbox', [])
                if len(box) == 4:
                    x1, y1, x2, y2 = map(int, box)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), self.box_color, self.line_thickness)
                    
                    # 添加标签
                    label = f"{detection.get('class', 'unknown')} {detection.get('confidence', 0):.2f}"
                    cv2.putText(frame, label, (x1, y1-10), self.font, self.font_scale, 
                               self.text_color, self.line_thickness)
                               
            return frame
            
        except Exception as e:
            self.logger.error(f"家具可视化失败: {str(e)}")
            return frame
            
    def _visualize_lighting(self, frame: np.ndarray, lighting_result: Dict) -> np.ndarray:
        """光照分析可视化"""
        try:
            # 添加光照质量
            quality = lighting_result.get('quality_score', 0)
            cv2.putText(frame, f"Light: {quality:.2f}", (10, 30), self.font,
                       self.font_scale, self.text_color, self.line_thickness)
                       
            # 添加亮度分布
            brightness = lighting_result.get('brightness', 0)
            cv2.putText(frame, f"Bright: {brightness:.2f}", (10, 60), self.font,
                       self.font_scale, self.text_color, self.line_thickness)
                       
            return frame
            
        except Exception as e:
            self.logger.error(f"光照可视化失败: {str(e)}")
            return frame
            
    def _visualize_scene(self, frame: np.ndarray, scene_result: Dict) -> np.ndarray:
        """场景分析可视化"""
        try:
            # 添加场景类型
            scene_type = scene_result.get('scene_type', 'unknown')
            confidence = scene_result.get('confidence', 0)
            cv2.putText(frame, f"Scene: {scene_type} ({confidence:.2f})", (10, 90),
                       self.font, self.font_scale, self.text_color, self.line_thickness)
                       
            return frame
            
        except Exception as e:
            self.logger.error(f"场景可视化失败: {str(e)}")
            return frame
            
    def _visualize_style(self, frame: np.ndarray, style_result: Dict) -> np.ndarray:
        """风格分析可视化"""
        try:
            # 添加风格信息
            style = style_result.get('main_style', 'unknown')
            cv2.putText(frame, f"Style: {style}", (10, 120), self.font,
                       self.font_scale, self.text_color, self.line_thickness)
                       
            return frame
            
        except Exception as e:
            self.logger.error(f"风格可视化失败: {str(e)}")
            return frame
            
    def _add_performance_info(self, frame: np.ndarray, metadata: Dict) -> np.ndarray:
        """添加性能信息"""
        try:
            # 添加FPS
            fps = metadata.get('fps', 0)
            cv2.putText(frame, f"FPS: {fps}", (frame.shape[1]-120, 30), self.font,
                       self.font_scale, self.text_color, self.line_thickness)
                       
            return frame
            
        except Exception as e:
            self.logger.error(f"性能信息添加失败: {str(e)}")
            return frame

class AnalysisVisualizer:
    """处理分析结果的可视化"""
    
    def create_visualization(self, result: Dict) -> np.ndarray:
        """创建可视化结果"""
        if result is None or 'frame' not in result:
            return None
            
        vis_image = result['frame'].copy()
        
        self._add_scene_visualization(vis_image, result.get('scene_type'))
        self._add_furniture_visualization(vis_image, result.get('furniture_info'))
        self._add_lighting_visualization(vis_image, result.get('lighting'))
        
        return vis_image
        
    def _add_scene_visualization(self, image, scene_type):
        """添加场景类型可视化"""
        if scene_type:
            cv2.putText(image, f"场景: {scene_type}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    def _add_furniture_visualization(self, image, furniture_info):
        """添加家具检测可视化"""
        if furniture_info:
            for item in furniture_info:
                x1, y1, x2, y2 = item['bbox']
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(image, item['class'], (x1, y1-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    def _add_lighting_visualization(self, image, lighting_info):
        """添加光照分析可视化"""
        if lighting_info:
            quality = lighting_info.get('quality', 'unknown')
            cv2.putText(image, f"光照: {quality}", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2) 
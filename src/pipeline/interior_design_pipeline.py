from typing import Dict, Any, Optional, List, Set
import torch
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torchvision.transforms as transforms
import torchvision.models as models
import os
import time
from collections import Counter
import traceback
from sklearn.cluster import KMeans  # 添加用于颜色分析
import math  # 用于数学计算
import inspect
try:
    import open3d as o3d  # 可选的3D可视化
    OPEN3D_AVAILABLE = True
except ImportError:
    OPEN3D_AVAILABLE = False

from .analyzers import (
    SceneAnalyzer,
    ColorAnalyzer,
    LightingAnalyzer,
    StyleAnalyzer,
    FurnitureDetector
)
from .strategies import (
    ComprehensiveStrategy,
    SingleAnalysisStrategy,
    RealtimeStrategy
)
from .strategies.base_strategy import BaseStrategy  # 改为从base_strategy导入
from .processors import (
    ImageProcessor,
    VideoProcessor,
    RealtimeProcessor,
    ResultProcessor
)
from .validators.input_validator import InputValidator
from .utils.result_types import AnalysisType
from .utils.suggestion_formatter import SuggestionFormatter
from .utils.model_config import ModelConfig
from .visualization.visualization_coordinator import VisualizationCoordinator
import logging
from .pipeline_coordinator import PipelineCoordinator
from .utils.suggestion_generator import SuggestionGenerator
from .utils.result_collector import ResultCollector
from .validators.data_validator import DataValidator
from .utils.error_handler import ErrorHandler
from .utils.output_manager import OutputManager

class InteriorDesignPipeline:
    _instance = None
    _initialized = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(InteriorDesignPipeline, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
            
        self.logger = logging.getLogger("pipeline.InteriorDesignPipeline")
        # 避免日志重复
        self.logger.propagate = False
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
            self.logger.addHandler(handler)
        
        try:
            self.logger.info("开始初始化 InteriorDesignPipeline...")
            
            # 初始化组件
            self.logger.debug("初始化 PipelineCoordinator...")
            self.coordinator = PipelineCoordinator()
            
            self.logger.debug("初始化其他组件...")
            self.data_validator = DataValidator()
            self.error_handler = ErrorHandler()
            self.output_manager = OutputManager()
            
            # 从协调器获取组件
            self.analyzers = self.coordinator.analyzers
            self.processors = self.coordinator.processors
            
            # 初始化其他组件
            self.visualization_coordinator = VisualizationCoordinator()
            self.input_validator = InputValidator()
            self.result_collector = ResultCollector()
            self.suggestion_generator = SuggestionGenerator()
            
            self.model_config = self.coordinator.model_config
            self.strategy = ComprehensiveStrategy(self.analyzers)
            
            self._initialized = True
            self.logger.info("InteriorDesignPipeline 初始化完成")
            
        except Exception as e:
            self.logger.error("Pipeline初始化失败", exc_info=True)
            raise

    def analyze_image(self, image: np.ndarray, mode: str = 'comprehensive', 
                     analyzer_type: str = None) -> Dict:
        """分析单张图片"""
        return self.coordinator.process_image(image, mode, analyzer_type)
        
    def analyze_video(self, video_path: str, mode: str = 'comprehensive',
                     analyzer_type: str = None) -> Dict:
        """分析视频"""
        return self.coordinator.process_video(video_path, mode, analyzer_type)
        
    def analyze_realtime(self, frame: np.ndarray) -> Dict:
        """实时分析（如摄像头输入）
        Args:
            frame: 实时输入帧
        Returns:
            分析结果
        """
        try:
            return self.processors['realtime'].process(frame)
        except Exception as e:
            self.logger.error(f"实时分析失败: {str(e)}")
            return {'error': str(e)}

    def process_video(self, video_path: str) -> Dict:
        """处理视频"""
        try:
            # 1. 获取视频处理器
            processor = self.processors['video']
            
            # 2. 处理视频帧
            frames_results = processor.process(video_path)
            
            # 3. 收集所有帧的结果 (使用已有的 result_collector)
            collected_results = self.result_collector.collect_frames(frames_results)
            
            # 4. 处理结果 (使用 coordinator 的结果处理器)
            processed_results = self.coordinator.processors['result'].process_video_results(collected_results)
            
            # 5. 生成建议 (使用已有的 suggestion_generator)
            suggestions = self.suggestion_generator.generate_suggestions(processed_results)
            
            # 6. 格式化最终结果
            final_results = {
                'metadata': {
                    'frame_count': processor.total_frames,
                    'processed_frames': len(frames_results)
                },
                'scene_analysis': processed_results.get('scene', {}),
                'furniture_analysis': processed_results.get('furniture', {}),
                'lighting_analysis': processed_results.get('lighting', {}),
                'color_analysis': processed_results.get('color', {}),
                'style_analysis': processed_results.get('style', {}),
                'suggestions': suggestions
            }
            
            # 7. 保存结果 (使用已有的 output_manager)
            self.output_manager.save_results(final_results)
            
            return final_results
            
        except Exception as e:
            self.logger.error(f"视频处理失败: {str(e)}")
            return {}

    def _collect_frames_results(self, frames_results: List[Dict]) -> Dict:
        """收集所有帧的分析结果"""
        try:
            # 直接使用已初始化的 result_collector
            return self.result_collector.collect_frames(frames_results)
        except Exception as e:
            self.logger.error(f"帧结果收集失败: {str(e)}")
            return {}

    def _process_results(self, collected_results: Dict) -> Dict:
        """处理收集到的结果"""
        try:
            # 使用 coordinator 中的结果处理器
            return self.coordinator.processors['result'].process_video_results(collected_results)
        except Exception as e:
            self.logger.error(f"结果处理失败: {str(e)}")
            return {}

    def _generate_suggestions(self, processed_results: Dict) -> List[str]:
        """生成改进建议"""
        try:
            return self.suggestion_generator.generate_suggestions(processed_results)
        except Exception as e:
            self.logger.error(f"建议生成失败: {str(e)}")
            return []

    def _save_results(self, results: Dict):
        """保存分析结果"""
        try:
            output_manager = OutputManager()
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            output_manager.save(
                results,
                f"output/video_analysis_{timestamp}.txt"
            )
        except Exception as e:
            self.logger.error(f"结果保存失败: {str(e)}")

    def analyze_single(self, image: np.ndarray, analyzer_type: str) -> Dict:
        """单项分析接口
        
        Args:
            image: 输入图片
            analyzer_type: 分析器类型
                - 'scene': 场景分析
                - 'furniture': 家具检测
                - 'lighting': 光照分析
                - 'style': 风格分析
                - 'color': 颜色分析
        """
        return self.coordinator.process_single_analysis(image, analyzer_type)
        
    def get_available_analyzers(self) -> Set[str]:
        """获取可用的分析器列表"""
        return self.coordinator.get_available_analyzers()

    def _init_analyzers(self):
        """初始化分析器"""
        try:
            self.analyzers = {
                'scene': SceneAnalyzer(self.model_config),
                'furniture': FurnitureDetector(self.model_config),
                'lighting': LightingAnalyzer(self.model_config),
                'style': StyleAnalyzer(self.model_config),
                'color': ColorAnalyzer(self.model_config)
            }
        except Exception as e:
            self.logger.error(f"分析器初始化失败: {str(e)}")
            raise

    def _init_strategies(self) -> Dict:
        """初始化策略
        Returns:
            策略字典
        """
        try:
            return {
                'comprehensive': ComprehensiveStrategy(self.analyzers),
                'single': SingleAnalysisStrategy(self.analyzers),
                'realtime': RealtimeStrategy(self.analyzers)
            }
        except Exception as e:
            self.logger.error(f"策略初始化失败: {str(e)}")
            raise

    def analyze(self, input_data: Any, input_mode: str, 
                analysis_type: str, **kwargs) -> Dict:
        """执行分析"""
        try:
            if input_mode == 'image':
                return self.process_image(input_data)
            elif input_mode == 'video':
                return self.process_video(input_data)
            elif input_mode == 'realtime':
                return self.process_realtime(input_data)
            else:
                raise ValueError(f"不支持的输入模式: {input_mode}")
        except Exception as e:
            self.logger.error(f"分析失败: {str(e)}")
            return {'error': str(e)}

    def _format_response(self, result, analysis_type, input_mode):
        """根据输入模式格式化响应"""
        if input_mode in ['image', 'video']:
            # 图片/视频模式：返回图片+文字建议
            return {
                'status': 'success',
                'data': {
                    'visualization': self._create_visualization(result),
                    'suggestions': self._format_suggestions(result)
                }
            }
        else:
            # 实时模式：返回实时分析结果
            return {
                'status': 'success',
                'data': result
            }

    def _preprocess_image(self, image):
        """图像预处理"""
        # 实现图像预处理的逻辑
        pass

    def _analyze_basic_info(self, frame):
        """基础分析"""
        try:
            # 1. 场景分析
            scene_result = self._analyze_scene(frame)
            scene_type = scene_result[0] if scene_result else None
            
            # 2. 家具检测
            furniture_info = self._detect_furniture(frame)
            
            # 3. 光照分析
            lighting_info = self.analyzers['lighting'].analyze_basic(frame)
            
            # 4. 空间维度分析
            dimensions = self._analyze_dimensions(frame, furniture_info) if furniture_info else None
            
            return {
                'frame': frame,
                'scene_type': scene_type,
                'furniture_info': furniture_info,
                'lighting': lighting_info,
                'dimensions': dimensions,
                'timestamp': time.time()  # 添加时间戳
            }
        except Exception as e:
            print(f"基础分析出错: {str(e)}")
            return None

    def _analyze_lighting_basic(self, frame):
        """基础光照分析"""
        try:
            # 转换到HSV空间
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            
            # 计算亮度
            brightness = np.mean(hsv[:,:,2])
            
            # 检测光源
            light_sources = self._detect_light_sources(frame)
            
            # 计算自然光
            natural_light = self._calculate_natural_light(frame)
            
            return {
                'brightness': brightness,
                'light_sources': light_sources,
                'natural_light': natural_light,
                'quality': self._evaluate_lighting_quality(brightness, light_sources, natural_light)
            }
        except Exception as e:
            print(f"光照分析出错: {str(e)}")
            return None

    def _detect_light_sources(self, frame):
        """检测光源"""
        try:
            # 转换到灰度图
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # 使用自适应阈值检测高亮区域
            thresh = cv2.adaptiveThreshold(
                gray, 255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY,
                11, 2
            )
            
            # 查找轮廓
            contours, _ = cv2.findContours(
                thresh,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE
            )
            
            light_sources = []
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area > 100:  # 过滤小区域
                    x, y, w, h = cv2.boundingRect(cnt)
                    light_sources.append({
                        'position': (x + w//2, y + h//2),
                        'intensity': np.mean(gray[y:y+h, x:x+w]),
                        'area': area
                    })
            
            return light_sources
            
        except Exception as e:
            print(f"光源检测出错: {str(e)}")
            return []

    def _evaluate_lighting_quality(self, brightness, light_sources, natural_light):
        """评估光照质量"""
        try:
            # 评分标准
            brightness_score = min(brightness / 128, 1.0)  # 亮度评分
            distribution_score = len(light_sources) / 5 if light_sources else 0  # 光源分布评分
            natural_light_score = natural_light  # 自然光评分
            
            # 综合评分
            total_score = (brightness_score * 0.4 + 
                          distribution_score * 0.3 + 
                          natural_light_score * 0.3)
            
            # 质量等级
            if total_score >= 0.8:
                return "excellent"
            elif total_score >= 0.6:
                return "good"
            elif total_score >= 0.4:
                return "fair"
            else:
                return "poor"
            
        except Exception as e:
            print(f"光照质量评估出错: {str(e)}")
            return "unknown"

    def _calculate_natural_light(self, frame):
        """计算自然光指数"""
        # 实现计算自然光指数的逻辑
        pass

    def _detect_artificial_lighting(self, frame):
        """检测人工照明"""
        # 实现检测人工照明的逻辑
        pass

    def _analyze_current_style(self, frame):
        """分析当前风格"""
        # 实现分析当前风格的逻辑
        pass

    def _generate_style_recommendations(self, current_style, scene_type):
        """生成风格推荐"""
        # 实现生成风格推荐的逻辑
        pass

    def _create_optimization_visualization(self, frame, suggestions):
        """创建空间优化建议的可视化"""
        # 实现创建空间优化建议可视化的逻辑
        pass

    def _create_lighting_visualization(self, frame):
        """创建光照分析的可视化"""
        # 实现创建光照分析可视化的逻辑
        pass

    def _create_style_visualization(self, frame, recommendations):
        """创建风格推荐的可视化"""
        # 实现创建风格推荐可视化的逻辑
        pass

    def measure_distance(self, frame1, frame2=None, mode='stereo', camera_params=None):
        """测量距离
        
        Args:
            frame1: 主摄像头图像
            frame2: 副摄像头图像(双摄模式需要)
            mode: 测距模式
            camera_params: 相机参数
        
        Returns:
            dict: 测距结果
                - distance: 距离(米)
                - confidence: 置信度
                - visualization: 可视化图像
        """
        try:
            if mode == 'depth':
                # 使用深度传感器测距
                return self._measure_with_depth_sensor(frame1, camera_params)
            elif mode == 'stereo':
                # 使用双摄像头测距
                if frame2 is None:
                    raise ValueError("双摄模式需要两个摄像头的图像")
                return self._measure_with_stereo(frame1, frame2, camera_params)
            else:
                raise ValueError(f"不支持的测距模式: {mode}")
            
        except Exception as e:
            print(f"测距失败: {str(e)}")
            return None

    def _measure_with_depth_sensor(self, frame, camera_params):
        """使用深度传感器测距"""
        try:
            # 获取深度图
            depth_map = camera_params.get('depth_map')
            if depth_map is None:
                raise ValueError("无法获取深度信息")
                
            # 计算距离
            distance = np.mean(depth_map[depth_map > 0])
            confidence = np.sum(depth_map > 0) / depth_map.size
            
            # 可视化
            visualization = self._visualize_depth(frame, depth_map)
            
            return {
                'distance': distance,
                'confidence': confidence,
                'visualization': visualization
            }
        except Exception as e:
            print(f"深度传感器测距失败: {str(e)}")
            return None

    def _measure_with_stereo(self, frame1, frame2, camera_params):
        """使用双摄像头测距"""
        try:
            # 获取相机参数
            focal_length = camera_params.get('focal_length')
            baseline = camera_params.get('baseline')
            if not all([focal_length, baseline]):
                raise ValueError("缺少必要的相机参数")
            
            # 计算视差
            disparity = self._compute_disparity(frame1, frame2)
            
            # 计算深度
            depth = (focal_length * baseline) / (disparity + 1e-6)
            
            # 计算置信度
            confidence = self._estimate_stereo_confidence(disparity)
            
            # 可视化
            visualization = self._visualize_stereo(frame1, depth)
            
            return {
                'distance': np.mean(depth),
                'confidence': confidence,
                'visualization': visualization
            }
        except Exception as e:
            print(f"双摄测距失败: {str(e)}")
            return None

    def _analyze_space_planning(self, frame):
        """空间规划分析逻辑"""
        # 实现空间规划分析的具体逻辑
        pass

    def _analyze_lighting(self, frame):
        """光照优化分析逻辑"""
        # 实现光照优化分析的具体逻辑
        pass

    def _analyze_color_matching(self, frame):
        """色彩搭配分析逻辑"""
        # 实现色彩搭配分析的具体逻辑
        pass

    def _analyze_style(self, frame):
        """装修风格分析逻辑"""
        # 实现装修风格分析的具体逻辑
        pass

    def _measure_distance_realtime(self, frame):
        """实时测距逻辑"""
        # 实现实时测距的具体逻辑
        pass

    def _estimate_area_realtime(self, frame):
        """实时面积估测逻辑"""
        # 实现实时面积估测的具体逻辑
        pass

    def _process_image(self, input_data):
        """处理图片输入"""
        # 实现图片处理的具体逻辑
        pass

    def _process_realtime(self, input_data):
        """处理实时输入"""
        try:
            # 确保传递正确的摄像头索引
            cap = cv2.VideoCapture(0)  # 0 表示默认摄像头
            if not cap.isOpened():
                raise ValueError("无法打开摄像头")
            
            # 读取帧
            ret, frame = cap.read()
            if not ret:
                raise ValueError("无法读取帧")
            
            # 处理帧的逻辑
            # ...

        except Exception as e:
            print(f"实时处理出错: {str(e)}")
        finally:
            cap.release()

    def _create_visualization(self, result):
        """创建可视化结果"""
        try:
            if result is None:
                return None
            
            frame = result.get('frame')
            if frame is None:
                return None
            
            # 创建可视化图像
            vis_image = frame.copy()
            
            # 添加分析结果的可视化
            if 'scene_type' in result:
                self._add_scene_visualization(vis_image, result['scene_type'])
            if 'furniture_info' in result:
                self._add_furniture_visualization(vis_image, result['furniture_info'])
            if 'lighting' in result:
                self._add_lighting_visualization(vis_image, result['lighting'])
            
            return vis_image
        except Exception as e:
            print(f"可视化创建失败: {str(e)}")
            return None

    def _add_scene_visualization(self, image, scene_type):
        """添加场景类型可视化"""
        try:
            # 在图像左上角添加场景类型文本
            cv2.putText(image, f"场景: {scene_type}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        except Exception as e:
            print(f"场景可视化失败: {str(e)}")

    def _add_furniture_visualization(self, image, furniture_info):
        """添加家具检测可视化"""
        try:
            for item in furniture_info:
                # 绘制边界框
                x1, y1, x2, y2 = item['bbox']
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                # 添加标签
                cv2.putText(image, item['class'], (x1, y1-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        except Exception as e:
            print(f"家具可视化失败: {str(e)}")

    def _add_lighting_visualization(self, image, lighting_info):
        """添加光照分析可视化"""
        try:
            # 添加光照质量信息
            quality = lighting_info.get('quality', 'unknown')
            cv2.putText(image, f"光照: {quality}", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # 标记光源位置
            for light in lighting_info.get('light_sources', []):
                pos = light['position']
                cv2.circle(image, pos, 5, (0, 255, 255), -1)
        except Exception as e:
            print(f"光照可视化失败: {str(e)}")

    def _format_suggestions(self, result):
        """格式化建议文本"""
        try:
            if result is None:
                return "无法生成建议"
            
            suggestions = []
            
            # 根据不同类型的分析结果生成建议
            if 'scene_type' in result:
                suggestions.append(f"场景类型: {result['scene_type']}")
            if 'lighting' in result:
                suggestions.append(f"光照质量: {result['lighting'].get('quality', 'unknown')}")
            if 'furniture_info' in result:
                suggestions.append(f"检测到的家具: {len(result['furniture_info'])}件")
            
            return "\n".join(suggestions)
        except Exception as e:
            print(f"建议格式化失败: {str(e)}")
            return "建议生成失败"

    def process_request(self, request: Dict) -> Dict:
        """处理分析请求"""
        try:
            # 处理请求
            result = self.strategy.process(request)
            
            # 添加可视化
            if 'frame' in request:
                result = self.vis_coordinator.process_result(
                    request['frame'], result
                )
                
            return result
            
        except Exception as e:
            print(f"请求处理失败: {str(e)}")
            return {'status': 'error', 'message': str(e)}
            
    def _visualize_scene(self, frame: np.ndarray, result: Dict) -> np.ndarray:
        """可视化场景分析结果"""
        return self.analyzers['scene'].visualize(frame, result)
        
    def _visualize_lighting(self, frame: np.ndarray, result: Dict) -> np.ndarray:
        """可视化光照分析结果"""
        return self.analyzers['lighting'].visualize(frame, result)
        
    def _visualize_colors(self, frame: np.ndarray, result: Dict) -> np.ndarray:
        """可视化色彩分析结果"""
        return self.analyzers['color'].visualize(frame, result)

    def analyze_frame(self, frame: np.ndarray) -> Dict:
        """分析单帧图像"""
        try:
            # 创建请求
            request = {
                'frame': frame,
                'type': 'frame'
            }
            
            # 处理请求
            return self.process_request(request)
            
        except Exception as e:
            print(f"图像分析失败: {str(e)}")
            return None

    def process_image(self, image_path: str) -> Dict:
        """处理图片"""
        try:
            self.logger.info(f"开始处理图片: {image_path}")
            result = self.processors['image'].process(image_path)
            self.logger.info("图片处理完成")
            return result
        except Exception as e:
            self.logger.error(f"图片处理失败: {str(e)}")
            return {'error': str(e)}
            
    def process_realtime(self, frame) -> Dict:
        """处理实时输入"""
        try:
            result = self.processors['realtime'].process(frame)
            return result
        except Exception as e:
            self.logger.error(f"实时处理失败: {str(e)}")
            return {'error': str(e)}

    def get_analyzer_info(self) -> Dict:
        """获取分析器信息"""
        return self.coordinator.get_available_analyzers()
        
    def get_processor_info(self) -> Dict:
        """获取处理器信息"""
        return {
            'image': self.processors['image'].__class__.__name__,
            'video': self.processors['video'].__class__.__name__,
            'realtime': self.processors['realtime'].__class__.__name__
        }
        
    def update_config(self, new_config: Dict) -> None:
        """更新配置"""
        self.config.update(new_config)

    def analyze_video_frame(self, frame: np.ndarray) -> Optional[Dict]:
        """分析视频帧
        Args:
            frame: 输入帧
        Returns:
            分析结果
        """
        try:
            if not isinstance(frame, np.ndarray):
                self.logger.error("输入必须是numpy数组")
                return None
            
            # 复制帧以避免修改原始数据
            frame = frame.copy()
            
            # 使用视频处理器
            return self.processors['video'].process_frame(frame)
            
        except Exception as e:
            self.logger.error(f"视频帧分析失败: {str(e)}")
            return None

    def process_input(self, input_data: Any, input_type: str) -> Dict:
        """统一的处理入口"""
        try:
            # 1. 输入验证
            if not self.input_validator.validate(input_data, input_type):
                return self.error_handler.handle_error(
                    ValueError("无效的输入数据"),
                    "输入验证"
                )
                
            # 2. 选择处理器和策略
            processor = self.processors.get(input_type)
            strategy = self.coordinator.select_strategy(input_type)
            if not processor or not strategy:
                return self.error_handler.handle_error(
                    ValueError(f"不支持的输入类型: {input_type}"),
                    "处理器选择"
                )
                
            # 3. 处理数据
            processed_data = processor.preprocess(input_data)
            
            # 4. 执行分析
            analysis_result = strategy.execute(processed_data)
            
            # 5. 收集和处理结果
            self.result_collector.collect(analysis_result)
            processed_result = self.result_collector.get_results()
            
            # 6. 生成建议
            suggestions = self.suggestion_generator.generate(processed_result)
            
            # 7. 格式化输出
            return self.output_manager.format_output({
                'analysis': processed_result,
                'suggestions': suggestions
            }, input_type)
            
        except Exception as e:
            return self.error_handler.handle_error(e, "处理流程")
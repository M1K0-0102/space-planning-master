import logging
from typing import Dict, Any, Optional, List, Set, Union
import numpy as np
import time
import os
from tqdm import tqdm
import json
from datetime import datetime

# 导入分析器
from .analyzers import (
    SceneAnalyzer,
    ColorAnalyzer,
    LightingAnalyzer,
    StyleAnalyzer,
    FurnitureDetector
)

# 导入处理器
from .processors import (
    ImageProcessor,
    VideoProcessor,
    RealtimeProcessor,
    ResultProcessor
)

# 导入策略
from .strategies import (
    ComprehensiveStrategy,
    SingleAnalysisStrategy,
    RealtimeStrategy
)

# 导入工具类
from .utils.result_collector import ResultCollector
from .utils.suggestion_generator import SuggestionGenerator
from .utils.model_config import ModelConfig
from .utils.result_types import AnalysisType, CollectedResults, ProcessedResult
from .validators.input_validator import InputValidator
from .validators.data_validator import DataValidator

# 导入其他组件
from .visualization.visualization_coordinator import VisualizationCoordinator
from .utils.suggestion_formatter import SuggestionFormatter
from .strategies.base_strategy import BaseStrategy
from .utils.error_handler import ErrorHandler
from .utils.output_manager import OutputManager
from .analyzers.base_analyzer import BaseAnalyzer
from .utils.result_formatter import ResultFormatter

class PipelineCoordinator:
    """管道协调器 - 协调整个分析流程"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)  # 默认使用 INFO 级别
        
        try:
            self.logger.info("初始化 PipelineCoordinator...")
            self.model_config = ModelConfig()
            
            # 初始化组件
            self.analyzers = self._init_analyzers()
            self.processors = self._init_processors()
            self.strategies = self._init_strategies()
            self.result_collector = ResultCollector()
            self.result_processor = ResultProcessor()
            self.suggestion_generator = SuggestionGenerator()
            self.result_formatter = ResultFormatter()
            self.suggestion_formatter = SuggestionFormatter()
            self.output_manager = OutputManager()
            
            self.logger.info("PipelineCoordinator 初始化完成")
            
        except Exception as e:
            self.logger.error("初始化失败", exc_info=True)
            raise
        
    def process(self, input_data: Union[str, np.ndarray], mode: str = 'comprehensive') -> ProcessedResult:
        """处理输入数据"""
        try:
            self.logger.info(f"开始处理，模式: {mode}")
            
            # 1. 获取处理策略
            strategy = self._get_strategy(mode)
            if not strategy:
                return ProcessedResult.error_result("无效的处理模式")
            
            # 2. 区分实时分析和其他模式
            if mode == 'realtime':
                # 实时分析模式
                return self._process_realtime(input_data, strategy)
            else:
                # 视频或图片分析模式
                return self._process_normal(input_data, strategy, mode)
            
        except Exception as e:
            self.logger.error(f"处理失败: {str(e)}")
            return ProcessedResult.error_result(str(e))

    def _process_realtime(self, frame: np.ndarray, strategy: BaseStrategy) -> ProcessedResult:
        """处理实时分析"""
        try:
            # 1. 执行实时分析
            analysis_result = strategy.execute(frame)
            
            # 2. 简化的结果处理，不生成报告
            if not analysis_result:
                return ProcessedResult.error_result("实时分析失败")
            
            # 3. 返回实时分析结果
            return ProcessedResult(
                success=True,
                results=analysis_result,
                suggestions=[],  # 实时模式不生成建议
                metadata={
                    'timestamp': datetime.now().isoformat(),
                    'mode': 'realtime',
                    'frame_results': analysis_result  # 用于实时显示
                }
            )
            
        except Exception as e:
            self.logger.error(f"实时处理失败: {str(e)}")
            return ProcessedResult.error_result(str(e))

    def _process_normal(self, input_data: Union[str, np.ndarray], strategy: BaseStrategy, mode: str) -> ProcessedResult:
        """处理普通分析（视频或图片）"""
        try:
            # 1. 执行分析
            if isinstance(input_data, str) and input_data.endswith(('.mp4', '.avi', '.mov')):
                # 视频处理
                self.logger.info("开始分析视频帧...")
                frame_results = self.processors['video'].process_video(input_data, strategy)
                
                # 收集结果
                self.logger.info("开始收集视频结果...")
                collected_results = self.result_collector.collect_video_results(frame_results)
                
            else:
                # 图片处理
                frame_result = strategy.execute(input_data)
                collected_results = self.result_collector.collect(frame_result)
            
            # 2. 处理结果
            processed_result = self.process_results(collected_results)
            if not processed_result.success:
                return processed_result
            
            # 3. 生成建议
            suggestions = self.suggestion_generator.generate_suggestions(processed_result.results)
            
            # 4. 格式化结果和建议
            formatted_results = self.result_formatter.format_results(processed_result.results)
            formatted_suggestions = self.suggestion_formatter.format_suggestions(suggestions)
            
            # 5. 输出结果
            self.output_manager.output_results(
                raw_results=processed_result.results,
                raw_suggestions=suggestions,
                formatted_results=formatted_results,
                formatted_suggestions=formatted_suggestions
            )
            
            # 6. 添加输出信息到结果中
            processed_result.metadata.update({
                'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S"),
                'mode': mode,
                'formatted_results': formatted_results,
                'formatted_suggestions': formatted_suggestions
            })
            
            # 7. 添加建议到 ProcessedResult
            processed_result.suggestions = suggestions
            
            return processed_result
            
        except Exception as e:
            self.logger.error(f"普通处理失败: {str(e)}")
            return ProcessedResult.error_result(str(e))

    def _collect_results(self, raw_results: Dict) -> CollectedResults:
        """收集分析结果"""
        try:
            collected = self.result_collector.collect(raw_results)
            return collected
        except Exception as e:
            self.logger.error(f"结果收集失败: {str(e)}")
            return {}

    def process_results(self, collected_results: CollectedResults) -> ProcessedResult:
        """处理收集的结果"""
        try:
            self.logger.info("开始处理收集的结果...")
            
            # 1. 处理结果
            processed_result = self.result_processor.process(collected_results)
            if not processed_result.success:
                raise ValueError(processed_result.error_message)
            
            # 2. 生成建议
            suggestions = self.suggestion_generator.generate_suggestions(processed_result.results)
            
            # 3. 格式化结果和建议
            formatted_results = self.result_formatter.format_results(processed_result.results)
            formatted_suggestions = self.suggestion_formatter.format_suggestions(suggestions)
            
            # 4. 输出结果
            self.output_manager.output_results(
                raw_results=processed_result.results,
                raw_suggestions=suggestions,
                formatted_results=formatted_results,
                formatted_suggestions=formatted_suggestions
            )
            
            # 5. 返回 ProcessedResult 对象
            return ProcessedResult(
                success=True,
                results=processed_result.results,
                error=None,
                metadata={
                    'timestamp': time.time(),
                    'version': '1.0.0'
                }
            )
            
        except Exception as e:
            self.logger.error(f"结果处理失败: {str(e)}")
            return ProcessedResult(
                success=False,
                results={},
                error=str(e)
            )

    def process_image(self, image: np.ndarray, mode: str = 'comprehensive', analyzer_type: str = None) -> Dict:
        """处理图片"""
        try:
            # 1. 验证输入
            if not self.input_validator.validate_image(image):
                return self.error_handler.handle_error(ValueError("无效的图片输入"), "输入验证")
                
            # 2. 选择策略
            strategy = self._get_strategy(mode, analyzer_type)
            if not strategy:
                return self.error_handler.handle_error(ValueError("无效的策略"), "策略选择")
                
            # 3. 预处理图片
            processed_image = self.processors['image'].preprocess(image)
            if processed_image is None:
                return self.error_handler.handle_error(ValueError("图片预处理失败"), "图片预处理")
                
            # 4. 执行分析
            analysis_result = strategy.execute(processed_image)
            if not analysis_result:
                return self.error_handler.handle_error(ValueError("分析失败"), "图片分析")
                
            # 5. 收集结果
            self.result_collector.collect(analysis_result)
            collected_results = self.result_collector.get_results()
            if not collected_results:
                return self.error_handler.handle_error(ValueError("结果收集失败"), "结果收集")
                
            # 6. 处理结果
            processed_result = self.result_processor.process(collected_results)
            if not processed_result:
                return self.error_handler.handle_error(ValueError("结果处理失败"), "结果处理")
                
            # 7. 生成报告
            report = self.output_manager.generate_report(processed_result)
            if not report:
                return self.error_handler.handle_error(ValueError("报告生成失败"), "报告生成")
                
            return report
            
        except Exception as e:
            return self.error_handler.handle_error(e, "图片处理流程")
            
    def process_video(self, video_path: str, mode: str = 'comprehensive', analyzer_type: str = None) -> ProcessedResult:
        """处理视频的全流程"""
        try:
            # 1. 输入验证
            if not self.input_validator.validate_video(video_path):
                return self._get_error_result("无效的视频输入")

            # 2. 提取处理后的视频帧
            video_processor = self.processors['video']
            frames = video_processor.extract_processed_frames(video_path)
            if not frames:
                return self._get_error_result("未提取到有效视频帧")

            # 3. 选择分析策略
            strategy = self._get_strategy(mode, analyzer_type)
            if not strategy:
                return self._get_error_result("无效的分析策略")

            # 4. 分析每一帧
            analysis_results = []
            for frame in frames:
                result = strategy.execute(frame)
                if result:
                    analysis_results.append(result)

            if not analysis_results:
                return self._get_error_result("视频分析未产生有效结果")

            # 5. 收集结果
            for result in analysis_results:
                self.result_collector.collect(result)
            collected_results = self.result_collector.get_results()

            # 6. 处理结果
            processed_result = self.result_processor.process(collected_results)
            if not processed_result:
                return self._get_error_result("结果处理失败")

            # 7. 生成报告
            report = self.output_manager.generate_report(processed_result)
            visualization = self.visualization_coordinator.generate_video_report(analysis_results)

            return ProcessedResult(
                success=True,
                report=report,
                visualization=visualization,
                metadata={
                    'frame_count': len(frames),
                    'processed_frames': len(analysis_results),
                    'timestamp': time.time()
                }
            )

        except Exception as e:
            self.logger.error(f"视频处理流程异常: {str(e)}")
            return self._get_error_result(f"视频处理失败: {str(e)}")
            
    def _get_strategy(self, mode: str, analyzer_type: str = None):
        return self.select_strategy(mode, analyzer_type)

    def get_available_analyzers(self) -> Set[str]:
        """获取可用的分析器列表
        Returns:
            分析器类型集合
        """
        return set(self.analyzers.keys())

    def _extract_video_frames(self, video_path: str) -> List[np.ndarray]:
        """提取视频关键帧
        Args:
            video_path: 视频文件路径
        Returns:
            关键帧列表
        """
        try:
            video_processor = self.processors['video']
            return video_processor.extract_keyframes(video_path)
        except Exception as e:
            self.logger.error(f"视频帧提取失败: {str(e)}")
            return []

    def _execute_analysis(self, image: np.ndarray, mode: str, analyzer_type: Optional[str] = None) -> Dict:
        """执行分析"""
        try:
            # 1. 图像预处理
            processed_image = self.processors['image'].preprocess(image)
            if processed_image is None:
                return None
                
            # 2. 选择策略
            strategy = self.select_strategy(mode, analyzer_type)
            if strategy is None:
                return None
                
            # 3. 执行分析
            result = strategy.execute(processed_image)
            if result is None:
                return None
                
            return result
            
        except Exception as e:
            self.logger.error(f"分析失败: {str(e)}")
            return None

    def _get_error_result(self, error_message: str) -> ProcessedResult:
        """获取错误结果"""
        self.logger.error(f"生成错误结果: {error_message}")
        return ProcessedResult.error_result(error_message)

    def analyze_frame(self, frame: np.ndarray) -> Dict:
        """分析单帧"""
        try:
            # 1. 预处理
            processed_frame = self._preprocess_frame(frame)
            if processed_frame is None:
                return {}
            
            # 2. 提取共享特征(只提取一次)
            features = {
                'lighting': self.feature_extractor.extract_lighting_features(processed_frame),
                'color': self.feature_extractor.extract_color_features(processed_frame),
                'texture': self.feature_extractor.extract_texture_features(processed_frame)
            }
            
            # 3. 分发特征给各个分析器
            results = {}
            for analyzer_name, analyzer in self.analyzers.items():
                try:
                    # 传入相关特征
                    result = analyzer.analyze(processed_frame, features)
                    if result:
                        results[analyzer_name] = result
                except Exception as e:
                    self.logger.error(f"{analyzer_name} 分析失败: {str(e)}")
            
            return results
            
        except Exception as e:
            self.logger.error(f"帧分析失败: {str(e)}")
            return {}

    def _get_analyzer(self, mode: str, analyzer_type: str = None) -> Union[BaseAnalyzer, BaseStrategy]:
        """获取分析器或策略
        Args:
            mode: 分析模式 ('comprehensive' 或 'single')
            analyzer_type: 分析器类型 (mode为'single'时必需)
        Returns:
            分析器实例或策略实例
        """
        try:
            if mode == 'comprehensive':
                # 综合分析模式返回综合策略
                return self.strategies['comprehensive']
            elif mode == 'single' and analyzer_type:
                # 单项分析模式返回指定分析器
                if analyzer_type not in self.analyzers:
                    raise ValueError(f"未找到分析器: {analyzer_type}")
                return self.analyzers[analyzer_type]
            else:
                raise ValueError(f"无效的分析模式: {mode}")
                
        except Exception as e:
            self.logger.error(f"获取分析器失败: {str(e)}")
            return None

    def select_strategy(self, mode: str, analyzer_type: str = None) -> BaseStrategy:
        """选择分析策略
        Args:
            mode: 分析模式
            analyzer_type: 分析器类型(可选)
        Returns:
            策略实例
        """
        try:
            if mode == 'comprehensive':
                return self.strategies['comprehensive']
            elif mode == 'single':
                # 如果没有指定分析器类型，让用户选择
                if not analyzer_type:
                    print("\n请选择分析器类型:")
                    available_analyzers = list(self.analyzers.keys())
                    for i, analyzer in enumerate(available_analyzers, 1):
                        print(f"{i}. {analyzer}")
                    
                    while True:
                        try:
                            choice = input("请输入分析器编号: ").strip()
                            idx = int(choice) - 1
                            if 0 <= idx < len(available_analyzers):
                                analyzer_type = available_analyzers[idx]
                                self.logger.info(f"选择了 {analyzer_type} 分析器")
                                break
                            else:
                                print("无效的选项，请重新输入")
                        except ValueError:
                            print("请输入有效的数字")
                
                # 创建单项分析策略
                if analyzer_type not in self.analyzers:
                    raise ValueError(f"未找到分析器: {analyzer_type}")
                return SingleAnalysisStrategy(self.analyzers, analyzer_type)
                
            elif mode == 'realtime':
                return self.strategies['realtime']
            else:
                raise ValueError(f"不支持的分析模式: {mode}")
                
        except Exception as e:
            self.logger.error(f"策略选择失败: {str(e)}")
            raise

    def _init_analyzers(self):
        """初始化分析器"""
        try:
            analyzers = {}
            analyzer_classes = [
                ('scene', SceneAnalyzer),
                ('furniture', FurnitureDetector),
                ('lighting', LightingAnalyzer),
                ('color', ColorAnalyzer),
                ('style', StyleAnalyzer)
            ]
            
            for name, analyzer_class in analyzer_classes:
                try:
                    self.logger.info(f"正在初始化 {name} 分析器...")
                    # 获取配置
                    config = self.model_config.get_analyzer_config(name)
                    self.logger.debug(f"{name} 分析器配置: {config}")
                    
                    # 检查必要的配置
                    if name in ['scene', 'furniture', 'style']:
                        model_path = config.get('model_path')
                        if not model_path:
                            self.logger.error(f"{name} 分析器缺少模型路径配置")
                            continue
                        
                        # 检查模型文件是否存在
                        project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
                        full_path = os.path.join(project_root, model_path)
                        self.logger.debug(f"检查模型文件: {full_path}")
                        if not os.path.exists(full_path):
                            self.logger.error(f"{name} 分析器模型文件不存在: {full_path}")
                            continue
                        
                        # 如果是场景分析器，还需要检查类别文件
                        if name == 'scene':
                            categories_path = config.get('categories_path')
                            if categories_path:
                                categories_full_path = os.path.join(project_root, categories_path)
                                self.logger.debug(f"检查类别文件: {categories_full_path}")
                                if not os.path.exists(categories_full_path):
                                    self.logger.error(f"场景类别文件不存在: {categories_full_path}")
                                    continue
                    
                    # 初始化分析器
                    self.logger.debug(f"开始创建 {name} 分析器实例...")
                    analyzer = analyzer_class(self.model_config)
                    self.logger.debug(f"{name} 分析器实例创建完成，检查初始化状态...")
                    
                    if not hasattr(analyzer, '_initialized'):
                        self.logger.error(f"{name} 分析器缺少 _initialized 属性")
                        continue
                        
                    if not analyzer._initialized:
                        self.logger.error(f"{name} 分析器初始化未完成，_initialized=False")
                        continue
                        
                    # 验证分析器是否有效初始化
                    if not hasattr(analyzer, 'analyze'):
                        raise RuntimeError(f"{name} 分析器缺少 analyze 方法")
                        
                    analyzers[name] = analyzer
                    self.logger.info(f"{name} 分析器初始化成功")
                    
                except Exception as e:
                    self.logger.error(f"{name} 分析器初始化失败: {str(e)}", exc_info=True)
                    continue
                    
            if not analyzers:
                raise ValueError("没有成功初始化任何分析器")
            
            self.logger.info(f"成功初始化的分析器: {list(analyzers.keys())}")
            return analyzers
            
        except Exception as e:
            self.logger.error("分析器初始化失败", exc_info=True)
            raise

    def _init_processors(self):
        """初始化处理器"""
        try:
            return {
                'image': ImageProcessor(self.model_config),
                'video': VideoProcessor(self.model_config),
                'realtime': RealtimeProcessor(self.model_config),
                'result': ResultProcessor()  # ResultProcessor 不需要 model_config
            }
        except Exception as e:
            self.logger.error("处理器初始化失败", exc_info=True)
            raise

    def _init_strategies(self):
        """初始化策略"""
        try:
            return {
                'comprehensive': ComprehensiveStrategy(self.analyzers),
                'single': None,  # 不在这里初始化，稍后动态创建
                'realtime': RealtimeStrategy(self.analyzers)
            }
        except Exception as e:
            self.logger.error("策略初始化失败", exc_info=True)
            raise 

    def _process_frame(self, frame: np.ndarray, analyzers: List[str]) -> Dict:
        """处理单帧图像"""
        try:
            results = {}
            for analyzer_name in analyzers:
                self.logger.debug(f"执行 {analyzer_name} 分析...")
                analyzer = self._get_analyzer(analyzer_name)
                if analyzer:
                    try:
                        # 执行分析
                        result = analyzer.analyze(frame)
                        if result:
                            results[analyzer_name] = result
                        self.logger.debug(f"{analyzer_name} 分析完成")
                    except Exception as e:
                        self.logger.error(f"{analyzer_name} 分析失败: {str(e)}")
                else:
                    self.logger.warning(f"找不到 {analyzer_name} 分析器")
                
            return results
            
        except Exception as e:
            self.logger.error(f"帧处理失败: {str(e)}")
            return {} 

    def _select_strategy(self, mode: str) -> BaseStrategy:
        """选择分析策略"""
        try:
            if mode == 'comprehensive':
                return self.strategies['comprehensive']
            elif mode == 'single':
                return self.strategies['single']
            elif mode == 'realtime':
                return self.strategies['realtime']
            else:
                raise ValueError(f"不支持的分析模式: {mode}")
        except Exception as e:
            self.logger.error(f"策略选择失败: {str(e)}")
            raise 

    def _validate_analyzer_results(self, results: Dict[str, Any]) -> bool:
        """验证分析器结果"""
        try:
            valid = True  # 用于跟踪所有验证结果
            
            for analyzer_type, result in results.items():
                if not result:
                    self.logger.error(f"{analyzer_type} 分析器返回空结果")
                    valid = False
                    continue
                    
                if not isinstance(result, dict):
                    self.logger.error(f"{analyzer_type} 分析器返回了非字典类型结果: {type(result)}")
                    valid = False
                    continue
                    
                if 'type' not in result:
                    self.logger.error(f"{analyzer_type} 分析器结果缺少 type 字段")
                    valid = False
                    continue
                    
                # 检查特定类型的必要字段
                if analyzer_type == 'lighting':
                    if 'overall_brightness' not in result or 'uniformity' not in result:
                        self.logger.error("光照分析结果缺少必要字段")
                        valid = False
                        
                elif analyzer_type == 'furniture':
                    if 'detected_items' not in result:
                        self.logger.warning("家具检测结果缺少 detected_items 字段")
                        result['detected_items'] = []
                    # 不要在这里提前返回
                    
                elif analyzer_type == 'style':
                    if 'style_type' not in result or 'confidence' not in result:
                        self.logger.error("风格分析结果缺少必要字段")
                        valid = False
                        
            return valid
            
        except Exception as e:
            self.logger.error(f"结果验证失败: {str(e)}")
            return False
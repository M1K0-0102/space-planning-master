import sys
import os
# 添加项目根目录到系统路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pathlib import Path
import logging
import cv2
import numpy as np
from typing import Dict, Any, Optional, Union, List, Tuple
import unittest
import time
import tkinter as tk
from tkinter import filedialog
from tqdm import tqdm
from src.pipeline.interior_design_pipeline import InteriorDesignPipeline
from src.pipeline.validators.input_validator import InputValidator
from src.pipeline.utils.result_types import ProcessedResult

# 导入必要的组件
from src.pipeline.strategies import (
    ComprehensiveStrategy,
    SingleAnalysisStrategy
)
from src.pipeline.processors import (
    VideoProcessor,
    ImageProcessor,
    RealtimeProcessor
)
from src.pipeline.utils.output_manager import OutputManager
from src.pipeline.utils.model_config import ModelConfig
from src.pipeline.analyzers import (
    SceneAnalyzer,
    ColorAnalyzer,
    LightingAnalyzer,
    StyleAnalyzer,
    FurnitureDetector
)
from src.pipeline.pipeline_coordinator import PipelineCoordinator

class VisualTest:
    _root = None  # 类变量，保存单例的 tk 根窗口
    
    def __init__(self):
        """初始化测试类"""
        self.logger = logging.getLogger("test.VisualTest")
        self.logger.info("初始化 VisualTest...")
        
        try:
            # 初始化 tkinter 根窗口（如果还没有初始化）
            if VisualTest._root is None:
                VisualTest._root = tk.Tk()
                VisualTest._root.withdraw()  # 隐藏主窗口
            
            # 初始化协调器
            self.logger.info("初始化 PipelineCoordinator...")
            self.coordinator = PipelineCoordinator()
            
            self.logger.info("测试初始化完成")
            
        except Exception as e:
            self.logger.error(f"初始化失败: {str(e)}")
            raise

    def run(self):
        """运行测试"""
        try:
            self.logger.info("开始运行测试...")
            
            # 1. 获取输入类型和策略
            input_type = self._get_input_type()
            strategy = self._get_strategy_type()
            self.logger.info(f"选择了{input_type}分析模式，{strategy}策略")
            
            # 2. 获取输入数据
            input_data = self._get_input_data(input_type)
            
            # 3. 执行分析
            result = self.coordinator.process(
                input_data=input_data,
                mode=strategy
            )
            
            # 4. 处理结果
            self._process_result(result)
            
        except Exception as e:
            self.logger.error(f"测试运行失败: {str(e)}")
            raise
        finally:
            self.logger.info("测试完成")

    def _get_input_type(self) -> str:
        """获取输入类型"""
        print("\n请选择输入类型:")
        print("1. video (批量分析)")
        print("2. image (图片分析)")
        print("3. realtime (实时分析)")
        
        while True:
            try:
                choice = input("请输入选项编号: ").strip()
                if choice == '1':
                    self.logger.info("选择了视频分析模式")
                    self._current_input_type = 'video'
                    return 'video'
                elif choice == '2':
                    self.logger.info("选择了图片分析模式")
                    self._current_input_type = 'image'
                    return 'image'
                elif choice == '3':
                    self.logger.info("选择了实时分析模式")
                    self._current_input_type = 'realtime'
                    return 'realtime'
                else:
                    print("无效的选项，请重新输入")
            except Exception as e:
                self.logger.error(f"输入类型选择失败: {str(e)}")
                print("输入错误，请重新选择")

    def _get_input_data(self, input_type: str) -> Union[str, np.ndarray]:
        """根据输入类型获取数据"""
        self.logger.info(f"获取{input_type}类型的输入数据...")
        
        try:
            if input_type == 'video':
                return self._get_video_path()
            elif input_type == 'image':
                return self._get_image_data()
            elif input_type == 'realtime':
                return self._get_realtime_data()
            else:
                raise ValueError(f"不支持的输入类型: {input_type}")
        except Exception as e:
            self.logger.error(f"输入数据获取失败: {str(e)}")
            raise

    def _get_video_path(self) -> str:
        """获取视频文件路径"""
        self.logger.info("请选择视频文件...")
        return self._get_file_path(
            title="选择视频文件",
            filetypes=[
                ("视频文件", "*.mp4;*.avi;*.mov"),
                ("所有文件", "*.*")
            ],
            valid_extensions=('.mp4', '.avi', '.mov')
        )

    def _get_image_data(self) -> np.ndarray:
        """获取图片数据"""
        self.logger.info("请选择图片文件...")
        image_path = self._get_file_path(
            title="选择图片文件",
            filetypes=[
                ("图片文件", "*.jpg;*.jpeg;*.png"),
                ("所有文件", "*.*")
            ],
            valid_extensions=('.jpg', '.jpeg', '.png')
        )
        
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"无法读取图片: {image_path}")
        return image

    def _get_file_path(self, title: str, filetypes: List, valid_extensions: Tuple[str, ...]) -> str:
        """通用文件选择方法"""
        try:
            file_path = filedialog.askopenfilename(
                title=title,
                filetypes=filetypes,
                initialdir=os.path.expanduser("~")
            )
            
            if not file_path:
                raise ValueError("未选择文件")
                
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"文件不存在: {file_path}")
                
            if not file_path.lower().endswith(valid_extensions):
                raise ValueError(f"不支持的文件格式，请选择以下格式: {', '.join(valid_extensions)}")
                
            self.logger.info(f"选择的文件: {file_path}")
            return file_path
            
        except Exception as e:
            self.logger.error(f"文件选择失败: {str(e)}")
            raise ValueError("文件选择失败") from e

    def _process_result(self, result: ProcessedResult):
        """处理分析结果"""
        try:
            if not result.success:
                self.logger.error(f"分析失败: {result.error}")
                return
                
            # 验证结果
            assert result.results, "分析结果为空"
            assert result.suggestions, "没有生成建议"
            assert result.metadata.get('formatted_results'), "格式化结果为空"
            assert result.metadata.get('formatted_suggestions'), "格式化建议为空"
            
            self.logger.info("测试通过")
                
        except AssertionError as e:
            self.logger.error(f"测试失败: {str(e)}")
            raise
        except Exception as e:
            self.logger.error(f"测试执行错误: {str(e)}")
            raise

    def _display_results(self, result: ProcessedResult):
        """显示分析结果"""
        try:
            print("\n=== 分析结果 ===")
            
            for analyzer_type, analysis in result.results.items():
                print(f"\n{analyzer_type}:")
                for key, value in analysis.items():
                    print(f"  {key}: {value}")
                    
            if result.suggestions:
                print("\n改进建议:")
                for suggestion in result.suggestions:
                    print(f"- {suggestion}")
                    
        except Exception as e:
            self.logger.error(f"结果显示失败: {str(e)}")

    def _get_strategy_type(self) -> str:
        """获取分析策略类型"""
        self.logger.info("请选择分析策略:")
        
        # 根据输入类型显示不同的选项
        if hasattr(self, '_current_input_type'):
            if self._current_input_type in ['video', 'image']:
                print("\n请选择分析策略:")
                print("1. comprehensive (综合分析)")
                print("2. single (单项分析)")
                
                while True:
                    try:
                        choice = input("请输入选项编号: ").strip()
                        if choice == '1':
                            self.logger.info("选择了综合分析策略")
                            return 'comprehensive'
                        elif choice == '2':
                            self.logger.info("选择了单项分析策略")
                            return 'single'
                        else:
                            print("无效的选项，请重新输入")
                    except Exception as e:
                        self.logger.error(f"策略选择失败: {str(e)}")
                        print("输入错误，请重新选择")
            
            elif self._current_input_type == 'realtime':
                self.logger.info("实时分析模式自动使用实时分析策略")
                return 'realtime'
        
        raise ValueError("未设置输入类型")

    def _get_realtime_data(self) -> str:
        """使用视频模拟实时输入流"""
        self.logger.info("开始模拟实时输入...")
        
        try:
            # 获取视频文件
            source = self._get_video_path()
            self.logger.info(f"选择视频文件: {source}")
            
            # 打开视频文件
            cap = cv2.VideoCapture(source)
            if not cap.isOpened():
                raise ValueError(f"无法打开视频文件: {source}")
            
            # 获取视频信息
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            self.logger.info(f"视频总帧数: {total_frames}, FPS: {fps}")
            
            # 创建显示窗口
            cv2.namedWindow('实时输入', cv2.WINDOW_NORMAL)
            cv2.namedWindow('实时分析', cv2.WINDOW_NORMAL)
            
            # 创建进度条
            pbar = tqdm(total=total_frames, desc="模拟实时输入进度")
            frame_count = 0
            
            while frame_count < total_frames:
                ret, frame = cap.read()
                if not ret:
                    self.logger.info("视频播放完成")
                    break
                
                frame_count += 1
                pbar.update(1)
                
                # 显示输入帧
                cv2.imshow('实时输入', frame)
                
                # 获取实时分析结果
                result = self.coordinator.process(
                    input_data=frame,
                    mode='realtime'
                )
                
                # 显示分析结果
                if result.success and result.results:
                    visualization = frame.copy()
                    
                    # 绘制检测框和标签
                    if 'furniture' in result.results:
                        for item in result.results['furniture']:
                            if 'bbox' in item:
                                x1, y1, x2, y2 = item['bbox']
                                cv2.rectangle(visualization, (x1, y1), (x2, y2), (0, 255, 0), 2)
                                cv2.putText(visualization, item.get('label', ''), 
                                          (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 
                                          0.5, (0, 255, 0), 2)
                    
                    # 添加分析结果文本
                    y_offset = 30
                    for analyzer_type, analysis in result.results.items():
                        if isinstance(analysis, dict):
                            text = f"{analyzer_type}: {str(analysis)}"
                            cv2.putText(visualization, text, 
                                      (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX,
                                      0.6, (255, 255, 255), 2)
                            y_offset += 25
                    
                    cv2.imshow('实时分析', visualization)
                
                # 控制播放速度并检查退出
                if cv2.waitKey(1000 // fps) & 0xFF == ord('q'):  # 按原始帧率播放
                    self.logger.info("用户中断播放")
                    break
            
            pbar.close()
            
            # 清理资源
            cap.release()
            cv2.destroyAllWindows()
            
            print("\n模拟实时输入完成")
            return source
            
        except Exception as e:
            self.logger.error(f"模拟实时输入失败: {str(e)}")
            raise

    def _save_test_report(self, results: Dict):
        """保存测试报告"""
        try:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            report_path = f"output/test_report_{timestamp}.txt"
            
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write("空间规划大师 - 测试报告\n")
                f.write("=" * 50 + "\n\n")
                
                # 写入基本信息
                f.write("测试信息:\n")
                f.write(f"测试时间: {timestamp}\n")
                f.write(f"测试模式: {results.get('mode', '未知')}\n\n")
                
                # 写入详细结果
                for key, value in results.items():
                    f.write(f"\n{key}:\n")
                    f.write("-" * 30 + "\n")
                    f.write(f"{value}\n")
                    
            self.logger.info(f"测试报告已保存至: {report_path}")
            
        except Exception as e:
            self.logger.error(f"保存测试报告失败: {str(e)}")

def main():
    # 创建输出目录
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'output', 'logs')
    os.makedirs(output_dir, exist_ok=True)
    
    # 生成日志文件名
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(output_dir, f'visual_test_{timestamp}.log')
    
    # 配置日志
    logging.basicConfig(
        level=logging.DEBUG,  # 改为 DEBUG 级别
        format='%(asctime)s - %(name)-25s - %(levelname)-8s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    
    # 单独设置场景分析器的日志级别为DEBUG
    scene_logger = logging.getLogger("src.pipeline.analyzers.scene_analyzer")
    scene_logger.setLevel(logging.DEBUG)
    
    # 过滤掉不重要的日志
    logging.getLogger("PIL").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    
    logging.info(f"日志文件保存在: {log_file}")
    logging.info("开始执行视觉测试...")
    
    try:
        test = VisualTest()
        test.run()
        logging.info("测试完成")
        
        # 添加等待用户输入
        input("\n按回车键退出...")
        
    except Exception as e:
        logging.error("测试失败", exc_info=True)
        raise

if __name__ == "__main__":
    main() 
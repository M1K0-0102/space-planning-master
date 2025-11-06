import cv2
import numpy as np
from typing import Optional, Dict, Any
import logging
import time
from queue import Queue
from threading import Thread, Lock
from .base_processor import BaseProcessor
from ..utils.model_config import ModelConfig

class RealtimeProcessor(BaseProcessor):
    """实时处理器 - 处理实时视频流"""
    
    _instance = None
    _initialized = False
    
    def __new__(cls, model_config: ModelConfig):
        if cls._instance is None:
            cls._instance = super(RealtimeProcessor, cls).__new__(cls)
        return cls._instance
    
    def __init__(self, model_config: ModelConfig):
        """初始化实时处理器
        Args:
            model_config: 模型配置
        """
        if self._initialized:
            return
            
        super().__init__(model_config)
        self.logger = logging.getLogger("pipeline.RealtimeProcessor")
        self.logger.propagate = False  # 避免重复日志
        
        # 帧处理队列
        self.frame_queue = Queue(maxsize=3)  # 限制队列大小避免内存溢出
        self.result_queue = Queue(maxsize=3)
        
        # 线程同步
        self.processing_lock = Lock()
        self.is_running = False
        
        # 性能监控
        self.frame_count = 0
        self.fps = 0
        self.last_time = time.time()
        
        # 初始化实时处理参数
        self.frame_buffer = []
        self.buffer_size = 5
        self.last_process_time = 0
        self.process_interval = 0.1  # 100ms
        
        # 从配置获取参数
        processor_config = self.model_config.get_processor_config('realtime')
        self.fps = processor_config.get('fps', 30)
        self.buffer_size = processor_config.get('buffer_size', 5)
        
        self._initialized = True
        
    def start(self):
        """启动实时处理"""
        self.is_running = True
        self.processing_thread = Thread(target=self._processing_loop)
        self.processing_thread.daemon = True
        self.processing_thread.start()
        
    def stop(self):
        """停止实时处理"""
        self.is_running = False
        if hasattr(self, 'processing_thread'):
            self.processing_thread.join()
            
    def process(self, input_data) -> Dict:
        """处理实时输入"""
        try:
            # 实现实时处理逻辑
            pass
        except Exception as e:
            self.logger.error("实时处理失败", exc_info=True)
            return {'error': str(e)}
            
    def preprocess(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """预处理视频帧"""
        try:
            if not isinstance(frame, np.ndarray):
                raise ValueError("输入必须是numpy数组")
                
            # 复制帧以避免修改原始数据
            frame = frame.copy()
            
            # 验证帧格式
            if len(frame.shape) != 3 or frame.shape[2] != 3:
                raise ValueError(f"无效的帧格式: {frame.shape}")
                
            # 确保是BGR格式和uint8类型
            if frame.dtype != np.uint8:
                frame = (frame * 255).astype(np.uint8)
                
            # 调整大小
            frame = cv2.resize(frame, (640, 480))
            
            # 颜色空间转换
            if len(frame.shape) == 2:
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            elif frame.shape[2] == 4:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
                
            return frame
            
        except Exception as e:
            self.logger.error("预处理失败", exc_info=True)
            return None
    
    def _postprocess(self, result: Dict) -> Dict:
        """后处理分析结果"""
        if not result:
            return {'error': '分析失败'}
            
        # 更新FPS
        current_time = time.time()
        self.fps = 1.0 / (current_time - self.last_time)
        self.last_time = current_time
        
        # 添加元数据
        result['metadata'] = {
            'fps': self.fps,
            'frame_count': self.frame_count,
            'timestamp': current_time
        }
        
        return result
            
    def _processing_loop(self):
        """处理循环"""
        while self.is_running:
            try:
                # 1. 获取待处理帧
                if self.frame_queue.empty():
                    time.sleep(0.01)  # 避免空转
                    continue
                    
                frame = self.frame_queue.get()
                
                # 2. 处理帧
                with self.processing_lock:
                    result = self._process_frame(frame)
                    
                # 3. 存储结果
                if result and not self.result_queue.full():
                    self.result_queue.put(result)
                    
            except Exception as e:
                self.logger.error(f"处理循环异常: {str(e)}")
                
    def _process_frame(self, frame: np.ndarray) -> Optional[Dict]:
        """处理单帧
        Args:
            frame: 预处理后的帧
        Returns:
            处理结果或None(处理失败)
        """
        try:
            # 这里应该实现具体的处理逻辑
            # 例如调用各种分析器进行分析
            result = {
                'frame': frame,
                'timestamp': time.time()
            }
            
            return result
            
        except Exception as e:
            self.logger.error(f"帧处理失败: {str(e)}")
            return None
            
    def get_performance_metrics(self) -> Dict[str, Any]:
        """获取性能指标"""
        return {
            'frame_count': self.frame_count,
            'fps': self.fps
        }

    def _update_performance_metrics(self):
        """更新性能指标"""
        try:
            # 更新帧计数
            self.frame_count += 1
            
            # 计算FPS
            current_time = time.time()
            time_diff = current_time - self.last_time
            if time_diff > 0:
                self.fps = 1.0 / time_diff
            self.last_time = current_time
            
            # 记录性能日志
            if self.frame_count % 30 == 0:  # 每30帧记录一次
                self.logger.info(f"性能指标 - FPS: {self.fps:.2f}, 总帧数: {self.frame_count}")
                
        except Exception as e:
            self.logger.error(f"性能指标更新失败: {str(e)}") 
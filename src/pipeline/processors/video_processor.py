from typing import Generator, Optional, List, Dict
import cv2
import numpy as np
import logging
from tqdm import tqdm
from ..utils.model_config import ModelConfig

class VideoProcessor:
    """视频处理器 - 负责视频帧的读取和预处理"""
    
    def __init__(self, model_config: ModelConfig):
        self.logger = logging.getLogger("pipeline.VideoProcessor")
        self.config = model_config
        
    def process_frames(self, video_path: str, sample_interval: Optional[int] = None) -> Generator[np.ndarray, None, None]:
        """处理视频帧
        Args:
            video_path: 视频文件路径
            sample_interval: 采样间隔（可选，默认每秒2帧）
        Yields:
            预处理后的视频帧
        """
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise ValueError(f"无法打开视频: {video_path}")
                
            # 获取视频信息
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            
            # 设置默认采样间隔（每秒2帧）
            if sample_interval is None:
                sample_interval = max(1, fps // 2)
                
            self.logger.info(f"开始处理视频: {video_path}")
            self.logger.info(f"视频信息: {total_frames}帧, {fps}fps, 采样间隔:{sample_interval}")
            
            frame_idx = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                    
                # 按间隔采样
                if frame_idx % sample_interval == 0:
                    # 预处理帧
                    processed_frame = self._preprocess_frame(frame)
                    if processed_frame is not None:
                        yield processed_frame
                        
                frame_idx += 1
                
            cap.release()
            
        except Exception as e:
            self.logger.error(f"视频处理失败: {str(e)}")
            raise
            
    def _preprocess_frame(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """预处理视频帧
        Args:
            frame: 原始视频帧
        Returns:
            预处理后的帧或None（如果处理失败）
        """
        try:
            if frame is None:
                return None
                
            # 1. 调整大小到标准尺寸
            target_size = (640, 480)  # 可以从配置读取
            frame = cv2.resize(frame, target_size)
            
            # 2. 色彩空间转换（如果需要）
            # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # 3. 其他预处理步骤...
            
            return frame
            
        except Exception as e:
            self.logger.error(f"帧预处理失败: {str(e)}")
            return None
        
    def process_video(self, video_path: str, strategy) -> List[Dict]:
        """处理整个视频文件
        Args:
            video_path: 视频文件路径
            strategy: 分析策略
        Returns:
            List[Dict]: 每一帧的分析结果列表
        """
        try:
            self.logger.info(f"开始处理视频: {video_path}")
            frame_results = []
            
            # 获取视频帧生成器
            frames = self.process_frames(video_path)
            
            # 使用tqdm显示进度
            for frame in tqdm(frames, desc="处理视频帧"):
                # 使用策略分析当前帧
                frame_result = strategy.execute(frame)
                if frame_result:
                    frame_results.append(frame_result)
                
            self.logger.info(f"视频处理完成，共处理 {len(frame_results)} 帧")
            return frame_results
            
        except Exception as e:
            self.logger.error(f"视频处理失败: {str(e)}")
            raise
        
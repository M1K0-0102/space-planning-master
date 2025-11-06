from typing import Union, Optional, Dict, Any, Tuple
import numpy as np
import cv2
import logging
from pathlib import Path
from enum import Enum

class AnalysisType(Enum):
    """分析类型枚举"""
    SCENE = 'scene'
    FURNITURE = 'furniture'
    LIGHTING = 'lighting'
    STYLE = 'style'
    COLOR = 'color'

class InputValidator:
    """输入验证器 - 验证各种输入的有效性"""
    
    # 支持的文件类型
    SUPPORTED_VIDEO_FORMATS = {
        '.mp4': ['avc1', 'hvc1'],  # H.264/HEVC
        '.avi': ['XVID', 'MJPG'],
        '.mov': ['avc1', 'mp4v'],
        '.mkv': ['avc1', 'hevc'],
        '.webm': ['vp9']
    }
    SUPPORTED_IMAGE_FORMATS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
    
    def __init__(self):
        self.logger = logging.getLogger("pipeline.InputValidator")
        
    def get_supported_video_formats(self) -> list:
        """获取支持的视频格式"""
        return [('视频文件', '*' + ';*'.join(self.SUPPORTED_VIDEO_FORMATS)),
                ('所有文件', '*.*')]
        
    def get_supported_image_formats(self) -> list:
        """获取支持的图片格式"""
        return [('图片文件', '*' + ';*'.join(self.SUPPORTED_IMAGE_FORMATS)),
                ('所有文件', '*.*')]

    def is_valid_video_format(self, file_path: Union[str, Path]) -> bool:
        """检查是否是支持的视频格式"""
        suffix = Path(file_path).suffix.lower()
        if suffix not in self.SUPPORTED_VIDEO_FORMATS:
            return False
        # 检查实际编解码器
        cap = cv2.VideoCapture(str(file_path))
        fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
        codec = chr(fourcc&0xff) + chr((fourcc>>8)&0xff) + chr((fourcc>>16)&0xff) + chr((fourcc>>24)&0xff)
        cap.release()
        return codec in self.SUPPORTED_VIDEO_FORMATS[suffix]

    def is_valid_image_format(self, file_path: Union[str, Path]) -> bool:
        """检查是否是支持的图片格式"""
        return Path(file_path).suffix.lower() in self.SUPPORTED_IMAGE_FORMATS

    def validate_image(self, image: Union[str, np.ndarray, Path]) -> Optional[np.ndarray]:
        """验证并加载图像输入
        Args:
            image: 图像路径或numpy数组
        Returns:
            验证后的图像数组或None(验证失败)
        """
        try:
            # 如果是路径，先加载图像
            if isinstance(image, (str, Path)):
                image_path = str(image)
                if not Path(image_path).exists():
                    self.logger.error(f"图像文件不存在: {image_path}")
                    return None
                image = cv2.imread(image_path)
                if image is None:
                    self.logger.error(f"图像加载失败: {image_path}")
                    return None
                
            # 验证图像数组
            if not isinstance(image, np.ndarray):
                self.logger.error(f"无效的图像类型: {type(image)}")
                return None
                
            if len(image.shape) != 3:
                self.logger.error(f"无效的图像维度: {image.shape}")
                return None
                
            if image.shape[2] != 3:
                self.logger.error(f"无效的图像通道数: {image.shape[2]}")
                return None
                
            # 新增分辨率检查
            min_resolution = (320, 240)
            if image.shape[0] < min_resolution[0] or image.shape[1] < min_resolution[1]:
                self.logger.error(f"图像分辨率过低: {image.shape[0]}x{image.shape[1]}")
                return None
                
            # 新增文件大小检查（当输入是路径时）
            if isinstance(image, (str, Path)):
                max_size_mb = 10  # 10MB
                file_size = Path(image).stat().st_size / (1024 * 1024)
                if file_size > max_size_mb:
                    self.logger.error(f"图像文件过大: {file_size:.1f}MB > {max_size_mb}MB")
                    return None
                
            return image
            
        except Exception as e:
            self.logger.error(f"图像验证失败: {str(e)}")
            return None
            
    def validate_video(self, video_path: Union[str, Path]) -> bool:
        """验证视频文件
        Args:
            video_path: 视频文件路径
        Returns:
            验证是否通过
        """
        try:
            video_path = str(video_path)
            # 检查文件格式
            if not self.is_valid_video_format(video_path):
                self.logger.error(f"不支持的视频格式: {Path(video_path).suffix}")
                return False
                
            # 检查文件大小
            max_size_mb = 500  # 500MB限制
            file_size = Path(video_path).stat().st_size / (1024 * 1024)
            if file_size > max_size_mb:
                self.logger.error(f"视频文件过大: {file_size:.1f}MB > {max_size_mb}MB")
                return False

            if not Path(video_path).exists():
                self.logger.error(f"视频文件不存在: {video_path}")
                return False
                
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                # 获取OpenCV错误码
                error_code = cap.get(cv2.CAP_PROP_POS_MSEC)
                self.logger.error(f"无法打开视频文件: {video_path} (错误码: {error_code})")
                return False
                
            # 检查视频基本信息
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # 检查视频时长
            max_duration = 300  # 5分钟
            duration = frame_count / fps if fps > 0 else 0
            if duration > max_duration:
                self.logger.error(f"视频时长过长: {duration//60:.0f}分{duration%60:.0f}秒 > {max_duration//60}分")
                return False

            if width <= 0 or height <= 0 or fps <= 0 or frame_count <= 0:
                self.logger.error(f"无效的视频参数: {width}x{height}@{fps}fps, {frame_count}帧")
                return False
                
            # 新增编解码器检查
            fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
            codec = ''.join([chr((fourcc >> 8 * i) & 0xFF) for i in range(4)])
            if codec not in ['avc1', 'hvc1']:  # H.264/HEVC
                self.logger.error(f"不支持的视频编解码器: {codec}")
                return False
                
            cap.release()
            return True
            
        except Exception as e:
            self.logger.error(f"视频验证失败: {str(e)}")
            return False
            
    def validate_config(self, config: Dict[str, Any]) -> bool:
        """验证配置参数"""
        try:
            required_fields = ['input_size', 'model_path', 'device']
            for field in required_fields:
                if field not in config:
                    self.logger.error(f"缺少必要的配置项: {field}")
                    return False
                    
            # 验证输入尺寸
            input_size = config['input_size']
            if not isinstance(input_size, (list, tuple)) or len(input_size) != 2:
                self.logger.error(f"无效的输入尺寸格式: {input_size}")
                return False
                
            # 验证模型路径
            model_path = Path(config['model_path'])
            if not model_path.exists():
                self.logger.error(f"模型文件不存在: {model_path}")
                return False
                
            # 验证设备
            device = config['device'].lower()
            if device not in ['cpu', 'cuda']:
                self.logger.error(f"无效的设备类型: {device}")
                return False
                
            return True
            
        except Exception as e:
            self.logger.error(f"配置验证失败: {str(e)}")
            return False
            
    def validate_frame(self, frame: np.ndarray) -> bool:
        """验证视频帧"""
        try:
            if frame is None:
                self.logger.error("帧为空")
                return False
                
            if not isinstance(frame, np.ndarray):
                self.logger.error(f"无效的帧类型: {type(frame)}")
                return False
                
            if len(frame.shape) != 3:
                self.logger.error(f"无效的帧维度: {frame.shape}")
                return False
                
            if frame.shape[2] not in [3, 4]:  # 允许BGR或BGRA
                self.logger.error(f"无效的帧通道数: {frame.shape[2]}")
                return False
                
            # 检查像素值范围
            if frame.dtype != np.uint8:
                self.logger.error(f"无效的数据类型: {frame.dtype}")
                return False
                
            # 检查有效像素比例
            valid_pixel_ratio = np.mean((frame >= 0) & (frame <= 255))
            if valid_pixel_ratio < 0.95:
                self.logger.error(f"无效像素比例过高: {valid_pixel_ratio*100:.1f}%")
                return False
                
            return True
            
        except Exception as e:
            self.logger.error(f"帧验证失败: {str(e)}")
            return False

    def validate_camera(self, camera_id: int = 0) -> Union[cv2.VideoCapture, None]:
        """验证并打开摄像头
        Args:
            camera_id: 摄像头ID
        Returns:
            视频捕获对象或None
        """
        try:
            cap = cv2.VideoCapture(camera_id)
            if not cap.isOpened():
                self.logger.error(f"摄像头打开失败: ID {camera_id}")
                return None
                
            return cap
            
        except Exception as e:
            self.logger.error(f"摄像头验证失败: {str(e)}")
            return None
            
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """预处理图片"""
        try:
            # 确保图片格式正确
            if len(image.shape) == 2:  # 灰度图转RGB
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            elif image.shape[2] == 4:  # RGBA转RGB
                image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
                
            # 统一图片大小
            image = cv2.resize(image, (640, 480))
            
            return image
            
        except Exception as e:
            self.logger.error(f"图片预处理失败: {str(e)}")
            return None
            
    def get_video_info(self, cap: cv2.VideoCapture) -> Tuple[int, int, float]:
        """获取视频信息
        Args:
            cap: 视频捕获对象
        Returns:
            (帧宽度, 帧高度, FPS)
        """
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        return width, height, fps

    def validate_analyzer_type(self, analyzer_type: str) -> bool:
        """验证分析器类型"""
        try:
            AnalysisType(analyzer_type)
            return True
        except ValueError:
            return False

    def validate_video_input(self, video_capture: cv2.VideoCapture) -> bool:
        """验证视频输入的有效性
        
        Args:
            video_capture: OpenCV视频捕获对象
            
        Returns:
            bool: 视频输入是否有效
        """
        try:
            if not isinstance(video_capture, cv2.VideoCapture):
                self.logger.error("输入必须是OpenCV的VideoCapture对象")
                return False
                
            if not video_capture.isOpened():
                self.logger.error("视频未成功打开")
                return False
                
            # 检查视频基本属性
            total_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = video_capture.get(cv2.CAP_PROP_FPS)
            width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            if total_frames <= 0:
                self.logger.error("视频帧数无效")
                return False
                
            if fps <= 0:
                self.logger.error("视频帧率无效")
                return False
                
            if width <= 0 or height <= 0:
                self.logger.error("视频分辨率无效")
                return False
                
            # 尝试读取第一帧
            ret, frame = video_capture.read()
            if not ret or frame is None:
                self.logger.error("无法读取视频帧")
                return False
                
            # 重置视频位置
            video_capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
            
            self.logger.info(f"视频验证通过 - 帧数: {total_frames}, FPS: {fps}, 分辨率: {width}x{height}")
            return True
            
        except Exception as e:
            self.logger.error(f"视频验证失败: {str(e)}")
            return False
            
    def validate_image_input(self, image: np.ndarray) -> bool:
        """验证图像输入的有效性"""
        try:
            if not isinstance(image, np.ndarray):
                self.logger.error("输入必须是numpy数组")
                return False
                
            if len(image.shape) != 3:
                self.logger.error("图像必须是3通道")
                return False
                
            if image.shape[2] != 3:
                self.logger.error("图像必须是BGR格式")
                return False
                
            if image.dtype != np.uint8:
                self.logger.error("图像必须是uint8类型")
                return False
                
            return True
            
        except Exception as e:
            self.logger.error(f"图像验证失败: {str(e)}")
            return False
            
    def validate_frame(self, frame: np.ndarray) -> bool:
        """验证单帧的有效性"""
        return self.validate_image_input(frame) 
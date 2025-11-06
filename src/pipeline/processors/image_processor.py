from .base_processor import BaseProcessor
import cv2
import numpy as np
from typing import Any, Dict, Optional
import logging
from ..utils import AnalysisType
import time
from ..utils.model_config import ModelConfig

class ImageProcessor(BaseProcessor):
    """图像处理器 - 处理单张图像"""
    
    def __init__(self, model_config: ModelConfig):
        """初始化图像处理器
        Args:
            model_config: 模型配置
        """
        super().__init__(model_config)
        self.logger = logging.getLogger("pipeline.ImageProcessor")
        # 从配置获取参数
        processor_config = self.model_config.get_processor_config('image')
        self.input_size = tuple(processor_config.get('input_size', [640, 480]))
        self.batch_size = processor_config.get('batch_size', 1)
        self.num_workers = processor_config.get('num_workers', 4)
        
    def preprocess(self, image: np.ndarray) -> Optional[np.ndarray]:
        """预处理图像
        Args:
            image: 输入图像
        Returns:
            处理后的图像或None(处理失败)
        """
        try:
            # 1. 尺寸调整
            target_size = (640, 480)  # 可以从配置中读取
            if image.shape[:2] != target_size:
                image = cv2.resize(image, target_size)
                
            # 2. 色彩空间转换
            if len(image.shape) == 2:  # 灰度图转RGB
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            elif image.shape[2] == 4:  # RGBA转RGB
                image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
                
            # 3. 数据类型转换
            if image.dtype != np.uint8:
                image = image.astype(np.uint8)
                
            # 4. 图像增强
            image = self._enhance_image(image)
            
            return image
            
        except Exception as e:
            self.logger.error(f"图像预处理失败: {str(e)}")
            return None
            
    def _enhance_image(self, image: np.ndarray) -> np.ndarray:
        """图像增强
        Args:
            image: 输入图像
        Returns:
            增强后的图像
        """
        try:
            # 1. 亮度和对比度调整
            alpha = 1.1  # 对比度
            beta = 5    # 亮度
            image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
            
            # 2. 降噪
            image = cv2.fastNlMeansDenoisingColored(image)
            
            return image
            
        except Exception as e:
            self.logger.error(f"图像增强失败: {str(e)}")
            return image

    def process(self, image: np.ndarray) -> Dict:
        """处理单张图片"""
        try:
            # 1. 预处理
            processed_image = self._preprocess(image)
            if processed_image is None:
                raise ValueError("图像预处理失败")
                
            # 2. 分析
            results = {}
            for name, analyzer in self.analyzers.items():
                try:
                    result = analyzer.analyze(processed_image)
                    if result:
                        results[name] = result
                except Exception as e:
                    self.logger.error(f"{name}分析失败: {str(e)}")
                    
            if not results:
                raise ValueError("所有分析器都失败了")
                
            # 3. 返回结果
            return {
                'type': 'image_analysis',
                'results': results,
                'metadata': {
                    'timestamp': time.time(),
                    'image_size': {
                        'width': processed_image.shape[1],
                        'height': processed_image.shape[0]
                    }
                }
            }
            
        except Exception as e:
            self.logger.error(f"图像处理失败: {str(e)}")
            return {'error': str(e)}

    def _analyze_image(self, image: np.ndarray) -> Any:
        """分析图像"""
        # 确保图像是BGR格式
        if len(image.shape) == 2:  # 灰度图转BGR
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        elif image.shape[2] == 4:  # BGRA转BGR
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        
        return image 
from typing import Dict, List
import numpy as np
import cv2
from sklearn.cluster import KMeans
from .model_config import ModelConfig

class FeatureExtractor:
    """统一的特征提取类"""
    
    def __init__(self, model_config: ModelConfig = None):
        """初始化特征提取器
        Args:
            model_config: 模型配置（可选）
        """
        self.model_config = model_config
        if model_config:
            # 从配置加载参数
            color_config = model_config.get_analyzer_config('color')
            self.color_clusters = color_config.get('clusters', 5)
            
            # 其他配置参数
            self.texture_method = 'sobel'  # 默认使用sobel算子
    
    @staticmethod
    def extract_lighting_features(frame: np.ndarray) -> Dict:
        """提取光照特征"""
        # 转换为HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        value = hsv[:, :, 2]
        
        # 计算有效区域
        valid_mask = (value > 20) & (value < 235)
        if np.sum(valid_mask) > 0:
            mean_brightness = np.mean(value[valid_mask])
            std_dev = np.std(value[valid_mask])
        else:
            mean_brightness = np.mean(value)
            std_dev = np.std(value)
            
        brightness = np.power(mean_brightness / 255.0, 0.6)  # gamma校正
        uniformity = 1.0 - (std_dev / 128.0)
        contrast = float((np.percentile(value, 90) - np.percentile(value, 10)) / 255.0)
        
        return {
            'brightness': float(brightness),
            'contrast': float(contrast),
            'uniformity': float(np.clip(uniformity, 0, 1))
        }
        
    def extract_color_features(self, frame: np.ndarray) -> Dict:
        """提取颜色特征"""
        try:
            # K-means聚类提取主要颜色
            pixels = frame.reshape(-1, 3)
            kmeans = KMeans(n_clusters=self.color_clusters, n_init=10)
            kmeans.fit(pixels)
            
            # 计算每个颜色的比例
            color_distribution = np.bincount(kmeans.labels_) / len(kmeans.labels_)
            
            return {
                'dominant_colors': kmeans.cluster_centers_.tolist(),
                'color_distribution': color_distribution.tolist()
            }
        except Exception as e:
            return {
                'dominant_colors': [],
                'color_distribution': []
            }
        
    @staticmethod
    def extract_texture_features(frame: np.ndarray) -> Dict:
        """提取纹理特征"""
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Sobel梯度
            sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
            sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1)
            
            # 计算纹理复杂度
            complexity = float(np.mean(np.sqrt(sobelx**2 + sobely**2)) / 255.0)
            
            return {
                'complexity': complexity,
                'gradient_x': float(np.mean(np.abs(sobelx)) / 255.0),
                'gradient_y': float(np.mean(np.abs(sobely)) / 255.0)
            }
        except Exception as e:
            return {
                'complexity': 0.0,
                'gradient_x': 0.0,
                'gradient_y': 0.0
            } 
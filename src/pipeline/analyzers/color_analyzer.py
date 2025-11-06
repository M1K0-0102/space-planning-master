from .base_analyzer import BaseAnalyzer
import cv2
import numpy as np
from typing import Dict, List, Tuple, Any, Optional, Union
from sklearn.cluster import KMeans
import torch
import logging
import hashlib
from collections import Counter
from ..utils.result_types import AnalyzerResult
import time
from ..utils.model_config import ModelConfig
from ..utils.feature_extractors import FeatureExtractor

class ColorAnalyzer(BaseAnalyzer):
    """颜色分析器"""
    _instances = {}
    
    def __new__(cls, model_config: ModelConfig):
        config_id = id(model_config)
        if config_id not in cls._instances:
            cls._instances[config_id] = super(ColorAnalyzer, cls).__new__(cls)
        return cls._instances[config_id]
    
    def __init__(self, model_config: ModelConfig):
        """初始化颜色分析器"""
        super().__init__(model_config)  # 调用父类初始化
        try:
            # 获取配置
            config = self.model_config.get_analyzer_config('color')
            self.num_colors = config.get('clusters', 5)
            
            # 初始化特征提取器
            self.feature_extractor = FeatureExtractor(model_config)
            
            self._initialized = True  # 设置初始化完成标志
            self.logger.info("颜色分析器初始化完成")
            
            # 添加输入尺寸
            self.input_size = (224, 224)
            
            # 颜色相关配置
            self.color_names = {
                'red': '红色', 'orange': '橙色', 'yellow': '黄色',
                'green': '绿色', 'blue': '蓝色', 'purple': '紫色',
                'brown': '棕色', 'white': '白色', 'black': '黑色',
                'gray': '灰色'
            }
            
            # 添加色温范围定义
            self.color_temp_ranges = {
                'sunset': (2000, 3500),  # 夕阳色温范围
                'daylight': (5500, 6500), # 日光色温范围
                'shade': (7000, 8000)     # 阴影区域色温范围
            }
            
            self.color_space = config.get('color_space', 'hsv')
            
        except Exception as e:
            self.logger.error(f"颜色分析器初始化失败: {str(e)}")
            raise

    def analyze(self, frame: np.ndarray) -> Dict:
        """分析图像颜色"""
        try:
            self.logger.debug(f"开始提取颜色特征: frame_shape={frame.shape}, dtype={frame.dtype}")
            
            # 转换到HSV颜色空间
            self.logger.debug("转换到HSV颜色空间")
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            self.logger.debug(f"HSV转换完成: shape={hsv.shape}")
            
            # 提取主要颜色
            main_colors = self._extract_main_colors(frame)
            avg_saturation = self._calculate_avg_saturation(hsv)
            harmony_score = self._calculate_harmony_score(main_colors)
            
            result = {
                'type': 'color',
                'confidence': harmony_score,
                'features': {
                    'main_colors': [
                        {
                            'rgb': color.tolist(),
                            'percentage': float(percentage)
                        }
                        for color, percentage in main_colors
                    ],
                    'avg_saturation': float(avg_saturation),
                    'harmony_score': float(harmony_score)
                },
                'metadata': {
                    'timestamp': time.time()
                }
            }
            
            self.logger.debug(f"颜色分析结果: {result}")
            return result
            
        except Exception as e:
            self.logger.error(f"颜色分析失败: {str(e)}")
            return {
                'type': 'color',
                'confidence': 0.0,
                'features': {},
                'metadata': {'error': str(e)}
            }

    def _extract_color_features(self, frame: np.ndarray) -> Dict:
        """提取颜色特征"""
        try:
            # 性能优化：降低分辨率
            scale = 0.5
            small_frame = cv2.resize(frame, None, fx=scale, fy=scale)
            
            self.logger.debug(f"开始提取颜色特征: frame_shape={small_frame.shape}, dtype={small_frame.dtype}")
            
            if not isinstance(small_frame, np.ndarray):
                self.logger.error("输入必须是numpy数组")
                return {}
            
            # 初始化特征字典
            features = {}
            
            # 确保是BGR格式和uint8类型
            if small_frame.dtype != np.uint8:
                self.logger.debug("转换为uint8类型")
                small_frame = (small_frame * 255).astype(np.uint8)
            
            # 确保是3通道
            if len(small_frame.shape) != 3 or small_frame.shape[2] != 3:
                self.logger.error(f"无效的帧格式: {small_frame.shape}")
                return {}
            
            # 转换到HSV颜色空间
            self.logger.debug("转换到HSV颜色空间")
            hsv = cv2.cvtColor(small_frame, cv2.COLOR_BGR2HSV)
            self.logger.debug(f"HSV转换完成: shape={hsv.shape}")
            
            # 提取颜色特征
            pixels = hsv.reshape(-1, 3)
            kmeans = KMeans(
                n_clusters=3,  # 减少聚类数
                random_state=42,
                n_init=5,      # 减少初始化次数
                max_iter=100   # 限制迭代次数
            )
            kmeans.fit(pixels)
            
            # 获取主要颜色及其比例
            colors = []
            for center in kmeans.cluster_centers_:
                color_name = self._get_color_name(center)
                colors.append(color_name)
            
            # 计算每种颜色的比例
            labels = kmeans.labels_
            color_counts = Counter(labels)
            total_pixels = len(labels)
            
            color_ratios = {}
            for label, count in color_counts.items():
                color_name = colors[label]
                ratio = count / total_pixels
                color_ratios[color_name] = float(ratio)
            
            features['main_colors'] = color_ratios
            
            # 计算整体色调
            features['avg_hue'] = float(np.mean(hsv[:, :, 0]))
            features['avg_saturation'] = float(np.mean(hsv[:, :, 1]))
            features['avg_value'] = float(np.mean(hsv[:, :, 2]))
            
            return features
            
        except Exception as e:
            self.logger.error(f"颜色特征提取失败: {str(e)}")
            return {}

    def _get_color_name(self, rgb: np.ndarray) -> str:
        """获取颜色名称
        Args:
            rgb: RGB颜色数组
        Returns:
            颜色名称
        """
        try:
            # 转换为HSV
            if isinstance(rgb, np.ndarray):
                rgb_reshaped = rgb[np.newaxis, np.newaxis, :]
                hsv = cv2.cvtColor(rgb_reshaped.astype(np.uint8), cv2.COLOR_RGB2HSV)[0, 0]
                h, s, v = hsv
                
                # 根据HSV值判断颜色
                if v < 75:  # 暗色
                    return '黑色'
                elif s < 50:  # 低饱和度
                    if v > 180:
                        return '白色'
                    else:
                        return '灰色'
                else:  # 有色彩
                    h = float(h)  # 确保是浮点数
                    if h <= 10 or h > 170:
                        return '红色'
                    elif 10 < h <= 25:
                        return '橙色'
                    elif 25 < h <= 35:
                        return '黄色'
                    elif 35 < h <= 85:
                        return '绿色'
                    elif 85 < h <= 135:
                        return '蓝色'
                    else:
                        return '紫色'
                    
            return '未知'
            
        except Exception as e:
            self.logger.error(f"颜色名称获取失败: {str(e)}")
            return '未知'

    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息"""
        return {
            'name': 'color',
            'type': 'ColorAnalyzer',
            'input_size': self.input_size,
            'color_names': self.color_names,
            'version': '1.0.0'
        }

    def validate_input(self, frame: np.ndarray) -> bool:
        """验证输入是否有效"""
        try:
            if frame is None:
                return False
            if not isinstance(frame, np.ndarray):
                return False
            if len(frame.shape) != 3:
                return False
            if frame.shape[2] != 3:
                return False
            if frame.dtype != np.uint8:
                return False
            return True
        except Exception:
            return False
            
    def _extract_main_colors(self, frame: np.ndarray) -> List[Tuple[np.ndarray, float]]:
        """提取主要颜色
        Args:
            frame: 输入图像
        Returns:
            List[Tuple[np.ndarray, float]]: 主要颜色列表，每个元素是(颜色数组, 占比)的元组
        """
        try:
            # 降低分辨率以提高性能
            small_frame = cv2.resize(frame, (64, 64))
            
            # 转换为RGB格式
            if len(small_frame.shape) == 2:
                small_frame = cv2.cvtColor(small_frame, cv2.COLOR_GRAY2RGB)
            elif small_frame.shape[2] == 4:
                small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGRA2RGB)
            elif small_frame.shape[2] == 3:
                small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
            
            # 重塑为像素列表
            pixels = small_frame.reshape(-1, 3)
            
            # 使用K-means聚类
            kmeans = KMeans(
                n_clusters=self.num_colors,
                random_state=42,
                n_init=5
            )
            kmeans.fit(pixels)
            
            # 获取聚类中心和标签
            colors = kmeans.cluster_centers_
            labels = kmeans.labels_
            
            # 计算每个颜色的占比
            unique_labels, counts = np.unique(labels, return_counts=True)
            percentages = counts / len(labels)
            
            # 返回颜色和占比的列表
            return [(colors[i].astype(np.uint8), float(percentages[i])) 
                    for i in range(len(colors))]
                
        except Exception as e:
            self.logger.error(f"主要颜色提取失败: {str(e)}")
            return []
        
    def _analyze_color_harmony(self, colors: list) -> Dict:
        """分析颜色和谐度"""
        # 实现颜色和谐度分析逻辑
        return {
            'score': 0.8,
            'suggestions': ['建议保持当前的配色方案']
        }

    def _create_model(self) -> None:
        """颜色分析不需要深度学习模型"""
        return None
        
    def _analyze_color_emotion(self, main_colors: List[Tuple[np.ndarray, float]]) -> Dict:
        """分析颜色情感"""
        try:
            # 初始化结果
            emotion = {
                'warmth': 0.0,  # 温暖度 (0-1)
                'energy': 0.0,  # 能量度 (0-1)
                'harmony': 0.0  # 和谐度 (0-1)
            }
            
            if not main_colors:
                return emotion
            
            # 计算温暖度
            total_warmth = 0.0
            total_weight = 0.0
            for color, weight in main_colors:
                # 转换为HSV
                color_bgr = color[np.newaxis, np.newaxis, :]
                hsv = cv2.cvtColor(color_bgr.astype(np.uint8), cv2.COLOR_BGR2HSV)[0, 0]
                h, s, v = float(hsv[0]), float(hsv[1]), float(hsv[2])
                
                # 暖色调权重 (红橙黄)
                if (h <= 30 or h >= 330) and s > 50:  # 红色
                    warmth = 1.0
                elif 30 < h <= 60:  # 橙黄
                    warmth = 0.8
                else:  # 其他颜色
                    warmth = 0.3
                    
                total_warmth += warmth * weight
                total_weight += weight
                
            emotion['warmth'] = total_warmth / total_weight if total_weight > 0 else 0.0
            
            # 计算能量度 (基于饱和度和亮度)
            total_energy = 0.0
            for color, weight in main_colors:
                color_bgr = color[np.newaxis, np.newaxis, :]
                hsv = cv2.cvtColor(color_bgr.astype(np.uint8), cv2.COLOR_BGR2HSV)[0, 0]
                s, v = float(hsv[1]), float(hsv[2])
                energy = (s/255.0 * 0.7 + v/255.0 * 0.3)  # 饱和度权重更大
                total_energy += energy * weight
                
            emotion['energy'] = total_energy / total_weight if total_weight > 0 else 0.0
            
            # 计算和谐度
            emotion['harmony'] = self._calculate_color_harmony(main_colors)
            
            return emotion
            
        except Exception as e:
            self.logger.error(f"颜色情感分析失败: {str(e)}")
            return {
                'warmth': 0.0,
                'energy': 0.0,
                'harmony': 0.0
            }
        
    def _analyze_color_relationship(self, colors: np.ndarray) -> Dict:
        """分析颜色之间的关系"""
        try:
            return {
                'harmony': self._calculate_color_harmony(colors),
                'contrast': self._calculate_color_contrast(colors),
                'balance': self._calculate_color_balance(colors)
            }
        except Exception as e:
            print(f"颜色关系分析失败: {str(e)}")
            return {} 

    def _calculate_color_harmony(self, colors: Union[np.ndarray, List[Tuple[np.ndarray, float]]]) -> float:
        """计算颜色和谐度
        Args:
            colors: 可以是以下两种格式之一:
                - 形状为 (n, 3) 的颜色数组，每行是一个 RGB 颜色
                - List[Tuple[np.ndarray, float]]，每个元素是(颜色数组, 占比)的元组
        Returns:
            float: 和谐度分数 (0-1)
        """
        try:
            # 如果输入是元组列表，提取颜色数组
            if isinstance(colors, list) and len(colors) > 0 and isinstance(colors[0], tuple):
                colors_array = np.array([color for color, _ in colors])
            else:
                colors_array = colors

            # 检查输入
            if len(colors_array) < 2:
                return 1.0  # 单色默认和谐
            
            # 转换为HSV颜色空间
            colors_hsv = []
            for color in colors_array:
                if isinstance(color, np.ndarray):
                    color_bgr = color[np.newaxis, np.newaxis, :]
                    color_hsv = cv2.cvtColor(color_bgr.astype(np.uint8), cv2.COLOR_BGR2HSV)
                    colors_hsv.append(color_hsv[0, 0])
            
            # 计算色相差异
            hue_diffs = []
            for i in range(len(colors_hsv)):
                for j in range(i + 1, len(colors_hsv)):
                    # 使用浮点数计算，并确保在0-180范围内
                    h1 = float(colors_hsv[i][0])
                    h2 = float(colors_hsv[j][0])
                    # 计算色相差异时考虑色环
                    diff = min(abs(h1 - h2), 180.0 - abs(h1 - h2))
                    hue_diffs.append(diff)
            
            # 计算和谐度分数
            avg_diff = float(np.mean(hue_diffs)) if hue_diffs else 0.0
            harmony_score = 1.0 - (avg_diff / 180.0)  # 归一化到0-1
            
            return float(harmony_score)
            
        except Exception as e:
            self.logger.error(f"和谐度计算失败: {str(e)}")
            return 0.0

    def _calculate_color_contrast(self, colors: np.ndarray) -> float:
        """计算颜色对比度
        
        Args:
            colors: 形状为 (n, 3) 的颜色数组，每行是一个 RGB 颜色
            
        Returns:
            float: 对比度分数 (0-1)
        """
        try:
            # 计算亮度
            luminance = np.dot(colors, [0.299, 0.587, 0.114])
            
            # 计算亮度对比
            contrasts = []
            for i in range(len(luminance)):
                for j in range(i + 1, len(luminance)):
                    l1, l2 = luminance[i], luminance[j]
                    contrast = abs(l1 - l2) / 255.0
                    contrasts.append(contrast)
            
            # 返回平均对比度
            if contrasts:
                return float(np.mean(contrasts))
            return 0.0
            
        except Exception as e:
            print(f"颜色对比度计算失败: {str(e)}")
            return 0.0

    def _calculate_color_balance(self, colors: np.ndarray) -> float:
        """计算颜色平衡度
        
        Args:
            colors: 形状为 (n, 3) 的颜色数组，每行是一个 RGB 颜色
            
        Returns:
            float: 平衡度分数 (0-1)
        """
        try:
            # 计算RGB通道的平均值
            rgb_means = np.mean(colors, axis=0)
            
            # 计算通道间的标准差
            channel_std = np.std(rgb_means)
            
            # 归一化到0-1范围
            balance = 1 - (channel_std / 255.0)
            
            return float(balance)
            
        except Exception as e:
            print(f"颜色平衡度计算失败: {str(e)}")
            return 0.0 

    def _get_harmony_level(self, score: float) -> str:
        """根据和谐度分数返回等级"""
        if score >= 0.8:
            return "极高"
        elif score >= 0.6:
            return "较高"
        elif score >= 0.4:
            return "一般"
        elif score >= 0.2:
            return "较低"
        else:
            return "极低" 

    def _get_complementary_color(self, rgb: List[int]) -> Dict:
        """获取互补色"""
        try:
            # 转换为HSV
            rgb_arr = np.array([[rgb]], dtype=np.uint8)
            hsv = cv2.cvtColor(rgb_arr, cv2.COLOR_RGB2HSV)
            h, s, v = hsv[0][0]
            
            # 计算互补色相
            h_comp = (h + 180) % 180
            
            # 转回RGB
            hsv_comp = np.array([[[h_comp, s, v]]], dtype=np.uint8)
            rgb_comp = cv2.cvtColor(hsv_comp, cv2.COLOR_HSV2RGB)[0][0]
            
            return {
                'rgb': rgb_comp.tolist(),
                'name': self._get_color_name(rgb_comp)
            }
            
        except Exception as e:
            print(f"互补色计算失败: {str(e)}")
            return {'rgb': [0, 0, 0], 'name': 'unknown'} 

    def _get_analogous_colors(self, rgb: List[int], angle: int = 30) -> List[Dict]:
        """获取类似色
        
        Args:
            rgb: RGB颜色值
            angle: 色相角度差（默认30度）
            
        Returns:
            List[Dict]: 类似色列表
        """
        try:
            # 转换为HSV
            rgb_arr = np.array([[rgb]], dtype=np.uint8)
            hsv = cv2.cvtColor(rgb_arr, cv2.COLOR_RGB2HSV)
            h, s, v = hsv[0][0]
            
            # 计算类似色相
            analogous_colors = []
            for delta in [-angle, angle]:
                h_similar = (h + delta) % 180
                hsv_similar = np.array([[[h_similar, s, v]]], dtype=np.uint8)
                rgb_similar = cv2.cvtColor(hsv_similar, cv2.COLOR_HSV2RGB)[0][0]
                
                analogous_colors.append({
                    'rgb': rgb_similar.tolist(),
                    'name': self._get_color_name(rgb_similar)
                })
                
            return analogous_colors
            
        except Exception as e:
            print(f"类似色计算失败: {str(e)}")
            return [] 

    def _calculate_warmth(self, colors: List[Dict]) -> float:
        """计算色彩温暖度
        
        Args:
            colors: 颜色列表，每个颜色包含 rgb 值和占比
            
        Returns:
            float: 温暖度分数 (0-1)，1表示最暖
        """
        try:
            warmth_scores = []
            for color in colors:
                rgb = np.array(color['rgb'])
                # 红色和黄色分量越高，越暖
                warmth = (rgb[0] + rgb[1] - rgb[2]) / (255 * 2)
                # 考虑颜色占比
                warmth_scores.append(warmth * color['percentage'])
            
            # 返回加权平均温暖度
            return float(max(0.0, min(1.0, sum(warmth_scores))))
            
        except Exception as e:
            print(f"温暖度计算失败: {str(e)}")
            return 0.0

    def _calculate_energy(self, colors: List[Dict]) -> float:
        """计算色彩能量度"""
        try:
            energy_scores = []
            for color in colors:
                rgb = np.array(color['rgb'])
                # 亮度和饱和度越高，能量越高
                brightness = np.mean(rgb) / 255
                saturation = (np.max(rgb) - np.min(rgb)) / 255
                energy = (brightness + saturation) / 2
                energy_scores.append(energy * color['percentage'])
            
            return float(max(0.0, min(1.0, sum(energy_scores))))
            
        except Exception as e:
            print(f"能量度计算失败: {str(e)}")
            return 0.0

    def _calculate_harmony(self, colors: List[Dict]) -> float:
        """计算色彩和谐度"""
        try:
            if len(colors) < 2:
                return 1.0
            
            # 提取RGB值和占比
            rgb_values = np.array([color['rgb'] for color in colors])
            percentages = np.array([color['percentage'] for color in colors])
            
            # 计算加权和谐度
            harmony = self._calculate_color_harmony(rgb_values)
            
            return float(harmony)
            
        except Exception as e:
            print(f"和谐度计算失败: {str(e)}")
            return 0.0 

    def visualize(self, frame: np.ndarray, result: Dict) -> np.ndarray:
        """可视化色彩分析结果"""
        try:
            vis = frame.copy()
            height, width = vis.shape[:2]
            
            # 绘制主要颜色条
            if 'main_colors' in result:
                bar_height = 50
                colors = result['main_colors']
                x_start = 0
                
                for color in colors:
                    # 计算颜色条宽度（基于颜色占比）
                    bar_width = int(width * color['percentage'])
                    
                    # 绘制颜色条
                    cv2.rectangle(vis,
                                (x_start, height-bar_height),
                                (x_start+bar_width, height),
                                color['rgb'],
                                -1)
                                
                    # 显示颜色名称和占比
                    cv2.putText(vis,
                               f"{self._get_color_name(color['rgb'])} ({color['percentage']:.1%})",
                               (x_start+5, height-10),
                               cv2.FONT_HERSHEY_SIMPLEX,
                               0.5, (255, 255, 255), 1)
                                
                    x_start += bar_width
                    
            # 显示色彩和谐度
            if 'harmony_score' in result:
                harmony_score = result['harmony_score']
                harmony_level = self._get_harmony_level(harmony_score)
                cv2.putText(vis,
                           f"和谐度: {harmony_score:.2f} ({harmony_level})",
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                           0.7, (0, 255, 0), 2)
                                
            # 显示色彩情感
            if 'emotion' in result:
                emotion = result['emotion']
                cv2.putText(vis,
                           f"主导情感: {emotion['dominant']}",
                           (10, 60), cv2.FONT_HERSHEY_SIMPLEX,
                           0.7, (0, 255, 0), 2)
                                
            return vis
            
        except Exception as e:
            self.log_error(f"色彩可视化失败: {str(e)}")
            return frame 

    def _get_color_scheme(self, hsv: np.ndarray) -> Dict[str, float]:
        """获取颜色方案"""
        try:
            # 定义颜色范围
            color_ranges = {
                'red': {'name': '红色', 'ranges': [(0, 10), (170, 180)]},
                'orange': {'name': '橙色', 'ranges': [(10, 25)]},
                'yellow': {'name': '黄色', 'ranges': [(25, 35)]},
                'green': {'name': '绿色', 'ranges': [(35, 85)]},
                'blue': {'name': '蓝色', 'ranges': [(85, 130)]},
                'purple': {'name': '紫色', 'ranges': [(130, 155)]},
                'pink': {'name': '粉色', 'ranges': [(155, 170)]}
            }
            
            # 计算每个颜色的比例
            color_ratios = {}
            total_pixels = hsv.shape[0] * hsv.shape[1]
            
            for color_info in color_ranges.values():
                mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
                for r in color_info['ranges']:
                    mask += cv2.inRange(hsv, (r[0], 50, 50), (r[1], 255, 255))
                ratio = np.count_nonzero(mask) / total_pixels
                color_ratios[color_info['name']] = float(ratio)
                
            return color_ratios
            
        except Exception as e:
            self.log_error(f"颜色方案提取失败: {str(e)}")
            return {} 

    def _get_dominant_colors(self, frame: np.ndarray, n_colors: int = 5) -> List[Dict[str, Any]]:
        """获取主要颜色"""
        try:
            # 将图像转换为RGB
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # 重塑为二维数组
            pixels = rgb.reshape(-1, 3)
            
            # K-means聚类
            kmeans = KMeans(n_clusters=n_colors, n_init=10, random_state=42)
            kmeans.fit(pixels)
            
            # 获取主要颜色
            colors = kmeans.cluster_centers_
            
            # 计算每个颜色的比例
            labels = kmeans.labels_
            counts = np.bincount(labels)
            percentages = counts / len(labels)
            
            # 转换为RGB值和百分比
            dominant_colors = []
            for color, percentage in zip(colors, percentages):
                dominant_colors.append({
                    'rgb': [int(c) for c in color],
                    'percentage': float(percentage)
                })
                
            return dominant_colors
            
        except Exception as e:
            self.log_error(f"主要颜色提取失败: {str(e)}")
            return [] 

    def _compute_frame_hash(self, frame: np.ndarray) -> str:
        """计算帧的哈希值"""
        # 降采样后再计算哈希以提高效率
        small_frame = cv2.resize(frame, (16, 16))
        return hashlib.md5(small_frame.tobytes()).hexdigest()
        
    def _preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """预处理帧"""
        # 1. 高斯模糊减少噪声
        blurred = cv2.GaussianBlur(frame, (5, 5), 0)
        
        # 2. 转换颜色空间
        rgb_frame = cv2.cvtColor(blurred, cv2.COLOR_BGR2RGB)
        
        # 3. 归一化
        normalized = rgb_frame.astype(float) / 255.0
        
        return normalized

    def _log_analysis_start(self):
        """记录分析开始"""
        self.logger.info("开始分析颜色")

    def _log_preprocessing(self, frame: np.ndarray):
        """记录预处理"""
        self.logger.info("预处理图像")

    def _log_analysis_result(self, result: Dict[str, Any]):
        """记录分析结果"""
        self.logger.info("分析结果: %s", result)

    def _log_error(self, e: Exception, action: str):
        """记录错误"""
        self.logger.error(f"{action}失败: {str(e)}")

    def _process_colors(self, colors: List[Dict]) -> Dict:
        """处理颜色分析结果"""
        stats = {
            'harmony_score': self._calculate_harmony(colors),
            'average_saturation': self._calculate_saturation(colors),
            'average_brightness': self._calculate_brightness(colors),
            'color_scheme': self._determine_color_scheme(colors)
        }
        
        return {
            'dominant_colors': colors,
            'stats': stats,
            'count': len(colors)
        }

    def _analyze(self, frame: np.ndarray) -> Dict:
        """分析颜色"""
        try:
            # 使用共享特征提取器
            color_features = self.feature_extractor.extract_color_features(frame)
            
            # 转换颜色为命名颜色
            named_colors = []
            for color in color_features['dominant_colors']:
                named_colors.append({
                    'rgb': color.tolist(),
                    'name': self._get_color_name(color),
                    'percentage': float(color_features['color_distribution'][len(named_colors)])
                })
            
            harmony_score = self._calculate_harmony_score(named_colors)
            avg_saturation = self._calculate_avg_saturation(cv2.cvtColor(frame, cv2.COLOR_BGR2HSV))
            
            # 扩展颜色分析
            color_emotion = self._analyze_color_emotion(named_colors)
            color_temp = self._estimate_color_temperature(frame)
            
            result = {
                'type': 'color',
                'confidence': harmony_score,
                'features': {
                    'main_colors': named_colors,
                    'color_metrics': {
                        'harmony_score': harmony_score,
                        'avg_saturation': avg_saturation,
                        'color_temperature': color_temp
                    },
                    'emotional_metrics': {
                        'warmth': color_emotion['warmth'],
                        'energy': color_emotion['energy'],
                        'harmony': color_emotion['harmony']
                    },
                    'color_scheme': self._analyze_color_scheme(named_colors)
                },
                'metadata': {'timestamp': time.time()}
            }
            
            self.logger.info(f"[颜色分析器] 输出: \n"
                            f"- 主要颜色 ({len(named_colors)} 种):\n" +
                            "\n".join([f"  • {color['name']}: {color['percentage']:.1%}"
                                     for color in named_colors[:3]]) + "\n"
                            f"- 颜色特征:\n"
                            f"  • 和谐度: {harmony_score:.2f}\n"
                            f"  • 饱和度: {avg_saturation:.2f}\n"
                            f"  • 色温: {color_temp:.0f}K\n"
                            f"- 色彩情感:\n"
                            f"  • 温暖度: {color_emotion['warmth']:.2f}\n"
                            f"  • 能量度: {color_emotion['energy']:.2f}")
            return result
        except Exception as e:
            self.logger.error(f"颜色分析失败: {str(e)}")
            return None

    def _is_sunset_lighting(self, frame: np.ndarray, color_temp: float) -> bool:
        """判断是否是夕阳照明场景"""
        try:
            # 1. 检查色温是否在夕阳范围内
            is_sunset_temp = self.color_temp_ranges['sunset'][0] <= color_temp <= self.color_temp_ranges['sunset'][1]
            
            # 2. 检查光照方向（通过亮度梯度）
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gradient_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)
            gradient_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)
            
            # 计算梯度方向
            gradient_direction = np.arctan2(gradient_y, gradient_x)
            avg_direction = np.mean(gradient_direction)
            
            # 3. 检查红橙色调的空间分布
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            red_orange_mask = cv2.inRange(hsv, (0, 50, 50), (30, 255, 255))
            red_orange_ratio = np.sum(red_orange_mask) / (frame.shape[0] * frame.shape[1])
            
            # 综合判断
            return (is_sunset_temp and 
                   (-np.pi/4 <= avg_direction <= np.pi/4) and  # 横向光照
                   red_orange_ratio > 0.2)  # 适当的红橙色比例
                   
        except Exception as e:
            self.logger.error(f"夕阳场景判断失败: {str(e)}")
            return False

    def _estimate_color_temperature(self, frame: np.ndarray) -> float:
        """估算场景色温"""
        try:
            # 转换到RGB空间
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # 计算平均RGB值
            avg_color = np.mean(rgb, axis=(0,1))
            r, g, b = avg_color
            
            # 使用McCamy's公式计算色温
            if b == 0:
                b = 1e-6
            x = (r - g) / (g - b)
            
            # McCamy's approximation
            n = (x - 0.3320) / (0.1858 - x)
            CCT = 449 * n**3 + 3525 * n**2 + 6823.3 * n + 5520.33
            
            return CCT
            
        except Exception as e:
            self.logger.error(f"色温估算失败: {str(e)}")
            return 6500  # 返回标准日光色温

    def _analyze_main_colors(self, frame: np.ndarray) -> Dict:
        """分析主色调"""
        # 实现分析主色调的逻辑
        return {}

    def _load_model(self, config: Dict) -> None:
        """加载颜色分析模型"""
        try:
            # 颜色分析不需要深度学习模型
            self.logger.info("颜色分析器初始化成功")
        except Exception as e:
            self.logger.error(f"颜色分析器初始化失败: {str(e)}")
            raise

    @property
    def confidence_threshold(self) -> float:
        """获取置信度阈值"""
        config = self.model_config.get_analyzer_config('color')
        threshold = config.get('confidence_threshold', 0.3)  # 颜色分析默认阈值
        self._confidence_threshold = threshold  # 更新基类中的值
        return self._confidence_threshold

    @confidence_threshold.setter
    def confidence_threshold(self, value: float):
        """设置置信度阈值"""
        self._confidence_threshold = value

    def _analyze_color_scheme(self, main_colors: List[Tuple[np.ndarray, float]]) -> str:
        """分析颜色方案"""
        try:
            # 获取和谐度分数
            harmony_score = self._calculate_color_harmony(main_colors)
            
            # 根据和谐度判断配色方案
            if harmony_score > 0.8:
                return "单色调和"
            elif 0.6 <= harmony_score <= 0.8:
                return "类似色调和"
            elif 0.4 <= harmony_score < 0.6:
                return "补色调和"
            else:
                return "对比色调和"
            
        except Exception as e:
            self.logger.error(f"颜色方案分析失败: {str(e)}")
            return "未知方案"

    def _calculate_harmony_score(self, colors: List[Dict]) -> float:
        """计算颜色和谐度分数
        Args:
            colors: 主要颜色列表，每个元素是(颜色数组, 占比)的元组
        Returns:
            和谐度分数 (0-1)
        """
        try:
            if len(colors) < 2:
                return 0.5  # 单个颜色应返回中性值0.5更合理
            
            # 转换为HSV颜色空间进行计算
            colors_hsv = []
            for color, _ in colors:
                if isinstance(color, np.ndarray):  # 确保是numpy数组
                    color_bgr = color[np.newaxis, np.newaxis, :]
                    color_hsv = cv2.cvtColor(color_bgr.astype(np.uint8), cv2.COLOR_BGR2HSV)
                    colors_hsv.append(color_hsv[0, 0])
            
            # 计算色相差异
            hue_diffs = []
            for i in range(len(colors_hsv)):
                for j in range(i + 1, len(colors_hsv)):
                    # 使用浮点数计算，并确保在0-180范围内
                    h1 = float(colors_hsv[i][0])
                    h2 = float(colors_hsv[j][0])
                    # 计算色相差异时考虑色环
                    diff = min(abs(h1 - h2), 180.0 - abs(h1 - h2))
                    hue_diffs.append(diff)
                
            # 计算和谐度分数
            avg_diff = float(np.mean(hue_diffs)) if hue_diffs else 0.0
            harmony_score = 1.0 - (avg_diff / 180.0)  # 归一化到0-1
            
            return float(harmony_score)
            
        except Exception as e:
            self.logger.error(f"和谐度计算失败: {str(e)}")
            return 0.0

    def _calculate_avg_saturation(self, hsv_img: np.ndarray) -> float:
        """计算平均饱和度"""
        try:
            # HSV中的S通道表示饱和度
            saturation = hsv_img[:, :, 1]
            return float(np.mean(saturation))
        except Exception as e:
            self.logger.error(f"饱和度计算失败: {str(e)}")
            return 0.0
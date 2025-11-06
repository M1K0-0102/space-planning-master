from .base_analyzer import BaseAnalyzer
import os
import torch
import torchvision.transforms as transforms
import torchvision.models as models
import numpy as np
import cv2
from typing import Dict, List, Any, Tuple, Optional
import torch.nn as nn
import torch.nn.functional as F
from ..utils.model_config import ModelConfig
from ..utils.result_processor import ResultProcessor
import logging
import time
from ..utils.result_types import (
    AnalysisResult,
    ImageAnalysisResult,
    VideoAnalysisResult,
    AnalyzerResult
)
from ..utils.feature_extractors import FeatureExtractor
from PIL import Image
import yaml  # 确保导入yaml库

class SceneAnalyzer(BaseAnalyzer):
    """场景分析器 - 专注于场景类型识别"""
    
    _instances = {}  # 使用字典存储不同配置的实例
    
    def __new__(cls, model_config: ModelConfig):
        config_id = id(model_config)
        if config_id not in cls._instances:
            cls._instances[config_id] = super(SceneAnalyzer, cls).__new__(cls)
        return cls._instances[config_id]
    
    def __init__(self, model_config: ModelConfig):
        """初始化场景分析器"""
        if hasattr(self, '_initialized') and self._initialized:
            return
            
        try:
            super().__init__(model_config)
            
            # 获取场景配置
            scene_config = self.model_config.get_analyzer_config('scene')
            
            # 设置日志
            self.logger.setLevel(logging.DEBUG)
            
            # 设置路径
            self.model_path = os.path.join(
                self.model_config.get_model_dir(),
                'resnet50_places365.pth'
            )
            
            # 获取类别文件路径
            self.categories_path = self.model_config.get_categories_path()
            
            # 加载类别
            self.categories = self._load_categories()
            if not self.categories:
                raise ValueError("无法加载场景类别")
                
            self.logger.info(f"成功加载 {len(self.categories)} 个场景类别")
            
            # 加载模型
            self.model = self._create_model()
            self.model.eval()
            
            # 调整阈值
            self.confidence_threshold = scene_config.get('confidence_threshold', 0.2)  # 降低默认阈值
            
            self._initialized = True
            self.logger.info("场景分析器初始化完成")
            
        except Exception as e:
            self.logger.error(f"场景分析器初始化失败: {str(e)}")
            raise

    def _load_categories(self) -> List[str]:
        """加载场景类别"""
        try:
            categories = []
            with open(self.categories_path, 'r', encoding='utf-8') as f:
                for line in f:
                    # 格式: /a/机场跑道 0
                    parts = line.strip().split()
                    if len(parts) >= 2:
                        # 提取纯场景名称（去掉路径前缀和索引）
                        category = parts[0].split('/')[-1]  # 只保留最后一个部分
                        categories.append(category)
                        
            if not categories:
                self.logger.error("未能加载任何类别")
                return []
                
            self.logger.info(f"成功加载 {len(categories)} 个中文场景类别")
            self.logger.debug(f"类别示例: {categories[:5]}")  # 输出前5个类别作为示例
            return categories
            
        except Exception as e:
            self.logger.error(f"加载场景类别失败: {str(e)}")
            return []

    def _create_model(self) -> nn.Module:
        """创建并加载模型"""
        try:
            # 创建模型
            model = models.resnet50(pretrained=False)
            model.fc = nn.Linear(model.fc.in_features, 365)  # Places365类别数
            
            # 加载权重
            checkpoint = torch.load(self.model_path, map_location=self.device)
            state_dict = checkpoint.get('state_dict', checkpoint)
            
            # 处理权重前缀
            new_state_dict = {
                k.replace('module.', ''): v 
                for k, v in state_dict.items()
            }
            
            # 加载权重
            model.load_state_dict(new_state_dict, strict=False)
            model = model.to(self.device)
            
            self.logger.info("模型加载成功")
            return model
            
        except Exception as e:
            self.logger.error(f"模型创建失败: {str(e)}")
            raise

    def _analyze(self, frame: np.ndarray) -> Dict[str, Any]:
        """实现基类的抽象方法"""
        try:
            self.logger.debug("开始场景分析...")
            
            # 1. 预处理
            self.logger.debug(f"输入图像形状: {frame.shape}")
            tensor = self._preprocess_image(frame)
            if tensor is None:
                self.logger.error("预处理返回空张量")
                return {}
            
            self.logger.debug(f"预处理后张量形状: {tensor.shape}, 类型: {tensor.dtype}")
            
            # 2. 推理
            with torch.no_grad():
                output = self.model(tensor)
                self.logger.debug(f"模型输出形状: {output.shape}")
                
                probs = F.softmax(output[0], dim=0)
                self.logger.debug(f"Softmax后概率形状: {probs.shape}")
                
                # 获取top1结果
                confidence, pred_idx = torch.max(probs, 0)
                confidence = confidence.item()
                pred_idx = pred_idx.item()
                
                self.logger.debug(f"预测索引: {pred_idx}, 置信度: {confidence}")
                
                # 获取场景类型
                scene_type = self._get_scene_type(pred_idx)
                
                # 应用置信度阈值
                if confidence < self.confidence_threshold:
                    self.logger.debug(f"置信度 {confidence} 低于阈值 {self.confidence_threshold}")
                    scene_type = "其他"
                    confidence = 0.0
            
            # 提取更多特征
            spatial_features = {
                'area': self._estimate_area(frame, scene_type),
                'symmetry': self._analyze_symmetry(frame),
                'wall_visibility': self._analyze_wall_visibility(frame),
                'natural_light': self._analyze_natural_light(frame)
            }

            result = {
                'type': 'scene',
                'confidence': confidence,
                'features': {
                    'scene_type': scene_type,
                    'scene_probs': {scene_type: confidence},
                    'spatial_features': spatial_features,
                    'texture_features': self._analyze_texture(frame),
                    'lighting_features': self._analyze_lighting(frame)
                },
                'metadata': {'timestamp': time.time()}
            }
            
            self.logger.info(f"[场景分析器] 输出: \n"
                            f"- 场景类型: {scene_type} (置信度={confidence:.2f})\n"
                            f"- 空间特征:\n"
                            f"  • 面积: {spatial_features['area']:.1f}㎡\n"
                            f"  • 对称性: {spatial_features['symmetry']:.2f}\n"
                            f"  • 墙面可见度: {spatial_features['wall_visibility']:.2f}\n"
                            f"  • 自然光评分: {spatial_features['natural_light']:.2f}")
            self.logger.info(f"所有类别的置信度: {probs.tolist()}")
            return result
            
        except Exception as e:
            self.logger.error(f"场景分析失败: {str(e)}")
            return {}

    def _preprocess_image(self, frame: np.ndarray) -> Optional[torch.Tensor]:
        """预处理图像"""
        try:
            # 1. BGR to RGB
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # 2. 调整大小
            resized = cv2.resize(rgb, (224, 224))
            
            # 3. 归一化
            img = resized.astype(np.float32) / 255.0  # 确保使用float32
            img = (img - np.array([0.485, 0.456, 0.406], dtype=np.float32)) / np.array([0.229, 0.224, 0.225], dtype=np.float32)
            
            # 4. 转换为tensor并指定dtype
            tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float()  # 明确指定为float
            
            return tensor.to(self.device)
            
        except Exception as e:
            self.logger.error(f"预处理失败: {str(e)}")
            return None

    def _error_result(self, message: str) -> Dict:
        """错误结果"""
        return {
            'type': 'sceneanalyzer',
            'confidence': 0.0,
            'features': {
                'scene_type': '其他',
                'scene_probs': {'其他': 0.0}
            },
            'metadata': {
                'error': True,
                'message': message,
                'timestamp': time.time()
            }
        }

    @property
    def confidence_threshold(self) -> float:
        """获取置信度阈值"""
        config = self.model_config.get_model_config('scene')
        threshold = config.get('confidence_threshold', 0.2)  # 降低阈值
        self._confidence_threshold = threshold  # 更新基类中的值
        return self._confidence_threshold

    @confidence_threshold.setter
    def confidence_threshold(self, value: float):
        """设置置信度阈值"""
        self._confidence_threshold = value

    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息"""
        return {
            'name': 'scene',
            'type': 'ResNet50-Places365',
            'input_size': (224, 224),
            'num_classes': len(self.categories),
            'scene_types': self.categories,
            'version': '1.0.0'
        }

    def _preprocess(self, frame: np.ndarray) -> torch.Tensor:
        """预处理输入图像"""
        try:
            if frame is None:
                self.logger.error("输入图像为空")
                return None
            
            # 调整大小
            frame = cv2.resize(frame, (224, 224))
            
            # 转换为 tensor
            frame = torch.from_numpy(frame).float()
            frame = frame.permute(2, 0, 1)  # HWC -> CHW
            frame = frame.unsqueeze(0)  # 添加 batch 维度
            
            # 归一化
            frame = frame / 255.0
            frame = self._normalize(frame)
            
            # 移动到正确的设备
            frame = frame.to(self.device)
            
            return frame
            
        except Exception as e:
            self.logger.error(f"预处理失败: {str(e)}")
            return None

    def _enhance_image(self, frame: np.ndarray) -> np.ndarray:
        """增强图像质量"""
        try:
            # 确保图像是uint8类型的BGR格式
            if frame.dtype != np.uint8:
                if frame.max() <= 1.0:
                    frame = (frame * 255).astype(np.uint8)
                else:
                    frame = frame.astype(np.uint8)
            
            # 确保是3通道BGR图像
            if len(frame.shape) == 2:
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            elif frame.shape[2] == 4:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
            
            # 1. 对比度增强
            lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            l = clahe.apply(l)
            enhanced = cv2.merge((l,a,b))
            enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
            
            # 2. 降噪
            enhanced = cv2.fastNlMeansDenoisingColored(enhanced)
            
            return enhanced
            
        except Exception as e:
            self.logger.error(f"图像增强失败: {str(e)}")
            return frame

    def _preprocess_numpy(self, frame: np.ndarray) -> np.ndarray:
        """预处理输入帧（返回numpy数组）"""
        try:
            # 调整大小
            frame = cv2.resize(frame, (224, 224))
            
            # 转换为RGB
            if len(frame.shape) == 2:
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
            elif frame.shape[2] == 3:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            elif frame.shape[2] == 4:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2RGB)
            
            # 归一化
            frame = frame.astype(np.float32) / 255.0
            
            return frame
            
        except Exception as e:
            self.logger.error(f"预处理失败: {str(e)}")
            return None

    def _extract_features(self, frame: np.ndarray) -> Dict:
        """提取场景特征"""
        try:
            return {
                'spatial': self._extract_spatial_features(frame),
                'texture': self._extract_texture_features(frame)
            }
        except Exception as e:
            self.logger.error(f"特征提取失败: {str(e)}")
            return {}

    def _extract_texture_features(self, frame: np.ndarray) -> Dict:
        """提取纹理特征"""
        try:
            return self.feature_extractor.extract_texture_features(frame)
        except Exception as e:
            self.logger.error(f"纹理特征提取失败: {str(e)}")
            return {}

    def _extract_spatial_features(self, frame: np.ndarray) -> Dict:
        """提取空间特征"""
        features = {}
        
        try:
            # 1. 空间特征
            h, w = frame.shape[:2]
            features['spatial'] = {
                'aspect_ratio': w/h,
                'area': w * h,
                'symmetry': self._calculate_symmetry(frame)
            }
            
            return features['spatial']
            
        except Exception as e:
            self.logger.error(f"空间特征提取失败: {str(e)}")
            return {}

    def _inference(self, frame: np.ndarray) -> Dict:
        """执行模型推理"""
        try:
            input_tensor = self.preprocess(frame)
            with torch.no_grad():
                output = self.model(input_tensor)
                probabilities = torch.softmax(output, dim=1)
                confidence, idx = torch.max(probabilities, dim=1)
                return {
                    'scene_type': self.categories[idx.item()],
                    'confidence': float(confidence.item())
                }
        except Exception as e:
            self.logger.error(f"模型推理失败: {str(e)}")
            return {'scene_type': '未知', 'confidence': 0.0}

    def _calculate_symmetry(self, frame: np.ndarray) -> float:
        """计算图像的对称性"""
        try:
            if not isinstance(frame, np.ndarray):
                self.logger.error("输入必须是numpy数组")
                return 0.0

            # 转换为灰度图
            if len(frame.shape) == 3:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            else:
                gray = frame

            # 获取图像中心线
            height, width = gray.shape
            mid = width // 2

            # 计算左右两边的差异
            left_side = gray[:, :mid]
            right_side = cv2.flip(gray[:, mid:], 1)  # 水平翻转右侧

            # 如果两边宽度不同，调整到相同宽度
            min_width = min(left_side.shape[1], right_side.shape[1])
            left_side = left_side[:, :min_width]
            right_side = right_side[:, :min_width]

            # 计算差异
            diff = np.abs(left_side.astype(float) - right_side.astype(float))
            symmetry = 1.0 - np.mean(diff) / 255.0

            return float(symmetry)

        except Exception as e:
            self.logger.error(f"对称性计算失败: {str(e)}")
            return 0.0

    def _calculate_spatial_match(self, current: Dict, target: Dict) -> float:
        """计算空间特征匹配度"""
        try:
            score = 0.0
            
            # 面积匹配
            if 'area' in current and 'min_area' in target:
                area_score = 1.0 if current['area'] >= target['min_area'] else 0.5
                score += 0.4 * area_score
                
            # 宽高比匹配
            if 'aspect_ratio' in current:
                ratio = current['aspect_ratio']
                ratio_score = 1.0 if 0.8 <= ratio <= 1.5 else 0.5
                score += 0.3 * ratio_score
                
            # 对称性匹配
            if 'symmetry' in current:
                symmetry = current['symmetry']
                symmetry_score = symmetry if target.get('openness') == 'high' else (1 - symmetry)
                score += 0.3 * symmetry_score
                
            return score
            
        except Exception as e:
            self.logger.error(f"空间特征匹配计算失败: {str(e)}")
            return 0.0

    def _calculate_functional_match(self, current: Dict, target: Dict) -> float:
        """计算功能特征匹配度"""
        try:
            score = 0.0
            
            # 家具类型匹配（提高权重）
            if 'furniture_types' in current and 'furniture' in target:
                current_furniture = set(current['furniture_types'])
                target_furniture = set(target['furniture'])
                
                # 计算关键家具的匹配度
                key_furniture_match = 0.0
                for furniture in target_furniture:
                    if furniture in current_furniture:
                        # 关键家具（如床对于卧室）权重更高
                        if furniture in ['床', '衣柜']:  # 卧室关键家具
                            key_furniture_match += 0.4
                        elif furniture in ['沙发', '电视柜']:  # 客厅关键家具
                            key_furniture_match += 0.3
                        else:
                            key_furniture_match += 0.2
                            
                score += 0.8 * key_furniture_match  # 提高家具匹配的权重
                
            # 布局合理性
            if 'furniture_layout' in current:
                layout_score = 0.2 if current['furniture_layout'].get('reasonable', False) else 0.0
                score += layout_score
                
            return score
            
        except Exception as e:
            self.logger.error(f"功能特征匹配计算失败: {str(e)}")
            return 0.0

    def _calculate_visual_match(self, current: Dict, target: Dict) -> float:
        """计算视觉特征匹配度"""
        try:
            score = 0.0
            
            # 自然光匹配
            if 'natural_light' in current:
                current_light = current['natural_light']
                target_light = float(target['natural_light'])  # 确保是数值
                # 使用差值计算匹配度
                light_score = 1.0 - min(abs(current_light - target_light), 1.0)
                score += 0.4 * light_score
            
            # 天花板高度匹配
            if 'ceiling_height' in current:
                current_height = current['ceiling_height']
                target_height = float(target['ceiling_height'])  # 确保是数值
                # 使用差值计算匹配度
                height_score = 1.0 - min(abs(current_height - target_height), 1.0)
                score += 0.3 * height_score
            
            # 墙面可见度匹配
            if 'wall_visibility' in current:
                current_wall = current['wall_visibility']
                target_wall = float(target['wall_visibility'])  # 确保是数值
                # 使用差值计算匹配度
                wall_score = 1.0 - min(abs(current_wall - target_wall), 1.0)
                score += 0.3 * wall_score
            
            return score
            
        except Exception as e:
            self.logger.error(f"视觉特征匹配计算失败: {str(e)}")
            return 0.0

    def _classify_scene(self, frame: np.ndarray) -> Dict:
        """场景分类"""
        try:
            # 预处理图像
            input_tensor = self._preprocess_image(frame)
            if input_tensor is None:
                return {'scene_type': '未知', 'confidence': 0.0}
            
            # 执行推理
            with torch.no_grad():
                output = self.model(input_tensor)
                probabilities = F.softmax(output, dim=1)
                
                # 获取最高置信度的预测
                confidence, predicted_idx = torch.max(probabilities, dim=1)
                
                # 获取场景类型
                scene_type = self.categories[predicted_idx.item()]
                
                return {
                    'scene_type': scene_type,
                    'confidence': float(confidence.item()),
                    'details': {
                        'predicted_class': predicted_idx.item(),
                        'confidence_score': float(confidence.item())
                    }
                }
            
        except Exception as e:
            self.logger.error(f"场景分类失败: {str(e)}")
            return {'scene_type': '未知', 'confidence': 0.0}

    def analyze_video(self, frames: List[np.ndarray]) -> Dict:
        """分析视频帧序列"""
        try:
            self.logger.debug(f"开始视频场景分析, 共 {len(frames)} 帧")
            frame_results = []
            
            for i, frame in enumerate(frames):
                self.logger.debug(f"分析第 {i+1}/{len(frames)} 帧")
                # 确保输入是 numpy 数组
                if not isinstance(frame, np.ndarray):
                    self.logger.error(f"第 {i+1} 帧不是numpy数组")
                    continue
                    
                result = self.analyze(frame)
                if result:
                    frame_results.append(result)
                    self.logger.debug(f"第 {i+1} 帧分析成功")
                else:
                    self.logger.warning(f"第 {i+1} 帧分析失败")

            self.logger.debug(f"视频分析完成, 成功分析 {len(frame_results)}/{len(frames)} 帧")
            
            return {
                'type': 'video_analysis',
                'analyzer': self.__class__.__name__.lower(),
                'results': {
                    'frame_results': frame_results,
                    'metadata': {
                        'frame_count': len(frame_results),
                        'total_frames': len(frames),
                        'timestamp': time.time()
                    }
                }
            }

        except Exception as e:
            self.logger.error(f"视频场景分析失败: {str(e)}", exc_info=True)
            return None

    def _analyze_spatial_features(self, frame: np.ndarray) -> Dict:
        """分析空间特征"""
        try:
            # 1. 计算面积
            area = self._calculate_area(frame)
            
            # 2. 分析对称性
            symmetry = self._analyze_symmetry(frame)
            
            # 3. 分析纹理复杂度
            texture = self._analyze_texture(frame)
            
            # 4. 分析长宽比
            aspect_ratio = frame.shape[1] / frame.shape[0]
            
            return {
                'area': float(area),
                'symmetry': float(symmetry),
                'texture_complexity': float(texture),
                'aspect_ratio': float(aspect_ratio),
                'lighting_quality': self._analyze_lighting(frame)
            }
        except Exception as e:
            self.logger.error(f"空间特征分析失败: {str(e)}")
            return {}

    def _analyze_visual_features(self, frame: np.ndarray) -> Dict:
        """分析视觉特征"""
        try:
            # 1. 分析墙面可见度
            wall_visibility = self._detect_walls(frame)
            
            # 2. 分析自然光
            natural_light = self._analyze_natural_light(frame)
            
            return {
                'wall_visibility': float(wall_visibility),
                'natural_light': float(natural_light)
            }
        except Exception as e:
            self.logger.error(f"视觉特征分析失败: {str(e)}")
            return {}

    def _detect_walls(self, frame: np.ndarray) -> float:
        """检测墙面"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=100)
        
        mask = np.zeros_like(gray)
        if lines is not None:
            for rho, theta in lines[:, 0]:
                a = np.cos(theta)
                b = np.sin(theta)
                x0 = a * rho
                y0 = b * rho
                x1 = int(x0 + 1000*(-b))
                y1 = int(y0 + 1000*(a))
                x2 = int(x0 - 1000*(-b))
                y2 = int(y0 - 1000*(a))
                cv2.line(mask, (x1, y1), (x2, y2), 255, 2)
        
        # 返回墙面占比
        return float(np.mean(mask > 0))

    def _analyze_symmetry(self, frame: np.ndarray) -> float:
        """分析场景对称性
        Args:
            frame: 输入图像
        Returns:
            对称性得分 (0-1)
        """
        try:
            # 转换为灰度图
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # 获取图像中心线
            height, width = gray.shape
            mid = width // 2
            
            # 计算左右两边的差异
            left = gray[:, :mid]
            right = cv2.flip(gray[:, mid:], 1)
            
            # 调整大小使左右两边相同
            min_width = min(left.shape[1], right.shape[1])
            left = left[:, :min_width]
            right = right[:, :min_width]
            
            # 计算对称性得分
            diff = np.abs(left.astype(float) - right.astype(float))
            symmetry = 1 - np.mean(diff) / 255
            
            return float(symmetry)
            
        except Exception as e:
            raise ValueError(f"对称性分析失败: {str(e)}")

    def _analyze_texture(self, frame: np.ndarray) -> float:
        """分析纹理复杂度
        Args:
            frame: 输入图像
        Returns:
            纹理复杂度得分 (0-1)
        """
        try:
            # 转换为灰度图
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # 计算梯度
            sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            
            # 计算梯度幅值
            magnitude = np.sqrt(sobelx**2 + sobely**2)
            
            # 归一化
            texture = np.mean(magnitude) / 255
            
            return float(texture)
            
        except Exception as e:
            raise ValueError(f"纹理分析失败: {str(e)}")

    def _analyze_lighting(self, frame: np.ndarray) -> float:
        """分析光照质量
        Args:
            frame: 输入图像
        Returns:
            光照质量得分 (0-1)
        """
        try:
            # 转换为HSV
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            
            # 提取亮度通道
            v = hsv[:, :, 2]
            
            # 计算亮度均值和标准差
            mean_v = np.mean(v)
            std_v = np.std(v)
            
            # 计算光照质量得分
            score = (mean_v / 255) * (1 - std_v / 255)
            
            return float(score)
            
        except Exception as e:
            raise ValueError(f"光照分析失败: {str(e)}")

    def _analyze_natural_light(self, frame: np.ndarray) -> float:
        """分析自然光
        Args:
            frame: 输入图像
        Returns:
            自然光得分 (0-1)
        """
        try:
            # 转换为HSV
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            
            # 计算饱和度和亮度的均值
            s = np.mean(hsv[:, :, 1]) / 255
            v = np.mean(hsv[:, :, 2]) / 255
            
            # 综合评分
            score = (v * 0.7 + s * 0.3)  # 亮度权重更大
            
            return float(score)
            
        except Exception as e:
            raise ValueError(f"自然光分析失败: {str(e)}")

    def _analyze_wall_visibility(self, frame: np.ndarray) -> float:
        """分析墙面可见度
        Args:
            frame: 输入图像
        Returns:
            墙面可见度得分 (0-1)
        """
        try:
            # 转换为HSV
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            
            # 提取亮度和饱和度
            s = hsv[:, :, 1]
            v = hsv[:, :, 2]
            
            # 墙面通常饱和度低、亮度适中
            wall_mask = (s < 30) & (v > 100) & (v < 200)
            
            # 计算墙面占比
            visibility = np.mean(wall_mask)
            
            return float(visibility)
            
        except Exception as e:
            raise ValueError(f"墙面分析失败: {str(e)}")

    def _estimate_area(self, frame: np.ndarray, scene_type: str) -> float:
        """估算场景面积
        Args:
            frame: 输入图像
            scene_type: 场景类型
        Returns:
            估算面积（平方米），如果场景类型不支持则返回-1
        """
        try:
            # 基准面积参考值（单位：平方米）
            base_areas = {
                '客厅': {
                    'min': 15.0,
                    'max': 30.0,
                    'default': 20.0
                },
                '卧室': {
                    'min': 9.0,
                    'max': 20.0,
                    'default': 15.0
                },
                '厨房': {
                    'min': 4.0,
                    'max': 12.0,
                    'default': 8.0
                },
                '浴室': {
                    'min': 3.0,
                    'max': 8.0,
                    'default': 5.0
                },
                '餐厅': {
                    'min': 10.0,
                    'max': 25.0,
                    'default': 15.0
                }
            }
            
            # 检查场景类型是否支持面积估算
            if scene_type not in base_areas:
                self.logger.info(f"场景类型 '{scene_type}' 暂不支持面积估算")
                return -1.0
            
            scene_config = base_areas[scene_type]
            
            # 分析图像特征以调整面积估算
            # 1. 分析墙面可见度
            wall_visibility = self._analyze_wall_visibility(frame)
            
            # 2. 分析空间开阔度
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            brightness = np.mean(hsv[:, :, 2]) / 255.0
            
            # 3. 根据特征调整面积
            area_factor = (wall_visibility * 0.6 + brightness * 0.4)  # 综合因子
            estimated_area = scene_config['default'] + (
                (scene_config['max'] - scene_config['default']) * area_factor if area_factor > 0.5
                else (scene_config['default'] - scene_config['min']) * (0.5 - area_factor) * 2
            )
            
            # 确保在合理范围内
            estimated_area = max(scene_config['min'], 
                               min(scene_config['max'], estimated_area))
            
            self.logger.debug(f"场景类型: {scene_type}, 估算面积: {estimated_area:.2f}㎡")
            return float(estimated_area)
            
        except Exception as e:
            self.logger.error(f"面积估算失败: {str(e)}")
            return -1.0  # 出错时返回-1

    def _normalize(self, frame: torch.Tensor) -> torch.Tensor:
        """归一化输入图像"""
        try:
            # 标准化
            frame = (frame - torch.tensor(self.mean).to(self.device)) / torch.tensor(self.std).to(self.device)
            return frame
        except Exception as e:
            self.logger.error(f"归一化失败: {str(e)}")
            return None

    def _calculate_area(self, frame: np.ndarray) -> float:
        """计算图像的面积"""
        try:
            return frame.shape[0] * frame.shape[1]
        except Exception as e:
            self.logger.error(f"面积计算失败: {str(e)}")
            return 0.0

    def _get_scene_type(self, class_idx: int) -> str:
        """通过索引直接获取场景类型"""
        try:
            if 0 <= class_idx < len(self.categories):
                scene_type = self.categories[class_idx]
                self.logger.debug(f"预测索引: {class_idx}, 对应场景: {scene_type}")
                return scene_type
            else:
                self.logger.warning(f"类别索引 {class_idx} 超出范围")
                return "其他"
        except Exception as e:
            self.logger.error(f"获取场景类型失败: {str(e)}")
            return "其他"

    def _postprocess(self, output: torch.Tensor) -> Tuple[str, float]:
        """后处理模型输出"""
        try:
            # 获取预测类别和置信度
            probs = F.softmax(output, dim=1)
            confidence, pred_idx = torch.max(probs, dim=1)
            
            # 获取场景类型
            scene_type = self._get_scene_type(pred_idx.item())
            confidence = float(confidence.item())
            
            # 调试信息
            self.logger.debug(f"原始预测索引: {pred_idx.item()}")
            self.logger.debug(f"预测场景类型: {scene_type}")
            self.logger.debug(f"置信度: {confidence:.3f}")
            
            # 应用阈值，但保留原始置信度
            if confidence < self.confidence_threshold:
                self.logger.debug(f"置信度 {confidence} 低于阈值 {self.confidence_threshold}")
                return "其他", confidence  # 只修改类型，保留置信度
            
            return scene_type, confidence
            
        except Exception as e:
            self.logger.error(f"后处理失败: {str(e)}")
            return "其他", 0.0  # 只在出错时将置信度设为0
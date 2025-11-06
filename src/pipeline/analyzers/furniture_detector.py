import os
from .base_analyzer import BaseAnalyzer
import cv2
import numpy as np
from typing import Dict, List, Any, Tuple, Union, Optional
import torch
import traceback
import logging
import torchvision
from ultralytics import YOLO
from ..utils.model_config import ModelConfig
from collections import Counter, defaultdict
from ..utils.result_types import AnalyzerResult
from ..utils.feature_extractors import FeatureExtractor
import time
import torch.nn as nn

class FurnitureDetector(BaseAnalyzer):
    """家具检测器 - 检测和识别室内家具"""
    
    _instances = {}  # 使用字典存储不同配置的实例
    
    def __new__(cls, model_config: ModelConfig):
        config_id = id(model_config)
        if config_id not in cls._instances:
            cls._instances[config_id] = super(FurnitureDetector, cls).__new__(cls)
        return cls._instances[config_id]
    
    def __init__(self, model_config: ModelConfig):
        if hasattr(self, '_initialized') and self._initialized:
            return
            
        try:
            # 设置正确的分析器类型
            self.analyzer_type = 'furniture'  # 确保与配置文件中的键名一致
            
            super().__init__(model_config)
            self.logger = logging.getLogger(__name__)
            self.logger.setLevel(logging.INFO)
            
            # 获取配置
            self.config = self.model_config.get_analyzer_config('furniture')
            
            # 设置基本参数
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
            self.model_path = os.path.join(project_root, self.config.get('model_path'))
            
            # 加载类别映射
            self.class_mapping = self.config.get('class_mapping', {})
            if not self.class_mapping:
                self.logger.warning("未找到类别映射配置，将使用默认映射")
                self.class_mapping = {
                    56: "椅子", 57: "沙发", 58: "盆栽", 59: "床",
                    60: "餐桌", 61: "马桶", 62: "电视", 63: "笔记本电脑",
                    64: "鼠标", 65: "遥控器", 66: "键盘", 67: "手机",
                    68: "微波炉", 69: "烤箱", 70: "水槽", 71: "冰箱"
                }
            
            # 将字符串键转换为整数键
            self.class_mapping = {int(k): v for k, v in self.class_mapping.items()}
            
            # 如果模型文件不存在，使用基础功能
            if not os.path.exists(self.model_path):
                self.logger.warning(f"模型文件不存在: {self.model_path}，将使用基础功能")
                self.model = None
            else:
                self.model = self._create_model()
                self.model.eval()
            
            # 初始化特征提取器
            self.feature_extractor = FeatureExtractor(model_config)
            
            self._initialized = True
            self.logger.info("家具检测器初始化完成")
            
        except Exception as e:
            self.logger.error("家具检测器初始化失败", exc_info=True)
            raise

    def _create_model(self) -> nn.Module:
        """创建模型实例"""
        try:
            # 使用 YOLO 模型
            model = YOLO(self.model_path)
            
            # 加载预训练权重
            if os.path.exists(self.model_path):
                checkpoint = torch.load(self.model_path, map_location='cpu')
                
                # 处理封装的权重格式
                if 'state_dict' in checkpoint:
                    state_dict = checkpoint['state_dict']
                else:
                    state_dict = checkpoint
                    
                # 移除模块前缀
                new_state_dict = {}
                for k, v in state_dict.items():
                    if k.startswith('module.'):
                        name = k[7:]  # 移除 'module.' 前缀
                    else:
                        name = k
                    new_state_dict[name] = v
                    
                model.load_state_dict(new_state_dict, strict=False)
                self.logger.info("加载预训练权重成功")
            
            return model
            
        except Exception as e:
            self.logger.error("模型创建失败", exc_info=True)
            raise
    
    def analyze(self, frame: np.ndarray) -> Dict:
        """分析家具"""
        try:
            self.logger.debug("开始家具分析...")
            
            # 执行检测
            results = self.model(frame)
            
            # 处理检测结果
            detected_items = []
            max_confidence = 0.0
            furniture_stats = defaultdict(lambda: {'count': 0, 'confidence': 0.0})  # 修改为更详细的统计
            
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    cls_id = int(box.cls[0].item())
                    conf = float(box.conf[0].item())
                    
                    if conf > self.confidence_threshold:
                        furniture_type = self.class_mapping.get(cls_id, '未知')
                        item = {
                            'type': furniture_type,
                            'confidence': conf,
                            'box': box.xyxy[0].tolist()
                        }
                        detected_items.append(item)
                        max_confidence = max(max_confidence, conf)
                        
                        # 更新家具统计
                        stats = furniture_stats[furniture_type]
                        stats['count'] += 1
                        stats['confidence'] += conf
            
            # 计算平均置信度
            for stats in furniture_stats.values():
                if stats['count'] > 0:
                    stats['confidence'] /= stats['count']
            
            # 分析布局
            layout_analysis = self._analyze_layout(detected_items)
            
            # 计算空间利用率
            space_utilization = self._calculate_space_utilization(detected_items)
            
            result = {
                'type': 'furniture',
                'confidence': max_confidence,
                'features': {
                    'detected_items': detected_items,
                    'item_count': len(detected_items),
                    'furniture_types': {
                        ftype: {
                            'count': stats['count'],
                            'avg_confidence': float(stats['confidence'])
                        }
                        for ftype, stats in furniture_stats.items()
                    },
                    'layout': {
                        'density': float(layout_analysis['density']),
                        'layout_score': float(layout_analysis['layout_score']),
                        'symmetry': bool(self._check_symmetry(detected_items)),
                        'linear_arrangement': bool(self._check_linear_arrangement(detected_items))
                    },
                    'space_utilization': space_utilization
                },
                'metadata': {
                    'timestamp': time.time(),
                    'model_info': self.get_model_info()
                }
            }
            
            self.logger.debug(f"家具分析结果: {result}")
            return result
            
        except Exception as e:
            self.logger.error(f"家具分析失败: {str(e)}")
            return {
                'type': 'furniture',
                'confidence': 0.0,
                'features': {},
                'metadata': {'error': str(e)}
            }

    def _count_furniture_types(self, detected_items: List[Dict]) -> Dict[str, Dict]:
        """统计各类家具数量和置信度"""
        type_stats = defaultdict(lambda: {'count': 0, 'confidence': 0.0})
        for item in detected_items:
            item_type = item['type']
            confidence = item['confidence']
            stats = type_stats[item_type]
            stats['count'] += 1
            stats['confidence'] += confidence
        
        # 计算平均置信度
        for stats in type_stats.values():
            if stats['count'] > 0:
                stats['confidence'] /= stats['count']
        
        return dict(type_stats)

    def _analyze_layout(self, detections: List[Dict]) -> Dict:
        """分析布局"""
        try:
            # 计算密度
            total_area = 640 * 480  # 假设标准图像尺寸
            furniture_area = sum(
                (box[2] - box[0]) * (box[3] - box[1])
                for det in detections
                for box in [det['box']]
            )
            density = furniture_area / total_area
            
            # 评估布局分数
            layout_score = self._evaluate_layout(density, detections)
            
            return {
                'density': float(density),  # 确保返回 Python float
                'layout_score': float(layout_score)  # 确保返回 Python float
            }
            
        except Exception as e:
            self.logger.error(f"布局分析失败: {str(e)}")
            return {
                'density': 0.0,
                'layout_score': 0.0
            }
    
    def _analyze(self, frame: np.ndarray) -> AnalyzerResult:
        """分析单帧图像中的家具"""
        try:
            # 执行检测
            results = self.model(frame)
            
            # 解析结果
            detections = []
            max_confidence = 0.0
            
            if len(results) > 0:
                result = results[0]  # 获取第一个结果
                
                # 遍历所有检测结果
                for det in result.boxes.data:  # 使用 .data 获取原始张量数据
                    # 解包检测结果：x1, y1, x2, y2, conf, cls
                    *xyxy, conf, cls = det.tolist()
                    
                    if conf > self.confidence_threshold:
                        cls = int(cls)  # 确保类别是整数
                        detection = {
                            'category': self.class_mapping.get(cls, 'unknown'),
                            'confidence': float(conf),
                            'bbox': [float(x) for x in xyxy]
                        }
                        detections.append(detection)
                        max_confidence = max(max_confidence, conf)
            
            # 分析布局
            layout_analysis = self._analyze_layout(detections)
            
            result = AnalyzerResult(
                analyzer_type='furniture',
                confidence=max_confidence,
                features={
                    'detections': detections,
                    'layout': layout_analysis,
                    'furniture_stats': self._count_furniture_types(detections),
                    'spatial_arrangement': {
                        'density': layout_analysis['density'],
                        'balance': self._check_symmetry(detections),
                        'accessibility': self._check_linear_arrangement(detections)
                    }
                },
                metadata={'timestamp': time.time()}
            )
            
            self.logger.info(f"[家具检测器] 输出: \n"
                            f"- 检测到 {len(detections)} 个物体 (最高置信度={max_confidence:.2f})\n"
                            f"- 家具布局:\n"
                            f"  • 空间密度: {layout_analysis['density']:.2f}\n"
                            f"  • 布局评分: {layout_analysis['layout_score']:.2f}\n"
                            f"- 检测到的家具:\n" +
                            "\n".join([f"  • {item['category']}: {item['confidence']:.2f}"
                                     for item in detections]))
            return result
            
        except Exception as e:
            self.logger.error(f"家具检测失败: {str(e)}")
            return AnalyzerResult(
                analyzer_type='furniture',
                confidence=0.0,
                features={},
                metadata={'error': str(e)}
            )
    
    def _detect_furniture(self, frame: np.ndarray) -> List[Dict]:
        """执行家具检测"""
        try:
            if frame is None:
                return []
            
            # YOLOv8 预测
            results = self.model.predict(
                source=frame,
                conf=self.confidence_threshold,
                iou=self.iou_threshold,
                max_det=self.max_detections,
                verbose=False
            )[0]
            
            # 处理检测结果
            detections = []
            if len(results.boxes) > 0:
                for r in results.boxes.data:
                    try:
                        if len(r) >= 6:
                            x1, y1, x2, y2, conf, cls_id = r.cpu().numpy()
                            cls_id = int(cls_id)
                            # 检查是否是我们关心的类别
                            if cls_id in self.class_mapping:
                                class_name = self.class_mapping[cls_id]
                                detections.append({
                                    'class': class_name,
                                    'confidence': float(conf),
                                    'box': [int(x1), int(y1), int(x2), int(y2)]
                                })
                            else:
                                self.logger.debug(f"跳过非目标类别: {cls_id}")
                    except Exception as e:
                        self.logger.warning(f"处理单个检测结果失败: {str(e)}")
                        continue
            
            return detections
            
        except Exception as e:
            self.logger.error(f"检测执行失败: {str(e)}")
            return []
    
    def _load_model(self, model_path: str, config_path: str) -> Any:
        """加载检测模型
        Args:
            model_path: 模型文件路径
            config_path: 配置文件路径
        Returns:
            加载的模型
        """
        try:
            # 使用 YOLOv8
            model = YOLO(model_path)
            
            # 设置模型参数
            model.conf = self.confidence_threshold  # 置信度阈值
            model.iou = self.iou_threshold  # NMS IOU阈值
            model.max_det = self.max_detections  # 最大检测数量
            
            self.logger.info(f"成功加载模型: {model_path}")
            return model
            
        except Exception as e:
            self.logger.error(f"模型加载失败: {str(e)}")
            return None
    
    def _get_empty_result(self) -> Dict:
        """返回空结果"""
        return {
            'type': 'furniture_analysis',
            'confidence': 0.0,
            'features': {
                'detections': [],
                'furniture_types': {},
                'total_count': 0
            }
        }

    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息"""
        config = self.model_config.get_analyzer_config('furniture')
        return {
            'name': 'YOLOv8n',
            'type': 'YOLOv8n',
            'input_size': (640, 640),
            'num_classes': len(self.class_mapping),
            'classes': list(self.class_mapping.values()),
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
            
    def visualize(self, frame: np.ndarray, result: Dict) -> np.ndarray:
        """可视化检测结果"""
        try:
            vis = frame.copy()
            if 'boxes' in result:
                for box in result['boxes']:
                    x1, y1, x2, y2 = box['box']
                    label = f"{box['class']} {box['confidence']:.2f}"
                    
                    # 绘制边界框
                    cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    
                    # 添加标签
                    cv2.putText(vis, label, (x1, y1-10),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                              
            return vis
            
        except Exception as e:
            self.logger.error(f"家具检测可视化失败: {str(e)}")
            return frame

    def _process_detections(self, results, frame: np.ndarray) -> List[Dict]:
        """处理检测结果"""
        detections = []
        for box in results.boxes:
            # 获取边界框坐标
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            # 添加其他处理逻辑...
        return detections
        
    def _evaluate_layout(self, density: float, detections: List[Dict]) -> float:
        """评估布局合理性"""
        try:
            # 1. 基础分数
            score = 1.0
            
            # 2. 评估家具密度
            if density > 0.7:  # 过于拥挤
                score *= 0.7
            elif density < 0.2:  # 过于空旷
                score *= 0.8
            
            # 3. 评估家具分布
            if len(detections) >= 2:
                # 检查对称性
                if not self._check_symmetry(detections):
                    score *= 0.9
                # 检查线性排列
                if not self._check_linear_arrangement(detections):
                    score *= 0.9
            
            # 4. 确保分数在0-1之间
            return float(max(0.1, min(1.0, score)))
            
        except Exception as e:
            self.logger.error(f"布局评估失败: {str(e)}")
            return 0.5

    def _check_symmetry(self, detections: List[Dict]) -> bool:
        """检查家具布局是否对称"""
        try:
            # 获取所有家具的中心点
            centers = []
            for det in detections:
                box = det['box']
                center_x = (box[0] + box[2]) / 2
                center_y = (box[1] + box[3]) / 2
                centers.append((center_x, center_y))
            
            # 计算中心点到中轴线的距离
            if not centers:
                return True
            
            center_x = sum(x for x, _ in centers) / len(centers)
            distances = [abs(x - center_x) for x, _ in centers]
            
            # 计算距离的标准差，标准差小说明比较对称
            std_dev = np.std(distances)
            
            # 将 numpy.bool_ 转换为 Python bool
            return bool(std_dev < 50)  # 距离标准差阈值
            
        except Exception as e:
            self.logger.error(f"对称性检查失败: {str(e)}")
            return bool(True)

    def _check_linear_arrangement(self, detections: List[Dict]) -> bool:
        """检查家具是否呈线性排列"""
        try:
            # 获取所有家具的中心点
            centers = []
            for det in detections:
                box = det['box']
                center_x = (box[0] + box[2]) / 2
                center_y = (box[1] + box[3]) / 2
                centers.append((center_x, center_y))
            
            if len(centers) < 3:
                return True
            
            # 使用线性回归检查点的分布
            x = np.array([p[0] for p in centers])
            y = np.array([p[1] for p in centers])
            
            # 计算相关系数
            correlation = np.corrcoef(x, y)[0, 1]
            
            # 将 numpy.bool_ 转换为 Python bool
            return bool(abs(correlation) > 0.7)  # 相关系数阈值
            
        except Exception as e:
            self.logger.error(f"线性排列检查失败: {str(e)}")
            return bool(True)

    @property
    def confidence_threshold(self) -> float:
        """获取置信度阈值"""
        config = self.model_config.get_analyzer_config('furniture')
        threshold = config.get('confidence_threshold', 0.5)
        self._confidence_threshold = threshold  # 更新基类中的值
        return self._confidence_threshold

    @confidence_threshold.setter
    def confidence_threshold(self, value: float):
        """设置置信度阈值"""
        self._confidence_threshold = value 

    def _calculate_space_utilization(self, detected_items: List[Dict]) -> Dict:
        """计算空间利用率"""
        try:
            # 假设标准图像尺寸
            total_area = 640 * 480
            
            # 计算地面区域使用率
            floor_items = ['沙发', '椅子', '茶几', '餐桌', '床']
            floor_area = sum(
                (box[2] - box[0]) * (box[3] - box[1])
                for item in detected_items
                if item['type'] in floor_items
                for box in [item['box']]
            )
            
            # 计算墙面区域使用率
            wall_items = ['电视', '书架', '装饰画']
            wall_area = sum(
                (box[2] - box[0]) * (box[3] - box[1])
                for item in detected_items
                if item['type'] in wall_items
                for box in [item['box']]
            )
            
            # 分析垂直空间分布
            heights = [
                (item['box'][3] + item['box'][1]) / 2
                for item in detected_items
            ]
            vertical_distribution = {
                'low': len([h for h in heights if h < 160]),  # 低于160像素
                'middle': len([h for h in heights if 160 <= h < 320]),
                'high': len([h for h in heights if h >= 320])
            }
            
            return {
                'floor_area': floor_area / total_area,
                'wall_area': wall_area / total_area,
                'vertical': vertical_distribution
            }
            
        except Exception as e:
            self.logger.error(f"空间利用率计算失败: {str(e)}")
            return {
                'floor_area': 0.0,
                'wall_area': 0.0,
                'vertical': {'low': 0, 'middle': 0, 'high': 0}
            }

    def _analyze_furniture_combinations(self, detected_items: List[Dict]) -> Dict[str, List[str]]:
        """分析家具组合"""
        try:
            # 获取每种家具类型的列表
            furniture_types = [item['type'] for item in detected_items]
            
            # 定义各个功能区的典型家具组合
            combinations = {
                'living_room': ['沙发', '茶几', '电视'],
                'dining_room': ['餐桌', '椅子'],
                'study_room': ['书桌', '椅子', '书架'],
                'bedroom': ['床', '衣柜', '床头柜']
            }
            
            # 检查每个功能区的家具组合
            detected_combinations = {}
            for area, required_items in combinations.items():
                detected = [item for item in required_items if item in furniture_types]
                if detected:  # 只记录检测到家具的功能区
                    detected_combinations[area] = detected
            
            return detected_combinations
            
        except Exception as e:
            self.logger.error(f"家具组合分析失败: {str(e)}")
            return {} 
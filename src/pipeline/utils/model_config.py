from dataclasses import dataclass
from typing import Dict, Optional, Any, List
import os
import yaml
import torch
import logging
from pathlib import Path

@dataclass
class ModelPaths:
    """模型路径配置"""
    # 场景分析模型 - ResNet50-Places365
    scene: str = 'pretrained_models/resnet50_places365.pth'
    
    # 物体检测模型 - YOLOv8
    furniture: str = 'pretrained_models/yolov8n.pt'
    
    # 风格分类模型 - EfficientNetV2
    style: str = 'pretrained_models/pre_efficientnetv2-m.pth'
    
    # 光照和颜色分析使用传统方法，不需要模型
    lighting: Optional[str] = None
    color: Optional[str] = None

    def get_path(self, model_type: str) -> Optional[str]:
        """获取指定类型的模型路径"""
        return getattr(self, model_type, None)

    def validate_path(self, model_type: str) -> bool:
        """验证模型文件是否存在"""
        path = self.get_path(model_type)
        if not path:
            return True  # 对于不需要模型的分析器返回True
        return os.path.exists(path)

    def get_absolute_path(self, model_type: str) -> Optional[str]:
        """获取模型的绝对路径"""
        path = self.get_path(model_type)
        if not path:
            return None
            
        if not os.path.isabs(path):
            # 从项目根目录开始查找
            project_root = Path(__file__).parent.parent.parent.parent
            path = project_root / path
            
        return str(path.resolve())

class ModelConfig:
    """模型配置管理类"""
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ModelConfig, cls).__new__(cls)
        return cls._instance
        
    def __init__(self):
        if not hasattr(self, 'initialized'):
            self.logger = logging.getLogger(__name__)
            self.config = self._load_config()
            # 从新的配置结构中获取场景配置
            scene_config = self.config.get('analyzers', {}).get('scene', {})
            self.scene_mapping = scene_config.get('scene_mapping', {})  # 确保加载 scene_mapping
            self.initialized = True
            
    def _load_config(self) -> Dict:
        """加载配置文件"""
        try:
            # 获取配置文件路径
            config_path = os.path.join('src', 'config', 'model_config.yaml')
            if not os.path.exists(config_path):
                raise FileNotFoundError(f"配置文件不存在: {config_path}")
                
            # 读取配置文件
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
                
            if not config:
                raise ValueError("配置文件为空")
                
            self.logger.info("配置文件加载成功")
            return config
            
        except Exception as e:
            self.logger.error("加载配置文件失败", exc_info=True)
            # 返回默认配置
            return {
                'analyzers': {
                    'scene': {
                        'model_path': 'pretrained_models/resnet50_places365.pth',
                        'categories_path': 'pretrained_models/places365_zh.txt',
                        'confidence_threshold': 0.3,
                        'input_size': [224, 224],
                        'device': 'cpu',
                        'batch_size': 1
                    }
                },
                'processors': {
                    'image': {
                        'input_size': [640, 480],
                        'batch_size': 1,
                        'num_workers': 4
                    },
                    'video': {
                        'sample_interval': 10,
                        'target_size': [640, 480]
                    },
                    'realtime': {
                        'fps': 30,
                        'buffer_size': 5
                    }
                },
                'defaults': {
                    'device': 'cpu',
                    'batch_size': 1,
                    'num_workers': 4,
                    'use_fp16': False
                }
            }
            
    def get(self, key: str, default: Any = None) -> Any:
        """获取配置项
        Args:
            key: 配置键
            default: 默认值
        Returns:
            配置值
        """
        try:
            # 支持多级键，如 'scene.model_path'
            keys = key.split('.')
            value = self.config
            for k in keys:
                value = value.get(k, default)
                if value is None:
                    return default
            return value
        except Exception as e:
            self.logger.error(f"获取配置项失败: {str(e)}")
            return default
            
    def get_processor_config(self, processor_type: str) -> Dict:
        """获取处理器配置"""
        try:
            if not self.config or 'processors' not in self.config:
                self.logger.warning("配置文件中缺少处理器配置")
                return {}
                
            processor_config = self.config['processors'].get(processor_type, {})
            if not processor_config:
                self.logger.warning(f"未找到 {processor_type} 处理器的配置")
                
            return processor_config
            
        except Exception as e:
            self.logger.error(f"获取处理器配置失败: {processor_type}", exc_info=True)
            return {}
            
    def get_analyzer_config(self, analyzer_type: str) -> Dict:
        """获取分析器配置"""
        try:
            if not self.config:
                self.logger.warning("配置为空")
                return {}
                
            # 从 analyzers 部分获取配置
            analyzer_config = self.config.get('analyzers', {}).get(analyzer_type, {})
            if not analyzer_config:
                self.logger.warning(f"未找到 {analyzer_type} 分析器的配置")
                
            # 合并全局配置
            global_config = self.config.get(analyzer_type, {})
            analyzer_config.update(global_config)
            
            return analyzer_config
            
        except Exception as e:
            self.logger.error(f"获取分析器配置失败: {analyzer_type}", exc_info=True)
            return {}

    def get_model_config(self, model_type: str) -> Optional[Dict[str, Any]]:
        """获取模型配置"""
        # 先尝试直接获取
        if model_type in self.config:
            return self.config[model_type]
            
        # 再尝试获取分析器配置
        return self.get_analyzer_config(model_type)
        
    def get_model_dir(self) -> str:
        """获取模型目录的绝对路径"""
        try:
            # 从配置中获取模型目录路径
            model_dir = self.config.get('model_dir', 'pretrained_models')
            
            # 如果是相对路径，转换为绝对路径
            if not os.path.isabs(model_dir):
                # 获取项目根目录（假设是当前文件的上三级目录）
                project_root = Path(__file__).parent.parent.parent.parent
                model_dir = project_root / model_dir
            
            # 确保目录存在
            model_dir = Path(model_dir)
            if not model_dir.exists():
                self.logger.warning(f"模型目录不存在: {model_dir}")
                model_dir.mkdir(parents=True, exist_ok=True)
                
            return str(model_dir.resolve())
            
        except Exception as e:
            self.logger.error(f"获取模型目录失败: {str(e)}")
            # 返回默认路径
            return str(Path(__file__).parent.parent.parent.parent / 'pretrained_models')

    def get_model_path(self, model_name: str) -> str:
        """获取模型文件路径"""
        try:
            # 从配置中获取模型路径
            model_path = self.config.get('models', {}).get(model_name)
            if not model_path:
                # 如果在models中找不到，尝试在分析器配置中查找
                analyzer_config = self.get_model_config(model_name)
                if analyzer_config:
                    model_path = analyzer_config.get('model_path')
                
                # 如果还是找不到，使用默认路径
                if not model_path:
                    model_path = os.path.join(self.get_model_dir(), f"{model_name}.pth")
            
            # 如果是相对路径，转换为绝对路径
            if not os.path.isabs(model_path):
                model_path = os.path.join(self.get_model_dir(), model_path)
                
            return model_path
            
        except Exception as e:
            self.logger.error(f"获取模型路径失败: {str(e)}")
            # 返回默认路径
            return os.path.join(self.get_model_dir(), f"{model_name}.pth")
        
    def get_scene_mapping(self) -> Dict[str, str]:
        """获取场景类型映射"""
        scene_config = self.get_model_config('scene')
        if not scene_config:
            return {}
        return scene_config.get('scene_mapping', {})
        
    def get_scene_types(self) -> list:
        """获取场景类型列表"""
        scene_config = self.get_model_config('scene')
        if not scene_config:
            return []
        return scene_config.get('scene_types', [])
        
    def get_device(self, model_type: str) -> str:
        """获取模型运行设备"""
        config = self.get_model_config(model_type)
        if not config:
            return 'cpu'
        return config.get('device', 'cpu')

    def get_model_type(self, model_type: str) -> Optional[str]:
        """获取模型类型"""
        try:
            model_config = self.config.get('models', {}).get(model_type, {})
            return model_config.get('type')
        except Exception as e:
            raise ValueError(f"获取模型类型失败: {str(e)}")
            return None
            
    def get_model_threshold(self, model_name: str) -> float:
        """获取模型阈值"""
        return self.config.get('models', {}).get(model_name, {}).get('threshold', 0.5)
        
    def get_model_params(self, model_name: str) -> Dict[str, Any]:
        """获取模型参数"""
        return self.config.get(model_name, {})
        
    def get_inference_params(self, model_name: str) -> Dict[str, Any]:
        """获取推理参数"""
        return self.config.get(f"{model_name}_inference", {})
    
    def validate_model_path(self, model_name: str) -> bool:
        """验证模型文件是否存在"""
        path = self.get_model_path(model_name)
        if not path:
            return False
        return os.path.exists(path)

    @property
    def model_params(self) -> Dict:
        """获取模型参数"""
        return {
            "device": self.device,
            "model_path": self.model_path,
            "batch_size": self.batch_size,
            "input_size": self.input_size
        }

    def get_categories_path(self) -> str:
        """获取场景类别文件路径"""
        try:
            # 从场景分析器配置中获取类别文件路径
            scene_config = self.get_analyzer_config('scene')
            categories_path = scene_config.get('categories_path', 'places365_zh.txt')
            
            # 获取项目根目录
            project_root = Path(__file__).parent.parent.parent.parent
            
            # 如果是相对路径，转换为绝对路径
            if not os.path.isabs(categories_path):
                # 移除可能重复的 pretrained_models 前缀
                if 'pretrained_models/' in categories_path:
                    categories_path = categories_path.replace('pretrained_models/', '')
                categories_path = os.path.join(project_root, 'pretrained_models', categories_path)
            
            # 检查文件是否存在
            if not os.path.exists(categories_path):
                self.logger.error(f"场景类别文件不存在: {categories_path}")
                raise FileNotFoundError(f"场景类别文件不存在: {categories_path}")
                
            self.logger.debug(f"使用场景类别文件: {categories_path}")
            return categories_path
            
        except Exception as e:
            self.logger.error(f"获取场景类别文件路径失败: {str(e)}")
            raise


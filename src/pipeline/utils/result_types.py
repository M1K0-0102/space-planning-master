from typing import Dict, List, Optional, Any, Union, TypedDict
from dataclasses import dataclass, field
from enum import Enum, auto
from datetime import datetime
import time

class AnalysisType(Enum):
    """分析类型枚举"""
    SCENE = auto()
    FURNITURE = auto()
    LIGHTING = auto()
    COLOR = auto()
    STYLE = auto()
    COMPREHENSIVE = auto()

@dataclass
class RawAnalysisResult:
    """原始分析结果"""
    type: str
    confidence: float
    features: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def create_empty(cls, analysis_type: str) -> 'RawAnalysisResult':
        """创建空结果"""
        return cls(
            type=f"{analysis_type}_analysis",
            confidence=0.0,
            features={},
            metadata={}
        )

    def to_dict(self) -> Dict:
        """转换为字典"""
        return {
            'type': self.type,
            'confidence': self.confidence,
            'features': self.features,
            'metadata': self.metadata
        }

@dataclass
class AnalysisDetails:
    """分析详情"""
    scene_details: Dict = field(default_factory=lambda: {
        'room_type': '',
        'area': 0,
        'layout': '',
        'features': []
    })
    
    lighting_details: Dict = field(default_factory=lambda: {
        'brightness': 0.0,
        'uniformity': 0.0,
        'contrast': 0.0,
        'color_temp': '',
        'issues': []
    })
    
    furniture_details: Dict = field(default_factory=lambda: {
        'total_count': 0,
        'categories': {},
        'layout_score': 0.0,
        'missing_items': []
    })
    
    style_details: Dict = field(default_factory=lambda: {
        'main_style': '',
        'elements': {},
        'consistency': 0.0
    })
    
    color_details: Dict = field(default_factory=lambda: {
        'main_colors': {},
        'color_scheme': '',
        'saturation': 0.0,
        'harmony_score': 0.0
    })

@dataclass
class SceneFeatures:
    """场景特征"""
    scene_type: str
    scene_probs: Dict[str, float]
    spatial_features: Dict[str, float] = field(default_factory=lambda: {
        'area': 0.0,           # 面积（平方米）
        'symmetry': 0.0,       # 对称性
        'wall_visibility': 0.0, # 墙面可见度
        'natural_light': 0.0   # 自然光评分
    })
    texture_features: Dict[str, float] = field(default_factory=dict)
    lighting_features: Dict[str, float] = field(default_factory=dict)

@dataclass
class FurnitureFeatures:
    """家具特征"""
    detections: List[Dict]  # 检测到的家具列表
    layout: Dict[str, float] = field(default_factory=lambda: {
        'density': 0.0,        # 布局密度
        'layout_score': 0.0    # 布局评分
    })
    furniture_stats: Dict[str, int] = field(default_factory=dict)  # 家具类型统计
    spatial_arrangement: Dict[str, float] = field(default_factory=lambda: {
        'density': 0.0,        # 空间密度
        'balance': 0.0,        # 布局平衡度
        'accessibility': 0.0   # 可达性
    })

@dataclass
class LightingFeatures:
    """光照特征"""
    basic_metrics: Dict[str, float] = field(default_factory=lambda: {
        'brightness': 0.0,     # 整体亮度
        'uniformity': 0.0,     # 均匀度
        'contrast': 0.0        # 对比度
    })
    quality: Dict[str, float] = field(default_factory=lambda: {
        'score': 0.0,          # 质量评分
        'color_temperature': 0.0  # 色温
    })
    light_sources: Dict[str, Any] = field(default_factory=lambda: {
        'natural_light_ratio': 0.0,  # 自然光比例
        'sources': []          # 光源列表
    })

@dataclass
class ColorFeatures:
    """颜色特征"""
    main_colors: List[Dict]    # 主要颜色列表
    color_metrics: Dict[str, float] = field(default_factory=lambda: {
        'harmony_score': 0.0,  # 和谐度
        'avg_saturation': 0.0, # 平均饱和度
        'color_temperature': 0.0  # 色温
    })
    emotional_metrics: Dict[str, float] = field(default_factory=lambda: {
        'warmth': 0.0,         # 温暖度
        'energy': 0.0,         # 能量度
        'harmony': 0.0         # 和谐度
    })
    color_scheme: str = ''     # 配色方案

@dataclass
class StyleFeatures:
    """风格特征"""
    primary_style: Dict[str, Any] = field(default_factory=lambda: {
        'type': '',           # 主要风格
        'confidence': 0.0,    # 置信度
        'consistency': 0.0    # 一致性
    })
    style_elements: Dict[str, float] = field(default_factory=lambda: {
        'texture_complexity': 0.0,  # 纹理复杂度
        'color_diversity': 0.0,     # 色彩多样性
        'shape_features': {}        # 形状特征
    })
    style_distribution: Dict[str, float] = field(default_factory=dict)  # 风格分布
    style_characteristics: Dict[str, Any] = field(default_factory=dict) # 风格特征

@dataclass
class AnalyzerResult:
    """分析器结果基类"""
    analyzer_type: str
    confidence: float
    features: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=lambda: {
        'timestamp': datetime.now().timestamp()
    })

    def to_dict(self) -> Dict:
        """转换为字典格式"""
        return {
            'analyzer_type': self.analyzer_type,
            'confidence': self.confidence,
            'features': self.features,
            'metadata': self.metadata
        }

@dataclass
class SceneResult(AnalyzerResult):
    """场景分析结果"""
    features: SceneFeatures = field(default_factory=SceneFeatures)

@dataclass
class FurnitureResult(AnalyzerResult):
    """家具分析结果"""
    features: FurnitureFeatures = field(default_factory=FurnitureFeatures)

@dataclass
class LightingResult(AnalyzerResult):
    """光照分析结果"""
    features: LightingFeatures = field(default_factory=LightingFeatures)

@dataclass
class ColorResult(AnalyzerResult):
    """颜色分析结果"""
    features: ColorFeatures = field(default_factory=ColorFeatures)

@dataclass
class StyleResult(AnalyzerResult):
    """风格分析结果"""
    features: StyleFeatures = field(default_factory=StyleFeatures)

@dataclass
class FrameResult:
    """单帧分析结果"""
    frame_id: int
    timestamp: float
    scene: Optional[SceneResult] = None
    furniture: Optional[FurnitureResult] = None
    lighting: Optional[LightingResult] = None
    color: Optional[ColorResult] = None
    style: Optional[StyleResult] = None

@dataclass
class CollectedResults:
    """收集的分析结果"""
    analyzer_results: Dict[str, List[AnalyzerResult]]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        """转换为字典格式"""
        return {
            'analyzer_results': {
                k: [r.to_dict() for r in v]
                for k, v in self.analyzer_results.items()
            },
            'metadata': self.metadata
        }

@dataclass
class ProcessedResult:
    """处理后的结果"""
    success: bool = True
    results: Dict[str, Any] = field(default_factory=dict)
    suggestions: List[str] = field(default_factory=list)
    error: Optional[str] = None
    visualization: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def error_result(cls, error_msg: str) -> 'ProcessedResult':
        """创建错误结果"""
        return cls(
            success=False,
            results={},
            suggestions=[],
            error=error_msg
        )

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            'success': self.success,
            'results': self.results,
            'suggestions': self.suggestions,
            'error': self.error,
            'visualization': self.visualization,
            'metadata': self.metadata
        }

@dataclass
class FormattedReport:
    """格式化后的分析报告"""
    video_info: Dict[str, Any]
    analysis_results: Dict
    suggestions: List[Dict]
    summary: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        """转换为字典格式"""
        # 处理 analysis_results 中的 AnalyzerResult 对象
        processed_results = {}
        for key, value in self.analysis_results.items():
            if isinstance(value, AnalyzerResult):
                processed_results[key] = {
                    'type': value.analyzer_type,
                    'confidence': value.confidence,
                    'features': value.features,
                    'metadata': value.metadata
                }
            else:
                processed_results[key] = value

        return {
            'video_info': self.video_info,
            'analysis_results': processed_results,
            'suggestions': self.suggestions,
            'summary': self.summary,
            'metadata': self.metadata
        }

@dataclass
class AnalysisResult:
    """分析结果类"""
    
    def __init__(self, scene_result=None, furniture_result=None, 
                 lighting_result=None, style_result=None, color_result=None):
        self.scene_result = scene_result
        self.furniture_result = furniture_result
        self.lighting_result = lighting_result
        self.style_result = style_result
        self.color_result = color_result
        
    def to_dict(self) -> Dict:
        """转换为字典格式"""
        return {
            'scene': self.scene_result,
            'furniture': self.furniture_result,
            'lighting': self.lighting_result,
            'style': self.style_result,
            'color': self.color_result
        }

@dataclass
class ImageAnalysisResult:
    """图像分析结果"""
    scene: Optional[SceneResult] = None
    furniture: Optional[FurnitureResult] = None
    lighting: Optional[LightingResult] = None
    style: Optional[StyleResult] = None
    color: Optional[ColorResult] = None
    suggestions: Optional[List[str]] = None
    timestamp: datetime = field(default_factory=datetime.now)

@dataclass
class VideoAnalysisResult:
    """视频分析结果"""
    frames: List[ImageAnalysisResult] = field(default_factory=list)
    summary: Dict = field(default_factory=dict)
    metadata: Dict = field(default_factory=dict)
    suggestions: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)

@dataclass
class RealtimeAnalysisResult:
    """实时分析结果"""
    current_frame: ImageAnalysisResult
    scene_history: List[Dict] = field(default_factory=list)
    performance_metrics: Dict = field(default_factory=dict)
    error: Optional[str] = None

# 分析建议类型
class SuggestionType(TypedDict):
    """分析建议类型"""
    category: str  # 建议类别
    priority: int  # 优先级
    content: str   # 建议内容
    reason: str    # 建议原因
    confidence: float  # 建议置信度

@dataclass
class AnalysisError:
    """分析错误"""
    error_type: str
    message: str
    details: Optional[Dict] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)

# 格式化结果类型
class FormattedResult(TypedDict):
    """格式化后的结果"""
    formatted_text: str  # 格式化后的文本
    raw_data: Dict  # 原始数据
    suggestions: Dict  # 建议信息

# 分析结果联合类型
AnalysisResultType = Union[
    ImageAnalysisResult,
    VideoAnalysisResult,
    RealtimeAnalysisResult,
    AnalysisError
]

@dataclass
class OutputData:
    """输出数据结构"""
    analysis_results: Dict[str, Any]  # 处理后的分析结果
    suggestions: List[str]  # 改进建议
    summary: str  # 报告摘要
    metadata: Dict[str, Any]  # 元数据

    def to_dict(self) -> Dict:
        """转换为字典格式"""
        return {
            'analysis_results': self.analysis_results,
            'suggestions': self.suggestions,
            'summary': self.summary,
            'metadata': self.metadata
        }

__all__ = [
    'AnalysisType',
    'AnalysisResult',
    'SceneResult',
    'FurnitureResult',
    'LightingResult',
    'StyleResult',
    'ColorResult',
    'ImageAnalysisResult',
    'VideoAnalysisResult',
    'RealtimeAnalysisResult',
    'AnalysisError',
    'SuggestionType',
    'FormattedResult',
    'AnalysisResultType',
    'OutputData'
] 
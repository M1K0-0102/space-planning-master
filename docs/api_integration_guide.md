## 一、系统概述
# 空间规划系统 API 集成指南
本系统提供室内场景的智能分析功能，采用模块化设计，通过 Pipeline 模式串联各个分析组件。主要功能包括：
- 场景识别
- 家具检测
- 光照分析
- 风格分析
- 颜色分析
## 二、核心数据流
mermaidgraph TDA[输入数据] --> B[InputValidator]B --> C[Processor处理]C --> D[Strategy策略]D --> E[Analyzer分析]E --> F[ResultCollector]F --> G[ResultProcessor]G --> H[SuggestionGenerator]H --> I[Formatter格式化]I --> J[OutputManager输出]
## 三、接口规范
### 1. 图片分析接口
pythondef analyze_image(image: np.ndarray,mode: str = 'comprehensive',analyzer_type: Optional[str] = None) -> Dict[str, Any]:"""分析单张图片Args:image: BGR格式的numpy数组，shape=(H,W,3)，dtype=uint8mode: 分析模式，可选值：
•  'comprehensive': 综合分析（默认）
•  'single': 单项分析analyzer_type: 单项分析时的分析器类型，可选值：
•  'scene': 场景分析
•  'furniture': 家具检测
•  'lighting': 光照分析
•  'style': 风格分析
•  'color': 颜色分析Returns:Dict: 分析结果，包含以下字段：{"success": bool,"error": Optional[str],"data": {"scene": {...},"furniture": [...],"lighting": {...},"style": {...},"color": {...}},"suggestions": {"formatted_text": str,"raw_data": List[str]}}Raises:ValueError: 输入数据格式错误RuntimeError: 处理过程错误"""
### 2. 视频分析接口
pythondef analyze_video(video_path: str,mode: str = 'comprehensive',analyzer_type: Optional[str] = None) -> Dict[str, Any]:"""分析视频文件Args:video_path: 视频文件路径mode: 同图片分析接口analyzer_type: 同图片分析接口Returns:Dict: 同图片分析接口Raises:FileNotFoundError: 视频文件不存在ValueError: 不支持的视频格式"""
### 3. 实时分析接口
pythondef analyze_realtime(frame: np.ndarray) -> Dict[str, Any]:"""实时分析视频帧Args:frame: 同图片分析接口的image参数Returns:Dict: 同图片分析接口，但不包含suggestions字段Raises:ValueError: 输入数据格式错误"""
## 四、返回数据详细说明
### 1. 场景分析结果
python{"scene": {"type": str, # 场景类型，如"客厅"、"卧室"等"confidence": float, # 置信度，范围[0,1]"features": {"area": float, # 空间面积(㎡)"symmetry": float, # 对称性评分[0,1]"wall_visibility": float, # 墙面可见度[0,1]"natural_light": float # 自然光评分[0,1]}}}
### 2. 家具检测结果
python{"furniture": {"items": [{"class_name": str, # 家具类别"confidence": float, # 置信度[0,1]"bbox": [x1,y1,x2,y2], # 边界框坐标"area": float # 占比[0,1]}],"layout": {"density": float, # 布局密度[0,1]"symmetry": float, # 布局对称性[0,1]"score": float # 布局评分[0,100]}}}
### 3. 光照分析结果
python{"lighting": {"basic_metrics": {"brightness": float, # 整体亮度[0,1]"uniformity": float, # 均匀度[0,1]"contrast": float # 对比度[0,1]},"quality": {"score": float, # 光照质量评分[0,1]"color_temperature": float # 色温(K)}}}
### 4. 风格分析结果
python{"style": {"primary_style": {"name": str, # 主要风格"confidence": float, # 置信度[0,1]"consistency": float # 风格一致性[0,1]},"style_distribution": { # 风格构成"现代简约": float,"北欧": float,"中式": float,# ...其他风格占比}}}
## 五、错误处理规范
### 1. 错误码定义
pythonERROR_CODES = {"E001": "输入数据格式错误","E002": "模型加载失败","E003": "处理过程错误","E004": "资源不足","E005": "文件不存在","E006": "不支持的格式"}
### 2. 错误返回格式
python{"success": False,"error": {"code": "E001","message": "输入图片格式错误","details": "Expected shape (H,W,3), got (H,W)"},"data": None}
## 六、使用示例
pythonfrom src.pipeline.interior_design_pipeline import InteriorDesignPipeline
初始化pipeline
pipeline = InteriorDesignPipeline()
图片分析
image = cv2.imread("room.jpg")result = pipeline.analyze_image(image=image,mode="comprehensive")
视频分析
video_result = pipeline.analyze_video(video_path="room_tour.mp4",mode="comprehensive")
实时分析
cap = cv2.VideoCapture(0)ret, frame = cap.read()if ret:realtime_result = pipeline.analyze_realtime(frame)
## 七、性能与限制
1. 处理时延
- 图片分析：200-300ms
- 视频分析：150-200ms/帧
- 实时分析：<300ms/帧
2. 资源限制
- 最大图片尺寸：1920x1080
- 最大视频时长：5分钟
- 最大文件大小：500MB
- 并发请求数：10
3. 硬件要求
- CPU：4核心以上
- 内存：8GB以上
- GPU：4GB显存以上
- 硬盘：10GB可用空间
## 八、版本信息
- API版本：1.0.0
- 更新日期：2024-02-08
- 文档维护：算法组


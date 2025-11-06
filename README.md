# 空间规划大师 - 室内设计智能分析系统

基于深度学习的室内设计分析系统，能够对室内空间图像进行多维度智能分析，包括场景识别、家具检测、光照分析、色彩分析和风格识别。

## 📋 项目特性

- 🏠 **场景识别**：基于 ResNet50-Places365 模型识别室内场景类型
- 🛋️ **家具检测**：使用 YOLOv8 检测和定位室内家具物品
- 💡 **光照分析**：评估室内光照条件和分布
- 🎨 **色彩分析**：提取主要色彩并进行色彩协调性分析
- 🖼️ **风格识别**：基于 EfficientNetV2 识别室内设计风格
- 📊 **综合建议**：生成专业的室内设计改进建议

## 🛠️ 技术栈

- Python 3.8+
- PyTorch 2.0+
- OpenCV
- YOLOv8 (Ultralytics)
- EfficientNetV2 (timm)
- scikit-learn

## 📦 安装指南

### 1. 克隆项目

```bash
# 从 GitHub 克隆
git clone https://github.com/M1K0-0102/space-planning-master.git

# 或从 Gitee 克隆
git clone https://gitee.com/M1K0-0102/space-planning-master.git

cd space-planning-master
```

### 2. 创建虚拟环境（推荐）

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

### 3. 安装依赖

```bash
pip install -r requirements.txt

# 或使用 setup.py 安装
pip install -e .
```

### 4. 下载预训练模型

由于模型文件较大，需要单独下载：

1. **ResNet50-Places365** (场景识别)
   - 下载地址: [Places365 Models](http://places2.csail.mit.edu/models_places365/)
   - 保存为: `pretrained_models/resnet50_places365.pth`

2. **YOLOv8n** (家具检测)
   - 下载地址: [YOLOv8](https://github.com/ultralytics/ultralytics)
   - 保存为: `pretrained_models/yolov8n.pt`
   - 或首次运行时自动下载

3. **EfficientNetV2-M** (风格识别)
   - 下载地址: [timm models](https://github.com/huggingface/pytorch-image-models)
   - 保存为: `pretrained_models/pre_efficientnetv2-m.pth`
   - 或使用预训练权重（自动下载）

### 5. 创建必要的目录

```bash
mkdir -p output/logs
mkdir -p pretrained_models
```

## 🚀 快速开始

### 基本使用

```python
from src.pipeline.interior_design_pipeline import InteriorDesignPipeline

# 创建分析管道
pipeline = InteriorDesignPipeline()

# 分析单张图像
result = pipeline.process_image("path/to/your/image.jpg")

# 打印结果
print(result)
```

### 批量处理

```python
# 处理多张图像
images = ["image1.jpg", "image2.jpg", "image3.jpg"]
for img_path in images:
    result = pipeline.process_image(img_path)
    print(f"分析完成: {img_path}")
```

### 视频分析

```python
from src.pipeline.processors.video_processor import VideoProcessor

processor = VideoProcessor()
results = processor.process("path/to/video.mp4")
```

## 📂 项目结构

```
空间规划大师/
├── src/                        # 源代码
│   ├── pipeline/              # 核心管道
│   │   ├── analyzers/        # 各类分析器
│   │   ├── processors/       # 图像/视频处理器
│   │   ├── strategies/       # 分析策略
│   │   ├── utils/            # 工具函数
│   │   ├── validators/       # 数据验证
│   │   └── visualization/    # 可视化工具
│   └── config/               # 配置文件
├── tests/                     # 测试文件
├── docs/                      # 文档
├── pretrained_models/        # 预训练模型（需下载）
├── output/                   # 输出结果
├── requirements.txt          # 依赖列表
├── setup.py                  # 安装脚本
└── README.md                 # 本文件
```

## 🧪 运行测试

```bash
# 运行所有测试
pytest tests/

# 运行可视化测试
python tests/visual_test.py
```

## ⚙️ 配置说明

配置文件位于 `src/config/model_config.yaml`，可以修改：

- 模型路径
- 置信度阈值
- 输入图像大小
- 设备选择 (CPU/GPU)
- 批处理大小

示例配置：

```yaml
analyzers:
  scene:
    confidence_threshold: 0.3
    device: cpu
  furniture:
    confidence_threshold: 0.25
    device: cpu
```

## 📊 输出说明

分析结果包含以下信息：

- **场景类型**：识别的室内场景（如客厅、卧室等）
- **家具列表**：检测到的家具及其位置
- **光照评估**：亮度、均匀度、对比度等指标
- **色彩分析**：主色调和配色方案
- **风格识别**：室内设计风格分类
- **改进建议**：基于分析结果的专业建议

结果会保存在 `output/` 目录下，包括 JSON 和文本格式。

## 🐛 常见问题

### 1. 模型加载失败

**问题**: `FileNotFoundError: pretrained_models/xxx.pth`

**解决**: 确保已下载所有预训练模型并放置在正确位置

### 2. CUDA 相关错误

**问题**: `RuntimeError: CUDA out of memory`

**解决**: 在配置文件中将 `device` 设置为 `cpu` 或减小 `batch_size`

### 3. OpenCV 导入错误

**问题**: `ImportError: libGL.so.1`

**解决** (Linux):
```bash
sudo apt-get install libgl1-mesa-glx
```

## 🤝 贡献指南

欢迎提交 Issue 和 Pull Request！

1. Fork 本项目
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启 Pull Request

## 📄 许可证

本项目采用 MIT 许可证 - 详见 [LICENSE](LICENSE) 文件

## 👥 团队

- 开发者: [miko]
- 联系方式: [15619352991@163.com]

## 🙏 致谢

- [Places365](http://places2.csail.mit.edu/) - 场景识别数据集
- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) - 物体检测
- [PyTorch Image Models](https://github.com/huggingface/pytorch-image-models) - 预训练模型

## 📝 更新日志

### v0.1.0 (2025-03-05)
- ✨ 初始版本发布
- 🎯 实现五大核心分析功能
- 📊 添加可视化支持
- 🧪 完善测试用例


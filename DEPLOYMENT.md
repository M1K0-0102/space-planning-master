# 部署指南 - 空间规划大师

本文档详细说明如何在不同环境中部署和配置"空间规划大师"项目。

## 📋 目录

- [系统要求](#系统要求)
- [环境准备](#环境准备)
- [详细安装步骤](#详细安装步骤)
- [模型下载指南](#模型下载指南)
- [配置说明](#配置说明)
- [验证安装](#验证安装)
- [常见问题](#常见问题)

## 🖥️ 系统要求

### 最低配置
- **操作系统**: Windows 10+, Linux (Ubuntu 18.04+), macOS 10.14+
- **Python**: 3.8 或更高版本
- **内存**: 8GB RAM
- **存储**: 5GB 可用空间（包括模型文件）
- **处理器**: 支持 AVX 指令集的 CPU

### 推荐配置
- **内存**: 16GB+ RAM
- **GPU**: NVIDIA GPU (支持 CUDA 11.0+) 可选，用于加速推理
- **存储**: 10GB+ 可用空间

## 🔧 环境准备

### 1. 安装 Python

#### Windows
```powershell
# 从官网下载 Python 3.8+
# https://www.python.org/downloads/

# 验证安装
python --version
pip --version
```

#### Linux (Ubuntu/Debian)
```bash
sudo apt update
sudo apt install python3.8 python3-pip python3-venv
python3 --version
pip3 --version
```

#### macOS
```bash
# 使用 Homebrew
brew install python@3.8

# 验证安装
python3 --version
pip3 --version
```

### 2. 安装 Git

#### Windows
```powershell
# 下载 Git for Windows
# https://git-scm.com/download/win

# 或使用 Chocolatey
choco install git

# 验证安装
git --version
```

#### Linux
```bash
sudo apt install git  # Ubuntu/Debian
sudo yum install git  # CentOS/RHEL
git --version
```

#### macOS
```bash
brew install git
git --version
```

### 3. 配置 Git (首次使用)

```bash
# 设置用户名和邮箱
git config --global user.name "你的名字"
git config --global user.email "your.email@example.com"

# 查看配置
git config --list
```

## 📥 详细安装步骤

### 步骤 1: 克隆仓库

```bash
# 从 GitHub 克隆
git clone https://github.com/your-username/space-planning-master.git

# 或从 Gitee 克隆（国内推荐）
git clone https://gitee.com/your-username/space-planning-master.git

# 进入项目目录
cd space-planning-master
```

### 步骤 2: 创建虚拟环境

#### Windows
```powershell
# 创建虚拟环境
python -m venv venv

# 激活虚拟环境
venv\Scripts\activate

# 确认激活（命令行前面应该显示 (venv)）
```

#### Linux/macOS
```bash
# 创建虚拟环境
python3 -m venv venv

# 激活虚拟环境
source venv/bin/activate

# 确认激活（命令行前面应该显示 (venv)）
```

### 步骤 3: 安装依赖

```bash
# 更新 pip
pip install --upgrade pip

# 安装项目依赖
pip install -r requirements.txt

# 或以开发模式安装
pip install -e .
```

### 步骤 4: 创建必要的目录

```bash
# Windows PowerShell
New-Item -ItemType Directory -Force -Path output, output\logs, pretrained_models

# Linux/macOS
mkdir -p output/logs pretrained_models
```

## 📦 模型下载指南

### 方法 1: 手动下载（推荐）

#### 1. ResNet50-Places365 (场景识别)

```bash
# 下载地址
# http://places2.csail.mit.edu/models_places365/resnet50_places365.pth.tar

# 保存到
# pretrained_models/resnet50_places365.pth
```

**步骤**:
1. 访问 [Places365 Models](http://places2.csail.mit.edu/models_places365/)
2. 下载 `resnet50_places365.pth.tar`
3. 解压并重命名为 `resnet50_places365.pth`
4. 移动到 `pretrained_models/` 目录

#### 2. YOLOv8n (家具检测)

```bash
# 使用 Python 脚本自动下载
python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"

# 或从官方仓库下载
# https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt

# 保存到
# pretrained_models/yolov8n.pt
```

#### 3. EfficientNetV2-M (风格识别)

```python
# 创建下载脚本 download_models.py
import timm
import torch

# 下载 EfficientNetV2-M 预训练权重
model = timm.create_model('tf_efficientnetv2_m', pretrained=True)
torch.save(model.state_dict(), 'pretrained_models/pre_efficientnetv2-m.pth')
print("模型下载完成！")
```

运行脚本:
```bash
python download_models.py
```

### 方法 2: 使用网盘下载

如果网络条件不好，可以从团队网盘下载所有模型：

```
网盘链接: [提供你的网盘链接]
提取码: [提取码]

包含文件:
- resnet50_places365.pth (约 97MB)
- yolov8n.pt (约 6MB)
- pre_efficientnetv2-m.pth (约 208MB)
- places365_zh.txt (分类标签)
```

下载后解压到 `pretrained_models/` 目录。

### 验证模型文件

```bash
# 检查模型文件是否存在
# Windows PowerShell
Get-ChildItem pretrained_models

# Linux/macOS
ls -lh pretrained_models/

# 应该看到以下文件:
# resnet50_places365.pth
# yolov8n.pt
# pre_efficientnetv2-m.pth
# places365_zh.txt
```

## ⚙️ 配置说明

### 修改配置文件

编辑 `src/config/model_config.yaml`:

```yaml
# 如果只有 CPU，确保所有 device 设置为 cpu
analyzers:
  scene:
    device: cpu
  furniture:
    device: cpu
  style:
    device: cpu

# 如果内存不足，可以减小批处理大小
defaults:
  batch_size: 1
```

### GPU 加速配置（可选）

如果有 NVIDIA GPU:

```bash
# 安装 CUDA 版本的 PyTorch
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# 修改配置文件
# device: cpu -> device: cuda
```

验证 GPU 可用性:
```python
import torch
print(f"CUDA 可用: {torch.cuda.is_available()}")
print(f"GPU 数量: {torch.cuda.device_count()}")
```

## ✅ 验证安装

### 1. 运行快速测试

创建测试脚本 `quick_test.py`:

```python
import sys
sys.path.insert(0, '.')

from src.pipeline.interior_design_pipeline import InteriorDesignPipeline

print("正在初始化分析管道...")
try:
    pipeline = InteriorDesignPipeline()
    print("✓ 初始化成功！")
    print("✓ 所有模型加载正常！")
    print("\n系统已准备就绪，可以开始使用。")
except Exception as e:
    print(f"✗ 初始化失败: {e}")
    print("\n请检查:")
    print("1. 所有依赖是否已安装")
    print("2. 预训练模型是否已下载")
    print("3. 配置文件是否正确")
```

运行测试:
```bash
python quick_test.py
```

### 2. 运行完整测试

```bash
# 运行单元测试
pytest tests/ -v

# 运行可视化测试（需要测试图像）
python tests/visual_test.py
```

### 3. 测试单张图像

```python
from src.pipeline.interior_design_pipeline import InteriorDesignPipeline

pipeline = InteriorDesignPipeline()

# 替换为你的测试图像路径
result = pipeline.process_image("test_image.jpg")
print(result)
```

## 🐛 常见问题

### 问题 1: pip 安装超时

**解决方案**: 使用国内镜像源

```bash
# 临时使用
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# 永久配置
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
```

### 问题 2: torch 安装失败

**解决方案**: 单独安装 PyTorch

```bash
# CPU 版本
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# GPU 版本 (CUDA 11.8)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### 问题 3: 模型加载失败

**错误**: `FileNotFoundError: pretrained_models/xxx.pth`

**解决方案**:
1. 确认模型文件已下载
2. 检查文件路径和名称是否正确
3. 验证文件完整性（不是损坏的下载）

### 问题 4: 内存不足

**错误**: `RuntimeError: [enforce fail at alloc_cpu.cpp:...] . DefaultCPUAllocator: not enough memory`

**解决方案**:
1. 关闭其他占用内存的程序
2. 减小 `batch_size` 设置
3. 使用更小的输入图像尺寸

### 问题 5: OpenCV 错误 (Linux)

**错误**: `ImportError: libGL.so.1: cannot open shared object file`

**解决方案**:
```bash
sudo apt-get install libgl1-mesa-glx libglib2.0-0
```

### 问题 6: 权限错误 (Linux/macOS)

**错误**: `PermissionError: [Errno 13] Permission denied`

**解决方案**:
```bash
# 修改目录权限
chmod -R 755 output/
chmod -R 755 pretrained_models/
```

## 🚀 部署到生产环境

### 使用 Docker (推荐)

创建 `Dockerfile`:

```dockerfile
FROM python:3.8-slim

WORKDIR /app

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 复制依赖文件
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 复制项目文件
COPY . .

# 创建必要的目录
RUN mkdir -p output/logs

CMD ["python", "your_main_script.py"]
```

构建和运行:
```bash
docker build -t space-planning-master .
docker run -v $(pwd)/output:/app/output space-planning-master
```

## 📞 获取帮助

如果遇到其他问题:

1. 查看 [README.md](README.md) 常见问题部分
2. 查看项目 Issues: [GitHub Issues](https://github.com/your-username/space-planning-master/issues)
3. 联系项目维护者: [your.email@example.com]

## 📝 下一步

安装完成后，你可以:

1. 阅读 [API 使用文档](docs/api_integration_guide.md)
2. 查看 [技术报告](docs/technical_report.md)
3. 运行测试用例了解功能
4. 开始分析你的室内设计图像！

祝使用愉快！ 🎉


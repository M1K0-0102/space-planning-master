#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
快速安装验证脚本
用于验证环境配置是否正确
"""

import sys
import os

def check_python_version():
    """检查 Python 版本"""
    print("=" * 60)
    print("1. 检查 Python 版本...")
    version = sys.version_info
    print(f"   当前版本: Python {version.major}.{version.minor}.{version.micro}")
    
    if version.major >= 3 and version.minor >= 8:
        print("   ✓ Python 版本符合要求 (3.8+)")
        return True
    else:
        print("   ✗ Python 版本过低，需要 3.8 或更高版本")
        return False

def check_dependencies():
    """检查依赖包是否已安装"""
    print("\n" + "=" * 60)
    print("2. 检查依赖包...")
    
    required_packages = {
        'numpy': 'numpy',
        'torch': 'torch',
        'cv2': 'opencv-python',
        'sklearn': 'scikit-learn',
        'PIL': 'Pillow',
        'yaml': 'PyYAML',
        'ultralytics': 'ultralytics',
        'timm': 'timm'
    }
    
    missing_packages = []
    
    for import_name, package_name in required_packages.items():
        try:
            __import__(import_name)
            print(f"   ✓ {package_name}")
        except ImportError:
            print(f"   ✗ {package_name} 未安装")
            missing_packages.append(package_name)
    
    if missing_packages:
        print(f"\n   缺少以下依赖包: {', '.join(missing_packages)}")
        print("   请运行: pip install -r requirements.txt")
        return False
    
    print("   ✓ 所有依赖包已安装")
    return True

def check_model_files():
    """检查模型文件是否存在"""
    print("\n" + "=" * 60)
    print("3. 检查预训练模型...")
    
    model_dir = "pretrained_models"
    required_models = {
        'resnet50_places365.pth': '场景识别模型',
        'yolov8n.pt': '家具检测模型',
        'pre_efficientnetv2-m.pth': '风格识别模型',
        'places365_zh.txt': '场景分类标签'
    }
    
    missing_models = []
    
    for filename, description in required_models.items():
        filepath = os.path.join(model_dir, filename)
        if os.path.exists(filepath):
            size_mb = os.path.getsize(filepath) / (1024 * 1024)
            print(f"   ✓ {description} ({filename}) - {size_mb:.1f} MB")
        else:
            print(f"   ✗ {description} ({filename}) 缺失")
            missing_models.append(filename)
    
    if missing_models:
        print(f"\n   缺少以下模型文件: {', '.join(missing_models)}")
        print("   请参考 DEPLOYMENT.md 中的模型下载指南")
        return False
    
    print("   ✓ 所有模型文件已准备")
    return True

def check_directories():
    """检查必要的目录是否存在"""
    print("\n" + "=" * 60)
    print("4. 检查项目目录...")
    
    required_dirs = ['output', 'output/logs', 'pretrained_models', 'src', 'tests']
    
    for dirname in required_dirs:
        if os.path.exists(dirname):
            print(f"   ✓ {dirname}/")
        else:
            print(f"   ! {dirname}/ 不存在，正在创建...")
            os.makedirs(dirname, exist_ok=True)
    
    print("   ✓ 所有必要目录已准备")
    return True

def test_pipeline():
    """测试分析管道是否能正常初始化"""
    print("\n" + "=" * 60)
    print("5. 测试分析管道初始化...")
    
    try:
        sys.path.insert(0, '.')
        from src.pipeline.interior_design_pipeline import InteriorDesignPipeline
        
        print("   正在初始化管道...")
        pipeline = InteriorDesignPipeline()
        print("   ✓ 分析管道初始化成功！")
        return True
        
    except Exception as e:
        print(f"   ✗ 初始化失败: {e}")
        print("\n   可能的原因:")
        print("   - 缺少依赖包")
        print("   - 模型文件未下载或路径不正确")
        print("   - 配置文件有误")
        return False

def main():
    """主函数"""
    print("\n🔧 空间规划大师 - 环境检查工具\n")
    
    results = []
    
    # 运行所有检查
    results.append(("Python 版本", check_python_version()))
    results.append(("依赖包", check_dependencies()))
    results.append(("项目目录", check_directories()))
    results.append(("预训练模型", check_model_files()))
    
    # 如果前面都通过了，再测试管道
    if all(r[1] for r in results):
        results.append(("分析管道", test_pipeline()))
    
    # 输出总结
    print("\n" + "=" * 60)
    print("📊 检查总结\n")
    
    for name, status in results:
        status_symbol = "✓" if status else "✗"
        print(f"   {status_symbol} {name}")
    
    all_passed = all(r[1] for r in results)
    
    if all_passed:
        print("\n🎉 太棒了！所有检查都通过了！")
        print("\n📚 接下来你可以:")
        print("   - 运行测试: python tests/visual_test.py")
        print("   - 分析图像: 查看 README.md 中的使用示例")
        print("   - 查看文档: 阅读 docs/ 目录下的文档")
    else:
        print("\n⚠️  存在一些问题需要解决")
        print("\n📖 请参考:")
        print("   - DEPLOYMENT.md - 详细的部署指南")
        print("   - README.md - 项目说明和快速开始")
        print("   - 或联系项目维护者获取帮助")
    
    print("\n" + "=" * 60)
    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())


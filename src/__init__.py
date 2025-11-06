"""
空间规划大师 - 室内设计分析系统
"""

import sys
import os

# 添加src目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 修改导入路径
from src.pipeline.interior_design_pipeline import InteriorDesignPipeline

__version__ = '0.1.0'
__all__ = ['InteriorDesignPipeline'] 
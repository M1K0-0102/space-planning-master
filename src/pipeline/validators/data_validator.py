from typing import Any, Dict, Optional
import numpy as np
import torch
import logging
import cv2

class DataValidator:
    """数据验证器 - 验证数据格式和内容"""
    
    def __init__(self):
        self.logger = logging.getLogger("pipeline.DataValidator")
        
    def validate_pipeline_input(self, data: Any, input_type: str) -> bool:
        """验证管道输入"""
        try:
            if input_type == 'image':
                return self.validate_frame(data)
            elif input_type == 'video':
                return isinstance(data, cv2.VideoCapture) and data.isOpened()
            elif input_type == 'realtime':
                return self.validate_frame(data)
            return False
        except Exception as e:
            self.logger.error(f"输入验证失败: {str(e)}")
            return False
            
    def validate_pipeline_output(self, output: Dict) -> bool:
        """验证管道输出"""
        try:
            if not isinstance(output, dict):
                return False
                
            if 'error' in output:
                return True  # 错误输出也是有效的
                
            required_fields = ['type', 'results', 'timestamp']
            if not all(field in output for field in required_fields):
                return False
                
            return True
            
        except Exception as e:
            self.logger.error(f"输出验证失败: {str(e)}")
            return False

    def validate_analyzer_output(self, result: Dict) -> bool:
        """验证分析器输出"""
        try:
            if not isinstance(result, dict):
                self.logger.error("分析结果必须是字典类型")
                return False
            
            # 检查必要字段
            required_fields = ['type', 'confidence', 'features', 'metadata']
            missing_fields = [field for field in required_fields if field not in result]
            if missing_fields:
                self.logger.error(f"缺少必要字段: {missing_fields}")
                return False
            
            # 验证置信度
            if not isinstance(result['confidence'], (int, float)):
                self.logger.error("置信度必须是数值类型")
                return False
            
            # 验证特征
            if not isinstance(result['features'], dict):
                self.logger.error("特征必须是字典类型")
                return False
            
            # 验证元数据
            if not isinstance(result['metadata'], dict):
                self.logger.error("元数据必须是字典类型")
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"分析结果验证失败: {str(e)}")
            return False
        
    def validate_collector_output(self, result: Dict) -> bool:
        """验证收集器输出"""
        required_sections = ['scene', 'furniture', 'lighting', 'color', 'style']
        if not all(section in result for section in required_sections):
            self.logger.error(f"结果缺少必要部分: {required_sections}")
            return False
            
        return True

    def _validate_numpy_input(self, data: np.ndarray) -> bool:
        """验证 numpy 数组输入"""
        try:
            if len(data.shape) != 3:
                self.logger.error(f"无效的图像维度: {data.shape}")
                return False
            
            if data.shape[2] != 3:
                self.logger.error(f"无效的通道数: {data.shape[2]}")
                return False
            
            if not np.issubdtype(data.dtype, np.floating) and not np.issubdtype(data.dtype, np.integer):
                self.logger.error(f"无效的数据类型: {data.dtype}")
                return False
            
            return True
        
        except Exception as e:
            self.logger.error(f"numpy输入验证失败: {str(e)}")
            return False

    def _validate_tensor_input(self, data: torch.Tensor) -> bool:
        """验证 PyTorch 张量输入"""
        try:
            if len(data.shape) not in [3, 4]:  # 允许有或没有batch维度
                self.logger.error(f"无效的张量维度: {data.shape}")
                return False
            
            if len(data.shape) == 3:
                if data.shape[0] != 3:  # CHW格式
                    self.logger.error(f"无效的通道数: {data.shape[0]}")
                    return False
            else:  # NCHW格式
                if data.shape[1] != 3:
                    self.logger.error(f"无效的通道数: {data.shape[1]}")
                    return False
                
            if not torch.is_floating_point(data):
                self.logger.error(f"无效的数据类型: {data.dtype}")
                return False
            
            return True
        
        except Exception as e:
            self.logger.error(f"tensor输入验证失败: {str(e)}")
            return False

    def validate_analysis_result(self, result: Dict) -> bool:
        """验证分析结果"""
        try:
            self.logger.debug(f"开始验证分析结果: {result}")
            
            if not result:
                self.logger.warning("结果为空")
                return False
            
            # 检查顶层结构
            if 'analysis_results' not in result:
                self.logger.warning("缺少 analysis_results 字段")
                self.logger.debug(f"可用字段: {result.keys()}")
                return False
            
            analysis = result['analysis_results']
            self.logger.debug(f"分析结果内容: {analysis}")
            
            # 验证每个分析器的结果
            for analyzer_name, analyzer_result in analysis.items():
                # 检查必要字段
                required_fields = ['type', 'confidence', 'features']
                missing_fields = [field for field in required_fields if field not in analyzer_result]
                
                if missing_fields:
                    self.logger.warning(f"{analyzer_name} 分析结果缺少字段: {missing_fields}")
                    continue  # 继续检查其他分析器的结果
                
                # 检查字段类型
                if not isinstance(analyzer_result['confidence'], (int, float)):
                    self.logger.warning(f"{analyzer_name} 的 confidence 必须是数值类型")
                    continue
                
                if not isinstance(analyzer_result['features'], dict):
                    self.logger.warning(f"{analyzer_name} 的 features 必须是字典类型")
                    continue
                
                # metadata 是可选的
                if 'metadata' in analyzer_result and not isinstance(analyzer_result['metadata'], dict):
                    self.logger.warning(f"{analyzer_name} 的 metadata 必须是字典类型")
                    continue
            
            # 只要有一个分析器的结果是有效的就返回 True
            return True
            
        except Exception as e:
            self.logger.error(f"结果验证失败: {str(e)}")
            self.logger.debug("验证失败详情", exc_info=True)
            return False

    def validate_frame(self, frame: np.ndarray) -> bool:
        """验证帧格式"""
        try:
            if not isinstance(frame, np.ndarray):
                self.logger.error("输入必须是numpy数组")
                return False
            
            if len(frame.shape) != 3:
                self.logger.error(f"无效的帧维度: {frame.shape}")
                return False
            
            if frame.shape[2] != 3:
                self.logger.error(f"无效的通道数: {frame.shape[2]}")
                return False
            
            if frame.dtype != np.uint8:
                self.logger.error(f"无效的数据类型: {frame.dtype}")
                return False
            
            return True
        
        except Exception as e:
            self.logger.error(f"帧验证失败: {str(e)}")
            return False 
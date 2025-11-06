import logging
import time
from typing import Dict

class AnalysisError(Exception):
    """分析错误基类"""
    pass

class ModelError(AnalysisError):
    """模型错误"""
    pass

class InputError(AnalysisError):
    """输入错误"""
    pass

class ErrorHandler:
    def __init__(self):
        self.logger = logging.getLogger("pipeline.ErrorHandler")
        
    def handle_error(self, error: Exception, context: str) -> Dict:
        """处理错误"""
        try:
            error_type = type(error).__name__
            error_msg = str(error)
            
            # 记录错误
            self.logger.error(f"{context} - {error_type}: {error_msg}")
            
            # 返回标准错误格式
            return {
                'error': {
                    'type': error_type,
                    'message': error_msg,
                    'context': context,
                    'timestamp': time.time()
                }
            }
            
        except Exception as e:
            self.logger.error(f"错误处理失败: {str(e)}")
            return {'error': '未知错误'} 
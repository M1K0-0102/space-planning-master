from typing import Dict, List, Any, Optional
import logging
import os
import time
import json

class OutputManager:
    """输出管理器 - 负责输出原始结果和格式化后的结果"""
    
    def __init__(self):
        self.logger = logging.getLogger("pipeline.OutputManager")

    def output_results(self, 
                      raw_results: Dict,
                      raw_suggestions: List[str],
                      formatted_results: str,
                      formatted_suggestions: str):
        """输出原始结果(JSON)和格式化后的结果(TXT)
        Args:
            raw_results: 原始分析结果
            raw_suggestions: 原始建议列表
            formatted_results: 格式化后的分析结果
            formatted_suggestions: 格式化后的建议
        """
        self.logger.info("开始输出分析结果和建议...")
        
        try:
            # 1. 输出原始JSON数据
            json_data = {
                'timestamp': time.time(),
                'analysis': {
                    'results': raw_results,
                    'suggestions': raw_suggestions,
                },
                'metadata': {
                    'version': '1.0.0',
                    'generated_at': time.strftime('%Y-%m-%d %H:%M:%S')
                }
            }
            
            json_path = self._get_output_path('json')
            self.logger.debug(f"JSON输出路径: {json_path}")
            
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(json_data, f, ensure_ascii=False, indent=2)
            
            self.logger.info(f"原始结果已保存到: {json_path}")
            
            # 2. 输出格式化文本
            txt_path = self._get_output_path('txt')
            self.logger.debug(f"文本输出路径: {txt_path}")
            
            with open(txt_path, 'w', encoding='utf-8') as f:
                self._write_formatted_results(f, formatted_results, formatted_suggestions)
            
            self.logger.info(f"格式化结果已保存到: {txt_path}")
            
        except Exception as e:
            self.logger.error(f"结果输出失败: {str(e)}", exc_info=True)
            raise

    def _write_formatted_results(self, file, formatted_results: str, formatted_suggestions: str):
        """写入格式化结果和建议到文件"""
        self.logger.debug("写入格式化结果")
        file.write(formatted_results)
        
        # 添加分隔行
        file.write("\n" + "=" * 50 + "\n\n")
        
        self.logger.debug("写入格式化建议")
        file.write(formatted_suggestions)

    def _get_output_path(self, format_type: str) -> str:
        """生成输出文件路径
        Args:
            format_type: 'json' 或 'txt'
        """
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_dir = "output"
        os.makedirs(output_dir, exist_ok=True)
        extension = '.json' if format_type == 'json' else '.txt'
        return os.path.join(output_dir, f"analysis_result_{timestamp}{extension}")

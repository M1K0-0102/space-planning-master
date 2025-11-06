from typing import Dict, List, Any, Optional, Union
import logging

class SuggestionFormatter:
    """建议格式化器 - 格式化建议的展示形式"""
    
    def __init__(self):
        self.logger = logging.getLogger("pipeline.SuggestionFormatter")
        
    def format_suggestions(self, suggestions: List[str]) -> str:
        """格式化建议"""
        try:
            self.logger.info("开始格式化改进建议...")
            
            if not suggestions:
                return "\n=== 改进建议 ===\n\n暂无改进建议\n"
            
            formatted = "\n=== 改进建议 ===\n"
            formatted += "-" * 40 + "\n"
            
            # 对建议进行分类
            current_category = None
            suggestion_count = 1
            
            for suggestion in suggestions:
                # 检查是否是新类别（以"，建议："结尾的行）
                if "，建议：" in suggestion:
                    # 新类别的主标题
                    if current_category:
                        formatted += "\n"  # 在类别之间添加空行
                    current_category = suggestion.split("，建议：")[0]
                    formatted += f"{suggestion_count}. {current_category}\n"
                    suggestion_count += 1
                
                # 检查是否是子建议（以" - "开头的行）
                elif suggestion.startswith("  - "):
                    formatted += f"   {suggestion}\n"
                
                # 普通建议
                else:
                    if current_category:
                        formatted += "\n"  # 在类别之间添加空行
                        current_category = None
                    formatted += f"{suggestion_count}. {suggestion}\n"
                    suggestion_count += 1
            
            self.logger.info("建议格式化完成")
            return formatted
            
        except Exception as e:
            self.logger.error(f"建议格式化失败: {str(e)}")
            return "\n=== 改进建议 ===\n\n格式化失败\n" 
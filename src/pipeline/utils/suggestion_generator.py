from __future__ import annotations
from typing import Dict, List, Set, TYPE_CHECKING, Any, Optional, Union
import os
import json
import requests
import logging
from .result_types import (
    ProcessedResult,
    RawAnalysisResult
)

if TYPE_CHECKING:
    from .result_types import AnalysisResult

# 配置代理服务器
PROXY_CONFIG = {
    'http': 'http://127.0.0.1:7890',  # 常用代理端口
    'https': 'http://127.0.0.1:7890'
}

# 配置超时时间（秒）
TIMEOUT_CONFIG = {
    'connect': 30,    # 连接超时增加到30秒
    'read': 120      # 读取超时增加到120秒
}

# API Key 应该通过环境变量或 .env 文件配置，不要硬编码在代码中
# 使用方法：
# 1. 创建 .env 文件：DEEPSEEK_API_KEY=your-api-key-here
# 2. 或设置环境变量：export DEEPSEEK_API_KEY=your-api-key-here

class SuggestionGenerator:
    """建议生成器 - 优先使用DeepSeek API生成建议，失败时降级到本地生成器"""
    
    def __init__(self, api_key: Optional[str] = None, proxies: Optional[Dict[str, str]] = None):
        self.logger = logging.getLogger("pipeline.SuggestionGenerator")
        self.api_base = "https://api.deepseek.com"
        self.proxies = proxies or PROXY_CONFIG
        self._configure_api(api_key)
    
    def _configure_api(self, api_key: Optional[str] = None) -> None:
        """配置API设置"""
        # 按优先级尝试获取API密钥：传入参数 > 环境变量
        self.api_key = api_key or os.getenv("DEEPSEEK_API_KEY")
        
        self.logger.debug("开始配置API...")
        if api_key:
            self.logger.debug("使用传入的API密钥")
        else:
            self.logger.debug("尝试从环境变量获取API密钥")
        
        if self.api_key:
            self.logger.debug(f"获取到API密钥: {self.api_key[:5]}...")
            if self._validate_api_key(self.api_key):
                self.logger.info("API密钥配置成功")
            else:
                self.logger.error("API密钥验证失败，将使用本地生成器")
                self.api_key = None
        else:
            self.logger.warning("未找到DeepSeek API密钥，请通过以下方式之一配置：")
            self.logger.warning("1. 在初始化时传入: SuggestionGenerator(api_key='your-key')")
            self.logger.warning("2. 设置环境变量: export DEEPSEEK_API_KEY='your-key'")
    
    def _validate_api_key(self, api_key: str) -> bool:
        """验证API密钥格式"""
        try:
            # DeepSeek API密钥格式验证
            if not isinstance(api_key, str):
                self.logger.debug("API密钥不是字符串类型")
                return False
                
            if not api_key.startswith("sk-"):
                self.logger.debug("API密钥必须以'sk-'开头")
                return False
                
            # DeepSeek的API密钥长度为35（包含"sk-"前缀）
            if len(api_key) != 35:
                self.logger.debug(f"API密钥长度不正确: 当前长度{len(api_key)}，应为35")
                return False
                
            # 验证密钥格式：应该只包含字母和数字
            if not all(c.isalnum() for c in api_key[3:]):
                self.logger.debug("API密钥包含非法字符（应只包含字母和数字）")
                return False
                
            self.logger.debug("API密钥验证通过")
            return True
            
        except Exception as e:
            self.logger.error(f"API密钥验证过程出错: {str(e)}")
            return False
    
    def generate_suggestions(self, results: Dict[str, Any]) -> List[str]:
        """生成建议，优先使用DeepSeek API"""
        try:
            if self.api_key:
                self.logger.info("尝试使用DeepSeek API生成建议...")
                suggestions = self._generate_with_deepseek(results)
                if suggestions:
                    self.logger.info("成功使用DeepSeek API生成建议")
                    return suggestions
                else:
                    self.logger.warning("DeepSeek API返回为空，降级到本地生成器")
            else:
                self.logger.info("未配置有效的DeepSeek API密钥，使用本地生成器")
            
            return self._generate_locally(results)
            
        except Exception as e:
            self.logger.error(f"建议生成失败: {str(e)}", exc_info=True)
            return self._generate_locally(results)
    
    def _generate_with_deepseek(self, results: Dict[str, Any]) -> List[str]:
        """使用DeepSeek API生成建议"""
        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            # 构建API请求
            prompt = self._build_deepseek_prompt(results)
            payload = {
                "model": "deepseek-chat",
                "messages": [
                    {"role": "system", "content": "你是一个专业的室内设计顾问，请基于分析结果提供具体的改进建议。"},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.7,
                "max_tokens": 1000,
                "stream": False
            }
            
            api_url = f"{self.api_base}/chat/completions"
            self.logger.info("准备发送DeepSeek API请求")
            self.logger.debug(f"API地址: {api_url}")
            self.logger.debug(f"请求头: {headers}")
            self.logger.debug(f"请求体: {json.dumps(payload, ensure_ascii=False)}")
            self.logger.debug(f"代理配置: {self.proxies}")
            
            try:
                response = requests.post(
                    api_url,
                    headers=headers,
                    json=payload,
                    proxies=self.proxies,
                    timeout=(TIMEOUT_CONFIG['connect'], TIMEOUT_CONFIG['read']),
                    verify=True
                )
                
                self.logger.info(f"API响应状态码: {response.status_code}")
                self.logger.debug(f"完整响应内容: {response.text}")
                
                if response.status_code == 200:
                    try:
                        content = response.json()
                        self.logger.debug(f"解析后的响应内容: {json.dumps(content, ensure_ascii=False, indent=2)}")
                        
                        if "choices" in content and content["choices"]:
                            message_content = content["choices"][0]["message"]["content"]
                            self.logger.info("成功获取API响应内容")
                            self.logger.debug(f"原始建议内容: {message_content}")
                            
                            # 处理建议内容
                            suggestions = [s.strip() for s in message_content.split("\n") if s.strip()]
                            self.logger.info(f"成功处理得到 {len(suggestions)} 条建议")
                            self.logger.debug(f"处理后的建议: {suggestions}")
                            
                            if suggestions:
                                return suggestions
                            else:
                                self.logger.warning("API返回内容处理后为空")
                                return []
                        else:
                            self.logger.error("API响应缺少必要的choices字段")
                            self.logger.debug(f"异常响应结构: {content}")
                            return []
                    except json.JSONDecodeError as je:
                        self.logger.error(f"API响应JSON解析失败: {str(je)}")
                        self.logger.debug(f"无法解析的响应内容: {response.text}")
                        return []
                elif response.status_code == 401:
                    self.logger.error("API认证失败，请检查API密钥")
                    return []
                elif response.status_code == 429:
                    self.logger.error("API请求超出限制")
                    return []
                else:
                    self.logger.error(f"API请求失败: {response.status_code}")
                    self.logger.debug(f"错误响应内容: {response.text}")
                    return []
                    
            except requests.exceptions.Timeout as e:
                self.logger.error(f"请求超时: {str(e)}")
                return []
            except requests.exceptions.ProxyError as e:
                self.logger.error(f"代理连接失败: {str(e)}")
                self.logger.debug("尝试不使用代理直接连接...")
                try:
                    response = requests.post(
                        api_url,
                        headers=headers,
                        json=payload,
                        timeout=(TIMEOUT_CONFIG['connect'], TIMEOUT_CONFIG['read']),
                        verify=True
                    )
                    # 处理直连响应...
                    if response.status_code == 200:
                        content = response.json()
                        if "choices" in content and content["choices"]:
                            suggestions = [s.strip() for s in content["choices"][0]["message"]["content"].split("\n") if s.strip()]
                            self.logger.info("直连成功获取响应")
                            return suggestions
                except Exception as direct_error:
                    self.logger.error(f"直接连接失败: {str(direct_error)}")
                    return []
            except Exception as e:
                self.logger.error(f"请求过程发生异常: {str(e)}")
                return []
                
        except Exception as e:
            self.logger.error(f"DeepSeek API调用过程发生异常: {str(e)}", exc_info=True)
            return []
    
    def _build_deepseek_prompt(self, results: Dict[str, Any]) -> str:
        """构建DeepSeek API的提示词"""
        prompt = "请基于以下空间分析结果，提供专业的改进建议：\n\n"
        
        for category, data in results.items():
            prompt += f"{category}分析结果：\n{json.dumps(data, ensure_ascii=False, indent=2)}\n\n"
        
        prompt += "请提供具体的、可操作的改进建议，包括但不限于：空间布局、照明设计、色彩搭配、家具选择等方面。"
        return prompt
    
    def _generate_locally(self, results: Dict[str, Any]) -> List[str]:
        """本地生成建议（原有逻辑）"""
        try:
            self.logger.info("开始使用本地生成器...")
            suggestions = []
            
            if 'scene' in results:
                scene_suggestions = self._generate_scene_suggestions(results['scene'])
                if scene_suggestions:
                    suggestions.extend(scene_suggestions)
            
            if 'furniture' in results:
                self.logger.debug("基于家具分析生成建议")
                furniture_suggestions = self._generate_furniture_suggestions(results['furniture'])
                if furniture_suggestions:
                    suggestions.extend(furniture_suggestions)
            
            if 'lighting' in results:
                self.logger.debug(f"光照分析结果: {results['lighting']}")
                lighting_suggestions = self._generate_lighting_suggestions(results['lighting'])
                if lighting_suggestions:
                    self.logger.debug(f"光照建议: {lighting_suggestions}")
                    suggestions.extend(lighting_suggestions)
            
            if 'style' in results:
                self.logger.debug("基于风格分析生成建议")
                style_suggestions = self._generate_style_suggestions(results['style'])
                if style_suggestions:
                    suggestions.extend(style_suggestions)
            
            if 'color' in results:
                self.logger.debug("基于颜色分析生成建议")
                color_suggestions = self._generate_color_suggestions(results['color'])
                if color_suggestions:
                    suggestions.extend(color_suggestions)

            # 如果没有生成任何建议，提供默认建议
            if not suggestions:
                suggestions.append("未能生成具体建议，请检查分析结果的完整性。")
            
            self.logger.info(f"生成了 {len(suggestions)} 条建议")
            self.logger.debug(f"所有建议: {suggestions}")
            return suggestions
            
        except Exception as e:
            self.logger.error(f"本地建议生成失败: {str(e)}", exc_info=True)
            return []

    def _generate_scene_suggestions(self, result: Dict) -> List[str]:
        """生成场景相关建议"""
        suggestions = []
        features = result.get('features', {})
        spatial = features.get('spatial_features', {})
        
        # 检查空间利用
        wall_visibility = spatial.get('wall_visibility', 0)
        if wall_visibility > 0.3:
            suggestions.extend([
                "墙面空间利用率较低，建议：",
                "  - 增加墙面装饰画或照片墙",
                "  - 安装壁挂式储物架或书架",
                "  - 考虑使用墙面收纳系统",
                "  - 添加艺术装饰品提升空间趣味性"
            ])
        
        # 检查自然光照
        natural_light = spatial.get('natural_light', 0)
        if natural_light < 0.4:
            suggestions.extend([
                "自然采光不足，建议：",
                "  - 选用浅色窗帘，增加光线透过率",
                "  - 合理摆放镜面装饰，反射自然光",
                "  - 避免在窗户附近放置大型家具",
                "  - 考虑增加天窗或扩大窗户面积"
            ])
        
        # 检查空间对称性
        symmetry = spatial.get('symmetry', 0)
        if symmetry < 0.6:
            suggestions.extend([
                "空间对称性不足，建议：",
                "  - 主要家具采用对称布置",
                "  - 装饰品成对使用增加平衡感",
                "  - 灯具布置注意左右均衡",
                "  - 色彩搭配保持视觉平衡"
            ])
        
        return suggestions
        
    def _generate_lighting_suggestions(self, result: Dict) -> List[str]:
        """生成光照相关建议"""
        suggestions = []
        features = result.get('features', {})
        metrics = features.get('basic_metrics', {})
        quality = features.get('quality', {})
        
        # 检查亮度
        brightness = metrics.get('brightness', 0)
        if brightness < 0.5:
            suggestions.extend([
                "整体亮度偏低，建议：",
                f"  - 当前亮度{brightness*100:.0f}%，建议提升到70%以上",
                "  - 增加主照明光源的功率",
                "  - 添加辅助照明，如落地灯、台灯",
                "  - 选用反光材质的装饰品增加光线反射"
            ])
        
        # 检查均匀度
        uniformity = metrics.get('uniformity', 0)
        if uniformity < 0.7:
            suggestions.extend([
                "光照分布不均匀，建议：",
                f"  - 当前均匀度{uniformity*100:.0f}%，建议提升到80%以上",
                "  - 调整灯具位置，避免光照死角",
                "  - 使用多个光源代替单一强光源",
                "  - 考虑安装射灯补充角落光照"
            ])
        
        # 检查色温
        color_temp = quality.get('color_temperature', 0)
        if color_temp < 2700 or color_temp > 6500:
            suggestions.extend([
                "光源色温不适宜，建议：",
                f"  - 当前色温{color_temp:.0f}K，建议调整到2700K-6500K范围",
                "  - 客厅建议使用3000K-4000K暖白光",
                "  - 书房建议使用5000K-6500K冷白光",
                "  - 卧室建议使用2700K-3000K暖黄光"
            ])
        
        return suggestions
        
    def _generate_style_suggestions(self, result: Dict) -> List[str]:
        """生成风格相关建议"""
        suggestions = []
        features = result.get('features', {})
        
        # 检查风格一致性
        consistency = features.get('consistency', 0)
        if consistency < 0.4:
            suggestions.extend([
                "空间风格一致性较低，建议：",
                "  - 确定一个主导风格，其他风格作为点缀",
                "  - 统一家具的设计风格",
                "  - 协调墙面、地面、天花板的装饰风格",
                "  - 选择风格一致的软装饰品"
            ])
        
        # 检查纹理复杂度
        texture = features.get('texture_complexity', 0)
        if texture < 0.3:
            suggestions.extend([
                "空间纹理层次感不足，建议：",
                "  - 增加不同材质的搭配，如木材、金属、布艺",
                "  - 添加具有肌理的墙面装饰",
                "  - 使用带有纹理的地毯或抱枕",
                "  - 选择富有质感的窗帘布料"
            ])
        
        return suggestions
        
    def _generate_color_suggestions(self, result: Dict) -> List[str]:
        """生成颜色相关建议"""
        suggestions = []
        features = result.get('features', {})
        
        # 检查色彩和谐度
        harmony = features.get('harmony_score', 0)
        if harmony < 0.7:
            suggestions.extend([
                "色彩搭配和谐度较低，建议：",
                "  - 选择同一色系的不同深浅度",
                "  - 运用60-30-10法则搭配主次色彩",
                "  - 避免过多鲜艳色彩同时使用",
                "  - 增加中性色调来平衡空间"
            ])
        
        # 检查饱和度
        saturation = features.get('avg_saturation', 0)
        if saturation > 150:
            suggestions.extend([
                "整体色彩饱和度过高，建议：",
                "  - 增加灰度色调中和饱和度",
                "  - 主要面积使用低饱和度的颜色",
                "  - 高饱和度色彩仅用于点缀",
                "  - 加入白色或米色等基础色"
            ])
        elif saturation < 50:
            suggestions.extend([
                "整体色彩饱和度偏低，建议：",
                "  - 通过装饰品增加色彩亮点",
                "  - 在沙发、窗帘等软装中加入彩色元素",
                "  - 添加绿植增添自然色彩",
                "  - 适当使用艺术画作增加色彩活力"
            ])
        
        return suggestions

    def _generate_furniture_suggestions(self, result: Dict) -> List[str]:
        """生成家具相关建议"""
        suggestions = []
        features = result.get('features', {})
        
        # 检查未识别物体
        furniture_types = features.get('furniture_types', {})
        if furniture_types.get('未知', {}).get('count', 0) > 0:
            suggestions.append(f"有{furniture_types['未知']['count']}件家具未能准确识别，建议调整拍摄角度")
        
        # 检查布局密度
        layout = features.get('layout', {})
        if layout.get('density', 0) < 0.2:
            suggestions.append("家具布局较为稀疏，建议适当增加家具或调整布局")
        
        return suggestions 
        return suggestions 
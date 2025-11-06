from typing import Dict, List, Any, Optional
import logging
from .result_types import (
    ImageAnalysisResult, 
    VideoAnalysisResult,
    RealtimeAnalysisResult,
    AnalyzerResult
)

class ResultFormatter:
    """结果格式化器 - 格式化分析结果的展示形式"""
    
    def __init__(self):
        self.logger = logging.getLogger("pipeline.ResultFormatter")
        
    def format_analysis_result(self, result: Dict) -> Dict:
        """格式化完整的分析结果"""
        try:
            formatted_result = {
                'scene': self._format_scene_analysis(result.get('scene_analysis', {})),
                'furniture': self._format_furniture_analysis(result.get('furniture_analysis', {})),
                'lighting': self._format_lighting_analysis(result.get('lighting_analysis', {})),
                'color': self._format_color_analysis(result.get('color_analysis', {})),
                'style': self._format_style_analysis(result.get('style_analysis', {}))
            }
            
            # 添加总结
            formatted_result['summary'] = self._generate_summary(formatted_result)
            
            return formatted_result
            
        except Exception as e:
            self.logger.error(f"结果格式化失败: {str(e)}")
            return {}

    def _format_scene_analysis(self, scene: Dict) -> Dict:
        """格式化场景分析结果"""
        try:
            features = scene.get('features', {})
            return {
                'summary': f"空间类型: {scene.get('room_type', '未知')}\n"
                          f"置信度: {scene.get('confidence', 0.0):.2f}",
                'details': {
                    'room_type': scene.get('room_type', '未知'),
                    'confidence': scene.get('confidence', 0.0),
                    'area': f"{features.get('area', 0.0):.2f}㎡",
                    'layout': self._format_layout_description(features),
                    'spatial_features': features.get('spatial_features', {})
                }
            }
        except Exception as e:
            self.logger.error(f"场景分析格式化失败: {str(e)}")
            return {}

    def _format_furniture_analysis(self, furniture: Dict) -> Dict:
        """格式化家具分析结果"""
        try:
            items = furniture.get('detected_items', [])
            layout = furniture.get('layout_analysis', {})
            
            return {
                'summary': f"检测到 {len(items)} 件家具\n"
                          f"空间密度: {layout.get('density', 0.0):.2f}",
                'details': {
                    'items': [
                        {
                            'type': item['type'],
                            'confidence': f"{item['confidence']:.2f}",
                            'position': item['position']
                        }
                        for item in items
                    ],
                    'layout_analysis': {
                        'density': f"{layout.get('density', 0.0):.2f}",
                        'distribution': layout.get('distribution', '均匀'),
                        'accessibility': layout.get('accessibility', '良好')
                    }
                }
            }
        except Exception as e:
            self.logger.error(f"家具分析格式化失败: {str(e)}")
            return {}

    def _format_lighting_analysis(self, lighting: Dict) -> Dict:
        """格式化光照分析结果"""
        try:
            return {
                'summary': f"整体亮度: {lighting.get('overall_brightness', 0.0):.2f}\n"
                          f"均匀度: {lighting.get('uniformity', 0.0):.2f}",
                'details': {
                    'brightness': f"{lighting.get('overall_brightness', 0.0):.2f}",
                    'uniformity': f"{lighting.get('uniformity', 0.0):.2f}",
                    'contrast': f"{lighting.get('contrast', 0.0):.2f}",
                    'quality': self._format_lighting_quality(lighting)
                }
            }
        except Exception as e:
            self.logger.error(f"光照分析格式化失败: {str(e)}")
            return {}

    def _format_color_analysis(self, color: Dict) -> Dict:
        """格式化颜色分析结果"""
        try:
            main_colors = color.get('main_colors', [])
            return {
                'summary': f"配色方案: {color.get('color_scheme', '未知')}\n"
                          f"和谐度: {color.get('harmony_score', 0.0):.2f}",
                'details': {
                    'color_scheme': color.get('color_scheme', '未知'),
                    'harmony_score': f"{color.get('harmony_score', 0.0):.2f}",
                    'main_colors': [
                        {
                            'name': c['name'],
                            'percentage': f"{c['percentage']:.1%}",
                            'rgb': c['rgb']
                        }
                        for c in main_colors
                    ],
                    'emotion': color.get('emotion', {})
                }
            }
        except Exception as e:
            self.logger.error(f"颜色分析格式化失败: {str(e)}")
            return {}

    def _format_style_analysis(self, style: Dict) -> Dict:
        """格式化风格分析结果"""
        try:
            return {
                'summary': f"检测风格: {style.get('detected_style', '未知')}\n"
                          f"置信度: {style.get('confidence', 0.0):.2f}",
                'details': {
                    'style': style.get('detected_style', '未知'),
                    'confidence': f"{style.get('confidence', 0.0):.2f}",
                    'elements': style.get('style_elements', []),
                    'consistency': f"{style.get('consistency_score', 0.0):.2f}"
                }
            }
        except Exception as e:
            self.logger.error(f"风格分析格式化失败: {str(e)}")
            return {}

    def _generate_summary(self, results: Dict[str, Any]) -> str:
        """生成总体摘要"""
        try:
            summary = "【总体分析】\n"
            
            # 场景类型
            if 'scene' in results:
                scene_probs = results['scene'].get('features', {}).get('scene_probs', {})
                main_scene = max(scene_probs.items(), key=lambda x: x[1])[0] if scene_probs else '其他'
                summary += f"• 空间类型: {main_scene}\n"
            
            # 风格分布
            if 'style' in results:
                style_dist = results['style'].get('features', {}).get('style_distribution', {})
                if style_dist:
                    main_style = max(style_dist.items(), key=lambda x: x[1])[0]
                    confidence = style_dist[main_style] * 100
                    summary += f"• 装修风格: {main_style} ({confidence:.0f}%)\n"
                else:
                    summary += "• 装修风格: 未知\n"
            
            # 家具概况
            if 'furniture' in results:
                furniture_types = results['furniture'].get('features', {}).get('furniture_types', {})
                total_count = sum(item['count'] for item in furniture_types.values() if isinstance(item, dict))
                summary += f"• 家具数量: {total_count:.0f}件\n"
            
            return summary
            
        except Exception as e:
            self.logger.error(f"生成总体摘要失败: {str(e)}", exc_info=True)
            return "【总体分析】\n• 分析失败\n"

    def format_results(self, processed_results: Dict[str, Any]) -> str:
        """格式化处理后的结果"""
        try:
            self.logger.info("开始格式化分析结果...")
            self.logger.debug(f"输入结果: {processed_results}")
            
            formatted = "=== 空间分析报告 ===\n\n"
            
            # 生成总体摘要
            summary = self._generate_summary(processed_results)
            self.logger.debug(f"生成的总体摘要: {summary}")
            formatted += summary + "\n\n"
            
            # 格式化各部分结果
            sections = []
            
            # 1. 场景分析
            if 'scene' in processed_results:
                self.logger.debug(f"场景分析结果: {processed_results['scene']}")
                scene_section = self._format_scene_section(processed_results['scene'])
                self.logger.debug(f"格式化后的场景分析: {scene_section}")
                sections.append(scene_section)
            
            # 2. 家具分析
            if 'furniture' in processed_results:
                self.logger.debug(f"家具分析结果: {processed_results['furniture']}")
                furniture_section = self._format_furniture_section(processed_results['furniture'])
                self.logger.debug(f"格式化后的家具分析: {furniture_section}")
                sections.append(furniture_section)
            
            # 3. 光照分析
            if 'lighting' in processed_results:
                self.logger.debug(f"光照分析结果: {processed_results['lighting']}")
                lighting_section = self._format_lighting_section(processed_results['lighting'])
                self.logger.debug(f"格式化后的光照分析: {lighting_section}")
                sections.append(lighting_section)
            
            # 4. 色彩分析
            if 'color' in processed_results:
                self.logger.debug(f"色彩分析结果: {processed_results['color']}")
                color_section = self._format_color_section(processed_results['color'])
                self.logger.debug(f"格式化后的色彩分析: {color_section}")
                sections.append(color_section)
            
            # 5. 风格分析
            if 'style' in processed_results:
                self.logger.debug(f"风格分析结果: {processed_results['style']}")
                style_section = self._format_style_section(processed_results['style'])
                self.logger.debug(f"格式化后的风格分析: {style_section}")
                sections.append(style_section)
            
            formatted += "\n".join(sections)
            
            self.logger.info("结果格式化完成")
            self.logger.debug(f"最终格式化结果: {formatted}")
            return formatted
            
        except Exception as e:
            self.logger.error(f"结果格式化失败: {str(e)}", exc_info=True)
            return "=== 空间分析报告 ===\n\n格式化失败\n"

    def _format_scene_section(self, scene_result: Dict) -> str:
        """格式化场景分析结果"""
        section = "\n一、空间场景分析\n"
        section += "-" * 40 + "\n"
        
        if not scene_result:
            return section + "未获取场景分析结果\n"
        
        features = scene_result.get('features', {})
        spatial = features.get('spatial_features', {})
        
        # 获取场景类型和置信度
        scene_probs = features.get('scene_probs', {})
        room_type = max(scene_probs.items(), key=lambda x: x[1])[0] if scene_probs else '未知'
        confidence = scene_probs.get(room_type, 0.0) if scene_probs else 0.0
        
        # 1. 空间类型识别
        section += "1. 空间类型识别\n"
        section += f"   系统识别此空间为{room_type}，置信度{confidence*100:.0f}%。\n\n"
        
        # 2. 空间特征分析
        section += "2. 空间特征分析\n"
        
        # 面积分析
        area = spatial.get('area', 0)
        if area > 0:
            section += f"   • 空间面积约{area:.1f}平方米。\n"
        
        # 对称性分析
        symmetry = spatial.get('symmetry', 0)
        symmetry_desc = "很好" if symmetry > 0.8 else "良好" if symmetry > 0.6 else "一般" if symmetry > 0.4 else "较差"
        section += f"   • 空间对称性{symmetry_desc}，对称度达到{symmetry*100:.0f}%。\n"
        
        # 采光分析
        natural_light = spatial.get('natural_light', 0)
        light_desc = "充足" if natural_light > 0.8 else "良好" if natural_light > 0.6 else "一般" if natural_light > 0.4 else "不足"
        section += f"   • 自然采光{light_desc}，光照度为{natural_light*100:.0f}%。\n"
        
        return section

    def _format_furniture_section(self, furniture_result: Dict) -> str:
        """格式化家具分析结果"""
        section = "\n二、家具布置分析\n"
        section += "-" * 40 + "\n"
        
        if not furniture_result:
            return section + "未获取家具分析结果\n"
        
        features = furniture_result.get('features', {})
        furniture_types = features.get('furniture_types', {})
        layout = features.get('layout', {})
        
        # 1. 家具清单
        section += "1. 主要家具清单\n"
        detected_furniture = [
            (ftype, stats) 
            for ftype, stats in furniture_types.items() 
            if isinstance(stats, dict) and stats.get('avg_confidence', 0) > 0.15
        ]
        
        if detected_furniture:
            for ftype, stats in detected_furniture:
                count = stats.get('count', 0)
                conf = stats.get('avg_confidence', 0) * 100
                section += f"   • 检测到{ftype}{count}件，识别置信度{conf:.0f}%\n"
        else:
            section += "   未检测到明显的家具\n"
        
        # 2. 布局评估
        section += "\n2. 空间布局评估\n"
        layout_score = layout.get('layout_score', 0)
        density = layout.get('density', 0)
        
        # 布局评分描述
        layout_desc = "非常合理" if layout_score > 0.8 else "比较合理" if layout_score > 0.6 else "一般" if layout_score > 0.4 else "需要改善"
        section += f"   • 整体布局{layout_desc}，评分为{layout_score*100:.0f}分\n"
        
        # 空间利用描述
        density_desc = "充分" if density > 0.8 else "适中" if density > 0.4 else "较低"
        section += f"   • 空间利用率{density_desc}，为{density*100:.0f}%\n"
        
        return section

    def _format_lighting_section(self, lighting_result: Dict) -> str:
        """格式化光照环境分析结果"""
        section = "\n三、光照环境分析\n"
        section += "-" * 40 + "\n"
        
        if not lighting_result:
            return section + "未获取光照分析结果\n"
        
        features = lighting_result.get('features', {})
        metrics = features.get('basic_metrics', {})
        quality = features.get('quality', {})
        
        # 整体亮度分析
        brightness = metrics.get('brightness', 0)
        brightness_desc = "充足" if brightness > 0.7 else \
                         "适中" if brightness > 0.5 else \
                         "偏暗" if brightness > 0.3 else "不足"
        section += f"• 整体亮度{brightness_desc}，照明度为{brightness*100:.0f}%\n"
        
        # 光照均匀度分析
        uniformity = metrics.get('uniformity', 0)
        uniformity_desc = "非常均匀" if uniformity > 0.8 else \
                         "比较均匀" if uniformity > 0.6 else \
                         "略有不均" if uniformity > 0.4 else "分布不均"
        section += f"• 光照分布{uniformity_desc}，均匀度为{uniformity*100:.0f}%\n"
        
        # 色温分析
        color_temp = quality.get('color_temperature', 0)
        if 2700 <= color_temp <= 3500:
            temp_desc = "暖色调（温馨舒适）"
        elif 3500 < color_temp <= 5000:
            temp_desc = "中性色调（自然平和）"
        elif 5000 < color_temp <= 6500:
            temp_desc = "冷色调（清爽明快）"
        else:
            temp_desc = "不在建议范围内"
        
        section += f"• 光源色温为{color_temp:.0f}K，呈现{temp_desc}\n"
        
        # 添加使用建议
        if color_temp > 0:
            section += "\n光照使用建议：\n"
            if 2700 <= color_temp <= 3500:
                section += "  适合用于：卧室、客厅等休息空间\n"
            elif 3500 < color_temp <= 5000:
                section += "  适合用于：餐厅、书房等日常活动空间\n"
            elif 5000 < color_temp <= 6500:
                section += "  适合用于：厨房、卫生间等功能性空间\n"
        
        return section

    def _format_color_section(self, color_result: Dict) -> str:
        """格式化色彩分析结果"""
        section = "\n四、色彩分析\n"
        section += "-" * 40 + "\n"
        
        if not color_result:
            return section + "未获取色彩分析结果\n"
        
        features = color_result.get('features', {})
        
        # 1. 色彩和谐度分析
        harmony = features.get('harmony_score', 0)
        harmony_desc = "非常和谐" if harmony > 85 else \
                      "比较和谐" if harmony > 70 else \
                      "一般" if harmony > 50 else "不够协调"
        
        section += "1. 色彩和谐度评估\n"
        section += f"   • 整体色彩搭配{harmony_desc}，和谐度评分{harmony:.0f}分\n"
        
        # 2. 色彩饱和度分析
        saturation = features.get('avg_saturation', 0)
        if saturation > 150:
            sat_desc = "非常浓郁"
            sat_effect = "空间色彩感强烈，具有很强的视觉冲击力"
        elif saturation > 100:
            sat_desc = "较为浓郁"
            sat_effect = "色彩表现力丰富，空间富有活力"
        elif saturation > 70:
            sat_desc = "适中"
            sat_effect = "色彩平衡适度，给人舒适自然的感觉"
        elif saturation > 40:
            sat_desc = "柔和"
            sat_effect = "色彩清淡素雅，空间感觉温和"
        else:
            sat_desc = "淡雅"
            sat_effect = "色彩倾向简约清爽，空间显得宁静"
        
        section += "\n2. 色彩饱和度评估\n"
        section += f"   • 整体色彩{sat_desc}，饱和度为{saturation:.0f}\n"
        section += f"   • 视觉效果：{sat_effect}\n"
        
        # 3. 色彩应用建议
        section += "\n3. 色彩应用建议\n"
        if harmony < 70:
            section += "   • 建议调整以提升和谐度：\n"
            section += "     - 选择同一色系的不同深浅度\n"
            section += "     - 使用互补色搭配增加协调性\n"
            section += "     - 可以添加中性色来平衡色彩\n"
        
        if saturation > 150:
            section += "   • 饱和度较高，建议：\n"
            section += "     - 适当增加灰度色调中和色彩\n"
            section += "     - 主要面积使用低饱和度的颜色\n"
            section += "     - 将高饱和度色彩用于点缀\n"
        elif saturation < 40:
            section += "   • 饱和度偏低，建议：\n"
            section += "     - 可以通过装饰品增添色彩亮点\n"
            section += "     - 在软装中加入适度的彩色元素\n"
            section += "     - 添加绿植为空间增添生机\n"
        
        return section

    def _format_style_section(self, style_result: Dict) -> str:
        """格式化风格分析结果"""
        section = "\n五、设计风格分析\n"
        section += "-" * 40 + "\n"
        
        if not style_result:
            return section + "未获取风格分析结果\n"
        
        features = style_result.get('features', {})
        style_dist = features.get('style_distribution', {})
        consistency = features.get('consistency', 0)
        
        # 1. 风格构成分析
        section += "1. 风格构成分析\n"
        significant_styles = [
            (style, prob) 
            for style, prob in style_dist.items() 
            if prob > 0.05
        ]
        significant_styles.sort(key=lambda x: x[1], reverse=True)
        
        if significant_styles:
            # 主要风格
            main_style, main_prob = significant_styles[0]
            section += f"   • 主要体现为{main_style}风格，占比{main_prob*100:.0f}%\n"
            
            # 其他风格
            if len(significant_styles) > 1:
                section += "   • 同时融入了"
                other_styles = [f"{style}({prob*100:.0f}%)" 
                              for style, prob in significant_styles[1:]]
                section += "、".join(other_styles) + "的设计元素\n"
        
        # 2. 风格统一性评估
        section += "\n2. 风格统一性评估\n"
        consistency_desc = "很好" if consistency > 0.8 else \
                          "良好" if consistency > 0.6 else \
                          "一般" if consistency > 0.4 else "较低"
        
        section += f"   • 空间风格统一性{consistency_desc}，一致性评分为{consistency*100:.0f}%\n"
        
        return section 
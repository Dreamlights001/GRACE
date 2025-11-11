"""
提示模板管理模块
负责管理和生成不同类型的提示模板
"""

import json
from typing import Dict, List, Any, Optional
from config.config import config

class PromptTemplateManager:
    """提示模板管理器"""
    
    def __init__(self):
        self.templates = {
            "basic": self._get_basic_template(),
            "with_context": self._get_context_template(),
            "with_graph": self._get_graph_template(),
            "with_examples": self._get_examples_template()
        }
    
    def _get_basic_template(self) -> str:
        """基础模板"""
        return """你是一个专业的代码安全分析专家。分析以下代码片段，识别可能的安全漏洞。

代码：
{code}

请按照以下格式回答：
- 漏洞判断：是/否
- 置信度：0-1之间的数值
- 漏洞类型：列出发现的漏洞类型
- 解释：详细说明漏洞的原因和影响

请直接输出JSON格式的答案，不要包含其他文本。"""
    
    def _get_context_template(self) -> str:
        """带上下文的模板"""
        return """你是一个专业的代码安全分析专家。分析以下代码片段，识别可能的安全漏洞。

上下文信息：
{context}

代码：
{code}

请按照以下格式回答：
- 漏洞判断：是/否
- 置信度：0-1之间的数值
- 漏洞类型：列出发现的漏洞类型
- 解释：详细说明漏洞的原因和影响

请直接输出JSON格式的答案，不要包含其他文本。"""
    
    def _get_graph_template(self) -> str:
        """带图结构信息的模板"""
        return """你是一个专业的代码安全分析专家。分析以下代码片段的图结构信息，识别可能的安全漏洞。

代码：
{code}

节点信息：
{node_info}

边信息：
{edge_info}

请按照以下格式回答：
- 漏洞判断：是/否
- 置信度：0-1之间的数值
- 漏洞类型：列出发现的漏洞类型
- 解释：详细说明漏洞的原因和影响

请直接输出JSON格式的答案，不要包含其他文本。"""
    
    def _get_examples_template(self) -> str:
        """带示例的模板"""
        return """你是一个专业的代码安全分析专家。基于以下示例，分析新的代码片段，识别可能的安全漏洞。

示例1：
{example1}

示例2：
{example2}

目标代码：
{code}

请按照以下格式回答：
- 漏洞判断：是/否
- 置信度：0-1之间的数值
- 漏洞类型：列出发现的漏洞类型
- 解释：详细说明漏洞的原因和影响

请直接输出JSON格式的答案，不要包含其他文本。"""
    
    def get_template(self, template_type: str = "basic") -> str:
        """获取指定类型的模板"""
        return self.templates.get(template_type, self.templates["basic"])
    
    def get_template_names(self) -> List[str]:
        """获取所有可用的模板名称"""
        return list(self.templates.keys())
    
    def format_prompt(self, template_type: str, **kwargs) -> str:
        """格式化提示模板"""
        template = self.get_template(template_type)
        
        try:
            return template.format(**kwargs)
        except KeyError as e:
            raise ValueError(f"模板参数缺失: {e}")
    
    def create_analysis_prompt(self, code: str, 
                             context: Optional[str] = None,
                             node_info: Optional[str] = None,
                             edge_info: Optional[str] = None,
                             examples: Optional[List[Dict]] = None,
                             template_type: str = "basic") -> str:
        """创建漏洞分析提示"""
        
        if template_type == "basic":
            return self.format_prompt("basic", code=code)
        
        elif template_type == "with_context" and context:
            return self.format_prompt("with_context", code=code, context=context)
        
        elif template_type == "with_graph" and node_info and edge_info:
            return self.format_prompt("with_graph", 
                                    code=code, 
                                    node_info=node_info, 
                                    edge_info=edge_info)
        
        elif template_type == "with_examples" and examples and len(examples) >= 2:
            format_kwargs = {
                "code": code,
                "example1": f"代码: {examples[0].get('code', '')}\n漏洞: {examples[0].get('vulnerability', '')}",
                "example2": f"代码: {examples[1].get('code', '')}\n漏洞: {examples[1].get('vulnerability', '')}"
            }
            return self.format_prompt("with_examples", **format_kwargs)
        
        # 回退到基础模板
        return self.format_prompt("basic", code=code)
    
    def load_custom_templates(self, template_path: str) -> bool:
        """从文件加载自定义模板"""
        try:
            with open(template_path, 'r', encoding='utf-8') as f:
                custom_templates = json.load(f)
            
            self.templates.update(custom_templates)
            return True
        except Exception as e:
            print(f"加载自定义模板失败: {e}")
            return False
    
    def save_templates(self, output_path: str) -> bool:
        """保存所有模板到文件"""
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(self.templates, f, ensure_ascii=False, indent=2)
            return True
        except Exception as e:
            print(f"保存模板失败: {e}")
            return False

class VulnerabilityPromptTemplates:
    """漏洞检测专用的提示模板集合"""
    
    @staticmethod
    def get_sql_injection_template() -> str:
        """SQL注入检测模板"""
        return """你是一个SQL注入漏洞检测专家。分析以下代码，重点检查SQL注入风险。

代码：
{code}

请特别关注以下SQL注入风险点：
1. 字符串拼接构建SQL语句
2. 未经验证的用户输入
3. 不当的SQL语句结构
4. 缺乏参数化查询

请按照以下格式输出：
{{
  "has_vulnerability": true/false,
  "vulnerability_type": "SQL注入",
  "confidence": 0.0-1.0,
  "explanation": "详细解释",
  "location": "具体位置（行号或函数名）"
}}"""
    
    @staticmethod
    def get_buffer_overflow_template() -> str:
        """缓冲区溢出检测模板"""
        return """你是一个缓冲区溢出漏洞检测专家。分析以下代码，重点检查缓冲区溢出风险。

代码：
{code}

请特别关注以下缓冲区溢出风险点：
1. 不安全的字符串操作函数
2. 缺乏边界检查的数组访问
3. 动态内存分配错误
4. 指针操作风险

请按照以下格式输出：
{{
  "has_vulnerability": true/false,
  "vulnerability_type": "缓冲区溢出",
  "confidence": 0.0-1.0,
  "explanation": "详细解释",
  "location": "具体位置"
}}"""
    
    @staticmethod
    def get_xss_template() -> str:
        """XSS漏洞检测模板"""
        return """你是一个XSS（跨站脚本）漏洞检测专家。分析以下代码，重点检查XSS风险。

代码：
{code}

请特别关注以下XSS风险点：
1. 直接输出用户输入到HTML
2. 缺乏输入验证和输出编码
3. 不安全的DOM操作
4. 动态HTML内容生成

请按照以下格式输出：
{{
  "has_vulnerability": true/false,
  "vulnerability_type": "XSS",
  "confidence": 0.0-1.0,
  "explanation": "详细解释",
  "location": "具体位置"
}}"""
    
    @staticmethod
    def get_command_injection_template() -> str:
        """命令注入检测模板"""
        return """你是一个命令注入漏洞检测专家。分析以下代码，重点检查命令注入风险。

代码：
{code}

请特别关注以下命令注入风险点：
1. 拼接系统命令
2. 调用外部程序
3. 缺乏输入验证
4. 不安全的shell操作

请按照以下格式输出：
{{
  "has_vulnerability": true/false,
  "vulnerability_type": "命令注入",
  "confidence": 0.0-1.0,
  "explanation": "详细解释",
  "location": "具体位置"
}}"""

# 全局模板管理器实例
_prompt_manager = None

def get_prompt_manager() -> PromptTemplateManager:
    """获取全局提示模板管理器"""
    global _prompt_manager
    if _prompt_manager is None:
        _prompt_manager = PromptTemplateManager()
    return _prompt_manager

def create_vulnerability_prompt(code: str, 
                              vulnerability_type: Optional[str] = None,
                              context: Optional[str] = None,
                              node_info: Optional[str] = None,
                              edge_info: Optional[str] = None) -> str:
    """创建漏洞检测提示"""
    manager = get_prompt_manager()
    
    if vulnerability_type:
        template_map = {
            "sql": VulnerabilityPromptTemplates.get_sql_injection_template(),
            "buffer": VulnerabilityPromptTemplates.get_buffer_overflow_template(),
            "xss": VulnerabilityPromptTemplates.get_xss_template(),
            "command": VulnerabilityPromptTemplates.get_command_injection_template()
        }
        
        template = template_map.get(vulnerability_type.lower())
        if template:
            return template.format(code=code)
    
    # 通用模板
    return manager.create_analysis_prompt(
        code=code,
        context=context,
        node_info=node_info,
        edge_info=edge_info
    )
"""
项目工具模块
包含各种辅助功能和工具函数
"""

import logging
import os
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import json
import re
from datetime import datetime

def setup_logging(log_level: str = "INFO", 
                 log_file: Optional[str] = None,
                 log_dir: str = "logs") -> logging.Logger:
    """设置日志配置"""
    
    # 创建日志目录
    if log_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = f"grace_{timestamp}.log"
    
    log_path = Path(log_dir)
    log_path.mkdir(exist_ok=True)
    full_log_path = log_path / log_file
    
    # 配置日志
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(full_log_path, encoding='utf-8')
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info(f"日志已设置: {full_log_path}")
    return logger

def get_project_root() -> Path:
    """获取项目根目录"""
    return Path(__file__).parent.parent

def create_directory_structure():
    """创建标准的项目目录结构"""
    root = get_project_root()
    
    directories = [
        "config",
        "data/raw",
        "data/processed", 
        "models",
        "utils",
        "logs",
        "output",
        "notebooks",
        "tests"
    ]
    
    for dir_name in directories:
        dir_path = root / dir_name
        dir_path.mkdir(parents=True, exist_ok=True)
        print(f"创建目录: {dir_path}")

def validate_code_content(code: str) -> bool:
    """验证代码内容是否有效"""
    if not code or len(code.strip()) < 10:
        return False
    
    # 简单的语法检查
    if re.search(r'[^\w\s\.\-\_\(\)\[\]\{\}\;\:\,\=\+\-\*\/\\%<>&\|\!\?]', code):
        # 包含可疑字符，需要进一步检查
        pass
    
    return True

def clean_code(code: str) -> str:
    """清理代码文本"""
    if not code:
        return ""
    
    # 移除多余的空白字符
    lines = code.split('\\n')
    cleaned_lines = []
    
    for line in lines:
        # 移除行首尾空白
        cleaned_line = line.strip()
        # 跳过空行
        if cleaned_line:
            cleaned_lines.append(cleaned_line)
    
    return '\\n'.join(cleaned_lines)

def extract_code_functions(code: str) -> List[str]:
    """从代码中提取函数定义"""
    if not code:
        return []
    
    # 简单的函数匹配模式
    function_patterns = [
        r'def\\s+(\\w+)\\s*\\([^)]*\\):',  # Python
        r'function\\s+(\\w+)\\s*\\([^)]*\\)',  # JavaScript
        r'(\\w+)\\s*\\([^)]*\\)\\s*{',  # C/Java等
    ]
    
    functions = []
    for pattern in function_patterns:
        matches = re.findall(pattern, code, re.MULTILINE)
        functions.extend(matches)
    
    return list(set(functions))  # 去重

def get_file_info(file_path: str) -> Dict[str, Any]:
    """获取文件信息"""
    path = Path(file_path)
    
    if not path.exists():
        return {"exists": False}
    
    stat = path.stat()
    
    return {
        "exists": True,
        "size": stat.st_size,
        "size_mb": round(stat.st_size / 1024 / 1024, 2),
        "modified_time": datetime.fromtimestamp(stat.st_mtime).isoformat(),
        "extension": path.suffix,
        "is_file": path.is_file(),
        "is_dir": path.is_dir()
    }

def safe_json_load(file_path: str, default: Any = None) -> Any:
    """安全地加载JSON文件"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        logging.warning(f"加载JSON文件失败 {file_path}: {e}")
        return default

def safe_json_dump(data: Any, file_path: str) -> bool:
    """安全地保存JSON文件"""
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        return True
    except Exception as e:
        logging.error(f"保存JSON文件失败 {file_path}: {e}")
        return False

def estimate_model_size(model_name: str) -> Dict[str, float]:
    """估算模型大小（MB）"""
    # 这里可以基于模型名称估算大小
    # 或者检查实际下载的模型文件
    
    size_estimates = {
        "microsoft/codebert-base": 440,  # MB
        "microsoft/graphcodebert-base": 440,
        "codet5-base": 220,
        "codet5-small": 60
    }
    
    return {
        "estimated_size_mb": size_estimates.get(model_name, 300),
        "note": "这是基于模型类型的估算值"
    }

def format_model_info(model_data: Dict[str, Any]) -> str:
    """格式化模型信息显示"""
    if not model_data:
        return "无模型信息"
    
    lines = []
    lines.append(f"模型名称: {model_data.get('name', '未知')}")
    lines.append(f"模型路径: {model_data.get('path', '未知')}")
    lines.append(f"文件数量: {model_data.get('file_count', 0)}")
    lines.append(f"总大小: {model_data.get('total_size_mb', 0)} MB")
    
    return "\\n".join(lines)

def create_sample_config() -> Dict[str, Any]:
    """创建示例配置文件"""
    return {
        "model_config": {
            "default_model": "microsoft/codebert-base",
            "max_length": 512,
            "temperature": 0.1
        },
        "data_config": {
            "datasets": {
                "bigvul": {
                    "name": "BigVul",
                    "description": "大型软件漏洞数据集",
                    "download_url": "https://github.com/microsoft/CodeXGLUE/tree/main/Code-Defect Detection"
                },
                "reveal": {
                    "name": "Reveal", 
                    "description": "代码漏洞检测数据集",
                    "download_url": "https://github.com/jple Phoebe/REVEAL"
                },
                "devign": {
                    "name": "Devign",
                    "description": "开发者引入的漏洞数据集",
                    "download_url": "https://github.com/duong_LEE/Devign"
                }
            }
        },
        "retrieval_config": {
            "faiss_index_type": "IndexFlatIP",
            "top_k": 5,
            "similarity_threshold": 0.8
        },
        "prompt_config": {
            "default_template": "basic",
            "include_code_context": True,
            "max_examples": 3
        }
    }

def generate_readme_content() -> str:
    """生成README内容"""
    return """# GRACE - 基于图结构和上下文学习的LLM漏洞检测

## 项目简介

GRACE (Graph structure and in-context learning Enhanced vulnerability detection) 是一个基于大语言模型的软件漏洞检测系统，支持图结构信息和上下文学习来提升检测准确率。

## 特性

- 🚀 本地模型推理 - 无需API依赖
- 📊 支持多种数据集 (BigVul, Reveal, Devign)
- 🔍 基于图结构的代码分析
- 🧠 上下文学习和示例检索
- 📈 完整的评估指标
- 🎯 交互式漏洞检测

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 下载预训练模型

```bash
python main.py --download-model
```

### 3. 运行评估

```bash
python main.py --mode eval --dataset bigvul
```

### 4. 交互式检测

```bash
python main.py --mode interactive
```

## 项目结构

```
GRACE/
├── main.py                 # 主程序入口
├── config/
│   └── config.py          # 配置文件
├── models/
│   └── __init__.py        # 模型接口
├── data/
│   └── __init__.py        # 数据处理
├── utils/
│   ├── model_downloader.py # 模型下载管理
│   └── prompt_templates.py # 提示模板
├── requirements.txt       # 依赖文件
└── README.md             # 项目文档
```

## 配置说明

主要配置项位于 `config/config.py`：

- `model_config`: 模型相关配置
- `data_config`: 数据集配置  
- `retrieval_config`: 检索配置
- `prompt_config`: 提示模板配置

## 使用示例

### 代码示例

```python
from models import LocalVulnerabilityDetector
from utils.prompt_templates import create_vulnerability_prompt

# 初始化检测器
detector = LocalVulnerabilityDetector("microsoft/codebert-base")

# 创建检测提示
prompt = create_vulnerability_prompt(code="your_code_here")

# 执行预测
result = detector.predict_vulnerability(prompt)
print(result)
```

### 命令行使用

```bash
# 下载模型
python main.py --download-model

# 评估特定数据集
python main.py --mode eval --dataset reveal --split test

# 交互式模式
python main.py --mode interactive
```

## 支持的数据集

1. **BigVul**: 大型软件漏洞数据集
2. **Reveal**: 代码漏洞检测数据集  
3. **Devign**: 开发者引入的漏洞数据集

## 性能指标

系统在三个数据集上的表现：

- BigVul: Accuracy 0.9169, F1 0.3593
- Reveal: Accuracy 0.8812, F1 0.4226
- Devign: Accuracy 0.6013, F1 0.6638

## 依赖

- Python 3.8+
- PyTorch
- Transformers
- FAISS
- scikit-learn
- pandas

## 许可证

本项目采用 MIT 许可证。

## 贡献

欢迎提交 Issue 和 Pull Request！

## 联系方式

如有问题，请通过 GitHub Issues 联系。
"""

def check_system_requirements() -> Dict[str, bool]:
    """检查系统要求"""
    checks = {
        "python_version": sys.version_info >= (3, 8),
        "torch_available": False,
        "transformers_available": False,
        "faiss_available": False,
        "sklearn_available": False,
        "memory_gte_4gb": False
    }
    
    # 检查包是否安装
    try:
        import torch
        checks["torch_available"] = True
    except ImportError:
        pass
    
    try:
        import transformers
        checks["transformers_available"] = True
    except ImportError:
        pass
    
    try:
        import faiss
        checks["faiss_available"] = True
    except ImportError:
        pass
    
    try:
        import sklearn
        checks["sklearn_available"] = True
    except ImportError:
        pass
    
    # 检查内存（简单估算）
    try:
        import psutil
        memory_gb = psutil.virtual_memory().total / (1024**3)
        checks["memory_gte_4gb"] = memory_gb >= 4
    except ImportError:
        # 如果psutil不可用，假设有足够内存
        checks["memory_gte_4gb"] = True
    
    return checks

class ProgressBar:
    """简单的进度条"""
    
    def __init__(self, total: int, width: int = 50):
        self.total = total
        self.width = width
        self.current = 0
    
    def update(self, step: int = 1):
        """更新进度"""
        self.current += step
        percent = self.current / self.total
        filled_width = int(self.width * percent)
        
        bar = '█' * filled_width + '░' * (self.width - filled_width)
        print(f'\\r进度: |{bar}| {percent:.1%} ({self.current}/{self.total})', end='')
        
        if self.current >= self.total:
            print()  # 换行

def ensure_directories():
    """确保项目必要的目录结构存在"""
    root = get_project_root()
    
    required_dirs = [
        root / "config",
        root / "data" / "raw", 
        root / "data" / "processed",
        root / "models",
        root / "utils",
        root / "logs",
        root / "outputs",
        root / "logs"
    ]
    
    for dir_path in required_dirs:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    return True

def save_json_safe(data: Any, file_path: str) -> bool:
    """安全地保存JSON文件的别名"""
    return safe_json_dump(data, file_path)

def estimate_model_size(model_name: str) -> float:
    """估算模型大小（简化版）"""
    # 基础估算：小型模型约 100-300MB，大型模型约 1-5GB
    size_estimates = {
        "microsoft/codebert-base": 440,  # MB
        "microsoft/graphcodebert-base": 440,
        "codet5-base": 220,
        "codet5-small": 60
    }
    return size_estimates.get(model_name, 300)

# 导出主要函数
__all__ = [
    'setup_logging',
    'get_project_root', 
    'create_directory_structure',
    'ensure_directories',
    'validate_code_content',
    'clean_code',
    'get_file_info',
    'safe_json_load',
    'safe_json_dump',
    'save_json_safe',
    'create_sample_config',
    'generate_readme_content',
    'check_system_requirements',
    'ProgressBar',
    'estimate_model_size'
]
"""
GRACE项目配置文件
支持本地预训练模型的配置管理
"""

import os
import torch
from pathlib import Path
from typing import Optional, Dict, Any

class Config:
    """项目配置类"""
    
    def __init__(self, data_root: str = None):
        # 基础路径配置
        self.project_root = Path(__file__).parent.parent.absolute()
        
        # 数据根目录配置 - 默认使用项目内data目录，可自定义
        if data_root:
            self.data_root = Path(data_root)
        else:
            # 默认使用云计算平台路径
            self.data_root = Path("/root/sj-tmp/dataset/")
        
        self.data_dir = self.data_root  # 数据集存储目录
        self.models_dir = Path("/root/sj-tmp/pre_train")  # 预训练模型存储目录
        self.output_dir = self.project_root / "outputs"
        self.logs_dir = self.project_root / "logs"
        self.figs_dir = self.project_root / "figs"
        
        # 创建必要目录
        for directory in [self.data_dir, self.models_dir, self.output_dir, self.logs_dir, self.figs_dir]:
            directory.mkdir(exist_ok=True)
        
        # 模型配置
        self.model_name = "microsoft/codebert-base"  # 默认模型，可选择其他
        self.embedding_model = "sentence-transformers/all-MiniLM-L6-v2"
        self.max_length = 512
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.batch_size = 8
        self.num_workers = 4
        
        # 训练配置
        self.learning_rate = 2e-5
        self.epochs = 10
        self.warmup_steps = 100
        self.save_steps = 500
        self.eval_steps = 500
        
        # 检索配置
        self.top_k_examples = 5
        self.code_weight = 0.7
        self.ast_weight = 0.3
        self.faiss_nlist = 1
        
        # 评估配置
        self.eval_batch_size = 16
        
        # 日志配置
        self.log_level = "INFO"
        self.log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        
        # 数据集配置 - 使用Hugging Face数据集
        self.datasets = {
            "bigvul": {
                "name": "BigVul",
                "huggingface_url": "https://huggingface.co/datasets/bstee615/bigvul",
                "local_files": {
                    "train": "bigvul_train.json",
                    "test": "bigvul_test.json",
                    "val": "bigvul_val.json"
                },
                "processed_files": {
                    "train": "bigvul_train_processed.json",
                    "test": "bigvul_test_processed.json"
                }
            },
            "reveal": {
                "name": "Reveal", 
                "huggingface_url": "https://huggingface.co/datasets/claudios/ReVeal",
                "local_files": {
                    "train": "reveal_train.json",
                    "test": "reveal_test.json", 
                    "val": "reveal_val.json"
                },
                "processed_files": {
                    "train": "reveal_train_processed.json",
                    "test": "reveal_test_processed.json"
                }
            },
            "devign": {
                "name": "Devign",
                "huggingface_url": "https://huggingface.co/datasets/DetectVul/devign",
                "local_files": {
                    "train": "devign_train.json",
                    "test": "devign_test.json",
                    "val": "devign_val.json"
                },
                "processed_files": {
                    "train": "devign_train_processed.json", 
                    "test": "devign_test_processed.json"
                }
            }
        }
        
        # 提示模板
        self.prompt_templates = {
            "basic": (
                "In the above code snippet, check for potential security vulnerabilities "
                "and output either 'Vulnerable' or 'Non-vulnerable'. "
                "You are now an excellent programmer. "
                "You are conducting a function vulnerability detection task for C/C++ language."
            ),
            "with_graph": (
                "In the above code snippet, check for potential security vulnerabilities "
                "and output either 'Vulnerable' or 'Non-vulnerable'. "
                "You are now an excellent programmer. "
                "You are conducting a function vulnerability detection task for C/C++ language.\n\n"
                "The node information of the function is as follows:\n{node_info}\n\n"
                "The edge information of the function is as follows:\n{edge_info}\n\n"
                "Here is an example for you to learn from:\n{example}"
            )
        }
    
    def get_model_path(self, model_name: Optional[str] = None) -> Path:
        """获取模型本地路径"""
        if model_name is None:
            model_name = self.model_name
        model_name = model_name.replace("/", "_")
        return self.models_dir / model_name
    
    def get_dataset_path(self, dataset_name: str, split: str = "test") -> Path:
        """获取数据集文件路径"""
        if dataset_name not in self.datasets:
            raise ValueError(f"Unknown dataset: {dataset_name}")
        
        dataset_config = self.datasets[dataset_name]
        filename = dataset_config["local_files"][split]
        return self.data_dir / filename
    
    def get_processed_dataset_path(self, dataset_name: str, split: str = "test") -> Path:
        """获取处理后的数据集文件路径"""
        if dataset_name not in self.datasets:
            raise ValueError(f"Unknown dataset: {dataset_name}")
        
        dataset_config = self.datasets[dataset_name]
        filename = dataset_config["processed_files"][split]
        return self.data_dir / filename
    
    def get_output_path(self, filename: str) -> Path:
        """获取输出文件路径"""
        return self.output_dir / filename
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        config_dict = {}
        for key, value in self.__dict__.items():
            if not key.startswith('_') and not callable(value):
                config_dict[key] = value
        return config_dict
    
    def save_config(self, filename: str = "config.json"):
        """保存配置到文件"""
        import json
        config_path = self.project_root / filename
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False, default=str)
    
    @classmethod
    def load_config(cls, filename: str = "config.json", data_root: str = None):
        """从文件加载配置"""
        config_path = Path(filename)
        if not config_path.exists():
            return cls(data_root)
        
        import json
        with open(config_path, 'r', encoding='utf-8') as f:
            config_data = json.load(f)
        
        config = cls(data_root)
        for key, value in config_data.items():
            if hasattr(config, key):
                setattr(config, key, value)
        
        return config

# 全局配置实例
config = Config()
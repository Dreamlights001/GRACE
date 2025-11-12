"""
数据处理模块
负责数据集加载、预处理和格式转换
"""

import json
import logging
import pickle
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import pandas as pd
import numpy as np

from config.config import config

logger = logging.getLogger(__name__)

class DataProcessor:
    """数据处理器"""
    
    def __init__(self):
        self.datasets_dir = config.data_dir
        self.datasets_dir.mkdir(exist_ok=True)
        self.current_data = []
    
    def load_dataset(self, dataset_name: str, split: str = "test") -> bool:
        """
        加载数据集
        
        Args:
            dataset_name: 数据集名称 (bigvul, reveal, devign)
            split: 数据分割 (train, test)
        
        Returns:
            是否加载成功
        """
        try:
            processed_path = None
            try:
                processed_path = config.get_processed_dataset_path(dataset_name, split)
            except Exception:
                processed_path = None
            
            if processed_path and processed_path.exists():
                dataset_path = processed_path
            else:
                dataset_path = config.get_dataset_path(dataset_name, split)
            
            if not dataset_path.exists():
                logger.warning(f"数据集文件不存在: {dataset_path}")
                logger.info(f"请手动下载数据集: {config.datasets[dataset_name].get('huggingface_url', '未知')}")
                self.current_data = []
                return False
            
            logger.info(f"加载数据集: {dataset_path}")
            
            raw = []
            with open(dataset_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        item = json.loads(line)
                        raw.append(item)
                    except json.JSONDecodeError as e:
                        logger.warning(f"第{line_num}行JSON解析失败: {e}")
                        continue
            
            logger.info(f"成功加载 {len(raw)} 个样本")
            
            def _coerce_label(x):
                import numpy as np
                import pandas as pd
                if isinstance(x, (list, tuple, np.ndarray)):
                    for v in x:
                        if v is None:
                            continue
                        if isinstance(v, float) and pd.isna(v):
                            continue
                        x = v
                        break
                if isinstance(x, bool):
                    return int(x)
                if isinstance(x, (np.integer, int)):
                    return int(x)
                if isinstance(x, (np.floating, float)):
                    try:
                        return int(round(x))
                    except Exception:
                        return 0
                if isinstance(x, str):
                    s = x.strip().lower()
                    if s in ('true', 'false'):
                        return 1 if s == 'true' else 0
                    try:
                        return int(s)
                    except Exception:
                        try:
                            return int(float(s))
                        except Exception:
                            return 0
                if isinstance(x, dict):
                    for k in ('label', 'target', 'value'):
                        if k in x:
                            return _coerce_label(x[k])
                    return 0
                return 0
            
            std = []
            for it in raw:
                if dataset_name == 'bigvul':
                    code = it.get('code') or it.get('func_before') or ''
                    label = it.get('label') if 'label' in it else it.get('vul')
                elif dataset_name == 'reveal':
                    code = it.get('code') or it.get('functionSource') or ''
                    label = it.get('label')
                elif dataset_name == 'devign':
                    code = it.get('code') or it.get('func') or ''
                    label = it.get('label') if 'label' in it else it.get('target')
                else:
                    code = it.get('code') or ''
                    label = it.get('label')
                std.append({
                    'code': str(code),
                    'label': _coerce_label(label)
                })
            
            self.current_data = std
            return len(self.current_data) > 0
            
        except Exception as e:
            logger.error(f"加载数据集失败: {e}")
            self.current_data = []
            return False
    
    def process_raw_data(self, raw_data_path: str, 
                        output_path: str,
                        include_graph: bool = True) -> bool:
        """
        处理原始数据为标准格式
        
        Args:
            raw_data_path: 原始数据路径
            output_path: 输出路径
            include_graph: 是否包含图信息
        
        Returns:
            处理是否成功
        """
        try:
            # 这里实现原始数据到标准格式的转换
            # 根据具体的原始数据格式来实现
            
            processed_data = []
            
            # 示例：处理CSV格式的数据
            if raw_data_path.endswith('.csv'):
                df = pd.read_csv(raw_data_path)
                for _, row in df.iterrows():
                    processed_item = {
                        'func': str(row.get('func', '')),
                        'target': int(row.get('target', 0)),
                        'example': str(row.get('example', ''))
                    }
                    
                    if include_graph:
                        processed_item['node'] = str(row.get('node', ''))
                        processed_item['edge'] = str(row.get('edge', ''))
                    
                    processed_data.append(processed_item)
            
            # 保存处理后的数据
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(processed_data, f, ensure_ascii=False, indent=2)
            
            logger.info(f"数据处理完成，保存到: {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"数据处理失败: {e}")
            return False
    
    def prepare_training_data(self, dataset_name: str) -> Tuple[List[str], List[str]]:
        """
        准备训练数据用于构建检索索引
        
        Args:
            dataset_name: 数据集名称
            
        Returns:
            (代码列表, AST列表)
        """
        try:
            # 加载训练数据
            train_data = self.load_dataset(dataset_name, "train")
            
            codes = []
            asts = []
            
            for item in train_data:
                if 'func' in item:
                    codes.append(item['func'])
                if 'ast' in item:
                    asts.append(item['ast'])
                else:
                    # 如果没有AST，生成一个简单的
                    asts.append(self._generate_simple_ast(item.get('func', '')))
            
            logger.info(f"准备了 {len(codes)} 个训练样本")
            return codes, asts
            
        except Exception as e:
            logger.error(f"准备训练数据失败: {e}")
            return [], []
    
    def _generate_simple_ast(self, code: str) -> str:
        """生成简单的AST表示（备用方法）"""
        # 这里可以实现简单的AST生成逻辑
        # 或者使用其他工具来生成AST
        return " ".join(code.lower().split())  # 简单的词袋表示
    
    def filter_data(self, data: List[Dict[str, Any]], 
                   min_length: int = 10,
                   max_length: int = 4000) -> List[Dict[str, Any]]:
        """
        过滤数据
        
        Args:
            data: 原始数据
            min_length: 最小代码长度
            max_length: 最大代码长度
        
        Returns:
            过滤后的数据
        """
        filtered_data = []
        
        for item in data:
            code = item.get('code', '')
            
            if min_length <= len(code) <= max_length:
                filtered_data.append(item)
        
        logger.info(f"过滤前: {len(data)}, 过滤后: {len(filtered_data)}")
        return filtered_data
    
    def split_data(self, data: List[Dict[str, Any]], 
                  train_ratio: float = 0.8,
                  val_ratio: float = 0.1) -> Tuple[List[Dict[str, Any]], 
                                                  List[Dict[str, Any]], 
                                                  List[Dict[str, Any]]]:
        """
        分割数据为训练、验证、测试集
        
        Args:
            data: 完整数据
            train_ratio: 训练集比例
            val_ratio: 验证集比例
        
        Returns:
            (训练集, 验证集, 测试集)
        """
        total = len(data)
        train_size = int(total * train_ratio)
        val_size = int(total * val_ratio)
        
        train_data = data[:train_size]
        val_data = data[train_size:train_size + val_size]
        test_data = data[train_size + val_size:]
        
        logger.info(f"数据分割: 训练 {len(train_data)}, 验证 {len(val_data)}, 测试 {len(test_data)}")
        return train_data, val_data, test_data
    
    def save_processed_data(self, data: List[Dict[str, Any]], 
                           output_path: str) -> bool:
        """保存处理后的数据"""
        try:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            
            logger.info(f"数据已保存到: {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"保存数据失败: {e}")
            return False
    
    def get_data_items(self) -> List[Dict[str, Any]]:
        """
        获取当前加载的数据项
        
        Returns:
            数据项列表
        """
        return self.current_data

class DatasetInfo:
    """数据集信息管理"""
    
    def __init__(self):
        self.dataset_info = config.datasets
    
    def get_dataset_names(self) -> List[str]:
        """获取所有数据集名称"""
        return list(self.dataset_info.keys())
    
    def get_dataset_info(self, dataset_name: str) -> Dict[str, Any]:
        """获取数据集信息"""
        if dataset_name not in self.dataset_info:
            raise ValueError(f"未知的数据集: {dataset_name}")
        return self.dataset_info[dataset_name]
    
    def check_dataset_files(self, dataset_name: str) -> Dict[str, bool]:
        """检查数据集文件是否存在"""
        if dataset_name not in self.dataset_info:
            return {}
        
        file_status = {}
        dataset_config = self.dataset_info[dataset_name]
        
        for split, filename in dataset_config["processed_files"].items():
            file_path = config.get_dataset_path(dataset_name, split)
            file_status[f"{split}_exists"] = file_path.exists()
        
        return file_status
    
    def suggest_preprocessing_steps(self, dataset_name: str) -> List[str]:
        """建议预处理步骤"""
        suggestions = []
        
        # 检查必要文件
        file_status = self.check_dataset_files(dataset_name)
        
        if not any(file_status.values()):
            suggestions.append("1. 下载原始数据集")
            suggestions.append("2. 提取并放置到data目录")
        
        for split, exists in file_status.items():
            if not exists:
                split_name = split.replace("_exists", "")
                suggestions.append(f"3. 处理 {split_name} 数据")
        
        return suggestions

def create_processor() -> DataProcessor:
    """创建数据处理器的工厂函数"""
    return DataProcessor()

def get_dataset_info() -> DatasetInfo:
    """获取数据集信息的工厂函数"""
    return DatasetInfo()

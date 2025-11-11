#!/usr/bin/env python3
"""
GRACE项目数据准备脚本
自动从Hugging Face下载和准备数据集

使用方法:
    python prepare_data.py --data-root /root/sj-tmp/dataset/
    python prepare_data.py --download-model  # 下载数据集
    python prepare_data.py --all  # 下载所有数据
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime

# 添加项目根目录到路径
sys.path.append(str(Path(__file__).parent))

try:
    from datasets import load_dataset, Dataset
    from datasets.exceptions import DatasetNotFoundError
    import pandas as pd
except ImportError as e:
    print(f"缺少依赖包: {e}")
    print("请运行: pip install -r requirements.txt")
    sys.exit(1)

from config.config import Config

class DataPreparator:
    """数据下载和准备器"""
    
    def __init__(self, data_root: str = "/root/sj-tmp/dataset/"):
        """
        初始化数据准备器
        
        Args:
            data_root: 数据集存储根目录
        """
        self.data_root = Path(data_root)
        self.config = Config(data_root=data_root)
        self.setup_logging()
        
        # 确保数据根目录存在
        self.data_root.mkdir(parents=True, exist_ok=True)
        
        print(f"数据根目录: {self.data_root}")
        print(f"配置数据集: {list(self.config.datasets.keys())}")
    
    def setup_logging(self):
        """设置日志"""
        log_dir = self.config.logs_dir
        log_dir.mkdir(exist_ok=True)
        
        log_file = log_dir / f"data_preparation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format=self.config.log_format,
            handlers=[
                logging.FileHandler(log_file, encoding='utf-8'),
                logging.StreamHandler()
            ]
        )
        
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"数据准备日志: {log_file}")
    
    def download_dataset(self, dataset_name: str) -> bool:
        """
        下载指定数据集
        
        Args:
            dataset_name: 数据集名称 (bigvul, reveal, devign)
            
        Returns:
            bool: 下载是否成功
        """
        if dataset_name not in self.config.datasets:
            self.logger.error(f"未知数据集: {dataset_name}")
            return False
        
        dataset_info = self.config.datasets[dataset_name]
        huggingface_url = dataset_info["huggingface_url"]
        
        self.logger.info(f"开始下载数据集: {dataset_name} ({huggingface_url})")
        
        try:
            # 从Hugging Face加载数据集
            if dataset_name == "bigvul":
                dataset = self._load_bigvul_dataset()
            elif dataset_name == "reveal":
                dataset = self._load_reveal_dataset()
            elif dataset_name == "devign":
                dataset = self._load_devign_dataset()
            else:
                self.logger.error(f"不支持的数据集: {dataset_name}")
                return False
            
            if dataset is None:
                self.logger.error(f"数据集 {dataset_name} 下载失败")
                return False
            
            # 保存数据集
            success = self._save_dataset(dataset, dataset_name)
            
            if success:
                self.logger.info(f"数据集 {dataset_name} 下载完成")
                return True
            else:
                self.logger.error(f"数据集 {dataset_name} 保存失败")
                return False
                
        except Exception as e:
            self.logger.error(f"下载数据集 {dataset_name} 时发生错误: {e}")
            return False
    
    def _load_bigvul_dataset(self) -> Optional[Dict[str, Dataset]]:
        """加载BigVul数据集"""
        self.logger.info("加载 BigVul 数据集...")
        try:
            # 加载训练集
            train_dataset = load_dataset("bstee615/bigvul", split="train[:80%]")
            # 加载测试集
            test_dataset = load_dataset("bstee615/bigvul", split="train[80%:]")
            # 创建验证集（从训练集分出一部分）
            val_dataset = load_dataset("bstee615/bigvul", split="train[:10%]")
            
            return {
                "train": train_dataset,
                "test": test_dataset,
                "val": val_dataset
            }
        except Exception as e:
            self.logger.error(f"加载BigVul数据集失败: {e}")
            return None
    
    def _load_reveal_dataset(self) -> Optional[Dict[str, Dataset]]:
        """加载Reveal数据集"""
        self.logger.info("加载 Reveal 数据集...")
        try:
            # 加载数据集
            full_dataset = load_dataset("claudios/ReVeal")
            
            # 分割数据集
            total_size = len(full_dataset["train"])
            train_size = int(total_size * 0.8)
            val_size = int(total_size * 0.1)
            
            train_dataset = full_dataset["train"].select(range(train_size))
            val_dataset = full_dataset["train"].select(range(train_size, train_size + val_size))
            test_dataset = full_dataset["train"].select(range(train_size + val_size))
            
            return {
                "train": train_dataset,
                "test": test_dataset,
                "val": val_dataset
            }
        except Exception as e:
            self.logger.error(f"加载Reveal数据集失败: {e}")
            return None
    
    def _load_devign_dataset(self) -> Optional[Dict[str, Dataset]]:
        """加载Devign数据集"""
        self.logger.info("加载 Devign 数据集...")
        try:
            # 加载数据集
            full_dataset = load_dataset("DetectVul/devign")
            
            # 分割数据集
            total_size = len(full_dataset["train"])
            train_size = int(total_size * 0.8)
            val_size = int(total_size * 0.1)
            
            train_dataset = full_dataset["train"].select(range(train_size))
            val_dataset = full_dataset["train"].select(range(train_size, train_size + val_size))
            test_dataset = full_dataset["train"].select(range(train_size + val_size))
            
            return {
                "train": train_dataset,
                "test": test_dataset,
                "val": val_dataset
            }
        except Exception as e:
            self.logger.error(f"加载Devign数据集失败: {e}")
            return None
    
    def _save_dataset(self, dataset: Dict[str, Dataset], dataset_name: str) -> bool:
        """
        保存数据集到本地文件
        
        Args:
            dataset: 数据集字典
            dataset_name: 数据集名称
            
        Returns:
            bool: 保存是否成功
        """
        try:
            local_files = self.config.datasets[dataset_name]["local_files"]
            
            for split, split_dataset in dataset.items():
                filename = local_files[split]
                filepath = self.data_root / filename
                
                self.logger.info(f"保存 {dataset_name}/{split} 到 {filepath}")
                
                # 转换为Pandas DataFrame并保存为JSON
                df = split_dataset.to_pandas()
                df.to_json(filepath, orient='records', lines=True, force_ascii=False)
                
                # 验证文件
                if filepath.exists():
                    file_size = filepath.stat().st_size
                    self.logger.info(f"文件保存成功: {filepath} ({file_size} bytes)")
                else:
                    self.logger.error(f"文件保存失败: {filepath}")
                    return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"保存数据集时发生错误: {e}")
            return False
    
    def preprocess_datasets(self) -> bool:
        """
        预处理数据集
        
        Returns:
            bool: 预处理是否成功
        """
        self.logger.info("开始预处理数据集...")
        
        try:
            for dataset_name in self.config.datasets.keys():
                self.logger.info(f"预处理数据集: {dataset_name}")
                success = self._preprocess_single_dataset(dataset_name)
                if not success:
                    self.logger.error(f"数据集 {dataset_name} 预处理失败")
                    return False
            
            self.logger.info("所有数据集预处理完成")
            return True
            
        except Exception as e:
            self.logger.error(f"预处理数据集时发生错误: {e}")
            return False
    
    def _preprocess_single_dataset(self, dataset_name: str) -> bool:
        """
        预处理单个数据集
        
        Args:
            dataset_name: 数据集名称
            
        Returns:
            bool: 预处理是否成功
        """
        try:
            local_files = self.config.datasets[dataset_name]["local_files"]
            processed_files = self.config.datasets[dataset_name]["processed_files"]
            
            processed_data = {}
            
            for split in ["train", "test"]:
                filename = local_files[split]
                filepath = self.data_root / filename
                
                if not filepath.exists():
                    self.logger.error(f"数据文件不存在: {filepath}")
                    continue
                
                # 读取原始数据
                df = pd.read_json(filepath, orient='records', lines=True)
                
                # 标准化字段名
                if dataset_name == "bigvul":
                    df = self._standardize_bigvul(df)
                elif dataset_name == "reveal":
                    df = self._standardize_reveal(df)
                elif dataset_name == "devign":
                    df = self._standardize_devign(df)
                
                processed_data[split] = df
                
                # 保存处理后的数据
                processed_filename = processed_files[split]
                processed_filepath = self.data_root / processed_filename
                
                processed_df = df.to_json(processed_filepath, orient='records', lines=True, force_ascii=False)
                self.logger.info(f"处理后的数据已保存: {processed_filepath}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"预处理数据集 {dataset_name} 时发生错误: {e}")
            return False
    
    def _standardize_bigvul(self, df: pd.DataFrame) -> pd.DataFrame:
        """标准化BigVul数据集格式"""
        # 重命名列以保持一致性
        column_mapping = {
            'func_before': 'code',
            'func_after': 'code_fixed',
            'vul': 'label',
            'project': 'project',
            'CVE ID': 'cve_id'
        }
        
        for old_col, new_col in column_mapping.items():
            if old_col in df.columns:
                df = df.rename(columns={old_col: new_col})
        
        # 确保必要的列存在
        if 'label' in df.columns:
            df['label'] = df['label'].astype(int)
        
        return df
    
    def _standardize_reveal(self, df: pd.DataFrame) -> pd.DataFrame:
        """标准化Reveal数据集格式"""
        # 重命名列以保持一致性
        column_mapping = {
            'functionSource': 'code',
            'label': 'label',
            'project': 'project'
        }
        
        for old_col, new_col in column_mapping.items():
            if old_col in df.columns:
                df = df.rename(columns={old_col: new_col})
        
        # 确保必要的列存在
        if 'label' in df.columns:
            df['label'] = df['label'].astype(int)
        
        return df
    
    def _standardize_devign(self, df: pd.DataFrame) -> pd.DataFrame:
        """标准化Devign数据集格式"""
        # 重命名列以保持一致性
        column_mapping = {
            'func': 'code',
            'target': 'label',
            'project': 'project'
        }
        
        for old_col, new_col in column_mapping.items():
            if old_col in df.columns:
                df = df.rename(columns={old_col: new_col})
        
        # 确保必要的列存在
        if 'label' in df.columns:
            df['label'] = df['label'].astype(int)
        
        return df
    
    def check_data_status(self) -> Dict[str, Dict[str, bool]]:
        """
        检查数据下载状态
        
        Returns:
            Dict: 数据集状态信息
        """
        status = {}
        
        for dataset_name, dataset_info in self.config.datasets.items():
            status[dataset_name] = {}
            
            # 检查原始文件
            for split, filename in dataset_info["local_files"].items():
                filepath = self.data_root / filename
                status[dataset_name][f"{split}_raw"] = filepath.exists()
            
            # 检查处理后文件
            for split, filename in dataset_info["processed_files"].items():
                filepath = self.data_root / filename
                status[dataset_name][f"{split}_processed"] = filepath.exists()
        
        return status
    
    def print_status(self):
        """打印数据状态"""
        status = self.check_data_status()
        
        print("\n" + "="*60)
        print("数据下载状态检查")
        print("="*60)
        
        for dataset_name, dataset_status in status.items():
            print(f"\n📊 {dataset_name.upper()} 数据集:")
            for file_type, exists in dataset_status.items():
                status_icon = "✅" if exists else "❌"
                file_size = ""
                
                if exists:
                    # 获取文件大小
                    filename = self.config.datasets[dataset_name]["local_files"].get(file_type.replace("_raw", ""), 
                                     self.config.datasets[dataset_name]["processed_files"].get(file_type.replace("_processed", ""), ""))
                    if filename:
                        filepath = self.data_root / filename
                        if filepath.exists():
                            size_mb = filepath.stat().st_size / (1024 * 1024)
                            file_size = f" ({size_mb:.1f}MB)"
                
                print(f"  {status_icon} {file_type}: {file_size}")
        
        print("\n" + "="*60)
    
    def run_full_preparation(self, datasets: List[str] = None):
        """
        运行完整的数据准备流程
        
        Args:
            datasets: 要准备的数据集列表，None表示准备所有数据集
        """
        if datasets is None:
            datasets = list(self.config.datasets.keys())
        
        print(f"开始数据准备流程，数据集: {datasets}")
        print(f"数据根目录: {self.data_root}")
        
        success_count = 0
        total_count = len(datasets)
        
        for dataset_name in datasets:
            print(f"\n{'='*20} 处理 {dataset_name} {'='*20}")
            
            # 下载数据集
            if self.download_dataset(dataset_name):
                success_count += 1
            else:
                print(f"❌ {dataset_name} 下载失败")
        
        print(f"\n{'='*20} 数据下载总结 {'='*20}")
        print(f"成功: {success_count}/{total_count}")
        
        if success_count == total_count:
            print("✅ 所有数据集下载成功，开始预处理...")
            # 预处理数据集
            if self.preprocess_datasets():
                print("✅ 数据预处理完成")
            else:
                print("❌ 数据预处理失败")
        else:
            print("❌ 部分数据集下载失败，跳过预处理")
        
        # 打印最终状态
        self.print_status()

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="GRACE项目数据准备工具")
    parser.add_argument("--data-root", type=str, default="/root/sj-tmp/dataset/",
                       help="数据存储根目录")
    parser.add_argument("--dataset", type=str, 
                       choices=["bigvul", "reveal", "devign"],
                       help="指定要下载的数据集")
    parser.add_argument("--all", action="store_true",
                       help="下载所有数据集")
    parser.add_argument("--check", action="store_true",
                       help="检查数据下载状态")
    parser.add_argument("--preprocess", action="store_true",
                       help="仅预处理已下载的数据")
    
    args = parser.parse_args()
    
    # 创建数据准备器
    preparator = DataPreparator(data_root=args.data_root)
    
    if args.check:
        # 检查状态
        preparator.print_status()
        
    elif args.preprocess:
        # 仅预处理
        if preparator.preprocess_datasets():
            print("✅ 数据预处理完成")
        else:
            print("❌ 数据预处理失败")
            
    elif args.dataset:
        # 下载指定数据集
        preparator.run_full_preparation([args.dataset])
        
    elif args.all:
        # 下载所有数据集
        preparator.run_full_preparation()
        
    else:
        # 默认行为：下载所有数据集
        print("未指定操作，默认下载所有数据集")
        preparator.run_full_preparation()

if __name__ == "__main__":
    main()
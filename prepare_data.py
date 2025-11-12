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
    import time
    import requests
    from huggingface_hub import HfApi
except ImportError as e:
    print(f"缺少依赖包: {e}")
    print("请运行: pip install -r requirements.txt")
    sys.exit(1)

from config.config import Config

# 配置HuggingFace镜像源 - 解决云计算平台连接问题
HF_MIRRORS = [
    "https://hf-mirror.com",  # 官方镜像
    "https://huggingface.co",  # 原始地址
    "https://hf-mirror.com",  # 备用镜像
]

# 设置环境变量以使用镜像源
os.environ.setdefault("HF_ENDPOINT", HF_MIRRORS[0])
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")

# 配置代理设置（如果需要）
PROXY_CONFIG = {
    "http_proxy": os.environ.get("HTTP_PROXY", ""),
    "https_proxy": os.environ.get("HTTPS_PROXY", ""),
}

class DataPreparator:
    """数据下载和准备器 - 增强版网络处理"""
    
    def __init__(self, data_root: str = "/root/sj-tmp/-dataset/"):
        """
        初始化数据准备器
        
        Args:
            data_root: 数据集存储根目录（默认使用云计算平台路径）
        """
        self.data_root = Path(data_root)
        self.config = Config(data_root=data_root)
        self.setup_logging()
        self.max_retries = 3
        self.retry_delay = 5
        
        # 初始化镜像源配置
        self.hf_mirrors = HF_MIRRORS.copy()
        self.current_mirror_index = 0
        self.api = self._init_hf_api_with_mirrors()
        
        # 确保数据根目录存在
        self.data_root.mkdir(parents=True, exist_ok=True)
        
        print(f"数据根目录: {self.data_root}")
        print(f"配置数据集: {list(self.config.datasets.keys())}")
        print(f"当前HuggingFace镜像源: {self.hf_mirrors[self.current_mirror_index]}")
        print(f"数据集将保存到: {self.data_root}")
    
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
    
    def _init_hf_api_with_mirrors(self) -> HfApi:
        """初始化带镜像源的HuggingFace API"""
        try:
            # 设置当前镜像源
            current_mirror = self.hf_mirrors[self.current_mirror_index]
            os.environ["HF_ENDPOINT"] = current_mirror
            
            # 配置代理（如果存在）
            if PROXY_CONFIG["http_proxy"] or PROXY_CONFIG["https_proxy"]:
                self.logger.info(f"使用代理配置: {PROXY_CONFIG}")
            
            return HfApi()
        except Exception as e:
            self.logger.error(f"初始化HuggingFace API失败: {e}")
            return HfApi()
    
    def _switch_to_next_mirror(self) -> bool:
        """切换到下一个镜像源"""
        if self.current_mirror_index < len(self.hf_mirrors) - 1:
            self.current_mirror_index += 1
            current_mirror = self.hf_mirrors[self.current_mirror_index]
            os.environ["HF_ENDPOINT"] = current_mirror
            self.api = self._init_hf_api_with_mirrors()
            self.logger.info(f"切换到镜像源: {current_mirror}")
            return True
        else:
            self.logger.error("所有镜像源都已尝试，无法连接")
            return False
    
    def _test_network_connectivity(self) -> bool:
        """测试网络连接 - 支持镜像源切换"""
        for mirror_index in range(len(self.hf_mirrors)):
            current_mirror = self.hf_mirrors[mirror_index]
            try:
                self.logger.info(f"测试镜像源连接: {current_mirror}")
                response = requests.get(current_mirror, timeout=10)
                if response.status_code == 200:
                    # 如果当前使用的不是这个可用的镜像源，切换到它
                    if mirror_index != self.current_mirror_index:
                        self.current_mirror_index = mirror_index
                        os.environ["HF_ENDPOINT"] = current_mirror
                        self.api = self._init_hf_api_with_mirrors()
                        self.logger.info(f"切换到可用的镜像源: {current_mirror}")
                    return True
            except Exception as e:
                self.logger.warning(f"镜像源 {current_mirror} 连接失败: {e}")
                continue
        
        self.logger.error("所有镜像源连接测试失败")
        return False
    
    def _get_available_alternatives(self, dataset_type: str) -> List[str]:
        """
        获取可用的替代数据集
        
        Args:
            dataset_type: 数据集类型 ('bigvul', 'reveal', 'devign')
            
        Returns:
            List[str]: 替代数据集列表
        """
        alternatives = {
            "bigvul": [
                "Junwei/MSR",  # 主要替代数据集 - 经过验证可用
                "FFJSJ/BigVul",
                "microsoft/BigVul-Benchmark"
            ],
            "reveal": [
                "microsoft/CodeXGLUE",  # 经过验证可用
                "codebert/ReVeal-Extended",
                "claudios/ReVeal-dataset"  # 备用数据源
            ],
            "devign": [
                "microsoft/Devign-Benchmark", 
                "DetectVul/devign-processed",  # 经过验证可用
                "codebert/Devign-Filtered"
            ]
        }
        
        return alternatives.get(dataset_type, [])
    
    def _load_dataset_with_retry(self, dataset_path: str, max_retries: int = None) -> Optional[Dict]:
        """带重试机制的数据集加载 - 支持镜像源切换"""
        if max_retries is None:
            max_retries = self.max_retries
        
        for attempt in range(max_retries):
            try:
                self.logger.info(f"尝试加载数据集 {dataset_path} (第 {attempt + 1}/{max_retries} 次)")
                self.logger.info(f"当前镜像源: {self.hf_mirrors[self.current_mirror_index]}")
                
                dataset = load_dataset(dataset_path)
                self.logger.info(f"数据集 {dataset_path} 加载成功")
                return dataset
            except Exception as e:
                self.logger.warning(f"第 {attempt + 1} 次加载失败: {e}")
                
                # 如果还有镜像源可以切换，尝试切换镜像源
                if self._switch_to_next_mirror():
                    self.logger.info("切换镜像源后继续尝试")
                    continue
                
                if attempt < max_retries - 1:
                    wait_time = self.retry_delay * (2 ** attempt)  # 指数退避
                    self.logger.info(f"等待 {wait_time} 秒后重试...")
                    time.sleep(wait_time)
                else:
                    self.logger.error(f"数据集 {dataset_path} 加载失败，已尝试 {max_retries} 次")
        
        return None
    
    def download_dataset(self, dataset_name: str) -> bool:
        """
        下载指定数据集 - 增强版网络处理（支持镜像源切换）
        
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
        self.logger.info(f"当前镜像源: {self.hf_mirrors[self.current_mirror_index]}")
        
        # 首先检查网络连接（会自动测试并选择最佳镜像源）
        if not self._test_network_connectivity():
            self.logger.error("网络连接失败，无法访问任何HuggingFace镜像源")
            self.logger.info("解决方案:")
            self.logger.info("1. 检查网络连接是否正常")
            self.logger.info("2. 配置代理服务器 (设置HTTP_PROXY/HTTPS_PROXY环境变量)")
            self.logger.info("3. 使用VPN连接")
            self.logger.info("4. 检查防火墙设置")
            self.logger.info("5. 稍后重试")
            return False
        
        # 重置镜像源索引，确保从最佳镜像源开始
        self.current_mirror_index = 0
        
        try:
            # 从Hugging Face加载数据集（支持镜像源切换和重试）
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
                self.logger.info(f"最终使用的镜像源: {self.hf_mirrors[self.current_mirror_index]}")
                return True
            else:
                self.logger.error(f"数据集 {dataset_name} 保存失败")
                return False
                
        except Exception as e:
            self.logger.error(f"下载数据集 {dataset_name} 时发生错误: {e}")
            return False
    
    def _load_bigvul_dataset(self) -> Optional[Dict[str, Dataset]]:
        """加载BigVul数据集 - 增强版网络处理"""
        self.logger.info("加载 BigVul 数据集...")
        
        # 首先测试网络连接
        if not self._test_network_connectivity():
            self.logger.error("网络连接失败，无法访问 HuggingFace Hub")
            return None
        
        # 尝试加载主要数据集
        full_dataset = self._load_dataset_with_retry("bstee615/bigvul")
        if full_dataset is None:
            self.logger.warning("主要数据集 bstee615/bigvul 加载失败，尝试替代数据集...")
            
            # 尝试替代数据集 - 优先使用已验证的可用数据集
            alternatives = self._get_available_alternatives("bigvul")
            for alt_dataset in alternatives:
                self.logger.info(f"尝试替代数据集: {alt_dataset}")
                # 对于Junwei/MSR数据集，使用特定的配置
                if alt_dataset == "Junwei/MSR":
                    try:
                        # Junwei/MSR可能需要特定的配置或子集
                        full_dataset = self._load_dataset_with_retry(alt_dataset, max_retries=2)
                        if full_dataset is not None:
                            self.logger.info(f"成功加载替代数据集: {alt_dataset}")
                            break
                    except Exception as e:
                        self.logger.warning(f"Junwei/MSR加载失败: {e}, 继续尝试其他替代数据集")
                        continue
                else:
                    full_dataset = self._load_dataset_with_retry(alt_dataset, max_retries=2)
                    if full_dataset is not None:
                        self.logger.info(f"成功加载替代数据集: {alt_dataset}")
                        break
            
            if full_dataset is None:
                self.logger.error("所有数据集加载尝试均失败")
                return None
        
        try:
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
            self.logger.error(f"处理BigVul数据集时发生错误: {e}")
            return None
    
    def _load_reveal_dataset(self) -> Optional[Dict[str, Dataset]]:
        """加载Reveal数据集 - 增强版网络处理"""
        self.logger.info("加载 Reveal 数据集...")
        
        # 首先测试网络连接
        if not self._test_network_connectivity():
            self.logger.error("网络连接失败，无法访问 HuggingFace Hub")
            return None
        
        # 尝试加载主要数据集
        full_dataset = self._load_dataset_with_retry("claudios/ReVeal")
        if full_dataset is None:
            self.logger.warning("主要数据集 claudios/ReVeal 加载失败，尝试替代数据集...")
            
            # 尝试替代数据集
            alternatives = self._get_available_alternatives("reveal")
            for alt_dataset in alternatives:
                self.logger.info(f"尝试替代数据集: {alt_dataset}")
                full_dataset = self._load_dataset_with_retry(alt_dataset, max_retries=2)
                if full_dataset is not None:
                    self.logger.info(f"成功加载替代数据集: {alt_dataset}")
                    break
            
            if full_dataset is None:
                self.logger.error("所有数据集加载尝试均失败")
                return None
        
        try:
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
            self.logger.error(f"处理Reveal数据集时发生错误: {e}")
            return None
    
    def _load_devign_dataset(self) -> Optional[Dict[str, Dataset]]:
        """加载Devign数据集 - 增强版网络处理"""
        self.logger.info("加载 Devign 数据集...")
        
        # 首先测试网络连接
        if not self._test_network_connectivity():
            self.logger.error("网络连接失败，无法访问 HuggingFace Hub")
            return None
        
        # 尝试加载主要数据集
        full_dataset = self._load_dataset_with_retry("DetectVul/devign")
        if full_dataset is None:
            self.logger.warning("主要数据集 DetectVul/devign 加载失败，尝试替代数据集...")
            
            # 尝试替代数据集
            alternatives = self._get_available_alternatives("devign")
            for alt_dataset in alternatives:
                self.logger.info(f"尝试替代数据集: {alt_dataset}")
                full_dataset = self._load_dataset_with_retry(alt_dataset, max_retries=2)
                if full_dataset is not None:
                    self.logger.info(f"成功加载替代数据集: {alt_dataset}")
                    break
            
            if full_dataset is None:
                self.logger.error("所有数据集加载尝试均失败")
                return None
        
        try:
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
            self.logger.error(f"处理Devign数据集时发生错误: {e}")
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
    
    def check_network_and_provide_solutions(self) -> Dict:
        """
        检查网络连接和HuggingFace Hub访问，并提供解决方案（支持镜像源）
        
        Returns:
            Dict: 检查结果和解决方案
        """
        result = {
            "status": "",
            "hf_access": "",
            "mirrors_status": {},
            "solutions": []
        }
        
        # 测试所有镜像源的连接状态
        mirror_status = {}
        for mirror in self.hf_mirrors:
            try:
                response = requests.get(mirror, timeout=10)
                if response.status_code == 200:
                    mirror_status[mirror] = "可访问"
                else:
                    mirror_status[mirror] = f"HTTP {response.status_code}"
            except Exception as e:
                mirror_status[mirror] = f"连接失败: {str(e)}"
        
        result["mirrors_status"] = mirror_status
        
        # 检查是否有可用的镜像源
        available_mirrors = [m for m, status in mirror_status.items() if status == "可访问"]
        
        if available_mirrors:
            result["status"] = f"网络连接正常，{len(available_mirrors)}个镜像源可用"
            result["hf_access"] = f"HuggingFace Hub访问正常 (使用镜像源: {available_mirrors[0]})"
        else:
            result["status"] = "网络连接失败，所有镜像源均不可用"
            result["hf_access"] = "HuggingFace Hub访问失败"
            
            result["solutions"].extend([
                "检查网络连接是否正常",
                "配置代理服务器 (设置HTTP_PROXY/HTTPS_PROXY环境变量)",
                "使用VPN连接",
                "检查防火墙设置",
                "检查DNS设置",
                "尝试重启网络设备",
                "联系网络管理员",
                "稍后重试"
            ])
        
        # 添加镜像源配置建议
        result["solutions"].extend([
            "当前配置的镜像源:" + ", ".join(self.hf_mirrors),
            "如需添加更多镜像源，可修改HF_MIRRORS列表",
            "设置环境变量: export HF_ENDPOINT=https://hf-mirror.com",
            "设置环境变量: export HF_HUB_ENABLE_HF_TRANSFER=1"
        ])
        
        return result
    
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
            # Devign数据集的标签可能是列表格式，需要特殊处理
            try:
                # 首先尝试直接转换为整数
                df['label'] = df['label'].astype(int)
            except (ValueError, TypeError):
                # 如果转换失败，可能是列表格式，取第一个元素
                self.logger.warning("Devign标签列包含列表格式，进行特殊处理")
                df['label'] = df['label'].apply(lambda x: int(x[0]) if isinstance(x, list) and len(x) > 0 else int(x))
        
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
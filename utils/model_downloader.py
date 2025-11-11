"""
模型下载器模块
负责从Hugging Face自动下载和管理预训练模型
"""

import os
import logging
import hashlib
import time
import requests
from pathlib import Path
from typing import Optional, Dict, Any, List
from huggingface_hub import hf_hub_download, snapshot_download, login, HfApi
import torch
from transformers import AutoTokenizer, AutoModel

import os
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from config.config import Config

logger = logging.getLogger(__name__)

class ModelDownloader:
    """模型下载和管理器 - 增强版网络处理"""
    
    def __init__(self, config: Optional[Config] = None, hf_token: Optional[str] = None):
        if config is None:
            config = Config()
        self.config = config
        self.model_dir = self.config.models_dir
        self.model_dir.mkdir(exist_ok=True)
        self.hf_token = hf_token
        self.max_retries = 3
        self.retry_delay = 5
        self.api = HfApi()
        
        if hf_token:
            try:
                login(token=hf_token)
                logger.info("Hugging Face登录成功")
            except Exception as e:
                logger.warning(f"Hugging Face登录失败: {e}")
    
    def _test_network_connectivity(self) -> bool:
        """测试网络连接性"""
        try:
            # 测试基本网络连接
            response = requests.get("https://huggingface.co", timeout=10)
            return response.status_code == 200
        except Exception as e:
            logger.warning(f"网络连接测试失败: {e}")
            return False
    
    def _get_available_alternatives(self, model_name: str) -> List[str]:
        """获取可用的替代模型"""
        alternatives = [
            "microsoft/codebert-base",
            "microsoft/graphcodebert-base", 
            "codet5-base",
            "codet5-small",
            "Salesforce/codet5-base",
            "huggingface/CodeBERTa-small-v1"
        ]
        
        # 过滤掉当前模型
        return [m for m in alternatives if m != model_name]
    
    def _download_with_retry(self, model_name: str, force: bool = False) -> Optional[str]:
        """带重试机制的下载"""
        model_path = self.model_dir / model_name
        
        for attempt in range(self.max_retries):
            try:
                logger.info(f"下载尝试 {attempt + 1}/{self.max_retries}: {model_name}")
                
                # 检查网络连接
                if attempt > 0:  # 重试时检查网络
                    if not self._test_network_connectivity():
                        logger.warning("网络连接不可用，等待后重试...")
                        time.sleep(self.retry_delay)
                        continue
                
                # 尝试下载
                snapshot_download(
                    repo_id=model_name,
                    local_dir=str(model_path),
                    local_dir_use_symlinks=False,
                    token=self.hf_token,
                    resume_download=True  # 启用断点续传
                )
                
                # 验证下载
                if self._verify_model(model_path, model_name):
                    logger.info(f"模型下载成功: {model_path}")
                    return str(model_path)
                else:
                    logger.warning("模型下载验证失败")
                    
            except Exception as e:
                logger.warning(f"下载尝试 {attempt + 1} 失败: {e}")
                if attempt < self.max_retries - 1:
                    logger.info(f"等待 {self.retry_delay} 秒后重试...")
                    time.sleep(self.retry_delay)
                else:
                    logger.error(f"所有重试尝试失败: {model_name}")
        
        return None
    
    def download_model(self, model_name: str, force: bool = False) -> Optional[str]:
        """
        下载预训练模型 - 增强版
        
        Args:
            model_name: 模型名称
            force: 是否强制重新下载
        
        Returns:
            本地模型路径，失败时返回None
        """
        model_path = self.model_dir / model_name
        
        if model_path.exists() and not force:
            if self._verify_model(model_path, model_name):
                logger.info(f"模型已存在且有效: {model_path}")
                return str(model_path)
            else:
                logger.warning("模型存在但验证失败，将重新下载")
        
        # 首先尝试下载目标模型
        result = self._download_with_retry(model_name, force)
        
        # 如果下载失败，尝试替代模型
        if result is None:
            logger.warning(f"下载模型 {model_name} 失败，尝试替代模型")
            alternatives = self._get_available_alternatives(model_name)
            
            for alt_model in alternatives:
                alt_path = self.model_dir / alt_model
                if alt_path.exists() and self._verify_model(alt_path, alt_model):
                    logger.info(f"使用现有替代模型: {alt_model}")
                    return str(alt_path)
                
                logger.info(f"尝试下载替代模型: {alt_model}")
                alt_result = self._download_with_retry(alt_model, force)
                if alt_result:
                    logger.info(f"替代模型下载成功: {alt_model}")
                    # 可选：创建符号链接或复制到目标名称
                    return alt_result
        
        return result
    
    def download_tokenizer(self, model_name: str, force: bool = False) -> Optional[str]:
        """
        下载分词器
        
        Args:
            model_name: 模型名称
            force: 是否强制重新下载
        
        Returns:
            本地分词器路径，失败时返回None
        """
        return self.download_model(model_name, force)
    
    def load_model(self, model_name: str, 
                  device: str = "auto", 
                  return_tokenizer: bool = True) -> Optional[Dict[str, Any]]:
        """
        加载模型到内存
        
        Args:
            model_name: 模型名称
            device: 设备类型
            return_tokenizer: 是否同时返回分词器
        
        Returns:
            包含模型和分词器的字典
        """
        try:
            # 下载模型（如果需要）
            model_path = self.download_model(model_name)
            if not model_path:
                return None
            
            # 决定设备
            if device == "auto":
                device = "cuda" if torch.cuda.is_available() else "cpu"
            
            logger.info(f"加载模型到设备: {device}")
            
            # 加载模型
            model = AutoModel.from_pretrained(model_path).to(device)
            model.eval()
            
            result = {"model": model}
            
            if return_tokenizer:
                tokenizer = AutoTokenizer.from_pretrained(model_path)
                result["tokenizer"] = tokenizer
            
            logger.info(f"模型加载成功: {model_name}")
            return result
            
        except Exception as e:
            logger.error(f"加载模型失败 {model_name}: {e}")
            return None
    
    def get_model_info(self, model_name: str) -> Optional[Dict[str, Any]]:
        """获取模型信息"""
        try:
            model_path = self.model_dir / model_name
            
            if not model_path.exists():
                return None
            
            # 获取文件信息
            file_info = {}
            for file_path in model_path.rglob("*"):
                if file_path.is_file():
                    relative_path = file_path.relative_to(model_path)
                    file_size = file_path.stat().st_size
                    file_info[str(relative_path)] = file_size
            
            # 计算目录大小
            total_size = sum(file_info.values())
            
            return {
                "name": model_name,
                "path": str(model_path),
                "exists": True,
                "file_count": len(file_info),
                "total_size_mb": round(total_size / 1024 / 1024, 2),
                "files": file_info
            }
            
        except Exception as e:
            logger.error(f"获取模型信息失败: {e}")
            return None
    
    def list_downloaded_models(self) -> Dict[str, Dict[str, Any]]:
        """列出已下载的模型"""
        models = {}
        
        try:
            for model_dir in self.model_dir.iterdir():
                if model_dir.is_dir():
                    model_name = model_dir.name
                    model_info = self.get_model_info(model_name)
                    if model_info:
                        models[model_name] = model_info
            
            return models
            
        except Exception as e:
            logger.error(f"列出模型失败: {e}")
            return {}
    
    def delete_model(self, model_name: str) -> bool:
        """删除下载的模型"""
        try:
            model_path = self.model_dir / model_name
            
            if not model_path.exists():
                logger.warning(f"模型不存在: {model_name}")
                return False
            
            import shutil
            shutil.rmtree(model_path)
            logger.info(f"模型已删除: {model_name}")
            return True
            
        except Exception as e:
            logger.error(f"删除模型失败 {model_name}: {e}")
            return False
    
    def clear_cache(self) -> bool:
        """清理所有缓存的模型"""
        try:
            import shutil
            shutil.rmtree(self.model_dir)
            self.model_dir.mkdir(exist_ok=True)
            logger.info("模型缓存已清理")
            return True
            
        except Exception as e:
            logger.error(f"清理缓存失败: {e}")
            return False
    
    def _verify_model(self, model_path: Path, model_name: str) -> bool:
        """验证模型文件完整性"""
        try:
            # 检查关键文件是否存在
            required_files = ["config.json"]
            for file_name in required_files:
                if not (model_path / file_name).exists():
                    return False
            
            # 检查模型文件
            model_files = list(model_path.glob("pytorch_model*.bin")) + \
                         list(model_path.glob("model*.safetensors"))
            
            if not model_files:
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"验证模型文件时出错: {e}")
            return False

class ModelManager:
    """模型管理器 - 更高层次的模型管理接口"""
    
    def __init__(self, config: Optional[Config] = None, hf_token: Optional[str] = None):
        if config is None:
            config = Config()
        self.config = config
        self.downloader = ModelDownloader(config, hf_token)
        self.loaded_models = {}
        self.model_config = self.config.model_config
    
    def ensure_model_available(self, model_name: str) -> bool:
        """确保模型可用"""
        if model_name in self.loaded_models:
            return True
        
        # 尝试下载
        model_path = self.downloader.download_model(model_name)
        return model_path is not None
    
    def get_model(self, model_name: str) -> Optional[Dict[str, Any]]:
        """获取已加载的模型"""
        return self.loaded_models.get(model_name)
    
    def load_model_if_needed(self, model_name: str) -> bool:
        """如果需要则加载模型"""
        if model_name in self.loaded_models:
            return True
        
        model_data = self.load_model(model_name)
        if model_data:
            self.loaded_models[model_name] = model_data
            return True
        
        return False
    
    def load_model(self, model_name: str) -> Optional[Dict[str, Any]]:
        """加载模型"""
        try:
            logger.info(f"加载模型: {model_name}")
            model_data = self.downloader.load_model(
                model_name, 
                return_tokenizer=True
            )
            return model_data
        except Exception as e:
            logger.error(f"加载模型失败: {e}")
            return None
    
    def unload_model(self, model_name: str) -> bool:
        """卸载模型"""
        try:
            if model_name in self.loaded_models:
                del self.loaded_models[model_name]
                logger.info(f"模型已卸载: {model_name}")
                return True
            return False
        except Exception as e:
            logger.error(f"卸载模型失败: {e}")
            return False
    
    def get_model_size_info(self, model_name: str) -> Optional[Dict[str, float]]:
        """获取模型大小信息（MB）"""
        model_info = self.downloader.get_model_info(model_name)
        if model_info:
            return {
                "total_size_mb": model_info["total_size_mb"],
                "file_count": model_info["file_count"]
            }
        return None

# 全局模型管理器实例
_model_manager = None

def get_model_manager(config: Optional[Config] = None, hf_token: Optional[str] = None) -> ModelManager:
    """获取全局模型管理器"""
    global _model_manager
    if _model_manager is None:
        _model_manager = ModelManager(config, hf_token)
    return _model_manager

def download_default_model(model_name: str = None, force: bool = False) -> bool:
    """下载模型
    
    Args:
        model_name: 模型名称，如果为None则使用默认模型
        force: 是否强制重新下载
        
    Returns:
        bool: 下载是否成功
    """
    config = Config()
    if model_name is None:
        model_name = config.model_config.get("default_model", "microsoft/codebert-base")
    downloader = ModelDownloader(config)
    model_path = downloader.download_model(model_name, force=force)
    return model_path is not None

def list_all_models() -> Dict[str, Dict[str, Any]]:
    """列出所有模型"""
    config = Config()
    downloader = ModelDownloader(config)
    return downloader.list_downloaded_models()

def clear_model_cache() -> bool:
    """清理模型缓存"""
    config = Config()
    downloader = ModelDownloader(config)
    return downloader.clear_cache()

# 便捷函数
def check_model_exists(model_name: str) -> bool:
    """检查模型是否存在
    
    Args:
        model_name: 模型名称
        
    Returns:
        bool: 模型是否存在
    """
    try:
        config = Config()
        model_path = config.models_dir / model_name
        return model_path.exists() and any(model_path.iterdir())
    except Exception:
        return False

def ensure_codebert_available() -> bool:
    """确保CodeBERT模型可用"""
    config = Config()
    model_manager = get_model_manager(config)
    return model_manager.ensure_model_available("microsoft/codebert-base")

def check_network_and_provide_solutions() -> Dict[str, str]:
    """检查网络连接并提供解决方案"""
    solutions = {}
    
    # 检查基本网络连接
    try:
        response = requests.get("https://huggingface.co", timeout=10)
        if response.status_code == 200:
            solutions["status"] = "网络连接正常"
        else:
            solutions["status"] = f"网络连接异常，状态码: {response.status_code}"
    except Exception as e:
        solutions["status"] = f"网络连接失败: {e}"
        solutions["solutions"] = [
            "1. 检查网络连接",
            "2. 配置代理（如需要）: export https_proxy=http://proxy:port",
            "3. 尝试使用HuggingFace镜像站点",
            "4. 检查防火墙设置"
        ]
    
    # 检查HuggingFace Hub访问
    try:
        api = HfApi()
        # 尝试获取模型信息来测试API访问
        model_info = api.model_info("microsoft/codebert-base", timeout=10)
        solutions["hf_access"] = "HuggingFace Hub访问正常"
    except Exception as e:
        solutions["hf_access"] = f"HuggingFace Hub访问失败: {e}"
        solutions["hf_solutions"] = [
            "1. 尝试设置HuggingFace镜像: export HF_ENDPOINT=https://hf-mirror.com",
            "2. 使用HuggingFace Token认证",
            "3. 清理缓存: rm -rf ~/.cache/huggingface/",
            "4. 升级huggingface_hub: pip install --upgrade huggingface_hub"
        ]
    
    return solutions
"""
模型管理模块
负责模型下载、加载和推理
"""

import os
import torch
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from transformers import (
    AutoTokenizer, 
    AutoModel, 
    AutoModelForCausalLM,
    pipeline
)
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# 使用配置实例，但避免循环导入问题
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.config import Config

# 创建配置实例但使用后续传递的方式
_global_config = None

def set_config(config_instance):
    """设置全局配置实例"""
    global _global_config
    _global_config = config_instance
    
def get_config():
    """获取全局配置实例"""
    return _global_config

logger = logging.getLogger(__name__)

class LocalVulnerabilityDetector:
    """本地漏洞检测器"""
    
    def __init__(self, config: Config, model_name: Optional[str] = None, 
                 embedding_model: Optional[str] = None):
        """
        初始化本地模型
        
        Args:
            config: 配置对象
            model_name: 主要推理模型名称
            embedding_model: 嵌入模型名称  
        """
        self.config = config
        self.model_name = model_name or config.model_name
        self.embedding_model = embedding_model or config.embedding_model
        self.device = config.device
        
        # 初始化组件
        self.tokenizer = None
        self.model = None
        self.embedding_model_obj = None
        self.faiss_index = None
        self.id2text = None
        
        # 加载模型
        self._load_models()
    
    def _load_models(self):
        """加载所有必要的模型"""
        logger.info(f"正在加载主要模型: {self.model_name}")
        
        # 加载主要模型和tokenizer
        try:
            # 检查模型是否已下载
            model_path = self.config.get_model_path(self.model_name)
            
            if model_path.exists() and list(model_path.iterdir()):
                logger.info(f"从本地路径加载模型: {model_path}")
                self.tokenizer = AutoTokenizer.from_pretrained(model_path)
                self.model = AutoModelForCausalLM.from_pretrained(model_path)
            else:
                logger.info(f"从HuggingFace下载模型: {self.model_name}")
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                self.model = AutoModelForCausalLM.from_pretrained(self.model_name)
                
                # 保存到本地
                model_path.mkdir(parents=True, exist_ok=True)
                self.model.save_pretrained(model_path)
                self.tokenizer.save_pretrained(model_path)
                logger.info(f"模型已保存到: {model_path}")
            
            self.model.to(self.device)
            
            # 添加pad token
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                
        except Exception as e:
            logger.error(f"加载主要模型失败: {e}")
            raise
        
        # 加载嵌入模型
        try:
            logger.info(f"正在加载嵌入模型: {self.embedding_model}")
            self.embedding_model_obj = SentenceTransformer(self.embedding_model)
        except Exception as e:
            logger.error(f"加载嵌入模型失败: {e}")
            raise
    
    def encode_text(self, text: str) -> np.ndarray:
        """编码文本为向量"""
        if isinstance(text, list):
            embeddings = self.embedding_model_obj.encode(text)
        else:
            embeddings = self.embedding_model_obj.encode([text])
        return embeddings
    
    def build_faiss_index(self, texts: List[str]):
        """构建FAISS索引"""
        logger.info(f"为 {len(texts)} 个文本构建FAISS索引")
        
        # 编码所有文本
        embeddings = self.encode_text(texts)
        embeddings = np.array(embeddings, dtype='float32')
        
        # 构建索引
        dimension = embeddings.shape[1]
        quantizer = faiss.IndexFlatIP(dimension)
        index = faiss.IndexIVFFlat(quantizer, dimension, 
                                   min(self.config.faiss_nlist, len(texts)))
        
        # 训练索引
        if len(texts) > 1:
            index.train(embeddings)
        
        # 添加向量
        ids = np.array(range(len(texts)), dtype='int64')
        index.add_with_ids(embeddings, ids)
        index.nprobe = 1  # 设置搜索的cluster数量
        
        self.faiss_index = index
        self.id2text = {idx: text for idx, text in enumerate(texts)}
        
        logger.info("FAISS索引构建完成")
    
    def search_similar(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """搜索相似文本"""
        if self.faiss_index is None:
            raise ValueError("请先构建FAISS索引")
        
        # 编码查询
        query_embedding = self.encode_text([query])
        query_embedding = np.array(query_embedding, dtype='float32')
        
        # 搜索
        scores, indices = self.faiss_index.search(query_embedding, top_k)
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx != -1:  # 有效的索引
                results.append({
                    'text': self.id2text[idx],
                    'score': float(score),
                    'index': int(idx)
                })
        
        return results
    
    def generate_prediction(self, prompt: str, max_length: int = 100) -> str:
        """使用本地模型生成预测"""
        try:
            # 编码输入
            inputs = self.tokenizer(prompt, return_tensors="pt", 
                                  truncation=True, max_length=self.config.max_length)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # 生成
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_length=max_length,
                    num_return_sequences=1,
                    temperature=0.1,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            # 解码输出
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # 提取模型响应部分
            if prompt in response:
                response = response.split(prompt, 1)[1].strip()
            
            return response
            
        except Exception as e:
            logger.error(f"生成预测失败: {e}")
            return "生成失败"
    
    def get_vulnerability_prediction(self, code: str, 
                                   node_info: str = "",
                                   edge_info: str = "",
                                   example: str = "",
                                   use_graph: bool = True) -> str:
        """获取漏洞预测"""
        
        if use_graph and node_info and edge_info:
            prompt = self.config.prompt_templates["with_graph"].format(
                node_info=node_info,
                edge_info=edge_info,
                example=example
            )
        else:
            prompt = self.config.prompt_templates["basic"]
        
        # 组合完整提示
        full_prompt = f"{code}\n\n{prompt}"
        
        # 生成预测
        prediction = self.generate_prediction(full_prompt)
        
        # 简单的结果解析
        if "vulnerable" in prediction.lower():
            return "1"  # Vulnerable
        elif "non-vulnerable" in prediction.lower() or "safe" in prediction.lower():
            return "0"  # Non-vulnerable
        else:
            # 尝试从数字判断
            if "1" in prediction and prediction.count("1") > prediction.count("0"):
                return "1"
            elif "0" in prediction and prediction.count("0") > prediction.count("1"):
                return "0"
            else:
                return "2"  # 不确定

class CodeRetriever:
    """代码检索器"""
    
    def __init__(self, detector: LocalVulnerabilityDetector):
        self.detector = detector
        self.examples_cache = {}
    
    def get_similar_examples(self, code: str, ast: str, 
                           top_k: int = 5,
                           code_weight: float = 0.7,
                           ast_weight: float = 0.3) -> List[Dict[str, Any]]:
        """获取相似示例"""
        
        # 缓存键
        cache_key = f"{hash(code)}_{hash(ast)}_{top_k}"
        if cache_key in self.examples_cache:
            return self.examples_cache[cache_key]
        
        # 搜索代码相似度
        code_results = self.detector.search_similar(code, top_k * 2)
        
        # 计算AST相似度（简化版）
        results = []
        for result in code_results[:top_k]:
            # 简单的AST相似度计算
            ast_sim = self._calculate_ast_similarity(ast, result['text'])
            
            # 综合分数
            final_score = code_weight * result['score'] + ast_weight * ast_sim
            
            results.append({
                'code': result['text'],
                'score': final_score,
                'code_similarity': result['score'],
                'ast_similarity': ast_sim
            })
        
        # 按分数排序
        results.sort(key=lambda x: x['score'], reverse=True)
        results = results[:top_k]
        
        # 缓存结果
        self.examples_cache[cache_key] = results
        
        return results
    
    def _calculate_ast_similarity(self, ast1: str, code2: str) -> float:
        """计算AST相似度（简化版）"""
        try:
            # 这里实现简化的AST相似度计算
            # 实际应用中应该使用更复杂的方法
            words1 = set(ast1.lower().split())
            words2 = set(code2.lower().split())
            
            intersection = words1.intersection(words2)
            union = words1.union(words2)
            
            if len(union) == 0:
                return 0.0
            
            return len(intersection) / len(union)
            
        except Exception:
            return 0.0

def create_detector(model_name: Optional[str] = None) -> LocalVulnerabilityDetector:
    """创建漏洞检测器的工厂函数"""
    return LocalVulnerabilityDetector(model_name)

def create_retriever(detector: LocalVulnerabilityDetector) -> CodeRetriever:
    """创建代码检索器的工厂函数"""
    return CodeRetriever(detector)
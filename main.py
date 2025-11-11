"""
GRACE漏洞检测主应用程序
使用本地预训练模型进行代码漏洞检测
"""

import argparse
import logging
import sys
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
import pandas as pd
from datetime import datetime

# 导入新模块化的组件
from config.config import config
from models import LocalVulnerabilityDetector, CodeRetriever
from data import DataProcessor, DatasetInfo
from utils.model_downloader import ensure_codebert_available, download_default_model
from utils.prompt_templates import create_vulnerability_prompt

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(config.log_dir / f'grace_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    ]
)
logger = logging.getLogger(__name__)

class GraceApplication:
    """GRACE漏洞检测应用程序主类"""
    
    def __init__(self):
        self.detector = None
        self.retriever = None
        self.data_processor = DataProcessor()
        self.dataset_info = DatasetInfo()
        
    def initialize(self, hf_token: Optional[str] = None) -> bool:
        """初始化应用程序"""
        try:
            logger.info("正在初始化GRACE应用程序...")
            
            # 确保模型目录存在
            config.model_dir.mkdir(exist_ok=True)
            config.log_dir.mkdir(exist_ok=True)
            
            # 下载并加载默认模型
            logger.info("正在准备预训练模型...")
            if not ensure_codebert_available():
                logger.warning("模型下载失败，尝试重新下载...")
                if not download_default_model():
                    logger.error("模型准备失败")
                    return False
            
            # 初始化检测器
            logger.info("正在初始化漏洞检测器...")
            self.detector = LocalVulnerabilityDetector(
                model_name="microsoft/codebert-base",
                hf_token=hf_token
            )
            
            # 初始化代码检索器
            logger.info("正在初始化代码检索器...")
            self.retriever = CodeRetriever(
                model_name="microsoft/codebert-base"
            )
            
            logger.info("GRACE应用程序初始化完成")
            return True
            
        except Exception as e:
            logger.error(f"初始化失败: {e}")
            return False
    
    def run_evaluation(self, dataset_name: str, split: str = "test", 
                      k_examples: int = 3) -> Dict[str, Any]:
        """运行评估"""
        try:
            logger.info(f"开始评估数据集: {dataset_name}")
            
            # 加载数据
            data = self.data_processor.load_dataset(dataset_name, split)
            if not data:
                logger.error(f"无法加载数据: {dataset_name}")
                return {}
            
            # 准备训练数据用于构建索引
            logger.info("正在构建检索索引...")
            train_codes, train_asts = self.data_processor.prepare_training_data(dataset_name)
            
            if train_codes and train_asts:
                self.retriever.build_index(train_codes, train_asts)
                logger.info("检索索引构建完成")
            
            # 处理数据
            results = []
            total = len(data)
            
            for i, item in enumerate(data):
                if i % 100 == 0:
                    logger.info(f"处理进度: {i}/{total}")
                
                try:
                    # 准备输入
                    code = item.get('func', '')
                    target = item.get('target', 0)
                    node_info = item.get('node', '')
                    edge_info = item.get('edge', '')
                    
                    # 获取相关示例
                    examples = []
                    if self.retriever and len(train_codes) > 0:
                        similar_codes, similar_asts = self.retriever.retrieve_similar(code, k=k_examples)
                        for j, (similar_code, similar_ast) in enumerate(zip(similar_codes, similar_asts)):
                            examples.append({
                                'code': similar_code,
                                'ast': similar_ast,
                                'vulnerability': '相关代码示例' if j < k_examples else ''
                            })
                    
                    # 创建提示
                    prompt = create_vulnerability_prompt(
                        code=code,
                        context="",
                        node_info=node_info,
                        edge_info=edge_info
                    )
                    
                    # 预测
                    prediction = self.detector.predict_vulnerability(prompt)
                    
                    result = {
                        'index': i,
                        'code': code[:200] + '...' if len(code) > 200 else code,
                        'target': target,
                        'prediction': prediction,
                        'node_info': node_info[:100] + '...' if len(node_info) > 100 else node_info,
                        'edge_info': edge_info[:100] + '...' if len(edge_info) > 100 else edge_info
                    }
                    results.append(result)
                    
                except Exception as e:
                    logger.warning(f"处理样本 {i} 时出错: {e}")
                    continue
            
            # 计算评估指标
            metrics = self._calculate_metrics(results)
            
            # 保存结果
            self._save_results(results, metrics, dataset_name, split)
            
            logger.info(f"评估完成 - {dataset_name}: {metrics}")
            return metrics
            
        except Exception as e:
            logger.error(f"评估失败: {e}")
            return {}
    
    def _calculate_metrics(self, results: List[Dict[str, Any]]) -> Dict[str, float]:
        """计算评估指标"""
        try:
            if not results:
                return {}
            
            y_true = []
            y_pred = []
            
            for result in results:
                target = result.get('target', 0)
                prediction = result.get('prediction', {})
                
                # 假设prediction包含has_vulnerability布尔值
                if isinstance(prediction, dict):
                    has_vuln = prediction.get('has_vulnerability', False)
                    pred_label = 1 if has_vuln else 0
                else:
                    pred_label = 0
                
                y_true.append(target)
                y_pred.append(pred_label)
            
            # 计算指标
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
            
            accuracy = accuracy_score(y_true, y_pred)
            precision = precision_score(y_true, y_pred, average='binary', zero_division=0)
            recall = recall_score(y_true, y_pred, average='binary', zero_division=0)
            f1 = f1_score(y_true, y_pred, average='binary', zero_division=0)
            
            return {
                'accuracy': round(accuracy, 4),
                'precision': round(precision, 4),
                'recall': round(recall, 4),
                'f1': round(f1, 4),
                'total_samples': len(results)
            }
            
        except Exception as e:
            logger.error(f"计算指标失败: {e}")
            return {}
    
    def _save_results(self, results: List[Dict], metrics: Dict[str, Any], 
                     dataset_name: str, split: str):
        """保存结果"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # 保存详细结果
            results_path = config.output_dir / f"{dataset_name}_{split}_results_{timestamp}.json"
            with open(results_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            
            # 保存评估指标
            metrics_path = config.output_dir / f"{dataset_name}_{split}_metrics_{timestamp}.json"
            with open(metrics_path, 'w', encoding='utf-8') as f:
                json.dump(metrics, f, ensure_ascii=False, indent=2)
            
            # 保存CSV格式的指标
            csv_path = config.output_dir / f"{dataset_name}_{split}_metrics_{timestamp}.csv"
            pd.DataFrame([metrics]).to_csv(csv_path, index=False)
            
            logger.info(f"结果已保存到: {config.output_dir}")
            
        except Exception as e:
            logger.error(f"保存结果失败: {e}")
    
    def run_interactive(self):
        """运行交互模式"""
        print("GRACE漏洞检测系统 - 交互模式")
        print("输入 'quit' 退出")
        
        while True:
            try:
                code = input("\\n请输入要检测的代码: ").strip()
                if code.lower() == 'quit':
                    break
                
                if not code:
                    continue
                
                # 检测漏洞
                prompt = create_vulnerability_prompt(code)
                prediction = self.detector.predict_vulnerability(prompt)
                
                print(f"\\n检测结果:")
                print(f"漏洞判断: {'是' if prediction.get('has_vulnerability', False) else '否'}")
                print(f"置信度: {prediction.get('confidence', 0.0)}")
                print(f"漏洞类型: {prediction.get('vulnerability_type', '无')}")
                print(f"解释: {prediction.get('explanation', '无')}")
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                logger.error(f"交互模式错误: {e}")
        
        print("感谢使用GRACE!")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="GRACE漏洞检测系统")
    parser.add_argument("--mode", choices=["eval", "interactive", "download"], 
                       default="eval", help="运行模式")
    parser.add_argument("--dataset", choices=["bigvul", "reveal", "devign"], 
                       default="bigvul", help="数据集")
    parser.add_argument("--split", choices=["train", "test"], 
                       default="test", help="数据分割")
    parser.add_argument("--k-examples", type=int, default=3, 
                       help="检索示例数量")
    parser.add_argument("--hf-token", type=str, 
                       help="Hugging Face API Token（可选）")
    parser.add_argument("--download-model", action="store_true", 
                       help="仅下载模型")
    
    args = parser.parse_args()
    
    # 创建应用
    app = GraceApplication()
    
    try:
        # 初始化
        if not app.initialize(args.hf_token):
            logger.error("应用程序初始化失败")
            sys.exit(1)
        
        # 运行模式
        if args.download_model:
            logger.info("模型下载完成")
            return
        
        if args.mode == "eval":
            # 评估模式
            metrics = app.run_evaluation(
                dataset_name=args.dataset,
                split=args.split,
                k_examples=args.k_examples
            )
            
            if metrics:
                print(f"\\n评估结果 - {args.dataset}:")
                for key, value in metrics.items():
                    print(f"{key}: {value}")
            else:
                logger.error("评估失败")
        
        elif args.mode == "interactive":
            # 交互模式
            app.run_interactive()
        
    except Exception as e:
        logger.error(f"程序运行错误: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
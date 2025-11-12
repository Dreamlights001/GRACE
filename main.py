#!/usr/bin/env python3
"""
GRACE主程序入口
支持本地预训练模型推理和漏洞检测

使用方法:
    python main.py --download-model  # 下载预训练模型
    python main.py --mode eval --dataset bigvul  # 评估BigVul数据集
    python main.py --mode interactive  # 交互式检测
    python main.py --download-data  # 下载数据集
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime

# 添加项目路径
sys.path.append(str(Path(__file__).parent))

# 导入项目模块
from config.config import Config
from models import LocalVulnerabilityDetector, CodeRetriever
from data import DataProcessor
from utils import setup_logging, ensure_directories, check_system_requirements, save_json_safe, estimate_model_size
from utils.model_downloader import download_default_model, check_model_exists
from utils.prompt_templates import create_vulnerability_prompt

class GraceApplication:
    """GRACE应用程序主类"""
    
    def __init__(self, data_root: str = None):
        """
        初始化应用程序
        
        Args:
            data_root: 数据集存储根目录，如果为None则使用配置文件默认值
        """
        # 初始化配置
        self.config = Config(data_root=data_root) if data_root else Config()
        
        # 设置日志和目录
        setup_logging(self.config.log_level, self.config.log_format)
        ensure_directories()
        
        # 初始化组件
        self.detector: Optional[LocalVulnerabilityDetector] = None
        self.retriever: Optional[CodeRetriever] = None
        self.data_processor: Optional[DataProcessor] = None
        
        # 评估结果存储
        self.evaluation_results: Dict[str, Any] = {}
        
        print("🚀 GRACE - 基于图结构和上下文学习的漏洞检测系统")
        print(f"📁 项目根目录: {self.config.project_root}")
        print(f"📊 数据目录: {self.config.data_dir}")
        print(f"🤖 模型目录: {self.config.models_dir}")
        print(f"💻 设备: {self.config.device}")
    
    def initialize_model(self, model_name: str = None) -> bool:
        """
        初始化模型
        
        Args:
            model_name: 模型名称，如果为None则使用配置默认值
            
        Returns:
            bool: 初始化是否成功
        """
        try:
            if model_name is None:
                model_name = self.config.model_name
            
            print(f"🔄 初始化模型: {model_name}")
            
            # 检查模型是否存在
            if not check_model_exists(model_name):
                print(f"❌ 模型 {model_name} 不存在")
                print("请运行: python main.py --download-model")
                return False
            
            # 初始化检测器
            self.detector = LocalVulnerabilityDetector(
                config=self.config,
                model_name=model_name
            )
            
            print("✅ 模型初始化成功")
            return True
            
        except Exception as e:
            print(f"❌ 模型初始化失败: {e}")
            return False
    
    def initialize_components(self, model_name: str = None) -> bool:
        """
        初始化所有组件
        
        Args:
            model_name: 模型名称
            
        Returns:
            bool: 初始化是否成功
        """
        try:
            # 初始化模型
            if not self.initialize_model(model_name):
                return False
            
            # 初始化数据处理器
            print("🔄 初始化数据处理器...")
            self.data_processor = DataProcessor()
            print("✅ 数据处理器初始化成功")
            
            # 初始化代码检索器
            print("🔄 初始化代码检索器...")
            self.retriever = CodeRetriever(self.detector)
            print("✅ 代码检索器初始化成功")
            
            return True
            
        except Exception as e:
            print(f"❌ 组件初始化失败: {e}")
            return False
    
    def download_model(self, model_name: str = None) -> bool:
        """
        下载预训练模型
        
        Args:
            model_name: 模型名称
            
        Returns:
            bool: 下载是否成功
        """
        try:
            if model_name is None:
                model_name = self.config.model_name
            
            print(f"🔄 开始下载模型: {model_name}")
            
            # 下载模型
            success = download_default_model(model_name)
            
            if success:
                print("✅ 模型下载成功")
                return True
            else:
                print("❌ 模型下载失败")
                return False
                
        except Exception as e:
            print(f"❌ 下载模型时发生错误: {e}")
            return False
    
    def run_evaluation(self, dataset_name: str, split: str = "test", 
                      output_file: str = None) -> bool:
        """
        运行数据集评估
        
        Args:
            dataset_name: 数据集名称
            split: 数据集分割
            output_file: 输出文件名
            
        Returns:
            bool: 评估是否成功
        """
        try:
            if self.detector is None:
                print("❌ 模型未初始化，请先运行 --download-model")
                return False
            
            print(f"🔄 开始评估数据集: {dataset_name} ({split})")
            
            # 加载数据
            if not self.data_processor.load_dataset(dataset_name, split):
                print(f"❌ 加载数据集 {dataset_name} 失败")
                return False
            
            # 获取数据
            data_items = self.data_processor.get_data_items()
            if not data_items:
                print(f"❌ 数据集 {dataset_name} 为空")
                return False
            
            print(f"📊 数据集大小: {len(data_items)} 条样本")
            
            # 评估设置
            total_samples = len(data_items)
            batch_size = self.config.eval_batch_size
            true_labels = []
            predictions = []
            confidences = []
            
            # 评估模型
            for i, item in enumerate(data_items):
                if i % 100 == 0:
                    print(f"📈 进度: {i}/{total_samples} ({i/total_samples*100:.1f}%)")
                
                # 获取代码
                code = item.get('code', '')
                if not code:
                    continue
                
                # 创建提示
                prompt = create_vulnerability_prompt(code=code)
                
                # 预测
                result = self.detector.predict_vulnerability(prompt)
                
                # 记录结果
                true_label = item.get('label', 0)
                pred_label = 1 if result.get('has_vulnerability', False) else 0
                confidence = result.get('confidence', 0.0)
                
                true_labels.append(true_label)
                predictions.append(pred_label)
                confidences.append(confidence)
            
            # 计算指标
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
            
            accuracy = accuracy_score(true_labels, predictions)
            precision = precision_score(true_labels, predictions, average='binary', zero_division=0)
            recall = recall_score(true_labels, predictions, average='binary', zero_division=0)
            f1 = f1_score(true_labels, predictions, average='binary', zero_division=0)
            
            # 保存结果
            results = {
                'dataset': dataset_name,
                'split': split,
                'total_samples': total_samples,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'timestamp': datetime.now().isoformat(),
                'model_name': self.config.model_name,
                'predictions': predictions,
                'true_labels': true_labels,
                'confidences': confidences
            }
            
            # 打印结果
            print(f"\n📊 评估结果 - {dataset_name} ({split}):")
            print(f"   Accuracy:  {accuracy:.4f}")
            print(f"   Precision: {precision:.4f}")
            print(f"   Recall:    {recall:.4f}")
            print(f"   F1-Score:  {f1:.4f}")
            
            # 保存到文件
            if output_file is None:
                output_file = f"{dataset_name}metrics{self.config.model_name.split('/')[-1]}.log"
            
            output_path = self.config.get_output_path(output_file)
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(f"Dataset: {dataset_name} ({split})\n")
                f.write(f"Model: {self.config.model_name}\n")
                f.write(f"Total Samples: {total_samples}\n")
                f.write(f"Accuracy: {accuracy:.4f}\n")
                f.write(f"Precision: {precision:.4f}\n")
                f.write(f"Recall: {recall:.4f}\n")
                f.write(f"F1-Score: {f1:.4f}\n")
                f.write(f"Timestamp: {datetime.now().isoformat()}\n")
            
            print(f"💾 结果已保存: {output_path}")
            
            # 存储结果
            self.evaluation_results[dataset_name] = results
            
            return True
            
        except Exception as e:
            print(f"❌ 评估失败: {e}")
            return False
    
    def run_interactive_mode(self):
        """运行交互式检测模式"""
        print("🎯 交互式漏洞检测模式")
        print("输入代码片段，系统将实时分析并提供漏洞检测结果")
        print("输入 'quit' 或 'exit' 退出\n")
        
        if self.detector is None:
            print("❌ 模型未初始化，请先运行 --download-model")
            return
        
        while True:
            try:
                # 获取用户输入
                print("请输入要检测的代码 (输入空行结束):")
                code_lines = []
                while True:
                    line = input()
                    if line.strip() == "":
                        break
                    code_lines.append(line)
                
                # 检查退出命令
                code = "\n".join(code_lines)
                if code.lower() in ['quit', 'exit', 'q']:
                    print("👋 再见！")
                    break
                
                if not code.strip():
                    print("⚠️ 代码为空，请重新输入")
                    continue
                
                # 创建提示
                prompt = create_vulnerability_prompt(code=code)
                
                # 预测
                print("🔄 分析中...")
                result = self.detector.predict_vulnerability(prompt)
                
                # 显示结果
                print("\n📊 检测结果:")
                print(f"   漏洞判断: {'是' if result.get('has_vulnerability', False) else '否'}")
                print(f"   置信度: {result.get('confidence', 0.0):.2f}")
                print(f"   漏洞类型: {result.get('vulnerability_type', '未知')}")
                print(f"   分析建议: {result.get('suggestion', '无')}")
                
                print("-" * 50)
                
            except KeyboardInterrupt:
                print("\n👋 程序被用户中断，再见！")
                break
            except Exception as e:
                print(f"❌ 处理出错: {e}")
                continue
    
    def run_data_preparation(self, dataset: str = None, data_root: str = None) -> bool:
        """
        运行数据准备流程
        
        Args:
            dataset: 要准备的数据集，None表示准备所有数据集
            data_root: 数据根目录
            
        Returns:
            bool: 数据准备是否成功
        """
        try:
            print("🔄 开始数据准备流程...")
            
            # 导入数据准备器
            from prepare_data import DataPreparator
            
            # 创建数据准备器
            preparator = DataPreparator(data_root=data_root or str(self.config.data_root))
            
            # 准备数据
            if dataset:
                preparator.run_full_preparation([dataset])
            else:
                preparator.run_full_preparation()
            
            print("✅ 数据准备完成")
            return True
            
        except Exception as e:
            print(f"❌ 数据准备失败: {e}")
            return False
    
    def check_system_status(self):
        """检查系统状态"""
        print("🔍 系统状态检查")
        print("=" * 50)
        
        # 检查目录
        print(f"📁 项目目录: {self.config.project_root}")
        print(f"📊 数据目录: {self.config.data_dir}")
        print(f"🤖 模型目录: {self.config.models_dir}")
        print(f"📄 输出目录: {self.config.output_dir}")
        
        # 检查模型
        model_exists = check_model_exists(self.config.model_name)
        model_status = "✅ 已下载" if model_exists else "❌ 未下载"
        print(f"🤖 预训练模型 {self.config.model_name}: {model_status}")
        
        # 检查系统要求
        requirements = check_system_requirements()
        print(f"💻 系统要求: {requirements}")
        
        # 检查数据状态
        try:
            from prepare_data import DataPreparator
            preparator = DataPreparator()
            preparator.print_status()
        except Exception as e:
            print(f"❌ 数据状态检查失败: {e}")
        
        print("=" * 50)
    
    def run_all_evaluations(self, datasets: List[str] = None):
        """
        运行所有数据集的评估
        
        Args:
            datasets: 要评估的数据集列表，None表示评估所有数据集
        """
        if datasets is None:
            datasets = ["bigvul", "reveal", "devign"]
        
        print(f"🔄 开始评估所有数据集: {datasets}")
        
        success_count = 0
        for dataset_name in datasets:
            print(f"\n{'=' * 20} 评估 {dataset_name} {'=' * 20}")
            
            if self.run_evaluation(dataset_name, split="test"):
                success_count += 1
                print(f"✅ {dataset_name} 评估成功")
            else:
                print(f"❌ {dataset_name} 评估失败")
        
        print(f"\n📊 评估总结: {success_count}/{len(datasets)} 成功")
        
        # 保存所有结果
        if self.evaluation_results:
            all_results_path = self.config.get_output_path("all_evaluation_results.json")
            with open(all_results_path, 'w', encoding='utf-8') as f:
                json.dump(self.evaluation_results, f, indent=2, ensure_ascii=False)
            print(f"💾 所有结果已保存: {all_results_path}")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="GRACE - 基于图结构和上下文学习的漏洞检测系统")
    
    # 基础参数
    parser.add_argument("--data-root", type=str, help="数据存储根目录")
    parser.add_argument("--model-name", type=str, help="使用的预训练模型名称")
    parser.add_argument("--device", type=str, choices=["cpu", "cuda"], help="计算设备")
    
    # 操作模式
    parser.add_argument("--download-model", action="store_true", help="下载预训练模型")
    parser.add_argument("--download-data", action="store_true", help="下载数据集")
    parser.add_argument("--check-status", action="store_true", help="检查系统状态")
    parser.add_argument("--eval-all", action="store_true", help="评估所有数据集")
    
    # 评估模式
    parser.add_argument("--mode", type=str, choices=["eval", "interactive"], help="运行模式")
    parser.add_argument("--dataset", type=str, choices=["bigvul", "reveal", "devign"], 
                       help="要评估的数据集")
    parser.add_argument("--split", type=str, default="test", choices=["train", "test", "val"],
                       help="数据集分割")
    parser.add_argument("--output", type=str, help="输出文件名")
    
    # 数据准备
    parser.add_argument("--dataset-for-data", type=str, choices=["bigvul", "reveal", "devign"],
                       help="要准备的数据集")
    
    args = parser.parse_args()
    
    # 创建应用程序
    app = GraceApplication(data_root=args.data_root)
    
    # 设置模型和设备
    if args.model_name:
        app.config.model_name = args.model_name
    if args.device:
        app.config.device = args.device
    
    try:
        # 处理各种操作
        if args.check_status:
            # 检查系统状态
            app.check_system_status()
            
        elif args.download_model:
            # 下载模型
            app.download_model(args.model_name)
            
        elif args.download_data:
            # 下载数据
            app.run_data_preparation(args.dataset_for_data, args.data_root)
            
        elif args.mode == "eval":
            # 评估模式
            if not args.dataset:
                print("❌ 评估模式需要指定 --dataset 参数")
                return
            
            if not app.initialize_components(args.model_name):
                return
            
            app.run_evaluation(args.dataset, args.split, args.output)
            
        elif args.mode == "interactive":
            # 交互式模式
            if not app.initialize_components(args.model_name):
                return
            
            app.run_interactive_mode()
            
        elif args.eval_all:
            # 评估所有数据集
            if not app.initialize_components(args.model_name):
                return
            
            app.run_all_evaluations()
            
        else:
            # 默认行为：显示帮助信息
            print("🚀 GRACE 漏洞检测系统")
            print("请指定操作模式：")
            print("  --download-model    下载预训练模型")
            print("  --download-data     下载数据集") 
            print("  --mode eval --dataset bigvul  评估数据集")
            print("  --mode interactive  交互式检测")
            print("  --eval-all          评估所有数据集")
            print("  --check-status      检查系统状态")
            print("\n示例:")
            print("  python main.py --download-model")
            print("  python main.py --mode eval --dataset bigvul")
            print("  python main.py --mode interactive")
            
    except Exception as e:
        print(f"❌ 程序运行错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
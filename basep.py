import openai
import pandas as pd
from tqdm import tqdm
import os
import json
import csv
import logging
import os
import pickle
import faiss
import torch
import numpy as np
import Levenshtein
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
import pandas as pd
from config.config import Config

openai.api_base = ""
openai.api_key = ""

templates = {
    1: 'In the above code snippet, check for potential security vulnerabilities and output either \'Vulnerable\' or \'Non-vulnerable\'. '
       'You are now an excellent programmer.'
       'You are conducting a function vulnerability detection task for C/C++ language.',
    2: 'The node information of the function is as follows:',
    3: 'The edge information of the function is as follows:',
    4: 'Here is an example for you to learn from:'
}

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

fh = logging.FileHandler('devignmetricsgpt4.log')
fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
fh.setFormatter(logging.Formatter(fmt))

logger.addHandler(fh)


def main():
    with open('F:/pycharmfile/vulllm/devign_data/devign_test_processed.json', 'r') as f:
        data = json.load(f)

    def calculate_metrics(predictions, ground_truth):
        true_positives = 0
        false_positives = 0
        false_negatives = 0
        true_negatives = 0

        for pred, target in zip(predictions, ground_truth):
            if pred == target == 1:
                true_positives += 1
            elif pred == target == 0:
                true_negatives += 1
            elif pred == 1 and target == 0:
                false_positives += 1
            elif pred == 0 and target == 1:
                false_negatives += 1

        accuracy = (true_positives + true_negatives) / len(predictions)
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        return accuracy, precision, recall, f1

    prediction_ls = []
    ground_truth = []

    for row in data[0:2000]:
        if 'func' in row:
            inputCode = row['func'][:4000]
        if 'node' in row:
            inputnode = row['node'][:2000]
        if 'edge' in row:
            inputedge = row['edge'][:2000]
        if 'func' in row:
            inputex = row['example'][:4000]


            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {"role": "user", "content": format(inputCode)+templates[1]}
                ]
            )
            prediction = response['choices'][0]['message']['content']
            print(prediction)

            with open('devignresultsgpt4.csv', 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['result'])
                writer.writerow(prediction)
                f.close()

            if prediction == "0" or prediction == "1":
                prediction = int(prediction)
            else:
                prediction = 2

            prediction_ls.append(prediction)
            ground_truth.append(row['target'])
            # print(inputCode)
            # print(inputnode)
            # print(inputedge)

        with open('devignresultsgpt4.csv', 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Prediction', 'Groundtruth'])
            writer.writerows(zip(prediction_ls, ground_truth))

        print(prediction_ls)
        print(ground_truth)

        accuracy, precision, recall, f1 = calculate_metrics(prediction_ls, ground_truth)
        print("Accuracy:", accuracy)
        print("Precision:", precision)
        print("Recall:", recall)
        print("F1 Score:", f1)

        logger.info("Accuracy: %f", accuracy)
        logger.info("Precision: %f", precision)
        logger.info("Recall: %f", recall)
        logger.info("F1 Score: %f", f1)

class ExampleGenerator:
    def __init__(self, config=None):
        if config is None:
            self.config = Config()
        else:
            self.config = config
        
        # 加载模型和分词器
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        self.model = AutoModel.from_pretrained(self.config.model_name)
        self.model.to(self.config.device)
        
        # 维度设置
        self.dim = self.model.config.hidden_size
        
        # 索引相关
        self.id2text = None
        self.vecs = None
        self.ids = None
        self.index = None
    
    def load_data(self, train_code_path, train_ast_path, test_code_path, test_ast_path):
        # 加载训练数据
        df = pd.read_csv(train_code_path, header=None)
        self.train_code_list = df[0].tolist()
        
        df = pd.read_csv(train_ast_path, header=None)
        self.train_ast_list = df[0].tolist()
        
        # 加载测试数据
        df = pd.read_csv(test_code_path, header=None)
        self.test_code_list = df[0].tolist()
        
        df = pd.read_csv(test_ast_path, header=None)
        self.test_ast_list = df[0].tolist()
    
    def encode_text(self, text):
        # 使用预训练模型编码文本
        inputs = self.tokenizer(
            text,
            padding=True,
            truncation=True,
            max_length=self.config.max_length,
            return_tensors="pt"
        ).to(self.config.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            # 使用CLS标记的输出作为文本表示
            embeddings = outputs.last_hidden_state[:, 0, :]
        
        return embeddings.cpu().numpy()
    
    def build_index(self, n_list=1):
        # 编码所有训练文本
        all_vecs = []
        for code in tqdm(self.train_code_list, desc="Encoding training data"):
            vec = self.encode_text(code)
            all_vecs.append(vec)
        
        self.vecs = np.concatenate(all_vecs, axis=0).astype("float32")
        self.ids = np.array(range(len(self.train_code_list)), dtype="int64")
        self.id2text = {idx: text for idx, text in enumerate(self.train_code_list)}
        
        # 构建FAISS索引
        quant = faiss.IndexFlatIP(self.dim)
        self.index = faiss.IndexIVFFlat(quant, self.dim, min(n_list, self.vecs.shape[0]))
        self.index.train(self.vecs)
        self.index.add_with_ids(self.vecs, self.ids)
        self.index.nprob = 1
    
    def jaccard_similarity(self, s1, s2):
        s1, s2 = set(s1), set(s2)
        intersection = s1.intersection(s2)
        union = s1.union(s2)
        return 1.0 * len(intersection) / len(union) if union else 0
    
    def find_similar_examples(self, code, ast, top_k=5):
        # 编码查询文本
        vec = self.encode_text(code)
        
        # 搜索相似文本
        _, sim_idx = self.index.search(vec, top_k)
        sim_idx = sim_idx[0].tolist()
        
        # 计算额外的相似度分数并选择最佳示例
        max_score = 0
        max_idx = 0
        
        for idx in sim_idx:
            # 代码相似度（Jaccard）
            code_score = self.jaccard_similarity(
                self.train_code_list[idx].split(), 
                code.split()
            )
            
            # AST相似度（Levenshtein）
            ast_score = Levenshtein.seqratio(
                str(self.train_ast_list[idx]).split(), 
                str(ast).split()
            )
            
            # 综合分数
            score = 0.7 * code_score + 0.3 * ast_score
            
            if score > max_score:
                max_score = score
                max_idx = idx
        
        return self.train_code_list[max_idx], self.train_ast_list[max_idx]

if __name__ == '__main__':
    config = Config()
    generator = ExampleGenerator(config)
    
    # 加载数据
    generator.load_data(
        train_code_path="data/train_function_clean.csv",
        train_ast_path="data/train_ast_clean.csv",
        test_code_path="data/test_function_clean.csv",
        test_ast_path="data/test_ast_clean.csv"
    )
    
    # 构建索引
    print("Building index...")
    generator.build_index(n_list=1)
    
    # 为测试数据生成示例
    print("Generating examples for test data...")
    similar_codes = []
    similar_asts = []
    
    for i in tqdm(range(len(generator.test_code_list))):
        sim_code, sim_ast = generator.find_similar_examples(
            generator.test_code_list[i], 
            generator.test_ast_list[i], 
            top_k=5
        )
        similar_codes.append(sim_code)
        similar_asts.append(sim_ast)
    
    # 保存结果
    os.makedirs(config.output_path, exist_ok=True)
    
    df = pd.DataFrame(similar_codes)
    df.to_csv(os.path.join(config.output_path, "similar_codes.csv"), index=False, header=None)
    
    df = pd.DataFrame(similar_asts)
    df.to_csv(os.path.join(config.output_path, "similar_asts.csv"), index=False, header=None)
    
    print("Examples generated and saved successfully!")
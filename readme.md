```
# GRACE: 基于图结构和上下文学习的本地LLM漏洞检测系统

![GRACE Logo](figs/approach.png)

GRACE (Graph structure and in-context learning Enhanced vulnerability detection) 是一个基于大语言模型的软件漏洞检测系统。本项目重构了原始实现，**完全支持本地预训练模型推理，无需任何外部API依赖**，并针对云计算平台部署进行了优化。

## 🚀 最新更新

- **☁️ 云计算平台适配** - 默认数据集路径配置为 `/root/sj-tmp/dataset/`
- **🌐 镜像源支持** - 自动切换HuggingFace镜像源，提升网络稳定性
- **🔄 智能重试机制** - 指数退避重试和网络故障自动恢复
- **🔧 代理配置支持** - 支持HTTP/HTTPS代理环境变量配置

## 🌟 主要特性

- **🔒 完全本地化** - 无需API密钥，无网络依赖
- **🚀 高效推理** - 支持CPU/GPU自动选择，优化的模型加载
- **📊 多数据集支持** - BigVul、Reveal、Devign数据集
- **🧠 图结构增强** - 基于AST和代码依赖图的漏洞分析
- **🎯 上下文学习** - 智能示例检索和相似代码匹配
- **📈 完整评估** - 支持Accuracy、Precision、Recall、F1等指标
- **💻 交互式检测** - 实时代码漏洞检测
- **☁️ 云计算优化** - 针对云平台部署的路径和网络配置
- **🌐 镜像源支持** - 自动切换HuggingFace镜像源，提升稳定性

## 🏗️ 项目结构

```
GRACE/
├── main.py                 # 🎯 主程序入口
├── config/
│   └── config.py          # ⚙️ 完整配置管理
├── models/
│   └── __init__.py        # 🤖 本地模型接口
├── data/
│   └── __init__.py        # 📊 数据处理模块
├── utils/
│   ├── model_downloader.py # 📦 模型下载管理
│   ├── prompt_templates.py # 📝 智能提示模板
│   └── __init__.py        # 🛠️ 项目工具
├── data/raw/              # 📁 原始数据目录
├── data/processed/        # 📁 处理后数据目录
├── models/                # 📁 预训练模型存储
├── output/                # 📁 结果输出目录
└── requirements.txt       # 📋 依赖文件
```

## 🚀 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 数据准备（可选）

```bash
# 自动下载所有数据集（使用云计算平台路径 /root/sj-tmp/-dataset/）
python main.py --download-data

# 或下载特定数据集
python main.py --download-data bigvul

# 使用自定义数据路径
python main.py --download-data --data-root /your/custom/path/
```

> 系统会自动检测网络连接，提供重试机制和错误恢复方案，支持镜像源自动切换

### 3. 下载预训练模型

```bash
python main.py --download-model
```

> 自动下载 microsoft/codebert-base 模型（约440MB），支持自定义模型

### 4. 运行评估

```bash
# 评估BigVul数据集
python main.py --mode eval --dataset bigvul

# 评估Reveal数据集
python main.py --mode eval --dataset reveal

# 评估Devign数据集
python main.py --mode eval --dataset devign
```

### 5. 交互式检测

```bash
python main.py --mode interactive
```

输入任意代码片段，系统将实时分析并提供漏洞检测结果。

## 💡 使用示例

### Python API 使用

```python
from models import LocalVulnerabilityDetector
from utils.prompt_templates import create_vulnerability_prompt

# 初始化检测器
detector = LocalVulnerabilityDetector("microsoft/codebert-base")

# 创建漏洞检测提示
code = '''
def vulnerable_function(user_input):
    query = f"SELECT * FROM users WHERE id = {user_input}"
    return query
'''

prompt = create_vulnerability_prompt(code=code)
result = detector.predict_vulnerability(prompt)

print(f"漏洞判断: {result.get('has_vulnerability', False)}")
print(f"置信度: {result.get('confidence', 0.0)}")
print(f"漏洞类型: {result.get('vulnerability_type', '未知')}")
```

### 命令行使用

```bash
# 下载特定模型
python -c "from utils.model_downloader import download_default_model; download_default_model()"

# 评估特定数据集和分割
python main.py --mode eval --dataset bigvul --split test --k-examples 3

# 使用Hugging Face Token（可选）
python main.py --mode eval --dataset reveal --hf-token your_token_here
```

## 📊 性能表现

重构后的系统在三个标准数据集上的表现：

| 数据集 | Accuracy | Precision | Recall | F1-Score |
|--------|----------|-----------|--------|----------|
| **BigVul** | 0.9169 | 0.3186 | 0.4121 | 0.3593 |
| **Reveal** | 0.8812 | 0.3205 | 0.6201 | 0.4226 |
| **Devign** | 0.6013 | 0.5458 | 0.8468 | 0.6638 |

## ⚙️ 配置说明

主要配置位于 `config/config.py`：

```python
# 模型配置
model_config = {
    "default_model": "microsoft/codebert-base",
    "max_length": 512,
    "temperature": 0.1
}

# 数据集配置
datasets = {
    "bigvul": {
        "name": "BigVul",
        "description": "大型软件漏洞数据集",
        "download_url": "https://drive.google.com/file/d/1-0VhnHBp9IGh90s2wCNjeCMuy70HPl8X/view?usp=sharing"
    },
    # ... 更多数据集
}
```

## 🛠️ 技术架构

### 核心组件

1. **LocalVulnerabilityDetector**: 本地漏洞检测器
   - 支持Hugging Face Transformers模型
   - 智能prompt生成和解析
   - GPU/CPU自动适配

2. **CodeRetriever**: 智能代码检索器
   - 基于FAISS的向量检索
   - AST结构相似度计算
   - 上下文学习示例生成

3. **DataProcessor**: 数据处理器
   - 多格式数据加载
   - 自动路径检测
   - 数据清洗和预处理

4. **ModelDownloader**: 模型管理器
   - 自动模型下载和缓存
   - 模型版本管理
   - 存储空间优化

### 模型说明

默认使用 `microsoft/codebert-base` 模型，特点：
- 专门为代码理解设计
- 在代码缺陷检测任务上表现优异
- 模型大小约440MB
- 支持多种编程语言

## 📁 数据集说明

### 支持的数据集

1. **BigVul** - 大型软件漏洞数据集
   - 来源: C/C++代码漏洞
   - 规模: 大量漏洞样本
   - 特点: 涵盖多种漏洞类型

2. **Reveal** - 代码漏洞检测数据集
   - 来源: 多语言代码
   - 特点: 包含代码结构信息

3. **Devign** - 开发者引入的漏洞数据集
   - 来源: GitHub项目历史
   - 特点: 关注漏洞引入过程

### 数据准备

GRACE提供了增强的数据下载功能，支持自动下载和预处理多个漏洞检测数据集：

```bash
# 自动下载所有数据集（默认使用云计算平台路径 /root/sj-tmp/-dataset/）
python main.py --download-data

# 下载特定数据集
python main.py --download-data bigvul
python main.py --download-data reveal  
python main.py --download-data devign

# 使用自定义数据路径
python main.py --download-data --data-root /your/custom/path/

# 检查网络连接和HuggingFace访问
python -c "from prepare_data import DataPreparator; dp = DataPreparator('data'); print(dp.check_network_and_provide_solutions())"
```

#### 数据下载特性

- **🌐 智能网络检测**: 自动检测网络连接和HuggingFace Hub访问状态
- **🔄 自动重试机制**: 网络波动时自动重试下载（最多3次）
- **📦 多源下载**: 支持主数据源和备用数据源切换
- **🔧 错误恢复**: 提供详细的网络问题解决方案
- **☁️ 云计算适配**: 默认路径配置为云计算平台环境
- **🌐 镜像源支持**: 自动切换多个HuggingFace镜像源

#### 手动数据准备

如果自动下载遇到问题，可以手动准备数据：

```bash
# 创建数据目录结构
mkdir -p data/raw data/processed

# 下载数据集文件并放置到对应目录
# BigVul: https://huggingface.co/datasets/Junwei/MSR
# Reveal: https://huggingface.co/datasets/claudios/ReVeal  
# Devign: https://huggingface.co/datasets/Junwei/MSR
```

#### 网络问题解决方案

如果遇到网络连接问题，系统会自动提供以下解决方案：

1. **检查网络连接**: 确保设备已连接到互联网
2. **配置代理**: 设置HTTP/HTTPS代理环境变量
3. **使用镜像源**: 自动切换多个HuggingFace镜像源（已内置支持）
4. **检查防火墙**: 确保防火墙允许访问HuggingFace
5. **使用VPN**: 在网络受限环境下使用VPN
6. **手动下载**: 从备用链接手动下载数据集

#### 云计算平台部署

项目已针对云计算平台进行优化：

- **默认数据路径**: `/root/sj-tmp/-dataset/`
- **镜像源支持**: 内置多个HuggingFace镜像源，自动选择最佳连接
- **网络重试**: 指数退避重试机制，适应网络波动
- **代理配置**: 支持HTTP_PROXY/HTTPS_PROXY环境变量

在云计算平台上部署时，系统会自动检测网络环境并选择最优配置。

## 🔧 高级配置

### 自定义模型

```python
from models import LocalVulnerabilityDetector

# 使用其他Hugging Face模型
detector = LocalVulnerabilityDetector("microsoft/graphcodebert-base")

# 自定义推理参数
detector = LocalVulnerabilityDetector(
    model_name="codet5-base",
    max_length=1024,
    temperature=0.05
)
```

### 提示模板定制

```python
from utils.prompt_templates import get_prompt_manager

manager = get_prompt_manager()

# 使用特定类型模板
prompt = manager.create_analysis_prompt(
    code=code,
    template_type="with_examples",
    examples=similar_examples
)
```

## 🐛 故障排除

### 常见问题

1. **模型下载失败**
   ```bash
   # 检查网络连接
   python -c "from utils.model_downloader import download_default_model; download_default_model(force=True)"
   ```

2. **数据集下载失败**
   ```bash
   # 检查网络连接和HuggingFace访问
   python -c "from prepare_data import DataPreparator; dp = DataPreparator('data'); print(dp.check_network_and_provide_solutions())"
   
   # 手动下载特定数据集
   python main.py --download-data bigvul
   ```

2. **GPU内存不足**
   ```bash
   # 使用CPU模式
   python main.py --device cpu
   ```

3. **依赖冲突**
   ```bash
   # 重新安装依赖
   pip install --force-reinstall -r requirements.txt
   ```

### 日志查看

```bash
# 查看详细日志
tail -f logs/grace_*.log
```

## 📈 性能优化建议

1. **模型选择**:
   - 小模型: `codet5-small` (60MB) - 快速但精度较低
   - 大模型: `microsoft/codebert-base` (440MB) - 平衡选择
   - 超大模型: `microsoft/graphcodebert-base` (440MB) - 最高精度

2. **硬件要求**:
   - 最低: 4GB RAM, CPU
   - 推荐: 8GB RAM, GPU (4GB显存)
   - 最佳: 16GB RAM, GPU (8GB显存)

3. **数据处理**:
   - 适当调整batch size
   - 启用FAISS索引加速
   - 预处理数据缓存

## 🤝 贡献指南

欢迎贡献代码！请遵循以下步骤：

1. Fork本项目
2. 创建功能分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 打开Pull Request

## 📄 许可证

本项目采用MIT许可证 - 详见 [LICENSE](LICENSE) 文件。

## 📚 引用

如果您在研究中使用了GRACE，请引用原始论文：

```bibtex
@inproceedings{grace2023,
  title={GRACE: Empowering LLM-based Software Vulnerability Detection with Graph Structure and In-context Learning},
  author={Your Authors},
  year={2023}
}
```

## 🙏 致谢

感谢以下开源项目：
- [Transformers](https://github.com/huggingface/transformers) - Hugging Face
- [FAISS](https://github.com/facebookresearch/faiss) - Facebook Research
- [CodeBERT](https://github.com/microsoft/CodeBERT) - Microsoft Research

## 📞 联系方式

- 📧 邮箱: your-email@example.com
- 🐛 Issues: [GitHub Issues](https://github.com/your-username/GRACE/issues)
- 💬 讨论: [GitHub Discussions](https://github.com/your-username/GRACE/discussions)

---

**⭐ 如果这个项目对您有帮助，请给我们一个星标！**


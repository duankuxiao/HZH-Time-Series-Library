<p align="center">
  <img src="figures/logo.png" alt="TimeLLM Logo" width="200"/>
</p>

<h1 align="center">TimeLLM 时间序列深度学习库</h1>

<p align="center">
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.8+-blue.svg" alt="Python Version"></a>
  <a href="https://pytorch.org/"><img src="https://img.shields.io/badge/PyTorch-2.0+-red.svg" alt="PyTorch"></a>
  <a href="https://github.com/duankuxiao/TimeLLM/stargazers"><img src="https://img.shields.io/github/stars/duankuxiao/TimeLLM?style=social" alt="GitHub Stars"></a>
  <a href="https://github.com/duankuxiao/TimeLLM/blob/master/LICENSE"><img src="https://img.shields.io/badge/license-MIT-green.svg" alt="License"></a>
</p>

<p align="center">
  一个功能全面的时间序列深度学习研究库，支持多种预测任务和20+种先进模型
</p>

---

## 目录

- [项目简介](#项目简介)
- [主要特性](#主要特性)
- [安装说明](#安装说明)
- [快速开始](#快速开始)
- [支持的模型](#支持的模型)
- [支持的数据集](#支持的数据集)
- [项目结构](#项目结构)
- [配置说明](#配置说明)
- [引用](#引用)
- [致谢](#致谢)
- [License](#license)

---

## 项目简介

这是一个基于 PyTorch 的时间序列深度学习研究库，集成了多种先进的深度学习模型，包括传统的 Transformer 架构和最新的大语言模型（LLM）增强方法。本库旨在为研究人员和工程师提供一个统一、灵活、易用的时间序列分析平台。

### 核心优势

- **模型丰富**：集成 20+ 种先进时间序列模型，包括 LLM 增强模型
- **任务全面**：支持预测、填补、分类、概率预测等多种任务
- **配置灵活**：两层配置系统，支持快速实验和参数调优
- **数据多样**：内置多种真实世界数据集支持
- **易于扩展**：模块化设计，便于添加新模型和数据集

---

## 主要特性

### 1. 长期时间序列预测 (Long-term Forecasting)
支持多步时间序列预测，适用于电力负荷、天气、交通流量等场景的长期预测。

### 2. 缺失值填补 (Missing Value Imputation)
提供多种缺失值填补策略：
- **RDO** (Random Deficiency Occurrence)
- **MCAR** (Missing Completely At Random)
- **MAR** (Missing At Random)

### 3. 时间序列分类 (Time Series Classification)
支持时间序列的分类任务，可用于异常检测、模式识别等应用。

### 4. 概率/区间预测 (Probabilistic Forecasting)
支持分位数回归和概率分布预测，提供预测的不确定性估计。

### 5. 小样本迁移学习 (Few-shot Transfer Learning)
支持预训练模型的迁移学习，在有限样本下实现高效预测。

---

## 安装说明

### 环境要求

- Python >= 3.8
- PyTorch >= 2.0
- CUDA >= 11.0 (可选，用于 GPU 加速)

### 依赖安装

```bash
# 克隆仓库
git clone https://github.com/duankuxiao/TimeLLM.git
cd TimeLLM

# 创建虚拟环境 (推荐)
conda create -n timellm python=3.10
conda activate timellm

# 安装依赖
pip install torch torchvision torchaudio
pip install numpy pandas scikit-learn matplotlib
pip install transformers  # LLM 模型支持
```

### LLM 模型配置 (可选)

如需使用 LLM 增强模型，请下载预训练权重：

```bash
# 权重存放目录
mkdir -p LLM/
# 下载 BERT/GPT2/LLAMA 等模型权重到 LLM/ 目录
```

---

## 快速开始

### 长期预测

```python
from configs.electricity_configs import args
from utils.setup_imputation import model_hyparameter_setup
from exp.exp_forecasting import Exp_Forecast

# 设置模型
args.model = 'TimesNet'
args.is_training = 1
args = model_hyparameter_setup(args)

# 创建实验实例
exp = Exp_Forecast(args)

# 训练
setting = f'{args.model}_{args.data}_{args.seq_len}_{args.pred_len}'
exp.train(setting)

# 测试
res_df, metrics_df = exp.test(setting)
```

或者直接运行脚本：

```bash
python run_long-term_forecast.py
```

### 缺失值填补

```bash
python run_imputation.py
```

### 时间序列分类

```bash
python run_classification.py
```

### 概率预测

```bash
python run_probabilistic_forecast.py
```

### 小样本学习

```bash
python few_shot_training.py
```

---

## 支持的模型

### 核心模型

| 模型 | 类型 | 描述 |
|------|------|------|
| **DLinear** | Linear | 基于分解的线性模型 |
| **Transformer** | Attention | 标准 Transformer 架构 |
| **Informer** | Attention | 稀疏注意力 Transformer |
| **Autoformer** | Attention | 序列分解 + 自相关注意力 |
| **TimesNet** | CNN | 基于 FFT 的周期性建模 |
| **PatchTST** | Attention | 基于 Patch 的时间序列 Transformer |
| **iTransformer** | Attention | 反向 Transformer (通道独立) |
| **TimeMixer** | MLP | 多尺度时间混合 |
| **RNN** | RNN | LSTM/GRU 循环网络 |

### LLM 增强模型

| 模型 | 基础 LLM | 描述 |
|------|----------|------|
| **TimeLLM** | BERT/GPT2/LLAMA | 时间序列 + 预训练语言模型 |
| **LLMformer** | BERT/GPT2 | LLM + Transformer 融合 |
| **AttLLM** | BERT/GPT2 | 注意力增强的 LLM 方法 |
| **AutoTimes** | BERT/GPT2 | 自动架构搜索 |

### 其他模型

- **Ablation**: 用于消融实验
- **Classification**: 分类专用模型

---

## 支持的数据集

### 能源与电力

| 数据集 | 特征数 | 描述 |
|--------|--------|------|
| 东京电力 | 21 | 日本东京地区电力负荷数据 |
| 九州电力 | 21 | 日本九州地区电力负荷数据 |
| 德州电力 | 21 | 美国德州电网数据 |
| HVAC | 25+ | 暖通空调系统运行数据 |

### 气象与环境

| 数据集 | 特征数 | 描述 |
|--------|--------|------|
| 太阳辐射 | 9 | 多城市太阳辐射数据 |
| 天气数据 | 多变量 | 温度、湿度、风速等 |

### 交通与其他

| 数据集 | 描述 |
|--------|------|
| PEMS | 加州交通流量数据 |
| ETT | 电力变压器温度数据集 |
| M4 | M4 预测竞赛数据集 |
| 价格数据 | 电力市场价格数据 |

---

## 项目结构

```
TimeLLM/
├── run_long-term_forecast.py    # 长期预测入口
├── run_imputation.py            # 缺失值填补入口
├── run_classification.py        # 分类任务入口
├── run_probabilistic_forecast.py # 概率预测入口
├── few_shot_training.py         # 小样本学习入口
│
├── configs/                     # 配置文件
│   ├── common_configs.py        # 通用配置
│   ├── electricity_configs.py   # 电力数据配置
│   ├── HVAC_configs.py          # HVAC 数据配置
│   └── ...
│
├── models/                      # 模型实现
│   ├── DLinear.py
│   ├── Transformer.py
│   ├── TimesNet.py
│   ├── TimeLLM.py
│   └── ...
│
├── exp/                         # 实验流程
│   ├── exp_basic.py             # 基础实验类
│   ├── exp_forecasting.py       # 预测实验
│   ├── exp_imputation.py        # 填补实验
│   └── exp_classification.py    # 分类实验
│
├── data_provider/               # 数据加载
│   ├── data_factory.py          # 数据工厂
│   ├── data_loader.py           # 数据加载器
│   └── data_loader_LLM.py       # LLM 数据加载器
│
├── layers/                      # 网络层组件
│   ├── Embed.py                 # 嵌入层
│   ├── Autoformer_EncDec.py     # 编解码器
│   └── SelfAttention_Family.py  # 注意力机制
│
├── utils/                       # 工具函数
│   ├── metrics.py               # 评估指标
│   ├── losses.py                # 损失函数
│   ├── tools.py                 # 工具函数
│   └── masking.py               # 掩码策略
│
├── dataset/                     # 数据集
│   └── prompt_bank/             # LLM 提示模板
│
├── LLM/                         # LLM 预训练权重
│
├── results/                     # 实验结果
│
└── figures/                     # 图片资源
```

---

## 配置说明

### 关键参数

```python
# 任务类型
task_name = 'long_term_forecast'  # 'imputation', 'classification', 'interval_forecast'

# 数据特征
features = 'M'     # 'M': 多变量, 'S': 单变量, 'MS': 多变量输入单变量输出

# 序列长度
seq_len = 96       # 输入序列长度
label_len = 48     # 标签序列长度
pred_len = 96      # 预测长度

# 模型架构
d_model = 512      # 模型维度
n_heads = 8        # 注意力头数
e_layers = 2       # 编码器层数
d_layers = 1       # 解码器层数

# LLM 配置
llm_model = 'GPT2' # 'BERT', 'GPT2', 'LLAMA'
llm_dim = 768      # LLM 隐藏维度
llm_layers = 6     # LLM 层数

# 训练参数
learning_rate = 0.0001
batch_size = 32
train_epochs = 10
```

### 配置优先级

1. **命令行参数** (最高优先级)
2. **数据集配置** (`configs/{dataset}_configs.py`)
3. **通用配置** (`configs/common_configs.py`)

---

## 引用

如果本项目对您的研究有帮助，请引用：

```bibtex
# 论文引用信息稍后补充
```

---

## 致谢

本项目参考和借鉴了以下优秀开源项目：

- [Time-Series-Library](https://github.com/thuml/Time-Series-Library)


---


---

<p align="center">
  <i>如有问题或建议，欢迎提交 Issue 或 Pull Request！</i>
</p>

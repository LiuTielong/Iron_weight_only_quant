# Weight-Only Quantization

这个文件夹包含了运行weight-only量化实验的核心代码文件，支持LLaMA和OPT模型族。

## 文件说明

- `main.py`: 主实验脚本，支持LLaMA和OPT模型的weight-only量化和PPL评测
- `utils.py`: 工具函数，包含模型和分词器加载功能
- `quant_funcs.py`: 量化核心函数，包含pseudo_quantize_tensor等基础量化操作
- `quant_wrapper.py`: 简化的量化包装器，专门用于weight-only量化
- `quant_linear.py`: 将一个线性层进行伪量化的核心代码

## 使用方法

### 基础用法
```bash
python main.py --model_path /path/to/model --datasets wikitext
```

### 自定义量化配置
```bash
# 测试4位和8位量化，使用per-channel量化
python main.py \
    --model_path /path/to/llama-7b \
    --w_bits 4 8 \
    --w_group_size -2 \
    --datasets wikitext ptb

# 使用per-group量化 (group_size=128)
python main.py \
    --model_path /path/to/opt-1.3b \
    --w_bits 16 8 4 \
    --w_group_size 128 \
    --datasets wikitext
```

## 主要参数

### 基础参数
- `--model_path`: 模型路径 (支持LLaMA和OPT模型族)
- `--datasets`: 评测数据集 (wikitext, ptb, c4)
- `--sample_size`: 样本数量 (默认1000)
- `--local_dataset_dir`: 本地数据集目录

### 量化参数
- `--w_bits`: 权重量化位宽列表 (默认: [16, 8, 4])
- `--w_group_size`: 量化粒度 (默认: -2)
  - `-1`: per-tensor量化
  - `-2`: per-channel量化
  - `32, 64, 128, 256`: per-group量化

## 支持的模型

- **LLaMA系列**: LLaMA-7B, LLaMA-13B, LLaMA-2-7B, LLaMA-2-13B等
- **OPT系列**: OPT-125M, OPT-1.3B, OPT-2.7B, OPT-6.7B, OPT-13B等

## 量化配置说明

1. **per-tensor量化** (`--w_group_size -1`): 整个权重张量共享一个量化参数
2. **per-channel量化** (`--w_group_size -2`): 每个输出通道使用独立的量化参数
3. **per-group量化** (`--w_group_size N`): 每N个权重共享一个量化参数

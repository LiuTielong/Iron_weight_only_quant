import random
import torch
import matplotlib.pyplot as plt

from quant_linear import _count_fp4_values, FP4_EXP_BITS, FP4_MANTISSA_BITS, FP6_EXP_BITS, FP6_MANTISSA_BITS


def collect_linears(model):
    """Collect Linear/QuantLinear modules with their names."""
    linears = []
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear) or module.__class__.__name__ == "QuantLinear":
            linears.append((name, module))
    return linears


def plot_random_fp4_dists(
    model,
    k=10,
    seed=0,
    save_path="./results/fp4_samples.png",
    exp_bits=FP4_EXP_BITS,
    mant_bits=FP4_MANTISSA_BITS,
    exp_bias=None,
):
    """
    随机抽取 k 个（最多 10 个）FP4 量化的线性层，绘制子图展示解码值分布。
    假设模型已量化为 FP4，且层上有 weight_fp4。
    """
    if exp_bias is None:
        exp_bias = (1 << (exp_bits - 1)) - 1
    random.seed(seed)
    linears = collect_linears(model)
    print(f"总 Linear/QuantLinear 数量: {len(linears)}")
    if len(linears) == 0:
        raise ValueError("模型中没有 Linear/QuantLinear 模块")

    picked = random.sample(linears, min(k, len(linears)))

    fig, axes = plt.subplots(2, 5, figsize=(18, 8))
    axes = axes.flatten()
    for ax, (name, layer) in zip(axes, picked):
        codes = getattr(layer, "weight_fp4", None)
        if codes is None:
            ax.set_title(f"{name}\n(no fp4)")
            ax.axis("off")
            continue
        values, counts = _count_fp4_values(codes, exp_bits=exp_bits, mant_bits=mant_bits, exp_bias=exp_bias)
        v = values.cpu().numpy()
        c = counts.cpu().numpy()
        ax.bar(range(len(v)), c, tick_label=[f"{x:.3g}" for x in v])
        ax.set_title(name, fontsize=10)
        # 为避免横坐标过于拥挤，只保留部分刻度标签
        step = max(1, len(v) // 8)
        for lbl in ax.get_xticklabels():
            lbl.set_visible(False)
        for i, lbl in enumerate(ax.get_xticklabels()):
            if i % step == 0:
                lbl.set_visible(True)
        ax.tick_params(axis='x', rotation=45)
    for ax in axes[len(picked):]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    print(f"示例子图已保存到 {save_path}")


def plot_random_fp6_dists(
    model,
    k=10,
    seed=0,
    save_path="./results/fp6_samples.png",
    exp_bits=FP6_EXP_BITS,
    mant_bits=FP6_MANTISSA_BITS,
    exp_bias=None,
):
    """
    随机抽取 k 个（最多 10 个）FP6 量化的线性层，绘制子图展示解码值分布。
    假设模型已量化为 FP6，且层上有 weight_fp6。
    """
    if exp_bias is None:
        exp_bias = (1 << (exp_bits - 1)) - 1
    random.seed(seed)
    linears = collect_linears(model)
    print(f"总 Linear/QuantLinear 数量: {len(linears)}")
    if len(linears) == 0:
        raise ValueError("模型中没有 Linear/QuantLinear 模块")

    picked = random.sample(linears, min(k, len(linears)))

    fig, axes = plt.subplots(2, 5, figsize=(18, 8))
    axes = axes.flatten()
    for ax, (name, layer) in zip(axes, picked):
        codes = getattr(layer, "weight_fp6", None)
        if codes is None:
            ax.set_title(f"{name}\n(no fp6)")
            ax.axis("off")
            continue
        values, counts = _count_fp4_values(codes, exp_bits=exp_bits, mant_bits=mant_bits, exp_bias=exp_bias)
        v = values.cpu().numpy()
        c = counts.cpu().numpy()
        ax.bar(range(len(v)), c, tick_label=[f"{x:.3g}" for x in v])
        ax.set_title(name, fontsize=10)
        ax.tick_params(axis='x', rotation=45)
    for ax in axes[len(picked):]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    print(f"FP6 值分布子图已保存到 {save_path}")


def plot_random_fp4_exponent_dists(
    model,
    k=10,
    seed=0,
    save_path="./results/fp4_exponent_samples.png",
    exp_bits=FP4_EXP_BITS,
    mant_bits=FP4_MANTISSA_BITS,
):
    """
    随机抽取 k 个（最多 10 个）FP4 量化的线性层，绘制指数字段分布。
    指数字段范围: [0, 2^exp_bits - 1]，0 代表次正规/零。
    """
    random.seed(seed)
    linears = collect_linears(model)
    print(f"总 Linear/QuantLinear 数量: {len(linears)}")
    if len(linears) == 0:
        raise ValueError("模型中没有 Linear/QuantLinear 模块")

    picked = random.sample(linears, min(k, len(linears)))

    fig, axes = plt.subplots(2, 5, figsize=(18, 8))
    axes = axes.flatten()
    for ax, (name, layer) in zip(axes, picked):
        codes = getattr(layer, "weight_fp4", None)
        if codes is None:
            ax.set_title(f"{name}\n(no fp4)")
            ax.axis("off")
            continue
        codes_flat = codes.reshape(-1).to(torch.uint8)
        exp_field = (codes_flat >> mant_bits) & ((1 << exp_bits) - 1)
        exp_vals, exp_counts = torch.unique(exp_field, sorted=True, return_counts=True)
        x = exp_vals.cpu().numpy()
        y = exp_counts.cpu().numpy()
        ax.bar(range(len(x)), y, tick_label=[str(int(v)) for v in x])
        ax.set_title(name, fontsize=10)
        ax.set_xlabel("exp field")
        ax.set_ylabel("count")
        ax.tick_params(axis='x', rotation=0)
    for ax in axes[len(picked):]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    print(f"指数分布子图已保存到 {save_path}")


def plot_random_fp6_exponent_dists(
    model,
    k=10,
    seed=0,
    save_path="./results/fp6_exponent_samples.png",
    exp_bits=FP6_EXP_BITS,
    mant_bits=FP6_MANTISSA_BITS,
):
    """
    随机抽取 k 个（最多 10 个）FP6 量化的线性层，绘制指数字段分布。
    指数字段范围: [0, 2^exp_bits - 1]，0 代表次正规/零。
    """
    random.seed(seed)
    linears = collect_linears(model)
    print(f"总 Linear/QuantLinear 数量: {len(linears)}")
    if len(linears) == 0:
        raise ValueError("模型中没有 Linear/QuantLinear 模块")

    picked = random.sample(linears, min(k, len(linears)))

    fig, axes = plt.subplots(2, 5, figsize=(18, 8))
    axes = axes.flatten()
    for ax, (name, layer) in zip(axes, picked):
        codes = getattr(layer, "weight_fp6", None)
        if codes is None:
            ax.set_title(f"{name}\n(no fp6)")
            ax.axis("off")
            continue
        codes_flat = codes.reshape(-1).to(torch.uint8)
        exp_field = (codes_flat >> mant_bits) & ((1 << exp_bits) - 1)
        exp_vals, exp_counts = torch.unique(exp_field, sorted=True, return_counts=True)
        x = exp_vals.cpu().numpy()
        y = exp_counts.cpu().numpy()
        ax.bar(range(len(x)), y, tick_label=[str(int(v)) for v in x])
        ax.set_title(name, fontsize=10)
        ax.set_xlabel("exp field")
        ax.set_ylabel("count")
        ax.tick_params(axis='x', rotation=0)
    for ax in axes[len(picked):]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    print(f"FP6 指数分布子图已保存到 {save_path}")

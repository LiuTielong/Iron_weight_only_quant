import random
import torch
import matplotlib.pyplot as plt
import numpy as np

from quant_linear import _count_fp4_values, FP4_EXP_BITS, FP4_MANTISSA_BITS, FP6_EXP_BITS, FP6_MANTISSA_BITS, FP8_EXP_BITS, FP8_MANTISSA_BITS


def collect_linears(model):
    """Collect Linear/QuantLinear modules with their names."""
    linears = []
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear) or module.__class__.__name__ == "QuantLinear":
            linears.append((name, module))
    return linears


def plot_random_fp_dists(
    model,
    k: int = 10,
    seed: int = 0,
    save_path: str | None = None,
    weight_attr: str = "weight_fp4",
    exp_bits: int = FP4_EXP_BITS,
    mant_bits: int = FP4_MANTISSA_BITS,
    exp_bias: int | None = None,
    title_prefix: str | None = None,
):
    """
    通用的随机子图绘制：展示指定 FP 码字解码值分布。
    假设量化层上缓存了对应码字（weight_attr）。
    """
    if exp_bias is None:
        exp_bias = (1 << (exp_bits - 1)) - 1
    if save_path is None:
        suffix = weight_attr.replace("weight_", "")
        save_path = f"./results/{suffix}_samples.png"
    if title_prefix is None:
        title_prefix = suffix if 'suffix' in locals() else weight_attr

    random.seed(seed)
    linears = collect_linears(model)
    print(f"总 Linear/QuantLinear 数量: {len(linears)}")
    if len(linears) == 0:
        raise ValueError("模型中没有 Linear/QuantLinear 模块")

    picked = random.sample(linears, min(k, len(linears)))

    fig, axes = plt.subplots(2, 5, figsize=(18, 8))
    axes = axes.flatten()
    for ax, (name, layer) in zip(axes, picked):
        codes = getattr(layer, weight_attr, None)
        if codes is None:
            ax.set_title(f"{name}\n(no {weight_attr})")
            ax.axis("off")
            continue
        values, counts = _count_fp4_values(codes, exp_bits=exp_bits, mant_bits=mant_bits, exp_bias=exp_bias)
        v = values.cpu().numpy()
        c = counts.cpu().numpy()
        ax.bar(range(len(v)), c)
        ax.set_title(name if title_prefix is None else f"{title_prefix}: {name}", fontsize=10)
        step = max(1, len(v) // 10)
        show_idx = list(range(0, len(v), step))
        if show_idx[-1] != len(v) - 1:
            show_idx.append(len(v) - 1)
        show_labels = [f"{v[i]:.3g}" for i in show_idx]
        ax.set_xticks(show_idx)
        ax.set_xticklabels(show_labels, rotation=45, ha="right")
        ax.tick_params(axis='x', labelsize=8)
    for ax in axes[len(picked):]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    print(f"值分布子图已保存到 {save_path}")


def plot_random_fp_exponent_dists(
    model,
    k: int = 10,
    seed: int = 0,
    save_path: str = "./results/fp_exponent_samples.png",
    weight_attr: str = "weight_fp4",
    exp_bits: int = FP4_EXP_BITS,
    mant_bits: int = FP4_MANTISSA_BITS,
):
    """
    通用的随机子图绘制：展示指定 FP 码字的指数字段分布。
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
        codes = getattr(layer, weight_attr, None)
        if codes is None:
            ax.set_title(f"{name}\n(no {weight_attr})")
            ax.axis("off")
            continue
        codes_flat = codes.reshape(-1).to(torch.uint8)
        exp_field = (codes_flat >> mant_bits) & ((1 << exp_bits) - 1)
        exp_vals, exp_counts = torch.unique(exp_field, sorted=True, return_counts=True)
        x = exp_vals.cpu().numpy()
        y = exp_counts.cpu().numpy()
        bars = ax.bar(range(len(x)), y, tick_label=[str(int(v)) for v in x])
        total = y.sum()
        if total > 0:
            for i, bar in enumerate(bars):
                perc = y[i] / total * 100
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{perc:.2f}", ha="center", va="bottom", fontsize=8)
        ax.set_title(name, fontsize=10)
        ax.set_xlabel("exp field")
        ax.set_ylabel("count")
        ax.tick_params(axis='x', rotation=0)
    for ax in axes[len(picked):]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    print(f"指数分布子图已保存到 {save_path}")


def plot_fp8_exponent_heatmaps(
    model,
    num_layers: int = 4,
    block_size: int = 64,
    seed: int = 0,
    save_path: str = "./results/fp8_exponent_heatmaps.png",
    exp_bits: int = FP8_EXP_BITS,
    mant_bits: int = FP8_MANTISSA_BITS,
):
    """
    随机选取若干 FP8 量化层，对每层截取一个 block_size x block_size 的码字块，
    提取指数字段并绘制热力图。
    """
    random.seed(seed)
    linears = collect_linears(model)
    linears = [(n, m) for n, m in linears if getattr(m, "weight_fp8", None) is not None]
    if len(linears) == 0:
        raise ValueError("模型中没有 FP8 量化的 Linear/QuantLinear 模块")

    picked = random.sample(linears, min(num_layers, len(linears)))
    fig_rows = 2
    fig_cols = 2
    fig, axes = plt.subplots(fig_rows, fig_cols, figsize=(12, 10))
    axes = axes.flatten()

    mask_bits = (1 << exp_bits) - 1
    for ax, (name, layer) in zip(axes, picked):
        codes = getattr(layer, "weight_fp8")
        if codes is None:
            ax.set_title(f"{name}\n(no fp8)")
            ax.axis("off")
            continue
        h_out, w_in = codes.shape
        bs_r = min(block_size, h_out)
        bs_c = min(block_size, w_in)
        start_r = 0 if h_out == bs_r else random.randint(0, h_out - bs_r)
        start_c = 0 if w_in == bs_c else random.randint(0, w_in - bs_c)
        block = codes[start_r:start_r + bs_r, start_c:start_c + bs_c].to(torch.int32)
        exp_field = (block >> mant_bits) & mask_bits
        im = ax.imshow(exp_field.cpu().numpy(), cmap="viridis", aspect="auto")
        ax.set_title(f"{name}\n[{start_r}:{start_r+bs_r}, {start_c}:{start_c+bs_c}]")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    for ax in axes[len(picked):]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    print(f"FP8 指数热力图已保存到 {save_path}")


def count_fp8_exponent_outliers(
    model,
    group_size: int = 4,
    threshold: int = 11,
    exp_bits: int = FP8_EXP_BITS,
    mant_bits: int = FP8_MANTISSA_BITS,
    k: int = 10,
    seed: int = 0,
):
    """Count FP8 exponent outlier groups (<= threshold) on k randomly sampled FP8 layers.

    For each FP8-quantized layer:
      - extract exponent field (same as heatmap: right shift mant_bits).
      - group exponents along columns by group_size (e.g., 4 -> row*width/group_size groups).
      - report percentage of groups containing 1/2/3/4 outliers (exponent <= threshold).
    """
    random.seed(seed)
    mask_bits = (1 << exp_bits) - 1
    linears = [(n, m) for n, m in collect_linears(model) if getattr(m, "weight_fp8", None) is not None]
    if len(linears) == 0:
        print("[fp8 outlier] no FP8 layers found")
        return []
    linears = random.sample(linears, min(k, len(linears)))

    results = []
    for name, layer in linears:
        codes = getattr(layer, "weight_fp8")
        h_out, w_in = codes.shape
        if w_in % group_size != 0:
            print(f"[warn] skip {name}: width {w_in} not divisible by group_size {group_size}")
            continue
        exp_field = (codes.to(torch.int32) >> mant_bits) & mask_bits
        outlier_mask = exp_field <= threshold
        groups = outlier_mask.view(h_out, w_in // group_size, group_size)
        counts = groups.sum(dim=2)
        c1 = int((counts == 1).sum().item())
        c2 = int((counts == 2).sum().item())
        c3 = int((counts == 3).sum().item())
        c4 = int((counts == 4).sum().item())
        total_groups = h_out * (w_in // group_size)
        pct = lambda c: (c / total_groups * 100.0) if total_groups > 0 else 0.0
        results.append({
            "layer": name,
            "total_groups": total_groups,
            "outlier_1_pct": pct(c1),
            "outlier_2_pct": pct(c2),
            "outlier_3_pct": pct(c3),
            "outlier_4_pct": pct(c4),
        })
        print(f"[fp8 outlier] {name}: groups={total_groups}, 1={pct(c1):.4f}%, 2={pct(c2):.4f}%, 3={pct(c3):.4f}%, 4={pct(c4):.4f}%")
    return results

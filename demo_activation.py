#!/usr/bin/env python3
"""
Demo: 4-bit weight-only (group) quantization and activation capture.

Usage example:
    python demo_activation.py \\
        --model_path /path/to/your/hf/model \\
        --prompt "Hello, how are you?" \\
        --layer_name model.layers.0.mlp.down_proj
"""

import argparse
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Optional, Tuple
import torch
import sys
import matplotlib.pyplot as plt

sys.path.append(".")
from utils import build_model_and_enc, get_module_by_name_suffix
from quant_wrapper import quantize_model
from quant_linear import QuantLinear


def parse_args():
    parser = argparse.ArgumentParser(description="4-bit grouped weight-only quantization demo with activation dump")
    parser.add_argument("--model_path", type=str, required=True, help="Local HuggingFace model path")
    parser.add_argument("--prompt", type=str, default="Hello, this is a short test prompt.")
    parser.add_argument("--layer_name", type=str, default=None,
                        help="Target linear layer name (exact or suffix). If omitted, the first QuantLinear/Linear is used.")
    parser.add_argument("--w_group_size", type=int, default=128,
                        help="Group size for weight quantization (>0 for per-group).")
    parser.add_argument("--device", type=str, default=None, help="Force device (cpu/cuda). Defaults to cuda if available.")
    parser.add_argument("--save_path", type=str, default="activation_distributions.png",
                        help="Where to save the per-token activation distribution plot.")
    parser.add_argument("--save_scatter_path", type=str, default="activation_dim_scatter.png",
                        help="Where to save the per-dimension scatter plot.")
    return parser.parse_args()


def build_quant_args(w_group_size: int) -> SimpleNamespace:
    """
    Construct the minimal namespace expected by quantize_model for RTN weight-only quantization.
    """
    return SimpleNamespace(
        w_bit=4,
        w_group_size=w_group_size,
        w_symmetric=True,
        w_format="int",  # grouped INT4
        a_bit=16,
        a_group_size=128,
        kv_bit=16,
        kv_group_size=128,
        mode=0,
        gptq=False,
        nsamples=0,
        percdamp=0.01,
        act_order=False,
        dataloader=None,
    )


def find_target_layer(model: torch.nn.Module, layer_name: Optional[str]) -> Tuple[str, torch.nn.Module]:
    if layer_name:
        module = get_module_by_name_suffix(model, layer_name)
        if module is None:
            # Try simple OPT/LLaMA alias mapping (e.g., down_proj -> fc2 for OPT)
            aliases = [
                ("mlp.down_proj", "fc2"),
                ("mlp.gate_proj", "fc1"),
                ("mlp.up_proj", "fc1"),
            ]
            for old, new in aliases:
                if old in layer_name:
                    alt_name = layer_name.replace(old, new)
                    module = get_module_by_name_suffix(model, alt_name)
                    if module is not None:
                        return alt_name, module

            # If still not found, surface a short list of similar module names to guide the user.
            similar = []
            tail = layer_name.split(".")[-1]
            for name, _ in model.named_modules():
                if tail in name or layer_name in name:
                    similar.append(name)
                if len(similar) >= 20:
                    break
            hint = "\n  - " + "\n  - ".join(similar) if similar else " (no similar names found)"
            raise ValueError(f"Layer '{layer_name}' not found in model. Similar candidates:{hint}")
        return layer_name, module

    # Fallback: pick the first quantized linear (or linear) layer that is not lm_head/output_layer.
    for name, module in model.named_modules():
        if isinstance(module, (QuantLinear, torch.nn.Linear)) and "lm_head" not in name and "output_layer" not in name:
            return name, module
    raise RuntimeError("No suitable linear layer found to hook.")


def main():
    args = parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    if not Path(args.model_path).exists():
        raise FileNotFoundError(f"Model path does not exist: {args.model_path}")

    torch.set_grad_enabled(False)

    # 1) Load model/tokenizer
    model, tokenizer = build_model_and_enc(args.model_path, use_flash_attn=False, kv_bit=16, kv_group_size=128)
    model.eval()

    # 2) Apply 4-bit grouped weight-only quantization
    quant_args = build_quant_args(args.w_group_size)
    model = quantize_model(model, quant_args)
    model = model.to(device)

    # 3) Locate target layer and register hook
    layer_full_name, target_layer = find_target_layer(model, args.layer_name)
    activation_store: Dict[str, torch.Tensor] = {}

    def _capture_activation(_module, _inp, output):
        activation_store["activation"] = output.detach().cpu()

    hook_handle = target_layer.register_forward_hook(_capture_activation)

    # 4) Run a short forward pass
    encoded = tokenizer(args.prompt, return_tensors="pt")
    encoded = {k: v.to(device) for k, v in encoded.items()}
    _ = model(**encoded)

    hook_handle.remove()

    if "activation" not in activation_store:
        raise RuntimeError("Activation was not captured; ensure the target layer is reached in the forward pass.")

    activation = activation_store["activation"]
    if activation.dim() > 2:
        # Flatten batch/sequence dims to a 2D matrix for easier viewing.
        activation_matrix = activation.reshape(-1, activation.shape[-1])
    else:
        activation_matrix = activation

    print(f"\nCaptured activation from layer '{layer_full_name}':")
    print(f"Original shape: {tuple(activation.shape)}; flattened matrix shape: {tuple(activation_matrix.shape)}")
    print(activation_matrix)

    # Print per-token extrema for quick inspection
    max_vals = activation_matrix.max(dim=1).values
    min_vals = activation_matrix.min(dim=1).values
    print("\nPer-token min/max:")
    for idx, (mn, mx) in enumerate(zip(min_vals.tolist(), max_vals.tolist())):
        print(f"  token {idx}: min={mn:.6f}, max={mx:.6f}")

    # 5) Plot per-token activation distributions
    num_tokens = activation_matrix.shape[0]
    fig, axes = plt.subplots(num_tokens, 1, figsize=(8, max(2.0, num_tokens * 1.5)), sharex=True)
    if num_tokens == 1:
        axes = [axes]
    for idx in range(num_tokens):
        axes[idx].hist(activation_matrix[idx].numpy().ravel(), bins=50, color="steelblue", alpha=0.8)
        axes[idx].set_ylabel(f"tok{idx}")
    axes[-1].set_xlabel("Activation value")
    fig.suptitle(f"Per-token activation distributions\nLayer: {layer_full_name}")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(args.save_path, dpi=200)
    print(f"\nSaved per-token activation distribution plot to: {args.save_path}")

    # 6) Scatter plot: dimensions on x-axis, token values on y-axis
    num_dims = activation_matrix.shape[1]
    seg_size = 512
    segments = list(range(0, num_dims, seg_size))
    fig2, axes2 = plt.subplots(len(segments), 1, figsize=(10, max(2.0, len(segments) * 2.0)), sharex=False, sharey=False)
    if len(segments) == 1:
        axes2 = [axes2]
    colors = plt.get_cmap("tab10")
    for seg_idx, start in enumerate(segments):
        end = min(start + seg_size, num_dims)
        x = list(range(start, end))
        ax = axes2[seg_idx]
        for t in range(num_tokens):
            y = activation_matrix[t, start:end].numpy()
            label = f"token {t}" if seg_idx == 0 else None
            ax.scatter(x, y, s=4, alpha=0.7, color=colors(t % 10), label=label)
        ax.set_ylabel(f"dims {start}-{end-1}")
        if seg_idx == len(segments) - 1:
            ax.set_xlabel("Dimension index")
        if seg_idx == 0:
            ax.legend(fontsize=8, ncol=min(num_tokens, 4))
    fig2.suptitle(f"Per-dimension activations (tokens as points)\nLayer: {layer_full_name}")
    fig2.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(args.save_scatter_path, dpi=200)
    print(f"Saved per-dimension scatter plot to: {args.save_scatter_path}")


if __name__ == "__main__":
    main()




"""
CUDA_VISIBLE_DEVICES=0  \
python demo_activation.py \
  --model_path /home/data/meta-llama/opt/6.7b/ \
  --prompt "Hello, how are you?" \
  --layer_name model.layers.0.mlp.down_proj \
  --w_group_size 128

"""

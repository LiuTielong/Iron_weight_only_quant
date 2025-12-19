#!/usr/bin/env python3
"""
Quantize a local OPT/LLaMA model, then evaluate lm-evaluation-harness tasks.

依赖：
- 量化代码：Iron_weight_only_quant 下的 quantize_model / build_model_and_enc / get_loaders
- lm-evaluation-harness 已在本仓库

用法示例：
python Iron_weight_only_quant/eval_quant_lm_eval.py \
  --model_path /home/data/meta-llama/opt/6.7b \
  --tasks arc_easy boolq piqa hellaswag winogrande arc_challenge lambada_standard sglue_rte \
  --w_bit 8 --w_group_size -2 --device cuda
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch

# 本脚本位于子目录，补充路径以便导入量化与评测代码
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))  # 允许 import lm_eval
sys.path.append(str(ROOT / "Iron_weight_only_quant"))  # 允许 import quant funcs
sys.path.append(str(ROOT / "Iron_weight_only_quant" / "gptq"))

from gptq.datautils import get_loaders
from lm_eval import evaluator, tasks
from lm_eval.models.huggingface import HFLM
from quant_wrapper import quantize_model
from utils import build_model_and_enc


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Quantize then eval with lm-evaluation-harness")
    p.add_argument("--model_path", type=str, required=True, help="原始 FP16 模型目录")
    p.add_argument(
        "--tasks",
        nargs="+",
        default=[
            "arc_easy",
            # "arc_challenge",
            # "boolq",
            # "piqa",
            # "hellaswag",
            # "winogrande",
            # "lambada_standard",
            # "sglue_rte",
        ],
        help="lm-eval 任务列表",
    )
    p.add_argument("--num_fewshot", type=int, default=0, help="few-shot 样本数")
    p.add_argument("--device", type=str, default="cuda", help='设备，如 "cuda" 或 "cpu"')
    p.add_argument("--batch_size", type=int, default=1, help="评测 batch size（会传给 HFLM）")
    p.add_argument("--max_batch_size", type=int, default=None, help="最大自适应 batch size")

    # 量化设置（与 Iron_weight_only_quant/main.py 保持一致的核心参数）
    p.add_argument("--w_bit", type=int, default=8, help="权重量化位宽，<16 则执行量化")
    p.add_argument(
        "--w_group_size",
        type=int,
        default=-2,
        choices=[-1, -2, 32, 64, 128, 256, 512, 1024],
        help="-1: per-tensor, -2: per-channel, >0: per-group",
    )
    p.add_argument("--w_symmetric", action="store_true", help="权重对称量化")
    p.add_argument("--w_format", type=str, default="int", choices=["int", "fp4", "fp6", "fp8", "bfp"], help="权重量化格式")
    p.add_argument("--quant_dim", type=int, default=0, choices=[0, 1], help="近似量化分组维度：0 按行/输入维分组，1 按列/输出维分组")
    p.add_argument("--mode", type=int, default=0, choices=[0, 1, 2], help="0: GPU, 1: FIGLUT-F, 2: FIGLUT-I")

    # GPTQ 相关
    p.add_argument("--gptq", action="store_true", help="是否使用 GPTQ")
    p.add_argument("--nsamples", type=int, default=128, help="GPTQ 校准样本数")
    p.add_argument("--percdamp", type=float, default=0.01, help="GPTQ damp 参数")
    p.add_argument("--act_order", action="store_true", help="GPTQ activation order")
    p.add_argument("--approximate", action="store_true", help="近似量化")
    p.add_argument("--calib_dataset", type=str, default="wikitext2", help="GPTQ 校准数据集 (get_loaders 名称)")

    # 离线/缓存
    p.add_argument("--offline", action="store_true", help="强制离线，只用本地缓存")
    p.add_argument("--hf_cache", type=str, default=None, help="可选：指定 HF 数据/模型缓存目录")
    return p.parse_args()


def prepare_env(args: argparse.Namespace) -> None:
    if args.offline:
        os.environ["HF_DATASETS_OFFLINE"] = "1"
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
        for key in ("HF_ENDPOINT", "HF_MIRROR", "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY"):
            os.environ.pop(key, None)
    if args.hf_cache:
        os.environ["HF_HOME"] = args.hf_cache
        os.environ["HF_DATASETS_CACHE"] = args.hf_cache


def build_quantized_model(args: argparse.Namespace):
    # 加载原始模型
    model, tokenizer = build_model_and_enc(
        args.model_path,
        use_flash_attn=False,
        kv_bit=16,
        kv_group_size=128,
    )

    if args.w_bit < 16:
        class QuantArgs:
            def __init__(self):
                self.w_bit = args.w_bit
                self.w_group_size = args.w_group_size
                self.w_symmetric = args.w_symmetric
                self.w_format = args.w_format
                self.approximate = args.approximate
                self.quant_dim = args.quant_dim
                self.a_bit = 16
                self.a_group_size = 128
                self.kv_bit = 16
                self.kv_group_size = 128
                self.mode = args.mode
                self.gptq = args.gptq
                self.nsamples = args.nsamples
                self.percdamp = args.percdamp
                self.act_order = args.act_order
                self.dataloader = None

        qargs = QuantArgs()
        if args.gptq:
            print(f"准备 GPTQ 校准数据集：{args.calib_dataset}")
            dataloader, _ = get_loaders(
                args.calib_dataset,
                nsamples=args.nsamples,
                seed=0,
                model=args.model_path,
                seqlen=2048,
            )
            qargs.dataloader = dataloader

        print(f"开始量化：w_bit={args.w_bit}, group={args.w_group_size}, format={args.w_format}")
        model = quantize_model(model, qargs)
    else:
        print("w_bit >= 16，跳过量化，直接使用原模型。")

    model = model.to(args.device).eval()
    return model, tokenizer


def run_eval(args: argparse.Namespace) -> None:
    prepare_env(args)

    model, tokenizer = build_quantized_model(args)

    # 将量化后的模型封装为 HFLM（传递已初始化的 model/tokenizer）
    hf_lm = HFLM(
        pretrained=model,
        tokenizer=tokenizer,
        backend="causal",
        device=args.device,
        batch_size=args.batch_size,
        max_batch_size=args.max_batch_size,
        dtype="auto",
        trust_remote_code=False,
    )

    task_manager = tasks.TaskManager()

    results = evaluator.simple_evaluate(
        model=hf_lm,
        tasks=args.tasks,
        num_fewshot=args.num_fewshot,
        batch_size=None,  # 避免与 HFLM 内部 batch_size 重复
        task_manager=task_manager,
    )

    print("=== 评测结果 ===")
    import json

    print(json.dumps(results["results"], indent=2, ensure_ascii=False))
    print("\nVersions:", json.dumps(results.get("versions", {}), indent=2))


if __name__ == "__main__":
    parsed_args = parse_args()
    run_eval(parsed_args)

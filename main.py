#!/usr/bin/env python3
"""
Unified entry: weight-only quantization + PPL evaluation or lm-evaluation-harness downstream tasks.

- eval_mode=ppl: compute perplexity on WikiText/PTB/C4 style datasets.
- eval_mode=lm_eval: run lm-evaluation-harness tasks (arc_easy, boolq, etc.).
"""

import argparse
import json
import math
import os
import time
from pathlib import Path
from typing import Optional, Tuple

import torch

os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
for env_key in ("HF_ENDPOINT", "HF_MIRROR", "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY"):
    os.environ.pop(env_key, None)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import sys
sys.path.append(".")
sys.path.append("./gptq")
from gptq.datautils import get_loaders
from quant_wrapper import quantize_model
from quant_linear import configure_fp_formats
from utils import build_model_and_enc
from visualize_utils import plot_random_fp8_exponent_dists, plot_fp8_exponent_heatmaps, count_fp8_exponent_outliers

# lm-eval imports（仅在 lm_eval 模式下使用）
from lm_eval import evaluator, tasks
from lm_eval.models.huggingface import HFLM


class SequentialPPLEvaluator:
    """Sequential perplexity evaluator matching GPTQ opt.py behavior."""

    DATASET_MAP = {
        "wikitext": "wikitext2",
        "ptb": "ptb",
        "c4": "c4",
    }

    def __init__(self, model, model_path, device, seqlen=None):
        self.model = model
        self.model_path = model_path
        self.device = device
        if seqlen is not None:
            self.seqlen = seqlen
        elif hasattr(model, "seqlen") and model.seqlen is not None:
            self.seqlen = int(model.seqlen)
        elif hasattr(model.config, "max_position_embeddings") and model.config.max_position_embeddings:
            self.seqlen = int(model.config.max_position_embeddings)
        else:
            self.seqlen = 2048
        self.test_cache = {}

    def _dataset_key(self, dataset_name):
        return self.DATASET_MAP.get(dataset_name.lower(), dataset_name)

    def _load_tokens(self, dataset_name):
        key = self._dataset_key(dataset_name)
        if key in self.test_cache:
            return self.test_cache[key]
        _, test_data = get_loaders(key, nsamples=1, seed=0, model=self.model_path, seqlen=self.seqlen)
        tokens = getattr(test_data, "input_ids", test_data)
        if not isinstance(tokens, torch.Tensor):
            tokens = torch.tensor(tokens, dtype=torch.long)
        if tokens.dim() == 1:
            tokens = tokens.unsqueeze(0)
        self.test_cache[key] = tokens.long()
        return self.test_cache[key]

    def calculate_ppl(self, dataset_name, max_chunks=None):
        tokens = self._load_tokens(dataset_name)
        total_len = tokens.shape[1]
        nsamples = total_len // self.seqlen
        if nsamples == 0:
            raise ValueError(f"Dataset {dataset_name} is shorter than the model sequence length ({self.seqlen}).")
        if max_chunks is not None and max_chunks > 0:
            nsamples = min(nsamples, max_chunks)
        model = self.model.to(self.device)
        model.eval()
        total_nll = 0.0
        total_tokens = 0
        with torch.no_grad():
            for i in range(nsamples):
                chunk = tokens[:, i * self.seqlen:(i + 1) * self.seqlen].to(self.device)
                outputs = model(chunk, labels=chunk)
                chunk_tokens = chunk.size(1)
                total_nll += outputs.loss.item() * chunk_tokens
                total_tokens += chunk_tokens
        if total_tokens == 0:
            return float('inf'), 0, nsamples
        ppl = math.exp(total_nll / total_tokens)
        return ppl, total_tokens, nsamples


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Quantize model then evaluate (PPL or lm-eval tasks)")
    p.add_argument("--eval_mode", type=str, default="ppl", choices=["ppl", "lm_eval"], help="选择评测模式：ppl 或 lm_eval")
    p.add_argument("--model_path", type=str, required=True, help="原始 FP16 模型目录")
    p.add_argument("--device", type=str, default="cuda", help="推理设备，如 cuda 或 cpu")
    p.add_argument("--use_flash_attn", action="store_true", help="是否启用 Flash Attention (构建模型时)")
    p.add_argument("--output_dir", type=str, default="./ppl_experiment_results", help="结果输出目录（ppl 会写入文件）")
    p.add_argument("--local_dataset_dir", type=str, default="/home/liutielong/Files_2025/data/ppl_datasets", help="PPL 数据集本地目录，可通过环境变量使用")

    # 量化参数
    p.add_argument("--w_bits", nargs="+", type=int, default=[16, 8, 4], help="权重量化位宽列表，<16 则执行量化")
    p.add_argument("--w_group_size", type=int, default=-2, choices=[-1, -2, 32, 64, 128, 256, 512, 1024], help="-1: per-tensor, -2: per-channel, >0: per-group")
    p.add_argument("--w_symmetric", action="store_true", help="权重对称量化")
    p.add_argument("--w_format", type=str, default="int", choices=["int", "fp4", "fp6", "fp8", "bfp"], help="权重量化格式")
    p.add_argument("--quant_dim", type=int, default=1, choices=[0, 1], help="近似量化分组维度：0 按行/输入维分组，1 按列/输出维分组")
    p.add_argument("--mode", type=int, default=0, choices=[0, 1, 2], help="0: GPU, 1: FIGLUT-F, 2: FIGLUT-I")
    p.add_argument("--approximate", action="store_true", help="近似量化")
    p.add_argument("--double_approximate", action="store_true", help="两层近似量化")
    # 可配置 FP4/FP6/FP8 结构
    p.add_argument("--fp4_exp_bits", type=int, default=2, help="FP4 指数字段位数")
    p.add_argument("--fp4_mantissa_bits", type=int, default=1, help="FP4 尾数字段位数（不含前导1）")
    p.add_argument("--fp6_exp_bits", type=int, default=3, help="FP6 指数字段位数")
    p.add_argument("--fp6_mantissa_bits", type=int, default=2, help="FP6 尾数字段位数（不含前导1）")
    p.add_argument("--fp8_exp_bits", type=int, default=4, help="FP8 指数字段位数")
    p.add_argument("--fp8_mantissa_bits", type=int, default=3, help="FP8 尾数字段位数（不含前导1）")
    # 近似 FP6/FP8 解码参数
    p.add_argument("--fp6_hi_align_start", type=int, default=4, help="FP6 近似解码：高指数对齐起始指数字段值")
    p.add_argument("--fp6_hi_align_exp_field", type=int, default=7, help="FP6 近似解码：高指数对齐到的指数字段值")
    p.add_argument("--fp6_tail_pad_bits", type=int, default=2, help="FP6 近似解码：尾数右移前补的低位0数量（可为负表示右移截断）")
    p.add_argument("--fp8_hi_align_start", type=int, default=12, help="FP8 近似解码：高指数对齐起始指数字段值")
    p.add_argument("--fp8_hi_align_exp_field", type=int, default=15, help="FP8 近似解码：高指数对齐到的指数字段值")
    p.add_argument("--fp8_tail_pad_bits", type=int, default=1, help="FP8 近似解码：尾数右移前补的低位0数量（可为负表示右移截断）")

    # GPTQ 相关
    p.add_argument("--gptq", action="store_true", help="是否使用 GPTQ")
    p.add_argument("--nsamples", type=int, default=128, help="GPTQ 校准样本数")
    p.add_argument("--percdamp", type=float, default=0.01, help="GPTQ damp 参数")
    p.add_argument("--act_order", action="store_true", help="GPTQ activation order")
    p.add_argument("--calib_dataset", type=str, default="wikitext2", help="GPTQ 校准数据集 (get_loaders 名称)")

    # PPL 评测参数
    p.add_argument("--datasets", nargs="+", default=["wikitext", "ptb", "c4"], choices=["wikitext", "ptb", "c4"], help="PPL 评测数据集")
    p.add_argument("--sample_size", type=int, default=None, help="每个数据集使用的 chunk 数（None 表示全量）")

    # lm-eval 评测参数
    p.add_argument("--tasks", nargs="+", default=["arc_easy"], help="lm-eval 任务列表")
    p.add_argument("--num_fewshot", type=int, default=0, help="few-shot 样本数")
    p.add_argument("--batch_size", type=int, default=1, help="lm-eval batch size（传给 HFLM）")
    p.add_argument("--max_batch_size", type=int, default=None, help="lm-eval 最大自适应 batch size")
    p.add_argument("--offline", action="store_true", help="lm-eval 强制离线，只用本地缓存")
    p.add_argument("--hf_cache", type=str, default=None, help="可选：指定 HF 数据/模型缓存目录")

    return p.parse_args()


def prepare_env(args: argparse.Namespace) -> None:
    """Prepare HF offline/cache env for lm-eval."""
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


def make_quant_args(args: argparse.Namespace, w_bit: int):
    class QuantArgs:
        def __init__(self):
            self.w_bit = w_bit
            self.w_group_size = args.w_group_size
            self.w_symmetric = args.w_symmetric
            self.w_format = args.w_format
            self.approximate = args.approximate
            self.double_approximate = args.double_approximate
            self.quant_dim = args.quant_dim
            self.fp8_hi_align_start = args.fp8_hi_align_start
            self.fp8_hi_align_exp_field = args.fp8_hi_align_exp_field
            self.fp8_tail_pad_bits = args.fp8_tail_pad_bits
            self.fp6_hi_align_start = args.fp6_hi_align_start
            self.fp6_hi_align_exp_field = args.fp6_hi_align_exp_field
            self.fp6_tail_pad_bits = args.fp6_tail_pad_bits
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
    return QuantArgs()


def build_and_quantize(args: argparse.Namespace, w_bit: int, device: str, calib_dataset: Optional[str] = None) -> Tuple[torch.nn.Module, object]:
    """Load model/tokenizer, run quantization (on CPU) if needed, and move to device."""
    # 配置 FP4/6/8 参数（影响 quant_linear 内部常量）
    configure_fp_formats(
        fp4_exp_bits=args.fp4_exp_bits,
        fp4_mantissa_bits=args.fp4_mantissa_bits,
        fp6_exp_bits=args.fp6_exp_bits,
        fp6_mantissa_bits=args.fp6_mantissa_bits,
        fp8_exp_bits=args.fp8_exp_bits,
        fp8_mantissa_bits=args.fp8_mantissa_bits,
    )

    model, tokenizer = build_model_and_enc(
        args.model_path,
        args.use_flash_attn,
        kv_bit=16,
        kv_group_size=128,
    )

    if w_bit < 16:
        print(f"⚙️ Applying quantization: w_bit={w_bit}, group={args.w_group_size}, format={args.w_format}")
        original_device = next(model.parameters()).device
        model = model.cpu()
        torch.cuda.empty_cache()
        qargs = make_quant_args(args, w_bit)
        if args.gptq:
            dataset_name = calib_dataset or args.calib_dataset
            # 兼容 wikitext -> wikitext2
            if dataset_name == "wikitext":
                dataset_name = "wikitext2"
            dataloader, _ = get_loaders(
                dataset_name,
                nsamples=args.nsamples,
                seed=0,
                model=args.model_path,
                seqlen=2048,
            )
            qargs.dataloader = dataloader
        model = quantize_model(model, qargs)
        # 可视化
        # if args.w_format == "fp4" and 4 in args.w_bits:
        #     plot_random_fp4_dists(model, k=10, seed=0, save_path="./Iron_weight_only_quant/results/fp4_dists.png")
        #     plot_random_fp4_exponent_dists(model, k=10, seed=0, save_path="./Iron_weight_only_quant/results/fp4_exponent_dists.png")
        # if args.w_format == "fp6" and 6 in args.w_bits:
        #     plot_random_fp6_dists(model, k=10, seed=0, save_path="./Iron_weight_only_quant/results/fp6_dists.png")
        #     plot_random_fp6_uniform_bins(model, k=10, seed=0, num_bins=16, save_path="./Iron_weight_only_quant/results/fp6_uniform_bins.png")
        #     plot_random_fp6_exponent_dists(model, k=10, seed=0, save_path="./Iron_weight_only_quant/results/fp6_exponent_dists.png")
        # if args.w_format == "fp8" and 8 in args.w_bits:
        #     plot_random_fp8_dists(model, k=10, seed=0, save_path="./Iron_weight_only_quant/results/fp8_dists.png")
        #     plot_random_fp8_uniform_bins(model, k=10, seed=0, num_bins=32, save_path="./Iron_weight_only_quant/results/fp8_uniform_bins.png")
        #     plot_random_fp8_exponent_dists(model, k=10, seed=0, save_path="./Iron_weight_only_quant/results/fp8_exponent_dists.png")
        #     plot_fp8_exponent_heatmaps(model, num_layers=4, block_size=64, seed=0, save_path="./Iron_weight_only_quant/results/fp8_exponent_heatmaps.png")
        #     count_fp8_exponent_outliers(model, group_size=4, threshold=11, exp_bits=args.fp8_exp_bits, mant_bits=args.fp8_mantissa_bits, k=10, seed=0)
        model = model.to(device)
        torch.cuda.empty_cache()
    else:
        model = model.to(device)

    model.eval()
    return model, tokenizer


def run_ppl(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results = {}

    if args.local_dataset_dir:
        os.environ["LOCAL_PPL_DATASET_DIR"] = args.local_dataset_dir

    if args.w_group_size == -1:
        granularity = "tensor"
    elif args.w_group_size == -2:
        granularity = "channel"
    else:
        granularity = f"group{args.w_group_size}"

    print(f"🚀 Starting PPL experiment on {args.datasets}")
    for w_bit in args.w_bits:
        config_name = f"w{w_bit}_{granularity}"
        print(f"\n{'='*50}\n🔄 Testing {config_name.upper()} Quantization\n{'='*50}")
        calib_ds = args.datasets[0] if args.gptq else None
        model, tokenizer = build_and_quantize(args, w_bit, args.device, calib_dataset=calib_ds)
        evaluator = SequentialPPLEvaluator(model.half(), args.model_path, args.device)
        results[config_name] = {}
        for dataset_name in args.datasets:
            print(f"\n📊 Evaluating on {dataset_name.upper()}...")
            start_time = time.time()
            ppl, token_count, chunk_count = evaluator.calculate_ppl(dataset_name, max_chunks=args.sample_size)
            elapsed = time.time() - start_time
            results[config_name][dataset_name] = {
                "perplexity": ppl,
                "num_tokens": token_count,
                "num_chunks": chunk_count,
                "eval_time": elapsed,
            }
            print(f"📝 {dataset_name}: chunks={chunk_count}, tokens={token_count}, ppl={ppl:.4f}, time={elapsed:.2f}s")
        del model, tokenizer
        torch.cuda.empty_cache()

    results_file = output_dir / "ppl_results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n💾 PPL results saved to: {results_file}")


def run_lm_eval(args: argparse.Namespace) -> None:
    prepare_env(args)
    all_results = {}
    for w_bit in args.w_bits:
        config_name = f"w{w_bit}_g{args.w_group_size}"
        print(f"\n{'='*50}\n🔄 Evaluating {config_name.upper()} on lm-eval tasks\n{'='*50}")
        model, tokenizer = build_and_quantize(args, w_bit, args.device, calib_dataset=args.calib_dataset)
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
        res = evaluator.simple_evaluate(
            model=hf_lm,
            tasks=args.tasks,
            num_fewshot=args.num_fewshot,
            batch_size=None,
            task_manager=task_manager,
        )
        all_results[config_name] = res["results"]
        print(json.dumps(res["results"], indent=2, ensure_ascii=False))
        del model, tokenizer, hf_lm
        torch.cuda.empty_cache()

    # 如需保存 lm-eval 结果
    out_path = Path(args.output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    results_file = out_path / "lm_eval_results.json"
    with open(results_file, "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\n💾 lm-eval results saved to: {results_file}")


def main():
    args = parse_args()
    if args.eval_mode == "ppl":
        run_ppl(args)
    else:
        run_lm_eval(args)


if __name__ == "__main__":
    main()

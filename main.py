#!/usr/bin/env python3
"""
Weight-only Quantization Perplexity Experiment
Supports LLaMA and OPT model families with configurable quantization settings.
Evaluates different bit widths and granularities on WikiText, PTB, and C4 datasets.
"""

import argparse
import os
import torch
import json
import time
import math
from pathlib import Path
import sys
sys.path.append(".")
sys.path.append("./gptq")
from gptq.datautils import get_loaders
from quant_wrapper import quantize_model
from utils import build_model_and_enc
from visualize_utils import *

def setup_args():
    parser = argparse.ArgumentParser(description="Weight-only Quantization PPL Experiment")
    parser.add_argument("--model_path", type=str,
                        required=True,
                        help="Path to the model (supports LLaMA and OPT families)")
    parser.add_argument("--output_dir", type=str,
                        default="./ppl_experiment_results",
                        help="Directory to save results")
    parser.add_argument("--use_flash_attn", action="store_true",
                        help="Use Flash Attention")
    parser.add_argument("--datasets", nargs="+",
                        default=["wikitext", "ptb", "c4"],
                        choices=["wikitext", "ptb", "c4"],
                        help="Datasets to evaluate on")
    parser.add_argument("--sample_size", type=int, default=None,
                        help="Maximum number of evaluation chunks (each chunk has length equal to the model sequence length)")
    parser.add_argument("--local_dataset_dir", type=str,
                        default="/home/liutielong/Files_2025/data/ppl_datasets",
                        help="Directory containing local datasets")
    parser.add_argument("--w_bits", nargs="+", type=int,
                        default=[16, 8, 4],
                        help="Weight quantization bit widths to test")
    parser.add_argument("--w_group_size", type=int,
                        default=-2,
                        choices=[-1, -2, 32, 64, 128, 256],
                        help="Weight quantization group size (-1: per-tensor, -2: per-channel, >0: per-group)")
    parser.add_argument("--w_symmetric", action="store_true",
                        help="Use symmetric quantization for weights")
    parser.add_argument("--w_format", type=str, default="int", choices=["int", "fp4", "fp6"],
                        help="Weight format: 'int' (原有整数量化) 或 'fp4'/'fp6'")
    parser.add_argument("--mode", type=int, default=0, choices=[0, 1, 2],
                        help="0 for GPU, 1 for FIGLUT-F and 2 for FIGLUT-I")
    # 关于gptq的参数
    parser.add_argument("--gptq", action="store_true",
                        help="Use GPTQ quantization")
    parser.add_argument("--nsamples", type=int, default=128,
                        help="Number of calibration samples for GPTQ")
    parser.add_argument("--percdamp", type=float, default=0.01,
                        help="Percent of average Hessian diagonal for dampening in GPTQ")
    parser.add_argument("--act_order", action="store_true",
                        help="Whether to apply activation order GPTQ heuristic")

    return parser.parse_args()

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


def run_quantization_experiment(args):
    """Main experiment function"""

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {}

    # Generate quantization configurations based on parameters
    quant_configs = {}

    # Determine granularity name
    if args.w_group_size == -1:
        granularity = "tensor"
    elif args.w_group_size == -2:
        granularity = "channel"
    else:
        granularity = f"group{args.w_group_size}"

    # Generate configs for each bit width
    for w_bit in args.w_bits:
        config_name = f"w{w_bit}_{granularity}"
        quant_configs[config_name] = {
            'w_bit': w_bit,
            'w_group_size': args.w_group_size
        }

    print(f"🚀 Starting Weight-only Quantization Experiment")
    print(f"📁 Model: {args.model_path}")
    print(f"📊 Datasets: {args.datasets}")
    print(f"🔢 Quantizations: {list(quant_configs.keys())}")

    for quant_name, config in quant_configs.items():
        print(f"\n{'='*50}")
        print(f"🔄 Testing {quant_name.upper()} Quantization")
        print(f"{'='*50}")

        # Load model and tokenizer
        print("📥 Loading model and tokenizer...")
        model, tokenizer = build_model_and_enc(
            args.model_path,
            args.use_flash_attn,
            kv_bit=16,  # Keep KV cache in FP16
            kv_group_size=128
        )

        # Apply quantization
        if config['w_bit'] < 16:
            print(f"⚙️ Applying {quant_name} quantization...")
            class QuantArgs:
                def __init__(self, w_bit, w_group_size, w_symmetric=False, mode=0, gptq=False, nsamples=128, percdamp=0.01, act_order=False, w_format="int"):
                    self.w_bit = w_bit
                    self.w_group_size = w_group_size
                    self.w_symmetric = w_symmetric
                    self.w_format = w_format
                    self.a_bit = 16  # Keep activation in FP16
                    self.a_group_size = 128
                    self.kv_bit = 16  # Keep KV cache in FP16
                    self.kv_group_size = 128
                    self.mode = mode
                    self.gptq = gptq
                    self.nsamples = nsamples
                    self.percdamp = percdamp
                    self.act_order = act_order

            quant_args = QuantArgs(
                config['w_bit'],
                config['w_group_size'],
                args.w_symmetric,
                args.mode,
                args.gptq,
                args.nsamples,
                args.percdamp,
                args.act_order,
                args.w_format
            )

            # 如果使用GPTQ，需要准备校准数据
            if args.gptq:
                print("📊 Preparing calibration data for GPTQ...")
                # 使用第一个数据集作为校准数据
                calib_dataset = args.datasets[0]
                if calib_dataset == "wikitext":
                    calib_dataset = "wikitext2"
                dataloader, _ = get_loaders(
                    calib_dataset,
                    nsamples=args.nsamples,
                    seed=0,
                    model=args.model_path,
                    seqlen=2048
                )
                quant_args.dataloader = dataloader
            else:
                quant_args.dataloader = None

            model = quantize_model(model, quant_args)
            # 可视化
            if args.w_format == "fp4" and 4 in args.w_bits:
                plot_random_fp4_dists(model, k=10, seed=0, save_path="./results/fp4_dists.png")
                plot_random_fp4_exponent_dists(model, k=10, seed=0, save_path="./results/fp4_exponent_dists.png")
            if args.w_format == "fp6" and 6 in args.w_bits:
                plot_random_fp6_dists(model, k=10, seed=0, save_path="./results/fp6_dists.png")
                plot_random_fp6_uniform_bins(model, k=10, seed=42, num_bins=16, save_path="./results/fp6_uniform_bins.png")
                plot_random_fp6_exponent_dists(model, k=10, seed=0, save_path="./results/fp6_exponent_dists.png")

        # Initialize evaluator
        device = next(model.parameters()).device
        model = model.half()
        evaluator = SequentialPPLEvaluator(model, args.model_path, device)

        results[quant_name] = {}

        # Evaluate on each dataset
        for dataset_name in args.datasets:
            print(f"\n📊 Evaluating on {dataset_name.upper()}...")

            start_time = time.time()
            ppl, token_count, chunk_count = evaluator.calculate_ppl(dataset_name, max_chunks=args.sample_size)
            end_time = time.time()

            results[quant_name][dataset_name] = {
                'perplexity': ppl,
                'num_tokens': token_count,
                'num_chunks': chunk_count,
                'eval_time': end_time - start_time
            }

            print(f"📝 Evaluated {chunk_count} chunks ({token_count} tokens)")
            print(f"📈 Perplexity: {ppl:.4f}")
            print(f"⏱️ Time: {end_time - start_time:.2f}s")

        # Clean up GPU memory
        del model, tokenizer
        torch.cuda.empty_cache()

    # Save results
    results_file = output_dir / "ppl_results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)

    # Print summary table
    print(f"\n{'='*70}")
    print("📊 PERPLEXITY RESULTS SUMMARY")
    print(f"{'='*70}")

    # Print header
    header = f"{'Dataset':<12}"
    for quant_name in quant_configs.keys():
        header += f"{quant_name.upper():>12s}"
    print(header)
    print("-" * 70)

    # Print results for each dataset
    for dataset in args.datasets:
        if any(dataset in results.get(quant_name, {}) for quant_name in quant_configs.keys()):
            row = f"{dataset.upper():<12}"
            for quant_name in quant_configs.keys():
                if dataset in results.get(quant_name, {}):
                    ppl = results[quant_name][dataset]['perplexity']
                    row += f"{ppl:12.2f}"
                else:
                    row += f"{'N/A':>12s}"
            print(row)

    print(f"\n💾 Results saved to: {results_file}")

    return results

if __name__ == "__main__":
    args = setup_args()

    # Validate model path
    if not Path(args.model_path).exists():
        print(f"❌ Model path does not exist: {args.model_path}")
        print("Please update --model_path to point to your model (LLaMA or OPT)")
        exit(1)

    print("🚀 Starting Weight-only Quantization PPL Experiment")
    print(f"📁 Model: {args.model_path}")
    print(f"📊 Datasets: {', '.join(args.datasets)}")
    print(f"🔢 Bit widths: {args.w_bits}")
    print(f"🔧 Group size: {args.w_group_size}")

    if args.local_dataset_dir:
        os.environ["LOCAL_PPL_DATASET_DIR"] = args.local_dataset_dir

    results = run_quantization_experiment(args)

import torch
from typing import Optional
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from fastchat.model import get_conversation_template

def get_module_by_name_suffix(model, module_name: str):
    for name, module in model.named_modules():
        if name.endswith(module_name):
            return module

# build model and tokenizer
def build_model_and_enc(model_path, use_flash_attn, kv_bit=16, kv_group_size=128):
    print(f"* Building model {model_path}")

    # weither trust remote code
    if 'chatglm' in model_path or 'mpt' in model_path or 'stable' in model_path:
        trust_remote_code = True
    else:
        trust_remote_code = False

    config = AutoConfig.from_pretrained(model_path, trust_remote_code=trust_remote_code)
    if use_flash_attn and 'chatglm' not in model_path and 'mpt' not in model_path:
        config._flash_attn_2_enabled = True
        config._attn_implementation = "flash_attention_2"
    elif use_flash_attn and 'mpt' in model_path:
        config.attn_config['attn_impl'] = 'triton'
    else:
        config._flash_attn_2_enabled = False
        config._attn_implementation = None

    # add the kv quantization parameters
    config.kv_bit = kv_bit
    config.kv_group_size = kv_group_size

    # load tokenizer
    if 'mpt' in model_path or 'stable' in model_path:
        use_fast = True
    else:
        use_fast = False
    enc = AutoTokenizer.from_pretrained(model_path, use_fast=use_fast, trust_remote_code=trust_remote_code)

    # load model
    kwargs = {"torch_dtype": torch.float16, "device_map": "balanced"}
    model = AutoModelForCausalLM.from_pretrained(model_path, config=config, trust_remote_code=trust_remote_code, **kwargs)
    return model, enc

def download_model(model_name, use_auth_token):
    # download tokenizer
    while True:
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=use_auth_token, trust_remote_code=True)
        except:
            continue
        break

    # download model
    while True:
        try:
            model = AutoModelForCausalLM.from_pretrained(model_name, use_auth_token=use_auth_token, device_map="auto", resume_download=True, trust_remote_code=True)
        except:
            continue
        break


def format_chat_prompt(input, model_name):
    if "longchat" in model_name.lower():
        conv = get_conversation_template("vicuna")
    else:
        conv = get_conversation_template(model_name)

    # add system call
    if 'llama' in model_name.lower():
        conv.set_system_message("You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Please ensure that your responses are socially unbiased and positive in nature. If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.")
    conv.append_message(conv.roles[0], input)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    return prompt


import transformers
import transformers.models.llama.modeling_llama
from torch.distributed import get_rank, is_initialized
from functools import partial
import matplotlib.pyplot as plt

def rank0_print(*args):
    if is_initialized():
        if get_rank() == 0:
            print(*args)
    else:
        print(*args)

class CondenseRotaryEmbedding(torch.nn.Module):
    def __init__(self, dim, ratio, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float().to(device) / dim))
        self.register_buffer("inv_freq", inv_freq)
        
        # Build here to make `torch.jit.trace` work.
        self.ratio = ratio
        max_position_embeddings *= ratio
        rank0_print(f"Condensing Positional embeddings from {max_position_embeddings} to {max_position_embeddings // ratio}")
        self.max_seq_len_cached = max_position_embeddings
        t = torch.arange(self.max_seq_len_cached, device=self.inv_freq.device, dtype=self.inv_freq.dtype) / ratio
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        dtype = torch.get_default_dtype()
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :].to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :].to(dtype), persistent=False)

    def forward(self, x, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        # This `if` block is unlikely to be run after we build sin/cos in `__init__`. Keep the logic here just in case.
        if seq_len > self.max_seq_len_cached:
            self.max_seq_len_cached = seq_len
            t = torch.arange(self.max_seq_len_cached, device=x.device, dtype=self.inv_freq.dtype) / self.ratio
            freqs = torch.einsum("i,j->ij", t, self.inv_freq)
            # Different from paper, but it uses a different permutation in order to obtain the same calculation
            emb = torch.cat((freqs, freqs), dim=-1).to(x.device)
            self.register_buffer("cos_cached", emb.cos()[None, None, :, :].to(x.dtype), persistent=False)
            self.register_buffer("sin_cached", emb.sin()[None, None, :, :].to(x.dtype), persistent=False)
        return (
            self.cos_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
            self.sin_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
        )

def replace_llama_with_condense(ratio):
    transformers.models.llama.modeling_llama.LlamaRotaryEmbedding = partial(CondenseRotaryEmbedding, ratio=ratio)


def visualize_fp16_bit_sparsity(
    data: torch.Tensor,
    save_path: Optional[str] = "fp16_mantissa_bit_sparsity.png",
):
    """
    将一组FP16数据解析为符号/指数/尾数, 按最大指数对齐尾数(补前导1, 次正规数补0),
    右移后截断到13位, 并绘制每个bit位为0的次数。
    Args:
        data: 任意形状的tensor, 将被视为FP16数据。
        save_path: 稀疏度柱状图的保存路径。
    Returns:
        一个dict, 包含解析出的各字段以及对齐后13位尾数的0计数。
    """
    if data.numel() == 0:
        raise ValueError("Input tensor is empty.")

    # 展平并确保是FP16
    x = data.detach().to(torch.float16).flatten()
    raw = x.view(torch.uint16)
    raw_i32 = raw.to(torch.int32)  # torch.uint16 lacks bitshift on some backends

    sign_bits = (raw_i32 >> 15) & 0x1               # 符号位
    exp_bits = (raw_i32 >> 10) & 0x1F               # 5位指数
    mant_bits = raw_i32 & 0x3FF                     # 10位尾数

    bias = 15
    is_subnormal = exp_bits == 0
    exp_unbiased = torch.where(
        is_subnormal,
        torch.full_like(exp_bits, 1 - bias, dtype=torch.int32),
        exp_bits.to(torch.int32) - bias,
    )
    max_exp = exp_unbiased.max()

    # 补前导位 (正规数补1, 次正规补0)
    leading = torch.where(is_subnormal, torch.zeros_like(mant_bits), torch.ones_like(mant_bits))
    mantissa_with_lead = (leading << 10) | mant_bits  # 11位
    mantissa_extended = (mantissa_with_lead.to(torch.int32) << 2)  # 添加两个低位0, 最多13位

    # 按最大指数右移对齐, 超过13位部分截断
    shift = (max_exp - exp_unbiased).clamp(min=0, max=31).to(torch.int32)
    aligned = mantissa_extended >> shift
    aligned = aligned & ((1 << 13) - 1)  # 仅保留低13位

    bit_positions = torch.arange(13, device=aligned.device)
    aligned_bits = ((aligned.unsqueeze(-1) >> bit_positions) & 0x1).to(torch.int32)  # [N, 13]
    zero_counts = (aligned_bits == 0).sum(dim=0).cpu()
    zero_counts = torch.flip(zero_counts, dims=[0])  # now index 0 corresponds to highest bit

    # 绘制稀疏度柱状图（可选）
    if save_path is not None:
        plt.figure(figsize=(8, 3.5))
        bits_for_plot = list(range(12, -1, -1))  # high bit on the left
        plt.bar(bits_for_plot, zero_counts.numpy(), color="slateblue", alpha=0.85)
        plt.xlabel("Aligned mantissa bit (12 = MSB, 0 = LSB after truncation)")
        plt.ylabel("Zero count")
        plt.title("FP16 aligned mantissa bit sparsity")
        plt.tight_layout()
        plt.savefig(save_path, dpi=200)
        plt.close()

    return {
        "sign_bits": sign_bits.cpu(),
        "exponent_bits": exp_bits.cpu(),
        "mantissa_bits": mant_bits.cpu(),
        "aligned_bits": aligned_bits.cpu(),  # 0/1, shape [N, 13]
        "zero_counts": zero_counts,
        "save_path": save_path,
    }


if __name__ == "__main__":
    x = torch.randn(128, dtype=torch.float16)
    out = visualize_fp16_bit_sparsity(x, save_path="bit_sparsity.png")
    print("Zero counts per bit:", out["zero_counts"])
    print("Plot saved to:", out["save_path"])

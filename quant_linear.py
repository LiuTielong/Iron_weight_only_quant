import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from quant_funcs import pseudo_quantize_tensor

MANTISSA_BITS = 12
FP4_EXP_BITS = 2
FP4_MANTISSA_BITS = 1
FP4_EXP_BIAS = 2 ** (FP4_EXP_BITS - 1) - 1
FP6_EXP_BITS = 3
FP6_MANTISSA_BITS = 2
FP6_EXP_BIAS = 2 ** (FP6_EXP_BITS - 1) - 1
FP8_EXP_BITS = 4
FP8_MANTISSA_BITS = 3
FP8_EXP_BIAS = 2 ** (FP8_EXP_BITS - 1) - 1


def _pre_align_fp_activation(input: torch.Tensor):
    """
    模拟FIGLUT-I中的激活值预对齐操作。
    将输入的浮点激活值对齐到每行（最后一个维度）的最大指数。
    Args:
        input: 浮点激活值张量, 为FP16。
    Returns:
        A tuple of (aligned_mantissas, exponents_max):
        - aligned_mantissas: 预对齐后的激活值（逻辑上的尾数部分）。
        - exponents_max: 用于对齐的最大指数。
    """
    assert input.dim() == 2
    device = input.device
    # ================== 1. 找到每行的最大指数 E_max ==================
    abs_input = torch.abs(input)
    nonzero_mask = abs_input > 0  # 避免对0取log2
    # 初始化指数为一个非常小的值
    exponents = torch.full_like(input, -float('inf'), dtype=torch.float32)
    if torch.any(nonzero_mask):
        min_nonezero = torch.min(abs_input[nonzero_mask])
        # 用非0最小值填充0值，以便log2计算。实际上该数的指数非常小，不影响其他数
        safe_abs_input = torch.where(nonzero_mask, abs_input, min_nonezero)  # 这句代码表示condition为True的位置取x的对应元素，否则取y的对应元素
        # 计算每个元素的指数
        calculated_exponents = torch.floor(torch.log2(safe_abs_input))
        # 仅对原始非0值应用计算出的指数
        exponents = torch.where(nonzero_mask, calculated_exponents, exponents)
    
    # 计算每行（最后一个维度）的最大指数
    exponents_max = torch.max(exponents, dim=-1, keepdim=True)[0]
    # 处理整行都是0的特殊情况，此时exponents_max为-inf
    exponents_max = torch.where(torch.isfinite(exponents_max), exponents_max, torch.zeros_like(exponents_max))
    exponents_max = exponents_max.to(input.dtype)

    # ================== 2. 计算对齐后的浮点尾数 ==================
    scaling_factors = torch.pow(2.0, exponents_max)
    aligned_mantissas = (input.to(torch.float32) / scaling_factors.to(torch.float32))

    # ================== 3. 转换为34位精度的定点整数 ==================
    # 乘以 2^33 将其缩放到一个大的整数范围，一共保留34位有效数
    # 使用.long()转换成64位整数以防溢出
    aligned_int_mantissas = torch.round(aligned_mantissas.double() * 2**(MANTISSA_BITS-1)).long()

    # ================== 4. 提取符号位 ==================
    # 1 表示负数, 0 表示正数或零
    sign_bit = (input < 0).long().unsqueeze(-1)  # 形状: [m, n, 1]

    # ================== 5. 提取34位尾数 (向量化实现) ==================
    abs_mantissas = torch.abs(aligned_int_mantissas).unsqueeze(-1)  # 形状: [m, n, 1]
    # 创建一个用于提取位的除数向量：[2^33, 2^32, ..., 2^0]
    bit_exponents = torch.arange(MANTISSA_BITS-1, -1, -1, device=device).long()
    divisors = torch.pow(2, bit_exponents)

    # 通过广播和整数除法/取模来高效地获取每一位
    # (abs_mantissas // divisors) -> [m, n, 34]
    # ... % 2 -> 得到每一位是0还是1
    mantissa_bits = (abs_mantissas // divisors) % 2 # 形状: [m, n, 34]

    # ================== 6. 拼接符号位和尾数位 ==================
    output_bits = torch.cat([sign_bit, mantissa_bits], dim=-1) # 形状: [m, n, 35]

    # 7. 返回结果
    # 将行指数的形状从 [m, 1] 压缩到 [m]
    return output_bits, exponents_max.squeeze(-1)


def configure_fp_formats(
    fp4_exp_bits: int = FP4_EXP_BITS,
    fp4_mantissa_bits: int = FP4_MANTISSA_BITS,
    fp6_exp_bits: int = FP6_EXP_BITS,
    fp6_mantissa_bits: int = FP6_MANTISSA_BITS,
    fp8_exp_bits: int = FP8_EXP_BITS,
    fp8_mantissa_bits: int = FP8_MANTISSA_BITS,
):
    """
    更新 FP4/FP6/FP8 的指数字段与尾数字段位宽，偏置随指数位宽自动计算。
    便于通过外部 args 进行配置，不改变默认值。
    """
    global FP4_EXP_BITS, FP4_MANTISSA_BITS, FP4_EXP_BIAS
    global FP6_EXP_BITS, FP6_MANTISSA_BITS, FP6_EXP_BIAS
    global FP8_EXP_BITS, FP8_MANTISSA_BITS, FP8_EXP_BIAS

    FP4_EXP_BITS = int(fp4_exp_bits)
    FP4_MANTISSA_BITS = int(fp4_mantissa_bits)
    FP4_EXP_BIAS = 2 ** (FP4_EXP_BITS - 1) - 1

    FP6_EXP_BITS = int(fp6_exp_bits)
    FP6_MANTISSA_BITS = int(fp6_mantissa_bits)
    FP6_EXP_BIAS = 2 ** (FP6_EXP_BITS - 1) - 1

    FP8_EXP_BITS = int(fp8_exp_bits)
    FP8_MANTISSA_BITS = int(fp8_mantissa_bits)
    FP8_EXP_BIAS = 2 ** (FP8_EXP_BITS - 1) - 1

def _rounding_rshift(val: torch.Tensor, shift: torch.Tensor | int) -> torch.Tensor:
    """带四舍五入的右移，shift 可为逐元素位移。"""
    if not torch.is_tensor(shift):
        shift = torch.tensor(shift, device=val.device, dtype=val.dtype)
    else:
        shift = shift.to(val.dtype)
    offset = torch.zeros_like(val)
    mask = shift > 0
    if mask.any():
        offset_masked = torch.bitwise_left_shift(torch.ones_like(val[mask]), shift[mask] - 1)
        offset[mask] = offset_masked
    return torch.bitwise_right_shift(val + offset, shift)


def _float_to_fp(x: torch.Tensor, exp_bits, mant_bits, exp_bias) -> torch.Tensor:
    """
    将浮点张量量化为带次正规支持的FP4/6/8码字。
    """
    sign = (x < 0).to(torch.uint8)
    x_abs = x.abs()
    zero_mask = x_abs == 0
    x_abs_safe = torch.where(zero_mask, torch.full_like(x_abs, 1e-8), x_abs)

    max_exp_field = (1 << exp_bits) - 1
    min_normal_exp = 1 - exp_bias  # 最小正规数的无偏指数。因为对于有偏指数编码，指数字段从1开始表示正规数。当它取1时，无偏指数就是1-exp_bias。

    # 原始无偏指数
    exp_val = torch.floor(torch.log2(x_abs_safe)).to(torch.int32)

    # 判定是否为次正规（真实指数低于可表示的正规数最小指数）
    is_subnormal = exp_val < min_normal_exp

    # 正规数路径：指数夹在可表示范围内，再根据夹后的指数计算尾数
    exp_clamped = torch.clamp(exp_val, min_normal_exp, max_exp_field - exp_bias)
    exp_unbiased = (exp_clamped + exp_bias).to(torch.uint8)
    mant_scale = 1 << mant_bits
    mant_normal = torch.round((x_abs_safe / torch.pow(2.0, exp_clamped.float()) - 1.0) * mant_scale)
    mant_normal = mant_normal.clamp(0, mant_scale - 1).to(torch.uint8)

    # 次正规路径：指数字段为0，尾数直接近似 x_abs / 2^(min_normal_exp)
    subnorm_denom = torch.pow(torch.tensor(2.0, device=x.device), float(min_normal_exp))
    mant_sub = torch.round((x_abs_safe / subnorm_denom) * mant_scale)
    mant_sub = mant_sub.clamp(0, mant_scale - 1).to(torch.uint8)

    exp_field = torch.where(is_subnormal, torch.zeros_like(exp_unbiased), exp_unbiased)
    mant_field = torch.where(is_subnormal, mant_sub, mant_normal)

    code = (sign << (exp_bits + mant_bits)) | (exp_field << mant_bits) | mant_field
    # 零值强制编码为0，便于解码时恢复0
    code = torch.where(zero_mask, torch.zeros_like(code), code)
    return code

# def _float_to_fp(x: torch.Tensor, exp_bits, mant_bits, exp_bias) -> torch.Tensor:
#     """
#     跟AxCore学的，做成两步量化。
#     将浮点张量量化为带次正规支持的 FP 码字，包括 FP4/FP6/FP8.
#       1) 用 log2 估计指数并生成 scales；
#       2) 先对幅值按 scales 量化；
#       3) 再按 FP 编码规则写入指数字段与尾数字段。
#     """
#     # 1. 记录符号并处理零
#     sign = (x < 0).to(torch.uint8)
#     x_abs = x.abs()
#     zero_mask = x_abs == 0
#     x_abs_safe = torch.where(zero_mask, torch.full_like(x_abs, 1e-8), x_abs)

#     # 2. 依据 _fp_scale 逻辑计算尺度并先量化幅值
#     if (exp_bits == 1 and mant_bits == 2):  # FP4-E1M2，PF6-E3M2, 要将指数clamp为1
#         tensor_log_scales = (torch.floor(torch.log2(x_abs_safe)) + exp_bias).detach()
#         tensor_log_scales = torch.clamp(tensor_log_scales, min=1)
#     else:  # 其他FP，PF4-E2M1, FP8.
#         tensor_log_scales = (torch.floor(torch.log2(x_abs_safe)) + exp_bias).detach()
#     scales = torch.pow(2.0, tensor_log_scales - mant_bits - exp_bias)
#     x_q_abs = torch.round(x_abs_safe / scales) * scales
#     x_q_abs = torch.where(zero_mask, torch.zeros_like(x_q_abs), x_q_abs)

#     # 3. 按 FP 编码生成指数字段与尾数字段（含次正规）
#     max_exp_field = (1 << exp_bits) - 1
#     min_normal_exp = 1 - exp_bias  # 最小正规数的无偏指数

#     exp_val = torch.floor(torch.log2(torch.where(x_q_abs == 0, torch.ones_like(x_q_abs), x_q_abs))).to(torch.int32)
#     is_subnormal = exp_val < min_normal_exp

#     exp_clamped = torch.clamp(exp_val, min_normal_exp, max_exp_field - exp_bias)
#     exp_unbiased = (exp_clamped + exp_bias).to(torch.uint8)
#     mant_scale = 1 << mant_bits
#     mant_normal = torch.round((x_q_abs / torch.pow(2.0, exp_clamped.float()) - 1.0) * mant_scale)
#     mant_normal = mant_normal.clamp(0, mant_scale - 1).to(torch.uint8)

#     subnorm_denom = torch.pow(torch.tensor(2.0, device=x.device), float(min_normal_exp))
#     mant_sub = torch.round((x_q_abs / subnorm_denom) * mant_scale)
#     mant_sub = mant_sub.clamp(0, mant_scale - 1).to(torch.uint8)

#     exp_field = torch.where(is_subnormal, torch.zeros_like(exp_unbiased), exp_unbiased)
#     mant_field = torch.where(is_subnormal, mant_sub, mant_normal)

#     code = (sign << (exp_bits + mant_bits)) | (exp_field << mant_bits) | mant_field
#     code = torch.where(zero_mask, torch.zeros_like(code), code)
#     return code


def _fp_to_float(code: torch.Tensor, exp_bits: int = FP4_EXP_BITS, mant_bits: int = FP4_MANTISSA_BITS, exp_bias: int = FP4_EXP_BIAS) -> torch.Tensor:
    """
    将 FP4/FP6/FP8 码字还原为浮点，支持次正规。输入为 uint8/long，返回 FP32。
    """
    code = code.to(torch.uint8)
    zero_mask = code == 0
    sign = ((code >> (exp_bits + mant_bits)) & 0x1).float()
    raw_exp = ((code >> mant_bits) & ((1 << exp_bits) - 1)).float()
    mant_field = (code & ((1 << mant_bits) - 1)).float()

    # 正规数：隐含1
    exp_normal = raw_exp - float(exp_bias)
    value_normal = (1.0 + mant_field / float(1 << mant_bits)) * torch.pow(2.0, exp_normal)

    # 次正规：无隐含1，指数固定为 1 - bias
    exp_sub = 1.0 - float(exp_bias)
    exp_sub_tensor = torch.tensor(exp_sub, device=code.device)
    value_sub = (mant_field / float(1 << mant_bits)) * torch.pow(torch.tensor(2.0, device=code.device), exp_sub_tensor)

    is_subnormal = raw_exp == 0
    value = torch.where(is_subnormal, value_sub, value_normal)
    value = ((-1.0) ** sign) * value
    return torch.where(zero_mask, torch.zeros_like(value), value)

def _fp_decode_aligned(
    code: torch.Tensor,
    hi_align_start: int,
    hi_align_exp_field: int,
    tail_pad_bits: int,
    exp_bits: int,
    mant_bits: int,
    exp_bias: int,
    align_subnorm_exp_as_one: bool = False,
    limit_align_exp_to_field: bool = True,
) -> torch.Tensor:
    """
    通用的 FP 近似解码：
      - 满足对齐条件的码字指数对齐到 hi_align_exp_field，并在尾数右移前按 tail_pad_bits 做补齐/截断。
      - 其他码字按常规方式解码（含次正规）。
    """
    code = code.to(torch.uint8)
    zero_mask = code == 0
    sign = ((code >> (exp_bits + mant_bits)) & 0x1).float()
    exp_field = ((code >> mant_bits) & ((1 << exp_bits) - 1)).int()
    mant_field = (code & ((1 << mant_bits) - 1)).int()

    align_exp = torch.where(exp_field == 0, torch.ones_like(exp_field), exp_field) if align_subnorm_exp_as_one else exp_field
    leading = torch.where(exp_field == 0, torch.zeros_like(exp_field), torch.ones_like(exp_field))
    mant_full = (leading << mant_bits) | mant_field

    if tail_pad_bits >= 0:
        mant_padded = mant_full << tail_pad_bits
    else:
        mant_padded = _rounding_rshift(mant_full, abs(tail_pad_bits))

    exp_unbiased = torch.where(exp_field == 0, 1 - exp_bias, exp_field - exp_bias).float()
    value_normal = mant_full.float() / (2.0 ** mant_bits) * torch.pow(torch.tensor(2.0, device=code.device), exp_unbiased)

    hi_mask = align_exp >= hi_align_start
    if limit_align_exp_to_field:
        hi_mask = hi_mask & (align_exp <= hi_align_exp_field)

    shift = torch.clamp(hi_align_exp_field - align_exp, min=0)
    mant_aligned = _rounding_rshift(mant_padded, shift)

    hi_unbiased = hi_align_exp_field - exp_bias
    value_hi = mant_aligned.float() / (2.0 ** (mant_bits + tail_pad_bits)) * (2.0 ** float(hi_unbiased))

    value = torch.where(hi_mask, value_hi, value_normal)
    value = torch.where(sign == 1, -value, value)
    return torch.where(zero_mask, torch.zeros_like(value), value)


def fp_decode_aligned_double_approx(
    code: torch.Tensor,
    hi_align_start: int,
    hi_align_exp_field: int,
    tail_pad_bits: int,
    exp_bits: int,
    mant_bits: int,
    exp_bias: int,
    align_subnorm_exp_as_one: bool = False,
    handle_max_outlier: bool = False,
) -> torch.Tensor:
    """
    通用的双近似 FP 解码（4 个一组）：
      - 按组统计 outlier（指数小于 hi_align_start 或大于 hi_align_exp_field）。
      - outlier_count <=1 的组，对齐到 hi_align_exp_field。
      - outlier_count >1 的组，对齐到组内最大指数。
      - handle_max_outlier=True 时，若组内存在最大指数 outlier，则整组对齐到最大指数。
      - align_subnorm_exp_as_one=True 时，对齐指数将次正规视为1。
    """
    code = code.to(torch.uint8).t()
    zero_mask = code == 0
    sign = ((code >> (exp_bits + mant_bits)) & 0x1).float()
    exp_field = ((code >> mant_bits) & ((1 << exp_bits) - 1)).int()
    mant_field = (code & ((1 << mant_bits) - 1)).int()

    align_exp = torch.where(exp_field == 0, torch.ones_like(exp_field), exp_field) if align_subnorm_exp_as_one else exp_field
    leading = torch.where(exp_field == 0, torch.zeros_like(exp_field), torch.ones_like(exp_field))
    mant_full = (leading << mant_bits) | mant_field
    if tail_pad_bits >= 0:
        mant_padded = mant_full << tail_pad_bits
    else:
        mant_padded = _rounding_rshift(mant_full, torch.full_like(mant_full, abs(tail_pad_bits)))

    flat_exp = align_exp.reshape(-1)
    flat_mant = mant_padded.reshape(-1)
    flat_sign = sign.reshape(-1)
    flat_zero = zero_mask.reshape(-1)
    if flat_exp.numel() % 4 != 0:
        raise ValueError("double approx requires total elements divisible by 4")
    exp_groups = flat_exp.view(-1, 4)
    mant_groups = flat_mant.view(-1, 4)
    sign_groups = flat_sign.view(-1, 4)
    zero_groups = flat_zero.view(-1, 4)

    outlier_mask = (exp_groups < hi_align_start) | (exp_groups > hi_align_exp_field)
    outlier_count = outlier_mask.sum(dim=1, keepdim=True)
    group_max = exp_groups.max(dim=1, keepdim=True).values

    target_exp = torch.where(outlier_count <= 1, torch.full_like(group_max, hi_align_exp_field), group_max)

    if handle_max_outlier:
        max_exp_val = (1 << exp_bits) - 1
        has_max_outlier = ((exp_groups == max_exp_val) & outlier_mask).any(dim=1, keepdim=True)
        target_exp = torch.where(has_max_outlier, torch.full_like(target_exp, max_exp_val), target_exp)

    shift = target_exp - exp_groups
    shift_right = torch.clamp(shift, min=0)
    shift_left = torch.clamp(-shift, min=0)
    mant_right = _rounding_rshift(mant_groups, shift_right)
    mant_left = torch.bitwise_left_shift(mant_groups, shift_left)

    if tail_pad_bits >= 0:
        cap = ((1 << (mant_bits + 1)) - 1) << tail_pad_bits
    else:
        cap = (1 << (mant_bits + 1)) - 1
        cap = cap >> abs(tail_pad_bits)
    mant_left = torch.clamp(mant_left, max=cap)
    mant_aligned = torch.where(shift >= 0, mant_right, mant_left)

    hi_unbiased = target_exp - exp_bias
    value = mant_aligned.float() / (2.0 ** (mant_bits + tail_pad_bits)) * torch.pow(torch.tensor(2.0, device=code.device), hi_unbiased.float())
    value = torch.where(sign_groups == 1, -value, value)
    value = torch.where(zero_groups, torch.zeros_like(value), value)
    return value.view(code.shape).t()


def _count_fp4_values(code: torch.Tensor, exp_bits: int = FP4_EXP_BITS, mant_bits: int = FP4_MANTISSA_BITS, exp_bias: int = FP4_EXP_BIAS):
    """
    统计 FP4 码字解码后的 15/16 个可表示值各自出现的次数。
    返回 (values, counts, fig)，两者 shape 相同，fig 为已绘制的条形图。
    """
    decoded = _fp_to_float(code, exp_bits=exp_bits, mant_bits=mant_bits, exp_bias=exp_bias).reshape(-1)
    values, counts = torch.unique(decoded, sorted=True, return_counts=True)
    values_np = values.cpu().numpy()
    counts_np = counts.cpu().numpy()

    # fig, ax = plt.subplots()
    # ax.bar(range(len(values_np)), counts_np, tick_label=[f"{v:.4g}" for v in values_np])
    # # 使用 ASCII 标签避免字体缺失警告
    # ax.set_xlabel("FP4 value")
    # ax.set_ylabel("Count")
    # ax.set_title("FP4 decoded value histogram")
    # ax.tick_params(axis='x', rotation=45)
    # fig.savefig("./results/fp4_values.png")
    return values, counts


class QuantLinear(nn.Module):
    """
    自定义量化线性层, 支持量化后的权重和FP16激活值之间的矩阵乘法
    """
    def __init__(self, in_features, out_features, bias=True, w_bit=4, w_group_size=128, symmetric=True, mode=0, weight_format: str = "int", approximate: bool = False, quant_dim: int = 0, fp8_hi_align_start: int = 12, fp8_hi_align_exp_field: int = 15, fp8_tail_pad_bits: int = 1, double_approximate: bool = False, fp6_hi_align_start: int = 4, fp6_hi_align_exp_field: int = 7, fp6_tail_pad_bits: int = 2, fp4_hi_align_start: int = 1, fp4_hi_align_exp_field: int = 1, fp4_tail_pad_bits: int = 0):
        """
        Args:
            in_features: 输入特征数
            out_features: 输出特征数
            bias: 是否包含偏置项
            w_bit: 权重量化位数（当 weight_format='int' 时生效）
            w_group_size: 权重量化分组大小(-1, per-tensor; -2, per-channel; >0: per-group)
            symmetric: 是否使用对称量化（当 weight_format='int' 时生效）
            mode: 如果为0, 表示forward过程使用GPU; 如果为1, 表示forward过程使用FIGLUT-F; 如果为2, 表示forward过程使用FIGLUT-I.
            weight_format: 权重量化格式，支持 'int'（默认）或 'fp4'。
        """
        super(QuantLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.w_bit = w_bit
        self.w_group_size = w_group_size
        self.symmetric = symmetric
        self.mode = mode
        self.quant_dim = quant_dim  # 0: 按行/输入维分组; 1: 按列/输出维分组（转置后分组）
        raw_format = weight_format.lower()
        self.weight_format = "bfp" if raw_format.startswith("bfp") else raw_format
        self.approximate = approximate
        self.double_approximate = double_approximate
        # 近似 FP8 对齐参数（可由外部注入）
        self.fp8_hi_align_start = fp8_hi_align_start
        self.fp8_hi_align_exp_field = fp8_hi_align_exp_field
        self.fp8_tail_pad_bits = fp8_tail_pad_bits
        self.fp6_hi_align_start = fp6_hi_align_start
        self.fp6_hi_align_exp_field = fp6_hi_align_exp_field
        self.fp6_tail_pad_bits = fp6_tail_pad_bits
        self.fp4_hi_align_start = fp4_hi_align_start
        self.fp4_hi_align_exp_field = fp4_hi_align_exp_field
        self.fp4_tail_pad_bits = fp4_tail_pad_bits
        if self.weight_format not in {"int", "fp4", "fp6", "fp8", "bfp"}:
            raise ValueError(f"Unsupported weight_format: {weight_format}")

        # 创建权重和偏置参数
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        
        # 量化相关的缓冲区，用于存储量化参数
        self.register_buffer("quantized", torch.tensor(False))
        self.register_buffer("scales", None)
        self.register_buffer("zeros", None)
        self.register_buffer("weight_fp4", None)
        self.register_buffer("weight_fp6", None)
        self.register_buffer("weight_fp8", None)
        self.register_buffer("weight_bfp_mantissa", None)
        self.register_buffer("weight_bfp_exponent", None)
        
        self.reset_parameters()

    def reset_parameters(self):
        """初始化参数"""
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / fan_in**0.5
            nn.init.uniform_(self.bias, -bound, bound)
    
    def quantize_weight_approximate(self):
        """一种近似量化权重的方案（列分组，需转置权重，w_group_size>0）。"""
        with torch.no_grad():
            original_shape = self.weight.shape  # [out_features, in_features]
            weight = self.weight.data
            if self.w_group_size <= 0:
                raise ValueError("approximate 仅支持分组量化，w_group_size 必须 > 0")

            if self.quant_dim == 1:
                weight_t = weight.t()  # [in_features, out_features]
                assert weight_t.shape[-1] % self.w_group_size == 0
                weight_grouped = weight_t.reshape(-1, self.w_group_size)
                regroup_shape = weight_t.shape
                regroup_transpose = True
            else:
                assert weight.shape[-1] % self.w_group_size == 0
                weight_grouped = weight.reshape(-1, self.w_group_size)
                regroup_shape = weight.shape
                regroup_transpose = False

            if self.weight_format == "fp4":
                fp_max_value = (1.0 + (2**FP4_MANTISSA_BITS - 1) / (2**FP4_MANTISSA_BITS)) * (2.0 ** ((1 << FP4_EXP_BITS) - 1 - FP4_EXP_BIAS))
                max_val = weight_grouped.abs().amax(dim=1, keepdim=True).clamp(min=1e-5)
                scales = (max_val / fp_max_value).clamp(min=1e-5)
                normalized = torch.clamp(weight_grouped / scales, min=-fp_max_value, max=fp_max_value)
                codes = _float_to_fp(normalized, exp_bits=FP4_EXP_BITS, mant_bits=FP4_MANTISSA_BITS, exp_bias=FP4_EXP_BIAS)
                if FP4_EXP_BITS == 1:
                    decoded = _fp_decode_aligned(
                        codes, 
                        hi_align_start=self.fp4_hi_align_start,
                        hi_align_exp_field=self.fp4_hi_align_exp_field,
                        tail_pad_bits=self.fp4_tail_pad_bits,
                        exp_bits=FP4_EXP_BITS,
                        mant_bits=FP4_MANTISSA_BITS,
                        exp_bias=FP4_EXP_BIAS,
                        align_subnorm_exp_as_one=True,
                        limit_align_exp_to_field=True,
                    ).to(self.weight.data.dtype)
                elif FP4_EXP_BITS == 2:
                    if self.double_approximate:
                        decoded = fp_decode_aligned_double_approx(
                            codes, 
                            hi_align_start=self.fp4_hi_align_start,
                            hi_align_exp_field=self.fp4_hi_align_exp_field,
                            tail_pad_bits=self.fp4_tail_pad_bits,
                            exp_bits=FP4_EXP_BITS, 
                            mant_bits=FP4_MANTISSA_BITS,
                            exp_bias=FP4_EXP_BIAS,
                            align_subnorm_exp_as_one=True,
                            handle_max_outlier=True,
                        ).to(self.weight.data.dtype)
                    else:
                        decoded = _fp_decode_aligned(
                            codes, 
                            hi_align_start=self.fp4_hi_align_start,
                            hi_align_exp_field=self.fp4_hi_align_exp_field,
                            tail_pad_bits=self.fp4_tail_pad_bits,
                            exp_bits=FP4_EXP_BITS,
                            mant_bits=FP4_MANTISSA_BITS,
                            exp_bias=FP4_EXP_BIAS,
                            align_subnorm_exp_as_one=True,
                            limit_align_exp_to_field=True,
                        ).to(self.weight.data.dtype)
            elif self.weight_format == "fp6":
                fp_max_value = (1.0 + (2**FP6_MANTISSA_BITS - 1) / (2**FP6_MANTISSA_BITS)) * (2.0 ** ((1 << FP6_EXP_BITS) - 1 - FP6_EXP_BIAS))
                max_val = weight_grouped.abs().amax(dim=1, keepdim=True).clamp(min=1e-5)
                scales = (max_val / fp_max_value).clamp(min=1e-5)
                normalized = torch.clamp(weight_grouped / scales, min=-fp_max_value, max=fp_max_value)
                codes = _float_to_fp(normalized, exp_bits=FP6_EXP_BITS, mant_bits=FP6_MANTISSA_BITS, exp_bias=FP6_EXP_BIAS)
                if self.double_approximate:
                    decoded = fp_decode_aligned_double_approx(
                        codes,
                        hi_align_start=self.fp6_hi_align_start,
                        hi_align_exp_field=self.fp6_hi_align_exp_field,
                        tail_pad_bits=self.fp6_tail_pad_bits,
                        exp_bits=FP6_EXP_BITS,
                        mant_bits=FP6_MANTISSA_BITS,
                        exp_bias=FP6_EXP_BIAS,
                        align_subnorm_exp_as_one=True,
                        handle_max_outlier=True,
                    ).to(self.weight.data.dtype)
                else:
                    decoded = _fp_decode_aligned(
                        codes,
                        hi_align_start=self.fp6_hi_align_start,
                        hi_align_exp_field=self.fp6_hi_align_exp_field,
                        tail_pad_bits=self.fp6_tail_pad_bits,
                        exp_bits=FP6_EXP_BITS,
                        mant_bits=FP6_MANTISSA_BITS,
                        exp_bias=FP6_EXP_BIAS,
                        align_subnorm_exp_as_one=True,
                        limit_align_exp_to_field=True,
                    ).to(self.weight.data.dtype)
            elif self.weight_format == "fp8":
                fp_max_value = (1.0 + (2**FP8_MANTISSA_BITS - 1) / (2**FP8_MANTISSA_BITS)) * (2.0 ** ((1 << FP8_EXP_BITS) - 1 - FP8_EXP_BIAS))
                max_val = weight_grouped.abs().amax(dim=1, keepdim=True).clamp(min=1e-5)
                scales = (max_val / fp_max_value).clamp(min=1e-5)
                normalized = torch.clamp(weight_grouped / scales, min=-fp_max_value, max=fp_max_value)
                codes = _float_to_fp(normalized, exp_bits=FP8_EXP_BITS, mant_bits=FP8_MANTISSA_BITS, exp_bias=FP8_EXP_BIAS)
                if self.double_approximate:
                    decoded = fp_decode_aligned_double_approx(
                        codes,
                        hi_align_start=self.fp8_hi_align_start,
                        hi_align_exp_field=self.fp8_hi_align_exp_field,
                        tail_pad_bits=self.fp8_tail_pad_bits,
                        exp_bits=FP8_EXP_BITS,
                        mant_bits=FP8_MANTISSA_BITS,
                        exp_bias=FP8_EXP_BIAS,
                        align_subnorm_exp_as_one=True,
                        handle_max_outlier=True,
                    ).to(self.weight.data.dtype)
                else:
                    decoded = _fp_decode_aligned(
                        codes,
                        hi_align_start=self.fp8_hi_align_start,
                        hi_align_exp_field=self.fp8_hi_align_exp_field,
                        tail_pad_bits=self.fp8_tail_pad_bits,
                        exp_bits=FP8_EXP_BITS,
                        mant_bits=FP8_MANTISSA_BITS,
                        exp_bias=FP8_EXP_BIAS,
                        align_subnorm_exp_as_one=True,
                        limit_align_exp_to_field=True,
                    ).to(self.weight.data.dtype)
            else:
                raise NotImplementedError("approximate 目前仅支持 fp4/fp6/fp8")

            self.scales = scales.view(-1, 1).half()
            self.zeros = None

            dequantized_grouped = decoded * scales
            if regroup_transpose:
                dequantized = dequantized_grouped.reshape(regroup_shape).t()
            else:
                dequantized = dequantized_grouped.reshape(weight.shape)

            # 缓存码字
            if self.weight_format == "fp4":
                self.weight_fp4 = codes.reshape(regroup_shape).t() if regroup_transpose else codes.reshape(weight.shape)
                self.weight_fp6 = None
                self.weight_fp8 = None
            elif self.weight_format == "fp6":
                self.weight_fp6 = codes.reshape(regroup_shape).t() if regroup_transpose else codes.reshape(weight.shape)
                self.weight_fp4 = None
                self.weight_fp8 = None
            elif self.weight_format == "fp8":
                self.weight_fp8 = codes.reshape(regroup_shape).t() if regroup_transpose else codes.reshape(weight.shape)
                self.weight_fp4 = None
                self.weight_fp6 = None

            self.weight.data = dequantized.view(original_shape)
            self.quantized.fill_(True)
            self.approximate = True
            return


    def quantize_weight(self):
        """量化权重"""
        with torch.no_grad():
            if self.approximate:
                return self.quantize_weight_approximate()
            quant_transpose = self.quant_dim == 1
            original_shape = self.weight.shape  # [out_features, in_features]
            weight_for_quant = self.weight.data.t() if quant_transpose else self.weight.data
            quant_shape = weight_for_quant.shape

            def _reshape_back(tensor: torch.Tensor) -> torch.Tensor:
                tensor = tensor.view(quant_shape)
                return tensor.t() if quant_transpose else tensor
            if self.weight_format == "bfp":
                if self.w_group_size <= 0:
                    raise ValueError("BFP 仅支持分组量化，请将 w_group_size 设为正数")

                weight_tensor = weight_for_quant
                assert quant_shape[-1] % self.w_group_size == 0
                weight_tensor_grouped = weight_tensor.reshape(-1, self.w_group_size)

                # 1) 先转为 FP16 并拆出 sign/exp/mantissa 字段
                weight_fp16 = weight_tensor_grouped.to(torch.float16)
                bits = weight_fp16.view(torch.int16).to(torch.int32)
                sign = (bits >> 15) & 0x1                 # [g, s]
                exp = (bits >> 10) & 0x1F                 # 5-bit 指数字段
                mant = bits & 0x3FF                       # 10-bit 尾数字段

                # 2) 尾数扩展前导1（正规数），次正规保持0
                leading = torch.where(exp == 0, torch.zeros_like(exp), torch.ones_like(exp))
                mant_with_leading = (leading << 10) | mant  # 11-bit 尾数

                # 3) 找到每组的最大指数，所有尾数按指数差右移对齐
                exp_block = exp.max(dim=1, keepdim=True)[0]
                shift = torch.clamp(exp_block - exp, min=0)
                mant_aligned = torch.bitwise_right_shift(mant_with_leading, shift)

                # 4) 按 w_bit 控制保留的尾数位数（含前导1），高位截断+舍入
                target_mant_bits = min(self.w_bit - 1, 11)  # 含前导1，最多11位
                shift_down = max(0, 11 - target_mant_bits)
                if shift_down > 0:
                    mant_rounded = mant_aligned >> shift_down
                else:
                    mant_rounded = mant_aligned
                mant_max = (1 << target_mant_bits) - 1
                mant_rounded = torch.clamp(mant_rounded, max=mant_max)
            
                # 6) 直接反量化：按共享指数左移，再按保留的小数位右移，还原近似值
                frac_bits_keep = target_mant_bits - 1  # 去掉前导1后的位数
                sign_factor = torch.where(sign == 1, -1.0, 1.0)
                exp_unbiased = exp_block.to(torch.int32) - 15  # FP16 bias = 15
                scale = torch.pow(torch.tensor(2.0, device=weight_tensor_grouped.device), exp_unbiased.float() - float(frac_bits_keep))
                dequantized = mant_rounded.to(weight_tensor_grouped.dtype) * scale * sign_factor.to(weight_tensor_grouped.dtype)

                self.weight.data = _reshape_back(dequantized)
                self.weight_bfp_mantissa = _reshape_back(mant_rounded).to(torch.int16)
                self.weight_bfp_exponent = exp_block.view(-1).to(torch.int16)
                self.scales = None
                self.zeros = None
                self.weight_fp4 = None
                self.weight_fp6 = None
                self.weight_fp8 = None
                self.quantized.fill_(True)
                return
            if self.weight_format == "fp4":
                weight_tensor = weight_for_quant

                if self.w_group_size > 0:
                    assert quant_shape[-1] % self.w_group_size == 0
                    weight_tensor_grouped = weight_tensor.reshape(-1, self.w_group_size)
                elif self.w_group_size == -1:
                    # per-tensor 量化
                    weight_tensor_grouped = weight_tensor.reshape(1, -1)
                elif self.w_group_size == -2:
                    # per-channel 量化
                    weight_tensor_grouped = weight_tensor.reshape(quant_shape[0], -1)
                else:
                    raise ValueError("Invalid w_group_size")

                # 计算 FP4 可表示的最大幅值，用于放缩避免过度拉伸
                fp4_max_value = (1.0 + (2**FP4_MANTISSA_BITS - 1) / (2**FP4_MANTISSA_BITS)) * (2.0 ** ((1 << FP4_EXP_BITS) - 1 - FP4_EXP_BIAS))

                # 对齐与整数量化相同的粒度，使用对称 scale 将 max 对齐到 FP4 可表示的最大值
                if self.symmetric:
                    max_val = weight_tensor_grouped.abs().amax(dim=1, keepdim=True).clamp(min=1e-5)
                    scales = (max_val / fp4_max_value).clamp(min=1e-5)
                    zeros = None
                    normalized = torch.clamp(weight_tensor_grouped / scales, min=-fp4_max_value, max=fp4_max_value)
                else:
                    max_val = weight_tensor_grouped.amax(dim=1, keepdim=True)
                    min_val = weight_tensor_grouped.amin(dim=1, keepdim=True)
                    mid_val = (max_val + min_val) * 0.5
                    span = ((max_val - min_val) * 0.5).clamp(min=1e-5)
                    scales = (span / fp4_max_value).clamp(min=1e-5)
                    zeros = mid_val
                    normalized = torch.clamp((weight_tensor_grouped - zeros) / scales, min=-fp4_max_value, max=fp4_max_value)

                fp4_codes = _float_to_fp(normalized, exp_bits=FP4_EXP_BITS, mant_bits=FP4_MANTISSA_BITS, exp_bias=FP4_EXP_BIAS)

                # 保存 scale 形状
                if self.w_group_size == -1:
                    self.scales = scales.view(1, 1).half()
                    self.zeros = zeros.view(1, 1).half() if zeros is not None else None
                elif self.w_group_size == -2:
                    self.scales = scales.view(quant_shape[0], 1).half()
                    self.zeros = zeros.view(quant_shape[0], 1).half() if zeros is not None else None
                else:
                    self.scales = scales.view(-1, 1).half()
                    self.zeros = zeros.view(-1, 1).half() if zeros is not None else None

                # 存 FP4 码字，并写回反量化后的权重便于前向
                self.weight_fp4 = _reshape_back(fp4_codes)
                self.weight_fp6 = None
                self.weight_fp8 = None
                dequantized = _fp_to_float(fp4_codes, exp_bits=FP4_EXP_BITS, mant_bits=FP4_MANTISSA_BITS, exp_bias=FP4_EXP_BIAS).to(self.weight.data.dtype) * scales
                if self.zeros is not None:
                    dequantized = dequantized + self.zeros.to(dequantized.dtype)
                self.weight.data = _reshape_back(dequantized)
                self.quantized.fill_(True)
                return

            if self.weight_format == "fp6":
                weight_tensor = weight_for_quant

                if self.w_group_size > 0:
                    assert quant_shape[-1] % self.w_group_size == 0
                    weight_tensor_grouped = weight_tensor.reshape(-1, self.w_group_size)
                elif self.w_group_size == -1:
                    weight_tensor_grouped = weight_tensor.reshape(1, -1)
                elif self.w_group_size == -2:
                    weight_tensor_grouped = weight_tensor.reshape(quant_shape[0], -1)
                else:
                    raise ValueError("Invalid w_group_size")

                fp6_max_value = (1.0 + (2**FP6_MANTISSA_BITS - 1) / (2**FP6_MANTISSA_BITS)) * (2.0 ** ((1 << FP6_EXP_BITS) - 1 - FP6_EXP_BIAS))
                if self.symmetric:
                    max_val = weight_tensor_grouped.abs().amax(dim=1, keepdim=True).clamp(min=1e-5)
                    scales = (max_val / fp6_max_value).clamp(min=1e-5)
                    zeros = None
                    normalized = torch.clamp(weight_tensor_grouped / scales, min=-fp6_max_value, max=fp6_max_value)
                else:
                    max_val = weight_tensor_grouped.amax(dim=1, keepdim=True)
                    min_val = weight_tensor_grouped.amin(dim=1, keepdim=True)
                    mid_val = (max_val + min_val) * 0.5
                    span = ((max_val - min_val) * 0.5).clamp(min=1e-5)
                    scales = (span / fp6_max_value).clamp(min=1e-5)
                    zeros = mid_val
                    normalized = torch.clamp((weight_tensor_grouped - zeros) / scales, min=-fp6_max_value, max=fp6_max_value)
                fp6_codes = _float_to_fp(normalized, exp_bits=FP6_EXP_BITS, mant_bits=FP6_MANTISSA_BITS, exp_bias=FP6_EXP_BIAS)

                if self.w_group_size == -1:
                    self.scales = scales.view(1, 1).half()
                    self.zeros = zeros.view(1, 1).half() if zeros is not None else None
                elif self.w_group_size == -2:
                    self.scales = scales.view(quant_shape[0], 1).half()
                    self.zeros = zeros.view(quant_shape[0], 1).half() if zeros is not None else None
                else:
                    self.scales = scales.view(-1, 1).half()
                    self.zeros = zeros.view(-1, 1).half() if zeros is not None else None

                self.weight_fp6 = _reshape_back(fp6_codes)
                self.weight_fp4 = None
                self.weight_fp8 = None
                dequantized = _fp_to_float(fp6_codes, exp_bits=FP6_EXP_BITS, mant_bits=FP6_MANTISSA_BITS, exp_bias=FP6_EXP_BIAS).to(self.weight.data.dtype) * scales
                if self.zeros is not None:
                    dequantized = dequantized + self.zeros.to(dequantized.dtype)
                self.weight.data = _reshape_back(dequantized)
                self.quantized.fill_(True)
                return

            if self.weight_format == "fp8":
                weight_tensor = weight_for_quant

                if self.w_group_size > 0:
                    assert quant_shape[-1] % self.w_group_size == 0
                    weight_tensor_grouped = weight_tensor.reshape(-1, self.w_group_size)
                elif self.w_group_size == -1:
                    weight_tensor_grouped = weight_tensor.reshape(1, -1)
                elif self.w_group_size == -2:
                    weight_tensor_grouped = weight_tensor.reshape(quant_shape[0], -1)
                else:
                    raise ValueError("Invalid w_group_size")

                fp8_max_value = (1.0 + (2**FP8_MANTISSA_BITS - 1) / (2**FP8_MANTISSA_BITS)) * (2.0 ** ((1 << FP8_EXP_BITS) - 1 - FP8_EXP_BIAS))
                if self.symmetric:
                    max_val = weight_tensor_grouped.abs().amax(dim=1, keepdim=True).clamp(min=1e-5)
                    scales = (max_val / fp8_max_value).clamp(min=1e-5)
                    zeros = None
                    normalized = torch.clamp(weight_tensor_grouped / scales, min=-fp8_max_value, max=fp8_max_value)
                else:
                    max_val = weight_tensor_grouped.amax(dim=1, keepdim=True)
                    min_val = weight_tensor_grouped.amin(dim=1, keepdim=True)
                    mid_val = (max_val + min_val) * 0.5
                    span = ((max_val - min_val) * 0.5).clamp(min=1e-5)
                    scales = (span / fp8_max_value).clamp(min=1e-5)
                    zeros = mid_val
                    normalized = torch.clamp((weight_tensor_grouped - zeros) / scales, min=-fp8_max_value, max=fp8_max_value)
                fp8_codes = _float_to_fp(normalized, exp_bits=FP8_EXP_BITS, mant_bits=FP8_MANTISSA_BITS, exp_bias=FP8_EXP_BIAS)

                if self.w_group_size == -1:
                    self.scales = scales.view(1, 1).half()
                    self.zeros = zeros.view(1, 1).half() if zeros is not None else None
                elif self.w_group_size == -2:
                    self.scales = scales.view(quant_shape[0], 1).half()
                    self.zeros = zeros.view(quant_shape[0], 1).half() if zeros is not None else None
                else:
                    self.scales = scales.view(-1, 1).half()
                    self.zeros = zeros.view(-1, 1).half() if zeros is not None else None

                self.weight_fp8 = _reshape_back(fp8_codes)
                self.weight_fp4 = None
                self.weight_fp6 = None
                dequantized = _fp_to_float(fp8_codes, exp_bits=FP8_EXP_BITS, mant_bits=FP8_MANTISSA_BITS, exp_bias=FP8_EXP_BIAS).to(self.weight.data.dtype) * scales
                if self.zeros is not None:
                    dequantized = dequantized + self.zeros.to(dequantized.dtype)
                self.weight.data = _reshape_back(dequantized)
                self.quantized.fill_(True)
                return

            if self.weight_format == "int":
                weight_tensor = weight_for_quant
                if self.w_bit >= 16:
                    self.quantized.fill_(False)
                    self.weight_fp4 = None
                    self.weight_fp6 = None
                    self.weight_fp8 = None
                    return
                
                # 计算量化参数

                if self.w_group_size > 0:
                    assert quant_shape[-1] % self.w_group_size == 0
                    weight_tensor = weight_tensor.reshape(-1, self.w_group_size)
                elif self.w_group_size == -1:
                    # per-tensor 量化
                    weight_tensor = weight_tensor.reshape(1, -1)
                elif self.w_group_size == -2:
                    # per-channel 量化
                    weight_tensor = weight_tensor.reshape(quant_shape[0], -1)
                else:
                    raise ValueError("Invalid w_group_size")
                
                # 计算scales和zeros并存储
                if self.symmetric:
                    max_val = weight_tensor.abs().amax(dim=1, keepdim=True)
                    max_val = max_val.clamp(min=1e-5)
                    max_int = 2 ** (self.w_bit - 1) - 1
                    min_int = -(2 ** (self.w_bit - 1))
                    scales = max_val / max_int
                    zeros = 0
                else:
                    max_val = weight_tensor.amax(dim=1, keepdim=True)
                    min_val = weight_tensor.amin(dim=1, keepdim=True)
                    max_int = 2**self.w_bit - 1
                    min_int = 0
                    scales = (max_val - min_val).clamp(min=1e-5) / max_int
                    zeros = (-min_val / scales).round().clamp(min_int, max_int)
                
                if self.w_group_size == -1:
                    self.scales = scales.view(1, 1)
                    self.zeros = zeros.view(1, 1) if not self.symmetric else None
                elif self.w_group_size == -2:
                    self.scales = scales.view(quant_shape[0], 1)
                    self.zeros = zeros.view(quant_shape[0], 1) if not self.symmetric else None
                else:
                    self.scales = scales.view(-1, 1)
                    self.zeros = zeros.view(-1, 1) if not self.symmetric else None

                # 量化权重
                quantized_weight = torch.clamp(
                    torch.round(weight_tensor / self.scales) + (self.zeros if self.zeros is not None else 0),
                    min_int,
                    max_int,
                )

                # 反量化写回，保持与其他格式一致的行为
                if self.zeros is not None:
                    dequantized_weight = (quantized_weight - self.zeros) * self.scales
                else:
                    dequantized_weight = quantized_weight * self.scales

                self.weight.data = _reshape_back(dequantized_weight)
                self.weight_fp4 = None
                self.weight_fp6 = None
                self.weight_fp8 = None
                self.quantized.fill_(True)
                return

            raise ValueError(f"Unsupported weight_format in quantize_weight: {self.weight_format}")
    
    def forward(self, input):
        """前向传播"""
        if not self.quantized:
            return F.linear(input, self.weight, self.bias)

        # 权重在量化阶段已解码到 self.weight，直接使用
        if self.weight_format in {"fp4", "fp6", "fp8", "bfp", "int"}:
            original_input_shape = input.shape
            weight = self.weight.to(input.dtype)
            out = F.linear(input, weight, self.bias)
            if input.dim() > 2:
                out = out.reshape(original_input_shape[:-1] + (self.out_features,))
            return out

    @classmethod
    def from_linear(
        cls,
        linear_layer,
        w_bit=4,
        w_group_size=128,
        symmetric=False,
        mode=0,
        weight_format: str = "int",
        approximate: bool = False,
        quant_dim: int = 0,
        fp8_hi_align_start: int = 12,
        fp8_hi_align_exp_field: int = 15,
        fp8_tail_pad_bits: int = 1,
        double_approximate: bool = False,
        fp6_hi_align_start: int = 4,
        fp6_hi_align_exp_field: int = 7,
        fp6_tail_pad_bits: int = 2,
        fp4_hi_align_start: int = 1,
        fp4_hi_align_exp_field: int = 1,
        fp4_tail_pad_bits: int = 0,
    ):
        """从现有的线性层创建量化线性层"""
        assert isinstance(linear_layer, nn.Linear), "Input layer must be nn.Linear"
        quant_linear = cls(
            in_features=linear_layer.in_features,
            out_features=linear_layer.out_features,
            bias=linear_layer.bias is not None,
            w_bit=w_bit,
            w_group_size=w_group_size,
            symmetric=symmetric,
            mode=mode,
            weight_format=weight_format,
            approximate=approximate,
            quant_dim=quant_dim,
            fp8_hi_align_start=fp8_hi_align_start,
            fp8_hi_align_exp_field=fp8_hi_align_exp_field,
            fp8_tail_pad_bits=fp8_tail_pad_bits,
            double_approximate=double_approximate,
            fp6_hi_align_start=fp6_hi_align_start,
            fp6_hi_align_exp_field=fp6_hi_align_exp_field,
            fp6_tail_pad_bits=fp6_tail_pad_bits,
            fp4_hi_align_start=fp4_hi_align_start,
            fp4_hi_align_exp_field=fp4_hi_align_exp_field,
            fp4_tail_pad_bits=fp4_tail_pad_bits,
        )
        # 复制权重和偏置
        quant_linear.weight.data = linear_layer.weight.data.clone()
        if linear_layer.bias is not None:
            quant_linear.bias.data = linear_layer.bias.data.clone()

        # 量化权重
        quant_linear.quantize_weight()
        return quant_linear

class GPTQQuantLinear(nn.Module):
    """
    简化的GPTQ量化线性层包装器, 用于在量化后自定义前向逻辑。
    """

    def __init__(self, weight: torch.Tensor, bias: torch.Tensor | None = None):
        super().__init__()
        assert weight.dim() == 2, 'Weight tensor must be 2-dimensional'
        self.out_features, self.in_features = weight.shape

        self.weight = nn.Parameter(weight.clone().detach(), requires_grad=False)
        if bias is not None:
            self.bias = nn.Parameter(bias.clone().detach(), requires_grad=False)
        else:
            self.register_parameter('bias', None)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return F.linear(input, self.weight, self.bias)

    @classmethod
    def from_linear(cls, linear_layer: nn.Linear) -> 'GPTQQuantLinear':
        assert isinstance(linear_layer, nn.Linear), 'Input layer must be nn.Linear'
        weight = linear_layer.weight.data
        bias = linear_layer.bias.data if linear_layer.bias is not None else None
        layer = cls(weight, bias)
        if hasattr(linear_layer, 'scales') and linear_layer.scales is not None:
            layer.scales = linear_layer.scales.clone().detach()
            # print(layer.scales.shape)
        if hasattr(linear_layer, 'zeros') and linear_layer.zeros is not None:
            layer.zeros = linear_layer.zeros.clone().detach()
        return layer

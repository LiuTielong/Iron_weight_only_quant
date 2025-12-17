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


def _float_to_fp4(x: torch.Tensor, exp_bits: int = FP4_EXP_BITS, mant_bits: int = FP4_MANTISSA_BITS, exp_bias: int = FP4_EXP_BIAS) -> torch.Tensor:
    """
    将浮点张量量化为带次正规支持的 FP4 (E2M1) 码字，存放在 uint8 的低 4 位。
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


def _fp4_to_float(code: torch.Tensor, exp_bits: int = FP4_EXP_BITS, mant_bits: int = FP4_MANTISSA_BITS, exp_bias: int = FP4_EXP_BIAS) -> torch.Tensor:
    """
    将 FP4 (E2M1) 码字还原为浮点，支持次正规。输入为 uint8/long，返回 FP32。
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


def _fp4_decode_aligned(code: torch.Tensor, target_exp_field: int = 2, exp_bits: int = FP4_EXP_BITS, mant_bits: int = FP4_MANTISSA_BITS, exp_bias: int = FP4_EXP_BIAS) -> torch.Tensor:
    """
    将 FP4 码字解码，并将指数对齐到 target_exp_field（偏移后的值）。
    步骤：
      1) 拆分符号、指数字段、尾数字段
      2) 正规数尾数前补1，次正规补0，得到2位尾数
      3) 尾数后补0，得到3位尾数
      4) 按无偏指数对齐到 target_exp_field，再转成 FP32
    """
    code = code.to(torch.uint8)
    sign = ((code >> (exp_bits + mant_bits)) & 0x1).float()
    exp_field = ((code >> mant_bits) & ((1 << exp_bits) - 1)).int()
    mant_field = (code & ((1 << mant_bits) - 1)).int()

    leading = torch.where(exp_field == 0, torch.zeros_like(exp_field), torch.ones_like(exp_field))
    mant2 = (leading << 1) | mant_field  # 2 位
    mant3 = mant2 << 1  # 补一位 0，变 3 位

    # 无偏指数，次正规取 1 - bias
    exp_unbiased = torch.where(exp_field == 0, 1 - exp_bias, exp_field - exp_bias)

    # 对齐到目标无偏指数（默认 2），理论上不会出现负移
    target_unbiased = target_exp_field
    shift = torch.clamp(target_unbiased - exp_unbiased, min=0)
    mant_shifted = torch.bitwise_right_shift(mant3, shift)

    value = mant_shifted.float() / 4.0 * (2.0 ** float(target_unbiased))
    value = torch.where(sign == 1, -value, value)
    return value


def _fp8_decode_aligned(
    code: torch.Tensor,
    hi_align_start: int = 12,
    hi_align_exp_field: int = 15,
    tail_pad_bits: int = 2,
    exp_bits: int = FP8_EXP_BITS,
    mant_bits: int = FP8_MANTISSA_BITS,
    exp_bias: int = FP8_EXP_BIAS,
) -> torch.Tensor:
    """
    FP8 (E4M3) 近似解码：
      - 指数字段 >= hi_align_start 的值，对齐到 hi_align_exp_field，并在尾数右移前右侧补 tail_pad_bits 位。
      - 其他值按常规方式解码。
    """
    code = code.to(torch.uint8)
    zero_mask = code == 0
    sign = ((code >> (exp_bits + mant_bits)) & 0x1).float()
    exp_field = ((code >> mant_bits) & ((1 << exp_bits) - 1)).int()
    mant_field = (code & ((1 << mant_bits) - 1)).int()

    leading = torch.where(exp_field == 0, torch.zeros_like(exp_field), torch.ones_like(exp_field))
    mant_full = (leading << mant_bits) | mant_field  # 前导 + 原尾数 (4 bits)
    mant_padded = mant_full << tail_pad_bits

    # 常规解码（含次正规）
    exp_unbiased = torch.where(exp_field == 0, 1 - exp_bias, exp_field - exp_bias).float()
    value_normal = mant_full.float() / (2.0 ** mant_bits) * torch.pow(torch.tensor(2.0, device=code.device), exp_unbiased)

    # 高指数对齐分支
    hi_mask = exp_field >= hi_align_start
    shift = torch.clamp(hi_align_exp_field - exp_field, min=0)
    mant_aligned = torch.bitwise_right_shift(mant_padded, shift)

    # debug
    # mant_aligned = torch.bitwise_right_shift(mant_aligned, 3)
    # mant_aligned = mant_aligned << 3
    
    hi_unbiased = hi_align_exp_field - exp_bias
    value_hi = mant_aligned.float() / (2.0 ** (mant_bits + tail_pad_bits)) * (2.0 ** float(hi_unbiased))

    value = torch.where(hi_mask, value_hi, value_normal)
    value = torch.where(sign == 1, -value, value)
    # 统计零值. 统计结果显示有15%~16%的值被零化
    # zero_mask = value == 0
    # zero_count = torch.sum(zero_mask)/(value.shape[0]*value.shape[1])
    # print(f"FP8 decode aligned: zero count = {zero_count.item()}")
    return torch.where(zero_mask, torch.zeros_like(value), value)


def _fp6e3m2_decode_aligned(
    code: torch.Tensor,
    hi_align_start: int = 4,
    hi_align_exp_field: int = 7,
    tail_pad_bits: int = 2,
    exp_bits: int = FP6_EXP_BITS,
    mant_bits: int = FP6_MANTISSA_BITS,
    exp_bias: int = FP6_EXP_BIAS,
) -> torch.Tensor:
    """
    FP6 (E3M2) 近似解码：
      - 指数字段 >= hi_align_start 的值，对齐到 hi_align_exp_field，并在尾数右移前右侧补 tail_pad_bits 位。
      - 其他值按常规方式解码。
    """
    code = code.to(torch.uint8)
    zero_mask = code == 0
    sign = ((code >> (exp_bits + mant_bits)) & 0x1).float()
    exp_field = ((code >> mant_bits) & ((1 << exp_bits) - 1)).int()
    mant_field = (code & ((1 << mant_bits) - 1)).int()

    leading = torch.where(exp_field == 0, torch.zeros_like(exp_field), torch.ones_like(exp_field))
    mant_full = (leading << mant_bits) | mant_field  # 前导 + 原尾数 (3 bits)
    mant_padded = mant_full << tail_pad_bits

    # 常规解码（含次正规）
    exp_unbiased = torch.where(exp_field == 0, 1 - exp_bias, exp_field - exp_bias).float()
    value_normal = mant_full.float() / (2.0 ** mant_bits) * torch.pow(torch.tensor(2.0, device=code.device), exp_unbiased)

    # 高指数对齐分支
    hi_mask = exp_field >= hi_align_start
    shift = torch.clamp(hi_align_exp_field - exp_field, min=0)
    mant_aligned = torch.bitwise_right_shift(mant_padded, shift)
    hi_unbiased = hi_align_exp_field - exp_bias
    value_hi = mant_aligned.float() / (2.0 ** (mant_bits + tail_pad_bits)) * (2.0 ** float(hi_unbiased))

    value = torch.where(hi_mask, value_hi, value_normal)
    value = torch.where(sign == 1, -value, value)
    return torch.where(zero_mask, torch.zeros_like(value), value)


def _count_fp4_values(code: torch.Tensor, exp_bits: int = FP4_EXP_BITS, mant_bits: int = FP4_MANTISSA_BITS, exp_bias: int = FP4_EXP_BIAS):
    """
    统计 FP4 码字解码后的 15/16 个可表示值各自出现的次数。
    返回 (values, counts, fig)，两者 shape 相同，fig 为已绘制的条形图。
    """
    decoded = _fp4_to_float(code, exp_bits=exp_bits, mant_bits=mant_bits, exp_bias=exp_bias).reshape(-1)
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

def _convert_bits_to_int(bits_tensor: torch.Tensor) -> torch.Tensor:
    """
    将[m, n, 35]的二进制张量转换回形状为[m, n]的逻辑整数张量。
    
    Args:
        bits_tensor: 形状为[m, n, 35]的二进制张量 (0/1)。
                     第0位是符号位 (1 for negative)。
    
    Returns:
        A tensor of shape [m, n] of type torch.long.
    """
    # 分离符号位和尾数位
    sign_bit = bits_tensor[..., 0]  # shape: [m, n]
    mantissa_bits = bits_tensor[..., 1:] # shape: [m, n, 34]
    
    # 创建用于计算的乘数向量: [2^33, 2^32, ..., 2^0]
    device = bits_tensor.device
    bit_exponents = torch.arange(MANTISSA_BITS-1, -1, -1, device=device).long()
    multipliers = torch.pow(2, bit_exponents) # shape: [34]
    
    # 将尾数位转换为整数
    # (mantissa_bits * multipliers) -> [m, n, 34]
    # .sum(dim=-1) -> [m, n]
    mantissa_int = (mantissa_bits * multipliers).sum(dim=-1)
    
    # 应用符号位
    # 符号位为1时，乘以-1；为0时，乘以1
    signs = 1 - 2 * sign_bit  # maps {0, 1} to {1, -1}
    
    return mantissa_int * signs


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


class QuantLinear(nn.Module):
    """
    自定义量化线性层, 支持量化后的权重和FP16激活值之间的矩阵乘法
    """
    def __init__(self, in_features, out_features, bias=True, w_bit=4, w_group_size=128, symmetric=True, mode=0, weight_format: str = "int", approximate: bool = False):
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
        self.weight_format = weight_format.lower()
        self.approximate = approximate
        if self.weight_format not in {"int", "fp4", "fp6", "fp8"}:
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
            weight_t = self.weight.data.t()     # [in_features, out_features]

            if self.w_group_size <= 0:
                raise ValueError("approximate 仅支持分组量化，w_group_size 必须 > 0")
            assert weight_t.shape[-1] % self.w_group_size == 0
            weight_grouped = weight_t.reshape(-1, self.w_group_size)

            if self.weight_format == "fp4":
                fp_max_value = (1.0 + (2**FP4_MANTISSA_BITS - 1) / (2**FP4_MANTISSA_BITS)) * (2.0 ** ((1 << FP4_EXP_BITS) - 1 - FP4_EXP_BIAS))
                max_val = weight_grouped.abs().amax(dim=1, keepdim=True).clamp(min=1e-5)
                scales = (max_val / fp_max_value).clamp(min=1e-5)
                normalized = torch.clamp(weight_grouped / scales, min=-fp_max_value, max=fp_max_value)
                codes = _float_to_fp4(normalized)
                decoded = _fp4_decode_aligned(codes, target_exp_field=2, exp_bits=FP4_EXP_BITS, mant_bits=FP4_MANTISSA_BITS, exp_bias=FP4_EXP_BIAS).to(self.weight.data.dtype)
            elif self.weight_format == "fp6":
                fp_max_value = (1.0 + (2**FP6_MANTISSA_BITS - 1) / (2**FP6_MANTISSA_BITS)) * (2.0 ** ((1 << FP6_EXP_BITS) - 1 - FP6_EXP_BIAS))
                max_val = weight_grouped.abs().amax(dim=1, keepdim=True).clamp(min=1e-5)
                scales = (max_val / fp_max_value).clamp(min=1e-5)
                normalized = torch.clamp(weight_grouped / scales, min=-fp_max_value, max=fp_max_value)
                codes = _float_to_fp4(normalized, exp_bits=FP6_EXP_BITS, mant_bits=FP6_MANTISSA_BITS, exp_bias=FP6_EXP_BIAS)
                if FP6_EXP_BIAS == 3:
                    decoded = _fp6e3m2_decode_aligned(codes, hi_align_start=4, hi_align_exp_field=7, tail_pad_bits=2, exp_bits=FP6_EXP_BITS, mant_bits=FP6_MANTISSA_BITS, exp_bias=FP6_EXP_BIAS).to(self.weight.data.dtype)
                elif FP6_EXP_BIAS == 2:
                    decoded = _fp4_to_float(codes, exp_bits=FP6_EXP_BITS, mant_bits=FP6_MANTISSA_BITS, exp_bias=FP6_EXP_BIAS).to(self.weight.data.dtype)
            elif self.weight_format == "fp8":
                fp_max_value = (1.0 + (2**FP8_MANTISSA_BITS - 1) / (2**FP8_MANTISSA_BITS)) * (2.0 ** ((1 << FP8_EXP_BITS) - 1 - FP8_EXP_BIAS))
                max_val = weight_grouped.abs().amax(dim=1, keepdim=True).clamp(min=1e-5)
                scales = (max_val / fp_max_value).clamp(min=1e-5)
                normalized = torch.clamp(weight_grouped / scales, min=-fp_max_value, max=fp_max_value)
                codes = _float_to_fp4(normalized, exp_bits=FP8_EXP_BITS, mant_bits=FP8_MANTISSA_BITS, exp_bias=FP8_EXP_BIAS)
                decoded = _fp8_decode_aligned(codes, hi_align_start=0, hi_align_exp_field=15, tail_pad_bits=0, exp_bits=FP8_EXP_BITS, mant_bits=FP8_MANTISSA_BITS, exp_bias=FP8_EXP_BIAS).to(self.weight.data.dtype)
            else:
                raise NotImplementedError("approximate 目前仅支持 fp4/fp8")

            self.scales = scales.view(-1, 1).half()
            self.zeros = None

            dequantized_grouped = decoded * scales
            dequantized_t = dequantized_grouped.view(weight_t.shape)
            dequantized = dequantized_t.t().contiguous()

            # 缓存码字（保持转置布局）
            self.weight_fp4 = codes if self.weight_format == "fp4" else None
            self.weight_fp6 = codes if self.weight_format == "fp6" else None
            self.weight_fp8 = codes if self.weight_format == "fp8" else None

            self.weight.data = dequantized.view(original_shape)
            self.quantized.fill_(True)
            self.approximate = True
            return


    def quantize_weight(self):
        """量化权重"""
        with torch.no_grad():
            if self.approximate:
                return self.quantize_weight_approximate()
            if self.weight_format == "fp4":
                original_shape = self.weight.shape  # [out_features, in_features]
                weight_tensor = self.weight.data

                if self.w_group_size > 0:
                    assert original_shape[-1] % self.w_group_size == 0
                    weight_tensor_grouped = weight_tensor.reshape(-1, self.w_group_size)
                elif self.w_group_size == -1:
                    # per-tensor 量化
                    weight_tensor_grouped = weight_tensor.reshape(1, -1)
                elif self.w_group_size == -2:
                    # per-channel 量化
                    weight_tensor_grouped = weight_tensor.reshape(original_shape[0], -1)
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

                fp4_codes = _float_to_fp4(normalized)

                # 保存 scale 形状
                if self.w_group_size == -1:
                    self.scales = scales.view(1, 1).half()
                    self.zeros = zeros.view(1, 1).half() if zeros is not None else None
                elif self.w_group_size == -2:
                    self.scales = scales.view(original_shape[0], 1).half()
                    self.zeros = zeros.view(original_shape[0], 1).half() if zeros is not None else None
                else:
                    self.scales = scales.view(-1, 1).half()
                    self.zeros = zeros.view(-1, 1).half() if zeros is not None else None

                # 存 FP4 码字，并写回反量化后的权重便于前向
                self.weight_fp4 = fp4_codes.view(original_shape)
                self.weight_fp6 = None
                self.weight_fp8 = None
                dequantized = _fp4_to_float(fp4_codes).to(self.weight.data.dtype) * scales
                if self.zeros is not None:
                    dequantized = dequantized + self.zeros.to(dequantized.dtype)
                self.weight.data = dequantized.view(original_shape)
                self.quantized.fill_(True)
                return

            if self.weight_format == "fp6":
                original_shape = self.weight.shape  # [out_features, in_features]
                weight_tensor = self.weight.data

                if self.w_group_size > 0:
                    assert original_shape[-1] % self.w_group_size == 0
                    weight_tensor_grouped = weight_tensor.reshape(-1, self.w_group_size)
                elif self.w_group_size == -1:
                    weight_tensor_grouped = weight_tensor.reshape(1, -1)
                elif self.w_group_size == -2:
                    weight_tensor_grouped = weight_tensor.reshape(original_shape[0], -1)
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
                fp6_codes = _float_to_fp4(normalized, exp_bits=FP6_EXP_BITS, mant_bits=FP6_MANTISSA_BITS, exp_bias=FP6_EXP_BIAS)

                if self.w_group_size == -1:
                    self.scales = scales.view(1, 1).half()
                    self.zeros = zeros.view(1, 1).half() if zeros is not None else None
                elif self.w_group_size == -2:
                    self.scales = scales.view(original_shape[0], 1).half()
                    self.zeros = zeros.view(original_shape[0], 1).half() if zeros is not None else None
                else:
                    self.scales = scales.view(-1, 1).half()
                    self.zeros = zeros.view(-1, 1).half() if zeros is not None else None

                self.weight_fp6 = fp6_codes.view(original_shape)
                self.weight_fp4 = None
                self.weight_fp8 = None
                dequantized = _fp4_to_float(fp6_codes, exp_bits=FP6_EXP_BITS, mant_bits=FP6_MANTISSA_BITS, exp_bias=FP6_EXP_BIAS).to(self.weight.data.dtype) * scales
                if self.zeros is not None:
                    dequantized = dequantized + self.zeros.to(dequantized.dtype)
                self.weight.data = dequantized.view(original_shape)
                self.quantized.fill_(True)
                return

            if self.weight_format == "fp8":
                original_shape = self.weight.shape  # [out_features, in_features]
                weight_tensor = self.weight.data

                if self.w_group_size > 0:
                    assert original_shape[-1] % self.w_group_size == 0
                    weight_tensor_grouped = weight_tensor.reshape(-1, self.w_group_size)
                elif self.w_group_size == -1:
                    weight_tensor_grouped = weight_tensor.reshape(1, -1)
                elif self.w_group_size == -2:
                    weight_tensor_grouped = weight_tensor.reshape(original_shape[0], -1)
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
                fp8_codes = _float_to_fp4(normalized, exp_bits=FP8_EXP_BITS, mant_bits=FP8_MANTISSA_BITS, exp_bias=FP8_EXP_BIAS)

                if self.w_group_size == -1:
                    self.scales = scales.view(1, 1).half()
                    self.zeros = zeros.view(1, 1).half() if zeros is not None else None
                elif self.w_group_size == -2:
                    self.scales = scales.view(original_shape[0], 1).half()
                    self.zeros = zeros.view(original_shape[0], 1).half() if zeros is not None else None
                else:
                    self.scales = scales.view(-1, 1).half()
                    self.zeros = zeros.view(-1, 1).half() if zeros is not None else None

                self.weight_fp8 = fp8_codes.view(original_shape)
                self.weight_fp4 = None
                self.weight_fp6 = None
                dequantized = _fp4_to_float(fp8_codes, exp_bits=FP8_EXP_BITS, mant_bits=FP8_MANTISSA_BITS, exp_bias=FP8_EXP_BIAS).to(self.weight.data.dtype) * scales
                if self.zeros is not None:
                    dequantized = dequantized + self.zeros.to(dequantized.dtype)
                self.weight.data = dequantized.view(original_shape)
                self.quantized.fill_(True)
                return

            if self.w_bit >= 16:
                self.quantized.fill_(False)
                self.weight_fp4 = None
                self.weight_fp6 = None
                self.weight_fp8 = None
                return
            
            # 计算量化参数
            original_shape = self.weight.shape  # [out_features, in_features]
            weight_tensor = self.weight.data

            if self.w_group_size > 0:
                assert original_shape[-1] % self.w_group_size == 0
                weight_tensor = weight_tensor.reshape(-1, self.w_group_size)
            elif self.w_group_size == -1:
                # per-tensor 量化
                weight_tensor = weight_tensor.reshape(1, -1)
            elif self.w_group_size == -2:
                # per-channel 量化
                weight_tensor = weight_tensor.reshape(original_shape[0], -1)
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
                self.scales = scales.view(original_shape[0], 1)
                self.zeros = zeros.view(original_shape[0], 1) if not self.symmetric else None
            else:
                self.scales = scales.view(-1, 1)
                self.zeros = zeros.view(-1, 1) if not self.symmetric else None

            # 量化权重
            quantized_weight = torch.clamp(torch.round(weight_tensor / self.scales) + (self.zeros if self.zeros is not None else 0),
                                    min_int, max_int)
            
            # 不要反量化，就存储int数据
            self.weight.data = quantized_weight.view(original_shape)
            self.weight_fp4 = None
            self.weight_fp6 = None
            self.weight_fp8 = None
            self.quantized.fill_(True)
    
    def forward(self, input):
        """前向传播"""
        if not self.quantized:
            return F.linear(input, self.weight, self.bias)

        # FP4 前向直接解码为浮点做矩阵乘法（与整数量化路径分离）
        if self.weight_format == "fp4":
            if self.approximate:
                if self.weight_fp4 is None or self.scales is None:
                    dequantized_weight = self.weight
                else:
                    codes_grouped = self.weight_fp4.reshape(-1, self.w_group_size)
                    scales = self.scales.reshape(-1, 1).to(self.weight.dtype)
                    decoded = _fp4_decode_aligned(codes_grouped, target_exp_field=2, exp_bits=FP4_EXP_BITS, mant_bits=FP4_MANTISSA_BITS, exp_bias=FP4_EXP_BIAS).to(self.weight.dtype)
                    dequantized_grouped = decoded * scales
                    weight_t_shape = (self.weight.shape[1], self.weight.shape[0])
                    dequantized_weight = dequantized_grouped.view(weight_t_shape).t()
                return F.linear(input, dequantized_weight, self.bias)
            original_input_shape = input.shape
            if self.weight_fp4 is None or self.scales is None:
                # 如果意外缺少码字或scale，退化为使用当前权重
                print("缺少scale!")
                dequantized_weight = self.weight
            else:
                if self.w_group_size > 0:
                    original_shape = self.weight.shape
                    codes_grouped = self.weight_fp4.reshape(-1, self.w_group_size)
                    scales = self.scales.reshape(-1, 1).to(self.weight.dtype)
                    dequantized_weight = _fp4_to_float(codes_grouped).to(self.weight.dtype) * scales
                    if self.zeros is not None:
                        zeros = self.zeros.reshape(-1, 1).to(self.weight.dtype)
                        dequantized_weight = dequantized_weight + zeros
                    dequantized_weight = dequantized_weight.view(original_shape)
                elif self.w_group_size == -1:
                    scales = self.scales.view(1, 1).to(self.weight.dtype)
                    dequantized_weight = _fp4_to_float(self.weight_fp4).to(self.weight.dtype) * scales
                    if self.zeros is not None:
                        dequantized_weight = dequantized_weight + self.zeros.view(1, 1).to(self.weight.dtype)
                elif self.w_group_size == -2:
                    original_shape = self.weight.shape
                    codes_grouped = self.weight_fp4.reshape(original_shape[0], -1)
                    scales = self.scales.reshape(original_shape[0], 1).to(self.weight.dtype)
                    dequantized_weight = _fp4_to_float(codes_grouped).to(self.weight.dtype) * scales
                    if self.zeros is not None:
                        zeros = self.zeros.reshape(original_shape[0], 1).to(self.weight.dtype)
                        dequantized_weight = dequantized_weight + zeros
                    dequantized_weight = dequantized_weight.view(original_shape)
                else:
                    raise ValueError("Invalid w_group_size")

            # FIGLUT-I 的整数累加不适用于 FP4，这里统一走标准线性层
            output = F.linear(input, dequantized_weight, self.bias)

            # 恢复输入的原始形状
            if input.dim() > 2:
                output = output.reshape(original_input_shape[:-1] + (self.out_features,))
            return output

        # FP6 前向
        if self.weight_format == "fp6":
            if self.approximate:
                if self.weight_fp6 is None or self.scales is None:
                    dequantized_weight = self.weight
                else:
                    codes_grouped = self.weight_fp6.reshape(-1, self.w_group_size)
                    scales = self.scales.reshape(-1, 1).to(self.weight.dtype)
                    decoded = _fp6e3m2_decode_aligned(codes_grouped, hi_align_start=4, hi_align_exp_field=7, tail_pad_bits=2, exp_bits=FP6_EXP_BITS, mant_bits=FP6_MANTISSA_BITS, exp_bias=FP6_EXP_BIAS).to(self.weight.dtype)
                    dequantized_grouped = decoded * scales
                    weight_t_shape = (self.weight.shape[1], self.weight.shape[0])
                    dequantized_weight = dequantized_grouped.view(weight_t_shape).t()
                return F.linear(input, dequantized_weight, self.bias)
            original_input_shape = input.shape
            if self.weight_fp6 is None or self.scales is None:
                dequantized_weight = self.weight
            else:
                if self.w_group_size > 0:
                    original_shape = self.weight.shape
                    codes_grouped = self.weight_fp6.reshape(-1, self.w_group_size)
                    scales = self.scales.reshape(-1, 1).to(self.weight.dtype)
                    dequantized_weight = _fp4_to_float(codes_grouped, exp_bits=FP6_EXP_BITS, mant_bits=FP6_MANTISSA_BITS, exp_bias=FP6_EXP_BIAS).to(self.weight.dtype) * scales
                    if self.zeros is not None:
                        zeros = self.zeros.reshape(-1, 1).to(self.weight.dtype)
                        dequantized_weight = dequantized_weight + zeros
                    dequantized_weight = dequantized_weight.view(original_shape)
                elif self.w_group_size == -1:
                    scales = self.scales.view(1, 1).to(self.weight.dtype)
                    dequantized_weight = _fp4_to_float(self.weight_fp6, exp_bits=FP6_EXP_BITS, mant_bits=FP6_MANTISSA_BITS, exp_bias=FP6_EXP_BIAS).to(self.weight.dtype) * scales
                    if self.zeros is not None:
                        dequantized_weight = dequantized_weight + self.zeros.view(1, 1).to(self.weight.dtype)
                elif self.w_group_size == -2:
                    original_shape = self.weight.shape
                    codes_grouped = self.weight_fp6.reshape(original_shape[0], -1)
                    scales = self.scales.reshape(original_shape[0], 1).to(self.weight.dtype)
                    dequantized_weight = _fp4_to_float(codes_grouped, exp_bits=FP6_EXP_BITS, mant_bits=FP6_MANTISSA_BITS, exp_bias=FP6_EXP_BIAS).to(self.weight.dtype) * scales
                    if self.zeros is not None:
                        zeros = self.zeros.reshape(original_shape[0], 1).to(self.weight.dtype)
                        dequantized_weight = dequantized_weight + zeros
                    dequantized_weight = dequantized_weight.view(original_shape)
                else:
                    raise ValueError("Invalid w_group_size")

            output = F.linear(input, dequantized_weight, self.bias)
            if input.dim() > 2:
                output = output.reshape(original_input_shape[:-1] + (self.out_features,))
            return output

        # FP8 前向
        if self.weight_format == "fp8":
            if self.approximate:
                if self.weight_fp8 is None or self.scales is None:
                    dequantized_weight = self.weight
                else:
                    codes_grouped = self.weight_fp8.reshape(-1, self.w_group_size)
                    scales = self.scales.reshape(-1, 1).to(self.weight.dtype)
                    decoded = _fp8_decode_aligned(codes_grouped, hi_align_start=12, hi_align_exp_field=15, tail_pad_bits=2, exp_bits=FP8_EXP_BITS, mant_bits=FP8_MANTISSA_BITS, exp_bias=FP8_EXP_BIAS).to(self.weight.dtype)
                    dequantized_grouped = decoded * scales
                    weight_t_shape = (self.weight.shape[1], self.weight.shape[0])
                    dequantized_weight = dequantized_grouped.view(weight_t_shape).t()
                return F.linear(input, dequantized_weight, self.bias)
            original_input_shape = input.shape
            if self.weight_fp8 is None or self.scales is None:
                dequantized_weight = self.weight
            else:
                if self.w_group_size > 0:
                    original_shape = self.weight.shape
                    codes_grouped = self.weight_fp8.reshape(-1, self.w_group_size)
                    scales = self.scales.reshape(-1, 1).to(self.weight.dtype)
                    dequantized_weight = _fp4_to_float(codes_grouped, exp_bits=FP8_EXP_BITS, mant_bits=FP8_MANTISSA_BITS, exp_bias=FP8_EXP_BIAS).to(self.weight.dtype) * scales
                    if self.zeros is not None:
                        zeros = self.zeros.reshape(-1, 1).to(self.weight.dtype)
                        dequantized_weight = dequantized_weight + zeros
                    dequantized_weight = dequantized_weight.view(original_shape)
                elif self.w_group_size == -1:
                    scales = self.scales.view(1, 1).to(self.weight.dtype)
                    dequantized_weight = _fp4_to_float(self.weight_fp8, exp_bits=FP8_EXP_BITS, mant_bits=FP8_MANTISSA_BITS, exp_bias=FP8_EXP_BIAS).to(self.weight.dtype) * scales
                    if self.zeros is not None:
                        dequantized_weight = dequantized_weight + self.zeros.view(1, 1).to(self.weight.dtype)
                elif self.w_group_size == -2:
                    original_shape = self.weight.shape
                    codes_grouped = self.weight_fp8.reshape(original_shape[0], -1)
                    scales = self.scales.reshape(original_shape[0], 1).to(self.weight.dtype)
                    dequantized_weight = _fp4_to_float(codes_grouped, exp_bits=FP8_EXP_BITS, mant_bits=FP8_MANTISSA_BITS, exp_bias=FP8_EXP_BIAS).to(self.weight.dtype) * scales
                    if self.zeros is not None:
                        zeros = self.zeros.reshape(original_shape[0], 1).to(self.weight.dtype)
                        dequantized_weight = dequantized_weight + zeros
                    dequantized_weight = dequantized_weight.view(original_shape)
                else:
                    raise ValueError("Invalid w_group_size")

            output = F.linear(input, dequantized_weight, self.bias)
            if input.dim() > 2:
                output = output.reshape(original_input_shape[:-1] + (self.out_features,))
            return output
        
        # 确保输入是2D的 (batch_size, in_features)
        original_input_shape = input.shape
        if input.dim() > 2:
            input = input.reshape(-1, input.shape[-1])

        # 根据mode选择前向计算方式
        if self.mode == 0 or self.mode == 1:
            # 1. 准备权重
            weight_int = self.weight        
            if self.w_group_size > 0: 
                # per-group 量化， 需要按组处理
                original_shape = self.weight.shape
                weight_int = self.weight.reshape(-1, self.w_group_size)

            # 2. 反量化，适用于所有粒度的情况
            if self.zeros is not None:
                dequantized_weight = (weight_int - self.zeros) * self.scales
            else:
                dequantized_weight = weight_int * self.scales

            if self.w_group_size > 0:
                dequantized_weight = dequantized_weight.view(original_shape)
            # FIGLUT-F模式和GPU模式都使用标准的线性层计算，因为GPU底层也是使用的FP32累加。
            return F.linear(input, dequantized_weight, self.bias)
        elif self.mode == 2:
            # FIGLUT-I模式
            # 1. 预对齐激活值
            aligned_activations, row_exponents = _pre_align_fp_activation(input)
            
            # 2. 将二进制表示转换回逻辑整数
            activations_int = _convert_bits_to_int(aligned_activations) # shape: [batch_size, in_features]

            # 3. 准备INT4权重
            weights_int = self.weight.long()  # shape: [out_features, in_features]

            if self.w_group_size <= 0:
                raise NotImplementedError("Only per-group quantization is supported in FIGLUT-I mode")
        
            num_groups = self.in_features // self.w_group_size
            scales_w = self.scales.reshape(self.out_features, num_groups)  # shape: [out_features, num_groups]

            # 4. 分组整数矩阵乘法
            # 4.1 Reshape A and W以进行分组计算
            # A: [m, n] -> [m, g, s] (m=batch_size, g=num_groups, s=group_size)
            activations_grouped = activations_int.reshape(-1, num_groups, self.w_group_size)
            # W: [o, n] -> [o, g, s] (o=out_features, g=num_groups, s=group_size)
            weights_grouped = weights_int.reshape(self.out_features, num_groups, self.w_group_size)

            # 4.2 使用einsum计算整数部分和
            # m: batch dim, o: output feature dim, g: group dim, s: group size dim
            # 我们对每个 group 内的元素 (s) 进行点积
            # 得到每个(batch, output_feature, group)的部分和
            # 'mgs,ogs->mog' 表示:
            #   - 对 m, g, s 和 o, g, s 进行操作
            #   - s 是求和的维度 (点积)
            #   - 结果的维度是 m, o, g
            activations_grouped_fp64 = activations_grouped.to(torch.float64)  # int64不被GPU支持，只能转换成FP64了
            weights_grouped_fp64 = weights_grouped.to(torch.float64)
            partial_sums_acc = torch.einsum('mgs,ogs->mog', activations_grouped_fp64, weights_grouped_fp64)

            # 5. 缩放部分和并用FP32累加
            # 5.1 准备用于广播的scales
            # scales_w: [o, g] -> [1, o, g]
            scales_w_broadcastable = scales_w.unsqueeze(0).to(torch.float32)

            # 5.2 将整数部分和乘以权重 scales，转换为 FP32
            # partial_sums_acc: [m, o, g]
            # scales_w_broadcastable: [1, o, g]
            # 结果 partial_sums_fp: [m, o, g]
            partial_sums_fp = partial_sums_acc.to(torch.float32) * scales_w_broadcastable
            partial_sums_fp /= 2**(MANTISSA_BITS-1)

            # 5.3 累加FP32部分和（在group维度上求和）
            # 模拟 FP32 累加器
            accumulated_fp = torch.sum(partial_sums_fp, dim=2) # shape: [m, o]

            # 6. 应用激活值的scale并添加偏置
            # 6.1 row_exponents: [m] -> [m, 1]
            row_exponents_fp32 = row_exponents.to(torch.float32)
            act_scales = torch.pow(2.0, row_exponents_fp32).unsqueeze(-1)
            output_rescaled = accumulated_fp * act_scales

            # 6.2 添加偏置并保证输出dtype与输入一致
            if self.bias is not None:
                output_rescaled = output_rescaled + self.bias.to(output_rescaled.dtype)
            output = output_rescaled.to(input.dtype)

        else:
            raise ValueError("Invalid mode")
        
        # 恢复输入的原始形状
        if original_input_shape != input.shape:
            output = output.reshape(original_input_shape[:-1] + (self.out_features,))

        return output

    @classmethod
    def from_linear(cls, linear_layer, w_bit=4, w_group_size=128, symmetric=False, mode=0, weight_format: str = "int", approximate: bool = False):
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

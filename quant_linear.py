import torch
import torch.nn as nn
import torch.nn.functional as F
from quant_funcs import pseudo_quantize_tensor

MANTISSA_BITS = 12
FP4_EXP_BITS = 2
FP4_MANTISSA_BITS = 1
FP4_EXP_BIAS = 1
FP6_EXP_BITS = 3
FP6_MANTISSA_BITS = 2
FP6_EXP_BIAS = 3


def _float_to_fp4(x: torch.Tensor, exp_bits: int = FP4_EXP_BITS, mant_bits: int = FP4_MANTISSA_BITS, exp_bias: int = FP4_EXP_BIAS) -> torch.Tensor:
    """
    将浮点张量量化为 FP4 (E2M1) 格式的 4bit 码字，存放在 uint8 的低 4 位。
    """
    sign = (x < 0).to(torch.uint8)
    x_abs = x.abs()
    zero_mask = x_abs == 0
    x_abs_safe = torch.where(zero_mask, torch.full_like(x_abs, 1e-8), x_abs)

    # 计算指数并加偏置
    exp_val = torch.floor(torch.log2(x_abs_safe)).to(torch.int32)
    exp_unbiased = (exp_val + exp_bias).clamp(0, (1 << exp_bits) - 1)

    # 调试：统计每组前八名的众数及出现次数（仅在二维输入时打印前10组）
    # if exp_unbiased.dim() == 2:
    #     num_groups = exp_unbiased.size(0)
    #     max_val = int(exp_unbiased.max()) + 1
    #     freq = torch.stack([
    #         torch.bincount(exp_unbiased[g], minlength=max_val) for g in range(num_groups)
    #     ], dim=0)
    #     topk_size = min(8, max_val)
    #     topk_vals, topk_idx = torch.topk(freq, k=topk_size, dim=1)
    #     show = min(10, num_groups)
    #     for g in range(show):
    #         pairs = []
    #         for k in range(topk_size):
    #             pairs.append(f"exp{int(topk_idx[g, k])}:{int(topk_vals[g, k])}")
    #         print(f"group {g}: " + ", ".join(pairs))

    # max_val = int(exp_unbiased.max()) + 1
    # modes = []
    # counts = []
    # top2_sum = []
    # for g in range(100):
    #     freq = torch.bincount(exp_unbiased[g], minlength=max_val)
    #     top2 = torch.topk(freq, k=min(3, freq.numel())).values     # 取前2个频次
    #     modes.append(top2)
    #     top2_sum.append(top2.sum())

    # top2_sum = torch.stack(top2_sum)  # [num_groups]

    # # 打印前10组的 top2 频次和
    # for i in range(100):
    #     print(f"group {i}: top2_freq_sum={int(top2_sum[i])}")
    # for i in range(100):
    #     print(exp_unbiased[i].sum())
    # # 计算这里面为1的指数的比例
    # exp_1 = exp_unbiased.sum() / (131072*128)

    # 计算尾数 (1 位)，用最邻近舍入
    mant_scale = 1 << mant_bits
    mant = torch.round((x_abs_safe / torch.pow(2.0, exp_val.float()) - 1.0) * mant_scale)
    mant = mant.clamp(0, mant_scale - 1).to(torch.uint8)

    code = (sign << (exp_bits + mant_bits)) | (exp_unbiased.to(torch.uint8) << mant_bits) | mant
    # code = code & 0xF  # 仅低 4 位有效
    # 零值强制编码为0，便于解码时恢复0
    code = torch.where(zero_mask, torch.zeros_like(code), code)
    return code


def _fp4_to_float(code: torch.Tensor, exp_bits: int = FP4_EXP_BITS, mant_bits: int = FP4_MANTISSA_BITS, exp_bias: int = FP4_EXP_BIAS) -> torch.Tensor:
    """
    将 FP4 (E2M1) 码字还原为浮点。输入为 uint8/long，返回 FP32。
    """
    code = code.to(torch.uint8)
    zero_mask = code == 0
    sign = ((code >> (exp_bits + mant_bits)) & 0x1).float()
    exp_field = ((code >> mant_bits) & ((1 << exp_bits) - 1)).float() - float(exp_bias)
    mant_field = (code & ((1 << mant_bits) - 1)).float() / float(1 << mant_bits)
    value = ((-1.0) ** sign) * (1.0 + mant_field) * torch.pow(2.0, exp_field)
    return torch.where(zero_mask, torch.zeros_like(value), value)

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
    def __init__(self, in_features, out_features, bias=True, w_bit=4, w_group_size=128, symmetric=True, mode=0, weight_format: str = "int"):
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
        if self.weight_format not in {"int", "fp4", "fp6"}:
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
        
        self.reset_parameters()

    def reset_parameters(self):
        """初始化参数"""
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / fan_in**0.5
            nn.init.uniform_(self.bias, -bound, bound)
    
    def quantize_weight(self):
        """量化权重"""
        with torch.no_grad():
            #----------------------------------------------------------------------
            # 将weight做个转置
            # self.weight.data = self.weight.data.t()
            #----------------------------------------------------------------------
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
                max_val = weight_tensor_grouped.abs().amax(dim=1, keepdim=True).clamp(min=1e-5)
                scales = (max_val / fp4_max_value).clamp(min=1e-5)

                # 归一化后限制在 FP4 表示范围内再编码
                normalized = torch.clamp(weight_tensor_grouped / scales, min=-fp4_max_value, max=fp4_max_value)
                fp4_codes = _float_to_fp4(normalized)

                # 保存 scale 形状
                if self.w_group_size == -1:
                    self.scales = scales.view(1, 1)
                elif self.w_group_size == -2:
                    self.scales = scales.view(original_shape[0], 1)
                else:
                    self.scales = scales.view(-1, 1)
                self.zeros = None

                # 存 FP4 码字，并写回反量化后的权重便于前向
                self.weight_fp4 = fp4_codes.view(original_shape)
                self.weight_fp6 = None
                dequantized = _fp4_to_float(fp4_codes).to(self.weight.data.dtype) * scales
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
                max_val = weight_tensor_grouped.abs().amax(dim=1, keepdim=True).clamp(min=1e-5)
                scales = (max_val / fp6_max_value).clamp(min=1e-5)

                normalized = torch.clamp(weight_tensor_grouped / scales, min=-fp6_max_value, max=fp6_max_value)
                fp6_codes = _float_to_fp4(normalized, exp_bits=FP6_EXP_BITS, mant_bits=FP6_MANTISSA_BITS, exp_bias=FP6_EXP_BIAS)

                if self.w_group_size == -1:
                    self.scales = scales.view(1, 1)
                elif self.w_group_size == -2:
                    self.scales = scales.view(original_shape[0], 1)
                else:
                    self.scales = scales.view(-1, 1)
                self.zeros = None

                self.weight_fp6 = fp6_codes.view(original_shape)
                self.weight_fp4 = None
                dequantized = _fp4_to_float(fp6_codes, exp_bits=FP6_EXP_BITS, mant_bits=FP6_MANTISSA_BITS, exp_bias=FP6_EXP_BIAS).to(self.weight.data.dtype) * scales
                self.weight.data = dequantized.view(original_shape)
                self.quantized.fill_(True)
                return

            if self.w_bit >= 16:
                self.quantized.fill_(False)
                self.weight_fp4 = None
                self.weight_fp6 = None
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
            self.quantized.fill_(True)
    
    def forward(self, input):
        """前向传播"""
        if not self.quantized:
            return F.linear(input, self.weight, self.bias)

        # FP4 前向直接解码为浮点做矩阵乘法（与整数量化路径分离）
        if self.weight_format == "fp4":
            original_input_shape = input.shape
            if self.weight_fp4 is None or self.scales is None:
                # 如果意外缺少码字或scale，退化为使用当前权重
                print("缺少scale!")
                dequantized_weight = self.weight
            else:
                if self.w_group_size > 0:
                    original_shape = self.weight.shape
                    codes_grouped = self.weight_fp4.reshape(-1, self.w_group_size)
                    scales = self.scales.reshape(-1, 1)
                    dequantized_weight = (_fp4_to_float(codes_grouped).to(self.weight.dtype) * scales).view(original_shape)
                elif self.w_group_size == -1:
                    scales = self.scales.view(1, 1)
                    dequantized_weight = _fp4_to_float(self.weight_fp4).to(self.weight.dtype) * scales
                elif self.w_group_size == -2:
                    original_shape = self.weight.shape
                    codes_grouped = self.weight_fp4.reshape(original_shape[0], -1)
                    scales = self.scales.reshape(original_shape[0], 1)
                    dequantized_weight = (_fp4_to_float(codes_grouped).to(self.weight.dtype) * scales).view(original_shape)
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
            original_input_shape = input.shape
            if self.weight_fp6 is None or self.scales is None:
                dequantized_weight = self.weight
            else:
                if self.w_group_size > 0:
                    original_shape = self.weight.shape
                    codes_grouped = self.weight_fp6.reshape(-1, self.w_group_size)
                    scales = self.scales.reshape(-1, 1)
                    dequantized_weight = (_fp4_to_float(codes_grouped, exp_bits=FP6_EXP_BITS, mant_bits=FP6_MANTISSA_BITS, exp_bias=FP6_EXP_BIAS).to(self.weight.dtype) * scales).view(original_shape)
                elif self.w_group_size == -1:
                    scales = self.scales.view(1, 1)
                    dequantized_weight = _fp4_to_float(self.weight_fp6, exp_bits=FP6_EXP_BITS, mant_bits=FP6_MANTISSA_BITS, exp_bias=FP6_EXP_BIAS).to(self.weight.dtype) * scales
                elif self.w_group_size == -2:
                    original_shape = self.weight.shape
                    codes_grouped = self.weight_fp6.reshape(original_shape[0], -1)
                    scales = self.scales.reshape(original_shape[0], 1)
                    dequantized_weight = (_fp4_to_float(codes_grouped, exp_bits=FP6_EXP_BITS, mant_bits=FP6_MANTISSA_BITS, exp_bias=FP6_EXP_BIAS).to(self.weight.dtype) * scales).view(original_shape)
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
    def from_linear(cls, linear_layer, w_bit=4, w_group_size=128, symmetric=False, mode=0, weight_format: str = "int"):
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

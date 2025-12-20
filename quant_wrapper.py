import torch
import sys
sys.path.append("./gptq")
from quant_funcs import pseudo_quantize_tensor
from quant_linear import QuantLinear, GPTQQuantLinear

def quantize_model(model, args, quant_mix_gate=False):
    """
    简化版的量化函数, 只支持Weight-only量化
    支持两种模式:
    1. RTN (Round-to-Nearest): 简单的四舍五入量化
    2. GPTQ: 基于Hessian的优化量化
    """
    # Weight-only quantization
    if (args.w_bit is not None and args.w_bit < 16) and (args.a_bit is None or args.a_bit >= 16):
        assert args.w_bit > 0 and args.w_bit < 16, "Weight bitwidth should be an integer between [1, 16] for weigth-only quantization, please check."
        w_format = getattr(args, 'w_format', 'int').lower()
        is_bfp = w_format.startswith("bfp")
        if is_bfp and getattr(args, "w_group_size", -1) <= 0:
            raise ValueError("BFP 量化仅支持分组量化，请设置正的 w_group_size（例如128）")

        # 判断使用哪种量化方法
        if hasattr(args, 'gptq') and args.gptq:
            if w_format != 'int':
                raise NotImplementedError("GPTQ 路径目前仅支持整数量化权重")
            print("使用GPTQ量化")
            from weight_only_quant.gptq_utils import apply_gptq
            model = apply_gptq(model, args)

            linear_modules = [
                (name, module)
                for name, module in model.named_modules()
                if isinstance(module, torch.nn.Linear) and 'lm_head' not in name and 'output_layer' not in name
            ]
            for name, module in linear_modules:
                gptq_linear = GPTQQuantLinear.from_linear(module)
                parent_module = model
                module_path = name.split('.')
                for module_name in module_path[:-1]:
                    parent_module = getattr(parent_module, module_name)
                setattr(parent_module, module_path[-1], gptq_linear)

            torch.cuda.empty_cache()
        else:
            if is_bfp:
                digits = "".join(ch for ch in w_format if ch.isdigit())
                bfp_bits = int(digits) if digits else getattr(args, "w_bit", 4)
                print(f"使用BFP分组量化（共享指数、每值{bfp_bits-1}bit尾数+符号，共{bfp_bits}bit）")
            else:
                print("使用RTN (Round-to-Nearest) 量化")
            # 用QuantLinear替换Linear层
            for name, module in model.named_modules():
                if isinstance(module, torch.nn.Linear) and 'lm_head' not in name and 'output_layer' not in name:
                    quant_linear = QuantLinear.from_linear(
                        linear_layer=module,
                        w_bit=args.w_bit,
                        w_group_size=args.w_group_size,
                        symmetric=args.w_symmetric,
                        mode=getattr(args, 'mode', 0),
                        weight_format=w_format,
                        approximate=getattr(args, "approximate", False),
                        quant_dim=getattr(args, "quant_dim", 0),
                        fp8_hi_align_start=getattr(args, "fp8_hi_align_start", 12),
                        fp8_hi_align_exp_field=getattr(args, "fp8_hi_align_exp_field", 15),
                        fp8_tail_pad_bits=getattr(args, "fp8_tail_pad_bits", 1),
                    )
                    # 替换
                    parent_module = model
                    module_path = name.split('.')
                    for module_name in module_path[:-1]:
                        parent_module = getattr(parent_module, module_name)
                    setattr(parent_module, module_path[-1], quant_linear)

                    torch.cuda.empty_cache()

    return model

import torch

FP4_E1M2_CLASS = {
    0.0: 0.0,
    0.5: 2**-16,
    1.0: 2**-15,
    1.5: 2**-15 * 1.5,
    2.0: 2**-14,
    2.5: 2**-14 * 1.25,
    3.0: 2**-14 * 1.5,
    3.5: 2**-14 * 1.75,
    -0.0: -0.0,
    -0.5: -2**-16,
    -1.0: -2**-15,
    -1.5: -2**-15 * 1.5,
    -2.0: -2**-14,
    -2.5: -2**-14 * 1.25,
    -3.0: -2**-14 * 1.5,
    -3.5: -2**-14 * 1.75,
}


def _fp4_e1m2_class_convert(tensor: torch.Tensor) -> torch.Tensor:
    device = tensor.device
    keys = torch.tensor(list(FP4_E1M2_CLASS.keys()), dtype=torch.float16, device=device)
    values = torch.tensor(list(FP4_E1M2_CLASS.values()), dtype=torch.float16, device=device)
    sorted_keys, indices = torch.sort(keys)
    sorted_values = values[indices]

    flat = tensor.to(device).view(-1).to(torch.float16)
    search_idx = torch.searchsorted(sorted_keys, flat)
    search_idx.clamp_(max=len(sorted_keys) - 1)
    mask = torch.isclose(sorted_keys[search_idx], flat, rtol=1e-3, atol=1e-5)
    return torch.where(mask, sorted_values[search_idx], flat).view(tensor.shape)


def _fp_scale(tensor, S, M, bias, max_float, min_float):
    tensor_unscaled = tensor / S
    tensor_unscaled = torch.clamp(tensor_unscaled, min_float, max_float)
    tensor_log_scales = torch.clamp((torch.floor(torch.log2(torch.abs(tensor_unscaled)) + bias)).detach(), 1.0)
    scales = 2.0 ** (tensor_log_scales - M - bias)
    tensor_q = (tensor_unscaled / scales).round()
    tensor_q = tensor_q * scales
    return tensor_q


def quantize_fp16_to_fp4_e1m2(tensor, group_size=128, per_tensor=False, return_scales=False):
    if tensor.dtype != torch.float16:
        tensor = tensor.to(torch.float16)
    if tensor.dim() != 2:
        raise ValueError("Expected a 2D tensor of shape [out_features, in_features].")

    org_shape = tensor.shape
    if group_size > 0:
        if org_shape[1] % group_size != 0:
            raise ValueError("in_features must be divisible by group_size.")
        tensor = tensor.reshape(-1, group_size)
    if per_tensor:
        tensor = tensor.reshape(1, -1)

    M = 1
    E = 2
    bias = 2 ** (E - 1) - 1
    max_float = (2 - 2 ** (-M)) * 2 ** (2**E - 1 - bias)
    min_float = -max_float

    max_val = tensor.abs().amax(dim=1, keepdim=True)
    max_val = max_val.clamp(min=1e-8)
    S = max_val / max_float

    block_q = _fp_scale(tensor, S, M, bias, max_float, min_float)
    return block_q*S

if __name__ == "__main__":
    torch.manual_seed(0)
    w = torch.randn(64, 128, dtype=torch.float16)
    w_q = quantize_fp16_to_fp4_e1m2(w, group_size=128)
    # print(w_q.dtype, w_q.shape)
    print(w[0])
    print(w_q[0])

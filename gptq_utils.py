"""Utility helpers to integrate GPTQ weight-only quantization with the FIGLUT codebase."""
import sys
from typing import List, Tuple

import torch
import torch.nn as nn

sys.path.append("./gptq")
from gptq.gptq import GPTQ
from gptq.quant import Quantizer
from gptq.modelutils import find_layers


def get_model_layers(model) -> Tuple[nn.ModuleList, str, nn.Module, nn.Module]:
    """Return the transformer block list and auxiliary modules for supported models."""
    model_type = model.config.model_type.lower()

    if "llama" in model_type:
        layers = model.model.layers
        prefix = "model.layers"
        embedding = model.model.embed_tokens
        norm = model.model.norm
    elif "opt" in model_type:
        layers = model.model.decoder.layers
        prefix = "model.decoder.layers"
        embedding = model.model.decoder.embed_tokens
        norm = model.model.decoder.final_layer_norm
    else:
        raise ValueError(f"Unsupported model type for GPTQ: {model_type}")

    return layers, prefix, embedding, norm


def _map_groupsize(w_group_size: int) -> int:
    if w_group_size is None:
        return -1
    if w_group_size in (-2, -1):
        return -1
    return w_group_size


def _collect_calib_batches(dataloader, limit: int) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    batches = []
    if dataloader is None:
        return batches
    for batch in dataloader:
        batches.append(batch)
        if len(batches) >= limit:
            break
    return batches


@torch.no_grad()
def apply_gptq(model, args):
    """Quantize the given causal LM in-place with GPTQ using the provided calibration data."""
    calib_batches = _collect_calib_batches(getattr(args, "dataloader", None), args.nsamples)
    if not calib_batches:
        raise ValueError("GPTQ quantization requires calibration data. Please provide a non-empty dataloader.")

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    if hasattr(model, "seqlen"):
        seqlen = model.seqlen
    elif hasattr(model.config, "max_position_embeddings"):
        seqlen = min(2048, model.config.max_position_embeddings)
    else:
        seqlen = 2048

    hidden_size = model.config.hidden_size
    nsamples = len(calib_batches)
    dtype = next(iter(model.parameters())).dtype

    layers, _, embedding, norm = get_model_layers(model)

    use_cache = getattr(model.config, "use_cache", False)
    model.config.use_cache = False

    embedding = embedding.to(dev)
    if norm is not None:
        norm = norm.to(dev)
    layers[0] = layers[0].to(dev)

    inps = torch.zeros((nsamples, seqlen, hidden_size), dtype=dtype, device=dev)
    outs = torch.zeros_like(inps)
    cache = {"i": 0, "attention_mask": None, "position_ids": None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            inps[cache["i"]] = inp
            cache["i"] += 1
            cache["attention_mask"] = kwargs.get("attention_mask")
            cache["position_ids"] = kwargs.get("position_ids")
            raise ValueError

    layers[0] = Catcher(layers[0])

    for batch in calib_batches:
        input_ids = batch[0].to(dev)
        try:
            model(input_ids)
        except ValueError:
            pass
        if cache["i"] >= nsamples:
            break

    layers[0] = layers[0].module
    layers[0] = layers[0].cpu()
    embedding = embedding.cpu()
    if norm is not None:
        norm = norm.cpu()
    torch.cuda.empty_cache()

    model.model.embed_tokens = embedding
    if norm is not None:
        if "llama" in model.config.model_type.lower():
            model.model.norm = norm
        elif "opt" in model.config.model_type.lower():
            model.model.decoder.final_layer_norm = norm

    attn_kwargs = {}
    if cache["attention_mask"] is not None:
        attn_kwargs["attention_mask"] = cache["attention_mask"]
    if cache["position_ids"] is not None:
        attn_kwargs["position_ids"] = cache["position_ids"]

    groupsize = _map_groupsize(getattr(args, "w_group_size", -1))
    per_channel = True
    if not per_channel and groupsize <= 0:
        raise ValueError('GPTQ requires a positive group size for per-group quantization.')

    print("Running GPTQ quantization...")
    for layer_idx, layer in enumerate(layers):
        print(f"Quantizing transformer layer {layer_idx}")
        layer = layer.to(dev)
        subset = find_layers(layer)
        if subset:
            gptq_modules = {}
            for name, module in subset.items():
                gptq_modules[name] = GPTQ(module)
                gptq_modules[name].quantizer = Quantizer()
                gptq_modules[name].quantizer.configure(
                    args.w_bit,
                    perchannel=per_channel,
                    sym=getattr(args, "w_symmetric", False),
                    mse=False,
                )

            def make_hook(name):
                def hook(_, inp, out):
                    gptq_modules[name].add_batch(inp[0].detach(), out.detach())
                return hook

            handles = []
            for name in subset:
                handles.append(subset[name].register_forward_hook(make_hook(name)))

            for sample_idx in range(nsamples):
                output = layer(inps[sample_idx].unsqueeze(0), **attn_kwargs)
                outs[sample_idx] = output[0] if isinstance(output, tuple) else output

            for handle in handles:
                handle.remove()

            for name in subset:
                gptq_modules[name].fasterquant(
                    percdamp=getattr(args, "percdamp", 0.01),
                    groupsize=groupsize,
                    actorder=getattr(args, "act_order", False),
                    static_groups=False,
                )
                quantizer = gptq_modules[name].quantizer
                module = subset[name]
                module.register_buffer('scales', quantizer.scale.detach().clone())  # 这里就注册好了
                module.register_buffer('zeros', quantizer.zero.detach().clone())
                gptq_modules[name].free()

            for sample_idx in range(nsamples):
                output = layer(inps[sample_idx].unsqueeze(0), **attn_kwargs)
                outs[sample_idx] = output[0] if isinstance(output, tuple) else output

        layers[layer_idx] = layer.cpu()
        torch.cuda.empty_cache()
        inps, outs = outs, inps

    model.to(dev)
    torch.cuda.empty_cache()

    model.config.use_cache = use_cache
    return model

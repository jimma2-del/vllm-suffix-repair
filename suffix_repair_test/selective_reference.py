# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Dense FP32 oracle for short selective-edit verification fixtures."""

from itertools import product

import torch

REDUCTIONS = tuple(product(("max", "sum"), repeat=3))
FORMATS = ("fp32", "fp16", "bf16", "fp8", "int8", "int4")


def represent(x, fmt):
    x = x.float()
    floats = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}
    if fmt in floats:
        return x.to(floats[fmt]).float()
    limit = {"int4": 7, "int8": 127, "fp8": 448}[fmt]
    # Round each division to FP32 independently of CUDA scalar reciprocal math.
    scale = (x.abs().amax(-1, keepdim=True).double() / limit).float()
    scale = scale.masked_fill(scale == 0, 1)
    code = (x.double() / scale.double()).float().clamp(-limit, limit)
    code = code.to(torch.float8_e4m3fn).float() if fmt == "fp8" else code.round()
    return code * scale


def dots(q, k, fmt):
    """Explicitly pair each query head with its KV head; no paged operations."""
    keys = represent(k, fmt)
    group_size = q.shape[1] // k.shape[1]
    scores = []
    for head in range(q.shape[1]):
        products = q[:, None, head].float() * keys[None, :, head // group_size]
        scores.append(products.sum(-1))
    return torch.stack(scores, dim=1)


def reduce_scores(layers, ops):
    def reduce(x, axis, op):
        return x.sum(axis) if op == "sum" else x.amax(axis)

    by_layer = [reduce(reduce(x, -1, ops[0]), -1, ops[1]) for x in layers]
    return reduce(torch.stack(by_layer), 0, ops[2])


def midpoint(scores):
    """Bisect distinct scores near the median, away from rounding ties."""
    values = scores.unique(sorted=True)
    assert values.numel() >= 2
    i = values.numel() // 2
    return (values[i - 1].item() + values[i].item()) / 2

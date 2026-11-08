# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from vllm.forward_context import set_forward_context
from vllm.model_executor.layers.rotary_embedding.base import RotaryEmbedding
from vllm.model_executor.layers.rotary_embedding.common import ApplyRotaryEmb
from vllm.utils.math_utils import cdiv
from vllm.v1.core.kv_cache_edit import KVCacheEdit
from vllm.v1.worker.gpu.attn_utils import (
    build_attn_metadata,
    build_slot_mappings_by_layer,
)
from vllm.v1.worker.utils import prepare_kernel_block_sizes

if TYPE_CHECKING:
    from vllm.v1.worker.gpu.model_runner import GPUModelRunner as GPUModelRunnerV2
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner


def _relocate_keys(
    keys: torch.Tensor, rope: RotaryEmbedding, old: torch.Tensor, new: torch.Tensor
) -> None:
    # Compose R(new) with the inverse of R(old). Scaled or rounded sin/cos
    # tables need the squared magnitude in the inverse.
    table = rope.cos_sin_cache
    cos0, sin0 = table[old].float().chunk(2, dim=-1)
    cos1, sin1 = table[new].float().chunk(2, dim=-1)
    norm = cos0.square() + sin0.square()
    cos = (cos1 * cos0 + sin1 * sin0) / norm
    sin = (sin1 * cos0 - cos1 * sin0) / norm
    rotary = keys[..., : rope.rotary_dim]
    rotary.copy_(
        ApplyRotaryEmb.forward_static(rotary.float(), cos, sin, rope.is_neox_style)
    )


def _prefill(
    runner: GPUModelRunner | GPUModelRunnerV2,
    token_ids: list[int],
    start_pos: int,
    pages: torch.Tensor,
    block_size: int,
) -> None:
    """Prefill a segment after the KV already present in the destination pages."""
    if not token_ids:
        return
    device = runner.device
    model = runner.get_model()
    ids = torch.tensor(token_ids, dtype=torch.long, device=device)
    pages = pages[: cdiv(start_pos + len(token_ids), block_size)].int()
    (kernel_size,) = prepare_kernel_block_sizes(
        runner.kv_cache_config, runner.attn_groups
    )
    ratio = block_size // kernel_size
    table = (
        pages[:, None] * ratio + torch.arange(ratio, dtype=pages.dtype, device=device)
    ).reshape(1, -1)
    chunk_size = runner.vllm_config.scheduler_config.max_num_batched_tokens
    for chunk in ids.split(chunk_size):
        count = chunk.numel()
        end = start_pos + count
        positions = torch.arange(start_pos, end, device=device)
        slots = pages[positions // block_size].long() * block_size
        slots += positions % block_size
        query_start = torch.tensor([0, count], dtype=torch.int32, device="cpu")
        seq_len = torch.tensor([end], dtype=torch.int32, device="cpu")
        slots = slots.unsqueeze(0)
        metadata = build_attn_metadata(
            runner.attn_groups,
            num_reqs=1,
            num_tokens=count,
            query_start_loc_gpu=query_start.to(device),
            query_start_loc_cpu=query_start,
            seq_lens=seq_len.to(device),
            seq_lens_cpu_upper_bound=seq_len,
            max_query_len=count,
            max_seq_len=end,
            block_tables=[table],
            slot_mappings=slots,
            kv_cache_config=runner.kv_cache_config,
            positions=positions,
            is_prefilling=torch.tensor([True], device="cpu"),
        )
        with set_forward_context(
            metadata,
            runner.vllm_config,
            num_tokens=count,
            slot_mapping=build_slot_mappings_by_layer(slots, runner.kv_cache_config),
            skip_compiled=True,
        ):
            model(input_ids=chunk, positions=positions)
        start_pos = end


def _copy_tokens(
    cache: torch.Tensor,
    rope: RotaryEmbedding,
    block_size: int,
    src_pages: torch.Tensor,
    dst_pages: torch.Tensor,
    src_start: int,
    dst_start: int,
    count: int,
) -> None:
    # Bound the temporary gather independently of the context length.
    for offset in range(0, count, 1024):
        steps = torch.arange(offset, min(offset + 1024, count), device=cache.device)
        old, new = steps + src_start, steps + dst_start
        src = src_pages[old // block_size] * block_size + old % block_size
        dst = dst_pages[new // block_size] * block_size + new % block_size
        kernel_size = cache.shape[1]
        kv = cache[src // kernel_size, src % kernel_size]
        if src_start != dst_start:
            _relocate_keys(kv[..., : rope.head_size], rope, old, new)
        cache[dst // kernel_size, dst % kernel_size] = kv


@torch.inference_mode()
def write_kv_cache_edit(
    runner: GPUModelRunner | GPUModelRunnerV2,
    edit: KVCacheEdit,
) -> None:
    model = runner.get_model()
    context = runner.vllm_config.compilation_config.static_forward_context
    layers = []
    for group in runner.attn_groups[0]:
        for name in group.layer_names:
            parent = model.get_submodule(name.rsplit(".", 1)[0])
            rope = getattr(parent, "rotary_emb", None)
            layer = context[name]
            cache = layer.kv_cache
            if (
                not isinstance(rope, RotaryEmbedding)
                or type(rope).forward_native is not RotaryEmbedding.forward_native
                or rope.cos_sin_cache.ndim != 2
                or rope.cos_sin_cache.shape[1] != rope.rotary_dim
                or cache.ndim != 4
                or cache.shape[1] != layer.num_kv_heads
                or cache.shape[-1] != 2 * rope.head_size
                or cache.dtype not in (torch.float16, torch.bfloat16, torch.float32)
                or edit.block_size % cache.shape[2]
                or getattr(layer, "kv_sharing_target_layer_name", None) is not None
            ):
                raise ValueError(
                    f"{name} needs a dense K/V cache and 1D RoPE for editing."
                )
            layers.append((cache.transpose(1, 2), rope))

    source, destination = (
        torch.tensor(pages, dtype=torch.long, device=runner.device)
        for pages in (edit.source, edit.destination)
    )
    # Whole prefix pages are shared. Seed the prefix in the first private page
    # for every layer before the replacement reads it.
    page_start = edit.start // edit.block_size * edit.block_size
    for cache, rope in layers:
        _copy_tokens(
            cache,
            rope,
            edit.block_size,
            source,
            destination,
            page_start,
            page_start,
            edit.start - page_start,
        )

    _prefill(runner, edit.replacement_ids, edit.start, destination, edit.block_size)
    suffix_start = edit.start + len(edit.replacement_ids)
    if edit.suffix_ids is not None:
        _prefill(runner, edit.suffix_ids, suffix_start, destination, edit.block_size)
    else:
        for cache, rope in layers:
            _copy_tokens(
                cache,
                rope,
                edit.block_size,
                source,
                destination,
                edit.end,
                suffix_start,
                edit.copy_end - suffix_start,
            )

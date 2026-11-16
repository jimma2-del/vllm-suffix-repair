# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.distributed as dist

from vllm.distributed.parallel_state import get_tp_group
from vllm.forward_context import set_forward_context
from vllm.model_executor.layers.rotary_embedding.base import RotaryEmbedding
from vllm.model_executor.layers.rotary_embedding.common import ApplyRotaryEmb
from vllm.utils.math_utils import cdiv
from vllm.v1.attention.ops.query_cache import QueryCache
from vllm.v1.core.kv_cache_edit import KVCacheEdit
from vllm.v1.engine import ScoreReduction
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
        slots = _slots(pages, positions, block_size)
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


def _slots(
    pages: torch.Tensor, positions: torch.Tensor, block_size: int
) -> torch.Tensor:
    return pages[positions // block_size].long() * block_size + positions % block_size


def _copy_tokens(
    cache: torch.Tensor,
    rope: RotaryEmbedding,
    block_size: int,
    src_pages: torch.Tensor,
    dst_pages: torch.Tensor,
    src_start: int,
    dst_start: int,
    count: int,
    query_cache: QueryCache | None = None,
) -> None:
    # Bound the temporary gather independently of the context length.
    for offset in range(0, count, 1024):
        steps = torch.arange(offset, min(offset + 1024, count), device=cache.device)
        old, new = steps + src_start, steps + dst_start
        src = _slots(src_pages, old, block_size)
        dst = _slots(dst_pages, new, block_size)
        kernel_size = cache.shape[1]
        kv = cache[src // kernel_size, src % kernel_size]
        if src_start != dst_start:
            _relocate_keys(kv[..., : rope.head_size], rope, old, new)
        cache[dst // kernel_size, dst % kernel_size] = kv
        if query_cache is not None and query_cache.data is not None:
            if src_start == dst_start:
                data = query_cache.data
                data[dst // kernel_size, :, dst % kernel_size] = data[
                    src // kernel_size, :, src % kernel_size
                ]
            else:
                query = query_cache.read(src)
                _relocate_keys(query, rope, old, new)
                query_cache.write(query, dst)


def _reduce(scores: torch.Tensor, dim: int, op: ScoreReduction) -> torch.Tensor:
    return scores.amax(dim) if op == "max" else scores.sum(dim)


def _accumulate(
    result: torch.Tensor, partial: torch.Tensor, op: ScoreReduction
) -> None:
    if op == "max":
        torch.maximum(result, partial, out=result)
    else:
        result.add_(partial)


def _score_key_span(
    cache: torch.Tensor,
    query_cache: QueryCache,
    key_slots: torch.Tensor,
    query_slots: torch.Tensor,
    head_scores: torch.Tensor,
    reduction: ScoreReduction,
) -> None:
    """Fold one old or new key span into this layer's per-query-head scores."""
    kernel_size, kv_heads = cache.shape[1:3]
    # Quantize a key tile once for all suffix queries and all heads in its group.
    for key_start in range(0, len(key_slots), 128):
        slots = key_slots[key_start : key_start + 128]
        keys = cache[
            slots // kernel_size, slots % kernel_size, :, : query_cache.head_size
        ]
        keys = query_cache.roundtrip_keys(keys).float()
        for start in range(0, len(query_slots), 256):
            tile = slice(start, start + 256)
            query = query_cache.read(query_slots[tile]).float()
            query = query.unflatten(1, (kv_heads, -1))
            # [suffix, KV head, query within group, affected key]
            dots = torch.einsum("tkgd,skd->tkgs", query, keys)
            partial = _reduce(dots, -1, reduction).flatten(1)
            _accumulate(head_scores[tile], partial, reduction)


def _selected_runs(
    edit: KVCacheEdit,
    layers: dict[str, tuple[torch.Tensor, RotaryEmbedding, QueryCache | None]],
    source: torch.Tensor,
    destination: torch.Tensor,
) -> list[list[int]]:
    """Return contiguous suffix-relative runs after all layers have scored."""
    options = edit.selection
    assert edit.suffix_ids is not None and options is not None
    count = len(edit.suffix_ids)
    removed = edit.end - edit.start if options["include_deleted"] else 0
    inserted = len(edit.replacement_ids)
    if not count or not (removed or inserted):
        return []

    device = destination.device
    offsets = torch.arange(count, device=device)

    def span_slots(pages: torch.Tensor, length: int) -> torch.Tensor:
        return _slots(
            pages, edit.start + torch.arange(length, device=device), edit.block_size
        )

    old_keys = span_slots(source, removed)
    new_keys = span_slots(destination, inserted)
    old_queries = _slots(source, edit.end + offsets, edit.block_size)
    new_queries = _slots(destination, edit.start + inserted + offsets, edit.block_size)
    key_op = options["key_reduction"]
    head_op = options["head_reduction"]
    layer_op = options["layer_reduction"]
    scores = torch.full(
        (count,),
        -torch.inf if layer_op == "max" else 0.0,
        dtype=torch.float32,
        device=device,
    )
    tp = get_tp_group()
    # Match per-layer collectives even when attention groups enumerate layers
    # in a different order on different ranks.
    for name in sorted(layers):
        cache, _, query_cache = layers[name]
        assert query_cache is not None
        by_head = scores.new_full(
            (count, query_cache.num_heads), -torch.inf if key_op == "max" else 0.0
        )
        _score_key_span(cache, query_cache, old_keys, old_queries, by_head, key_op)
        _score_key_span(cache, query_cache, new_keys, new_queries, by_head, key_op)
        by_layer = _reduce(by_head, 1, head_op)
        if tp.world_size > 1:
            op = dist.ReduceOp.MAX if head_op == "max" else dist.ReduceOp.SUM
            dist.all_reduce(by_layer, op=op, group=tp.device_group)
        _accumulate(scores, by_layer, layer_op)

    mask = torch.nn.functional.pad(scores > options["threshold"], (1, 1))
    boundaries = (mask[1:] != mask[:-1]).nonzero().reshape(-1, 2)
    return boundaries.tolist()


@torch.inference_mode()
def write_kv_cache_edit(
    runner: GPUModelRunner | GPUModelRunnerV2,
    edit: KVCacheEdit,
) -> None:
    model = runner.get_model()
    context = runner.vllm_config.compilation_config.static_forward_context
    layers = {}
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
            query_cache = getattr(layer, "query_cache", None)
            if edit.selection is not None and (
                query_cache is None
                or query_cache.data is None
                or query_cache.head_size != rope.head_size
                or query_cache.num_heads % layer.num_kv_heads
            ):
                raise ValueError(
                    f"{name} requires a bound per-head Q cache for selection."
                )
            layers[name] = (cache.transpose(1, 2), rope, query_cache)

    source, destination = (
        torch.tensor(pages, dtype=torch.long, device=runner.device)
        for pages in (edit.source, edit.destination)
    )
    # Whole prefix pages are shared. Seed the prefix in the first private page
    # for every layer before the replacement reads it.
    page_start = edit.start // edit.block_size * edit.block_size
    for cache, rope, query_cache in layers.values():
        _copy_tokens(
            cache,
            rope,
            edit.block_size,
            source,
            destination,
            page_start,
            page_start,
            edit.start - page_start,
            query_cache,
        )

    _prefill(runner, edit.replacement_ids, edit.start, destination, edit.block_size)
    suffix_start = edit.start + len(edit.replacement_ids)
    if edit.suffix_ids is not None and edit.selection is None:
        _prefill(runner, edit.suffix_ids, suffix_start, destination, edit.block_size)
        return
    for cache, rope, query_cache in layers.values():
        _copy_tokens(
            cache,
            rope,
            edit.block_size,
            source,
            destination,
            edit.end,
            suffix_start,
            edit.copy_end - suffix_start,
            query_cache,
        )
    if edit.selection is not None:
        assert edit.suffix_ids is not None
        for start, end in _selected_runs(edit, layers, source, destination):
            _prefill(
                runner,
                edit.suffix_ids[start:end],
                suffix_start + start,
                destination,
                edit.block_size,
            )

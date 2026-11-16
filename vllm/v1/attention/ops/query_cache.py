# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Mapping
from typing import Any

import torch

from vllm.config.cache import QueryCacheDType
from vllm.distributed.parallel_state import get_dcp_group
from vllm.triton_utils import tl, tldevice, triton
from vllm.utils.math_utils import cdiv

_FLOAT_DTYPES = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}
_QUANT_MAX = {"fp8": 448, "int8": 127, "int4": 7}


@triton.jit
def _store_bytes(page, plane_stride, byte, values, mask, WIDTH: tl.constexpr):
    dest = page + (byte // WIDTH).to(tl.int64) * plane_stride + byte % WIDTH
    tl.store(dest, values, mask=mask)


@triton.jit
def _e4m3_bytes(x):
    # E4M3FN, including subnormals, on devices without native FP8 conversion.
    mag = tl.abs(x)
    exponent = tl.maximum((mag.to(tl.int32, bitcast=True) >> 23) - 127, -6)
    inv_step = ((130 - exponent) << 23).to(tl.float32, bitcast=True)
    mantissa = tldevice.rint(mag * inv_step).to(tl.int32)
    sign = (x.to(tl.uint32, bitcast=True) >> 24) & 128
    return sign | (mantissa + ((exponent + 6) << 3))


@triton.jit
def _store_query(
    src,
    dst,
    slots,
    src_token_stride,
    src_head_stride: tl.constexpr,
    src_dim_stride: tl.constexpr,
    block_stride,
    plane_stride,
    token_stride: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    PLANE_BYTES: tl.constexpr,
    ROW_BYTES: tl.constexpr,
    HEAD_SIZE: tl.constexpr,
    START: tl.constexpr,
    FORMAT: tl.constexpr,
    LIMIT: tl.constexpr,
    TILE: tl.constexpr,
):
    token = tl.program_id(0).to(tl.int64)
    head = tl.program_id(1)
    slot = tl.load(slots + token).to(tl.int64)
    if slot < 0:
        return
    i = tl.arange(0, TILE)
    start = START + head * ROW_BYTES
    src += token * src_token_stride + head * src_head_stride
    page = (
        dst + (slot // BLOCK_SIZE) * block_stride + (slot % BLOCK_SIZE) * token_stride
    )
    if FORMAT == "raw":
        row = tl.load(src + i, mask=i < ROW_BYTES, other=0)
        _store_bytes(page, plane_stride, start + i, row, i < ROW_BYTES, PLANE_BYTES)
    else:
        q = tl.load(src + i * src_dim_stride, mask=i < HEAD_SIZE, other=0).to(
            tl.float32
        )
        scale = tl.div_rn(tl.max(tl.abs(q), axis=0), LIMIT)
        scale = tl.where(scale > 0, scale, 1.0)
        s = tl.arange(0, 4)
        scale_bytes = scale.to(tl.uint32, bitcast=True) >> (s * 8)
        _store_bytes(page, plane_stride, start + s, scale_bytes, True, PLANE_BYTES)
        x = tl.clamp(tl.div_rn(q, scale), -LIMIT, LIMIT)
        codes = _e4m3_bytes(x) if FORMAT == "fp8" else tldevice.rint(x).to(tl.int8)
        if FORMAT == "int4":
            lo, hi = tl.split(tl.reshape(codes, (TILE // 2, 2)))
            payload = (lo & 15) | (hi << 4)
            j = tl.arange(0, TILE // 2)
        else:
            payload = codes
            j = i
        _store_bytes(
            page, plane_stride, start + 4 + j, payload, j < ROW_BYTES - 4, PLANE_BYTES
        )


class QueryCache:
    """Per-token query heads packed into the trailing head slots of a KV page."""

    def __init__(
        self, dtype: QueryCacheDType, num_heads: int, head_size: int, dcp_size: int = 1
    ):
        self.dtype = dtype
        self.local_num_heads = num_heads
        self.num_heads = num_heads * dcp_size
        self.head_size = head_size
        if dtype in _FLOAT_DTYPES:
            self.row_bytes = head_size * _FLOAT_DTYPES[dtype].itemsize
        else:
            if dtype not in _QUANT_MAX:
                raise ValueError(f"Unsupported query cache dtype {dtype!r}")
            self.row_bytes = 4 + (cdiv(head_size, 2) if dtype == "int4" else head_size)
        self.size_bytes = self.row_bytes * self.num_heads
        self.byte_offset = 0
        self.pages: torch.Tensor | None = None
        self.data: torch.Tensor | None = None

    def bind(
        self, pages: torch.Tensor, kv_heads: int, *, packed_nvfp4: bool = False
    ) -> torch.Tensor:
        self.pages = pages
        kv = pages[:, :kv_heads]
        data = pages[:, kv_heads:]
        if packed_nvfp4 and pages.stride(1) == pages.shape[-1]:
            # FlashInfer's NHD NVFP4 pages contain contiguous K/V data and
            # scale planes. Place Q after those planes within each block.
            width = pages.shape[-1]
            kv = kv.as_strided(kv.shape, (pages.stride(0), width, kv_heads * width, 1))
            data = data.as_strided(
                data.shape,
                (pages.stride(0), width, data.shape[1] * width, 1),
                pages.storage_offset() + kv_heads * pages.shape[2] * width,
            )
        self.data = data.view(torch.uint8)
        assert self.byte_offset + self.size_bytes <= (
            self.data.shape[1] * self.data.shape[-1]
        )
        return kv

    def unbind(self) -> None:
        self.pages = self.data = None

    def _encode(self, query: torch.Tensor) -> torch.Tensor:
        if self.dtype in _FLOAT_DTYPES:
            data = query.to(_FLOAT_DTYPES[self.dtype])
            if data.stride(-1) != 1:
                data = data.clone(memory_format=torch.contiguous_format)
            return data.view(torch.uint8)
        q = query.float()
        limit = _QUANT_MAX[self.dtype]
        scale = q.abs().amax(dim=-1, keepdim=True) / limit
        scale = torch.where(scale > 0, scale, 1.0)
        values = (q / scale).clamp(-limit, limit)
        if self.dtype == "fp8":
            payload = values.to(torch.float8_e4m3fn).view(torch.uint8)
        else:
            payload = values.round().to(torch.int8).view(torch.uint8)
            if self.dtype == "int4":
                if self.head_size % 2:
                    payload = torch.nn.functional.pad(payload, (0, 1))
                payload = (payload[..., ::2] & 15) | (payload[..., 1::2] << 4)
        return torch.cat((scale.view(torch.uint8), payload), dim=-1)

    def _indices(self, slots: torch.Tensor) -> tuple[torch.Tensor, ...]:
        assert self.data is not None
        _, _, block_size, plane_bytes = self.data.shape
        byte = torch.arange(self.size_bytes, device=slots.device) + self.byte_offset
        return (
            slots[:, None] // block_size,
            byte // plane_bytes,
            slots[:, None] % block_size,
            byte % plane_bytes,
        )

    def record(self, query: torch.Tensor, slots: torch.Tensor) -> None:
        """Gather the query heads for the tokens owned by this DCP rank."""
        if self.num_heads != self.local_num_heads and self.data is not None:
            local = query.view(-1, self.local_num_heads, self.head_size)
            query = get_dcp_group().all_gather(local.contiguous(), dim=1)
        self.write(query, slots)

    def write(self, query: torch.Tensor, slots: torch.Tensor) -> None:
        """Write all query heads using the KV writer's physical slot mapping."""
        data = self.data
        if data is None or slots.numel() == 0:
            return
        query = query.view(-1, self.num_heads, self.head_size)[: slots.numel()]
        assert query.shape[0] == slots.numel()
        if data.is_cuda:
            self._write_cuda(query, slots, data, self.byte_offset)
        else:
            valid = slots >= 0
            valid_slots = slots[valid]
            if valid_slots.numel():
                data[self._indices(valid_slots)] = self._encode(query[valid]).flatten(1)

    def _write_cuda(
        self,
        vectors: torch.Tensor,
        slots: torch.Tensor,
        data: torch.Tensor,
        byte_offset: int,
    ) -> None:
        raw = self.dtype in _FLOAT_DTYPES
        src = self._encode(vectors) if raw else vectors
        _store_query[(len(slots), vectors.shape[1])](
            src,
            data,
            slots,
            *src.stride(),
            *data.stride()[:3],
            data.shape[2],
            data.shape[3],
            self.row_bytes,
            self.head_size,
            byte_offset,
            "raw" if raw else self.dtype,
            _QUANT_MAX.get(self.dtype, 0),
            triton.next_power_of_2(self.row_bytes if raw else max(2, self.head_size)),
        )

    def read(self, slots: torch.Tensor) -> torch.Tensor:
        """Decode the specified physical slots, e.g. to relocate post-RoPE Q."""
        assert self.data is not None
        rows = self.data[self._indices(slots)].view(-1, self.num_heads, self.row_bytes)
        return self._decode(rows)

    def roundtrip_keys(self, keys: torch.Tensor) -> torch.Tensor:
        """Represent each KV head in the query format for edit scoring."""
        if self.dtype in _FLOAT_DTYPES:
            return keys.to(_FLOAT_DTYPES[self.dtype])
        if keys.is_cuda and keys.numel():
            # Use the Q writer's exact scale and tie rounding. Each temporary
            # page contains one token and one byte plane per KV head.
            tokens, heads, _ = keys.shape
            rows = keys.new_empty((tokens, heads, 1, self.row_bytes), dtype=torch.uint8)
            self._write_cuda(keys, torch.arange(tokens, device=keys.device), rows, 0)
            return self._decode(rows.squeeze(2))
        return self._decode(self._encode(keys))

    def _decode(self, rows: torch.Tensor) -> torch.Tensor:
        if self.dtype in _FLOAT_DTYPES:
            return rows.view(_FLOAT_DTYPES[self.dtype])
        scale = rows[..., :4].flatten().view(torch.float32).view(*rows.shape[:-1], 1)
        payload = rows[..., 4:]
        if self.dtype == "fp8":
            q = payload.view(torch.float8_e4m3fn).float()
        else:
            q = payload.view(torch.int8)
            if self.dtype == "int4":
                q = torch.stack(((q << 4) >> 4, q >> 4), dim=-1).flatten(-2)
                q = q[..., : self.head_size]
        return q * scale


def get_cache_pages(layer: Any) -> torch.Tensor:
    """The complete bound pages for cache lifecycle operations."""
    cache = getattr(layer, "query_cache", None)
    if cache is not None and cache.pages is not None:
        return cache.pages
    return layer.kv_cache


def get_query_cache_sizes(layers: Mapping[str, Any]) -> dict[str, int]:
    """Assign independent Q records to layers sharing the same KV pages."""
    sizes: dict[str, int] = {}
    for name, layer in layers.items():
        if (cache := getattr(layer, "query_cache", None)) is not None:
            owner = layer.kv_sharing_target_layer_name or name
            cache.byte_offset = sizes.get(owner, 0)
            sizes[owner] = cache.byte_offset + cache.size_bytes
    return sizes

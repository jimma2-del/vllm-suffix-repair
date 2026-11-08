# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any
from uuid import uuid4

from vllm.config import VllmConfig
from vllm.sampling_params import SamplingParams
from vllm.utils.math_utils import cdiv
from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.kv_cache_utils import (
    BlockHash,
    KVCacheBlock,
    generate_block_hash_extra_keys,
    get_request_block_hasher,
    hash_block_tokens,
)
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheConfig, KVQuantMode
from vllm.v1.request import Request


@dataclass
class KVCacheEdit:
    """Physical page tables for one synchronous, copy-on-write edit."""

    block_size: int
    source: list[int]
    scratch: list[int]
    destination: list[int]
    copy_end: int
    replacement_ids: list[int]
    start: int
    end: int


class KVCacheEditor:
    """Edit local, dense, causal attention pages using the APC block pool."""

    salt_prefix = "kv-edit-"

    def __init__(self, pool: BlockPool, hash_fn: Callable[[Any], bytes]):
        self.pool = pool
        self.block_size = pool.hash_block_size
        self.hash_fn = hash_fn
        self.block_hasher = get_request_block_hasher(self.block_size, hash_fn)

    @classmethod
    def for_config(
        cls,
        config: VllmConfig,
        cache: KVCacheConfig,
        pool: BlockPool,
        hash_fn: Callable[[Any], bytes],
    ) -> "KVCacheEditor | None":
        parallel = config.parallel_config
        if (
            not pool.enable_caching
            or config.device_config.device_type != "cuda"
            or len(cache.kv_cache_groups) != 1
            or config.kv_transfer_config is not None
            or config.speculative_config is not None
            or config.lora_config is not None
            or parallel.data_parallel_size != 1
            or parallel.pipeline_parallel_size != 1
            or parallel.decode_context_parallel_size != 1
            or parallel.prefill_context_parallel_size != 1
            or config.model_config.runner_type != "generate"
        ):
            return None
        spec = cache.kv_cache_groups[0].kv_cache_spec
        if (
            type(spec) is not FullAttentionSpec
            or spec.non_causal
            or spec.kv_quant_mode != KVQuantMode.NONE
            or spec.block_size != pool.hash_block_size
        ):
            return None
        return cls(pool, hash_fn)

    def _hash_at(self, request: Request, end: int) -> BlockHash:
        page, tail = divmod(end, self.block_size)
        if not tail:
            return request.block_hashes[page - 1]
        start = page * self.block_size
        extra, _ = generate_block_hash_extra_keys(request, start, end, 0)
        return hash_block_tokens(
            self.hash_fn,
            request.block_hashes[page - 1] if page else None,
            request.all_token_ids[start:end],
            extra,
        )

    def _publish(self, request: Request, end: int, block: KVCacheBlock) -> None:
        self.pool.cache_block_alias(request, block, self._hash_at(request, end), end)

    def cache_tail(
        self, request: Request, end: int, blocks: list[KVCacheBlock]
    ) -> None:
        if end % self.block_size and not (
            request.mm_features or request.prompt_embeds is not None
        ):
            self._publish(request, end, blocks[end // self.block_size])

    def extend_hit(
        self, request: Request, blocks: list[KVCacheBlock], hit: int, limit: int
    ) -> int:
        """Match an edited partial page when resending or extending its prompt."""
        for end in range(min(hit + self.block_size - 1, limit), hit, -1):
            block_hash = self._hash_at(request, end)
            cached = self.pool.get_cached_block(block_hash, [0])
            if cached:
                blocks.append(cached[0])
                if (
                    self.pool.enable_kv_cache_events
                    and request.kv_cache_report_mode == "full"
                ):
                    self.pool.emit_block_alias_event(request, block_hash, end)
                return end
        return hit

    def _request(self, ids: list[int], salt: str | None) -> Request:
        return Request(
            request_id="kv-cache-edit",
            prompt_token_ids=ids,
            sampling_params=SamplingParams(max_tokens=1),
            pooling_params=None,
            cache_salt=salt,
            block_hasher=self.block_hasher,
        )

    def edit(
        self,
        original_ids: list[int],
        replacement_ids: list[int],
        start: int,
        end: int,
        cache_salt: str | None,
        write: Callable[[KVCacheEdit], object],
    ) -> str:
        if not 0 <= start <= end <= len(original_ids):
            raise ValueError("Expected 0 <= start <= end <= len(original_token_ids).")
        original = self._request(original_ids, cache_salt)
        salt = self.salt_prefix + uuid4().hex
        edited = self._request(
            original_ids[:start] + replacement_ids + original_ids[end:], salt
        )
        size = self.block_size
        source = []
        for pos in range(0, len(original_ids), size):
            boundary = min(pos + size, len(original_ids))
            blocks = self.pool.get_cached_block(self._hash_at(original, boundary), [0])
            if blocks is None:
                raise ValueError(
                    f"Original KV is not cached at tokens [{pos}, {boundary})."
                )
            source.append(blocks[0])

        num_pages = cdiv(edited.num_tokens, size)
        # A length-preserving edit can also share pages beyond its last write.
        changed_end = (
            end if edited.num_tokens == len(original_ids) else edited.num_tokens
        )
        if changed_end == start:
            first = last = num_pages
        else:
            first, last = start // size, cdiv(changed_end, size)
        num_scratch = cdiv(len(replacement_ids), size)
        self.pool.touch(source)
        allocated: list[KVCacheBlock] = []
        try:
            allocated = self.pool.get_new_blocks(num_scratch + last - first)
            destination = (
                source[:first] + allocated[num_scratch:] + source[last:num_pages]
            )
            if first < last:
                write(
                    KVCacheEdit(
                        block_size=size,
                        source=[b.block_id for b in source],
                        scratch=[b.block_id for b in allocated[:num_scratch]],
                        destination=[b.block_id for b in destination],
                        copy_end=min(last * size, edited.num_tokens),
                        replacement_ids=replacement_ids,
                        start=start,
                        end=end,
                    )
                )
            for i, block in enumerate(destination):
                self._publish(edited, min((i + 1) * size, edited.num_tokens), block)
            # The sampler must run the last prompt token to obtain its logits.
            resend_end = edited.num_tokens - 1
            if resend_end > 0 and resend_end % size:
                self._publish(edited, resend_end, destination[resend_end // size])
        finally:
            self.pool.free_blocks(reversed(allocated))
            self.pool.free_blocks(reversed(source))
        return salt

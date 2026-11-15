# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Time one layer of Qwen2.5-1.5B query-cache writes on the current GPU."""

from statistics import median

import torch
from flashinfer.testing import bench_gpu_time_with_cupti

from vllm.v1.attention.ops.query_cache import QueryCache
from vllm.v1.kv_cache_interface import FullAttentionSpec


def main() -> None:
    torch.manual_seed(7)
    print(torch.cuda.get_device_name())
    print("tokens  format  median_us")
    for tokens in (1, 32, 512):
        query = torch.randn(tokens, 12, 128, device="cuda", dtype=torch.float16)
        slots = torch.randperm(1024, device=query.device)[:tokens]
        for fmt, bound in (
            ("fp16", 0),
            ("fp8", 0.04),
            ("int8", 0.004),
            ("int4", 0.072),
        ):
            cache = QueryCache(fmt, 12, 128)
            spec = FullAttentionSpec(
                block_size=16,
                num_kv_heads=2,
                head_size=128,
                dtype=query.dtype,
                query_bytes_per_token=cache.size_bytes,
            )
            pages = torch.zeros(
                64, 16, spec.num_heads, 256, device="cuda", dtype=query.dtype
            )
            cache.bind(pages.transpose(1, 2), spec.num_kv_heads)
            cache.write(query, slots)
            error = (cache.read(slots).float() - query.float()).abs()
            assert (error <= query.float().abs().amax(-1, True) * bound).all()
            elapsed_ms = bench_gpu_time_with_cupti(
                cache.write,
                input_args=(query, slots),
                use_cuda_graph=True,
                dry_run_iters=8,
                repeat_iters=40,
            )
            print(f"{tokens:6}  {fmt:6}  {median(elapsed_ms) * 1000:9.2f}", flush=True)


if __name__ == "__main__":
    main()

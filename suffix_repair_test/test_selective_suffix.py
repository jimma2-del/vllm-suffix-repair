# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from suffix_repair_test.selective_reference import (
    FORMATS,
    REDUCTIONS,
    dots,
    midpoint,
    reduce_scores,
    represent,
)
from vllm.v1.attention.ops.query_cache import QueryCache
from vllm.v1.core.kv_cache_edit import KVCacheEdit
from vllm.v1.engine import SuffixSelection
from vllm.v1.worker.kv_cache_edit import _selected_runs


def fixture(
    device, fmt, kv_heads, removed, inserted, suffix, rank=None, replicated=True
):
    heads, dim, block = 4, 7, 8
    start = 3
    pages = (start + max(removed, inserted) + suffix + block - 1) // block
    # Nonadjacent manager blocks, each covering two kernel blocks.
    source = torch.arange(2 * pages - 2, -1, -2, device=device)
    dest = torch.arange(1, 2 * pages, 2, device=device)
    kernel = block // 2
    shape = (4 * pages, kernel)
    seed = 178 if replicated else 178 + rank
    generator = torch.Generator(device=device).manual_seed(seed)
    edit = KVCacheEdit(
        block,
        source.tolist(),
        dest.tolist(),
        start + inserted + suffix,
        [5] * inserted,
        [6] * suffix,
        start,
        start + removed,
    )
    layers, reference = {}, []

    def slots(table, begin, count):
        return [
            int(table[p // block]) * block + p % block
            for p in range(begin, begin + count)
        ]

    for i in range(2):
        qcache = QueryCache(fmt, heads, dim)
        width = 2 * dim * 4
        planes = kv_heads + (qcache.size_bytes + width - 1) // width
        storage = torch.zeros(shape[0], planes, kernel, 2 * dim, device=device)
        kv = qcache.bind(storage, kv_heads).transpose(1, 2)
        keys = torch.randn(*shape, kv_heads, dim, generator=generator, device=device)
        keys[::5, 0] = 0
        kv[..., :dim] = keys
        q = torch.randn(*shape, heads, dim, generator=generator, device=device)
        if rank is not None and replicated:
            # Replicated KV with distinct query heads on the actual Gloo ranks.
            q = q * rank
        q.reshape(-1, heads, dim)[::19] = 0
        q = q.reshape(-1, heads, dim)
        qcache.write(q, torch.arange(len(q), device=device))
        layers[str(i)] = (kv, None, qcache)
        keys = keys.reshape(-1, kv_heads, dim)
        parts = []
        for table, length in ((source.tolist(), removed), (dest.tolist(), inserted)):
            sq = slots(table, start + length, suffix)
            sk = slots(table, start, length)
            parts.append(dots(represent(q[sq], fmt), keys[sk], fmt))
        reference.append(torch.cat(parts, dim=-1))
    return edit, layers, source, dest, reference


def check_mask(fix, ops, threshold, expected):
    edit, layers, src, dst, _ = fix
    edit.selection = SuffixSelection(
        key_reduction=ops[0],
        head_reduction=ops[1],
        layer_reduction=ops[2],
        threshold=threshold,
        include_deleted=True,
    )
    got = _selected_runs(edit, layers, src, dst)
    selected = [p for a, b in got for p in range(a, b)]
    assert selected == expected.nonzero().flatten().tolist(), (ops, threshold, got)


def tp_rank(rank, rendezvous, backend):
    device = "cpu"
    if backend == "nccl":
        torch.cuda.set_device(rank)
        device = f"cuda:{rank}"
    dist.init_process_group(backend, init_method=rendezvous, rank=rank, world_size=2)
    try:
        group = SimpleNamespace(world_size=2, device_group=dist.group.WORLD)
        with patch("vllm.v1.worker.kv_cache_edit.get_tp_group", return_value=group):
            for replicated in (True, False):
                # Build the global oracle independently of the score collective.
                shards = [
                    fixture(
                        device, "int4", 1 if replicated else 2, 3, 4, 21, r, replicated
                    )
                    for r in range(2)
                ]
                refs = [
                    torch.cat(pair, dim=1) for pair in zip(shards[0][-1], shards[1][-1])
                ]
                for ops in REDUCTIONS:
                    scores = reduce_scores(refs, ops)
                    threshold = midpoint(scores)
                    check_mask(shards[rank], ops, threshold, scores > threshold)
                if replicated:
                    scores = reduce_scores(refs, ("max",) * 3)
                    threshold = midpoint(scores)
                    assert threshold > 0 and (scores > threshold).any()
                    assert not (
                        reduce_scores(shards[0][-1], ("max",) * 3) > threshold
                    ).any()
    finally:
        dist.destroy_process_group()


class SelectiveScoringTest(unittest.TestCase):
    def matrix(self, device):
        group = SimpleNamespace(world_size=1)
        with patch("vllm.v1.worker.kv_cache_edit.get_tp_group", return_value=group):
            for fmt in FORMATS:
                for kv_heads in (1, 2, 4):
                    with self.subTest(device=device, fmt=fmt, kv_heads=kv_heads):
                        fix = fixture(device, fmt, kv_heads, 3, 5, 21)
                        for ops in REDUCTIONS:
                            scores = reduce_scores(fix[-1], ops)
                            threshold = midpoint(scores)
                            check_mask(fix, ops, threshold, scores > threshold)
            fix = fixture(device, "int8", 2, 129, 130, 257)
            for ops in REDUCTIONS:
                scores = reduce_scores(fix[-1], ops)
                threshold = midpoint(scores)
                check_mask(fix, ops, threshold, scores > threshold)

    def test_cpu(self):
        self.matrix("cpu")

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_cuda(self):
        self.matrix("cuda")

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_key_quantization_matches_query_writer(self):
        generator = torch.Generator(device="cuda").manual_seed(117)
        for fmt in FORMATS:
            cache = QueryCache(fmt, 2, 15)
            cache.bind(torch.empty(64, 3, 4, 64, device="cuda", dtype=torch.float16), 1)
            slots = torch.arange(256, device="cuda")
            for dtype in (torch.float16, torch.bfloat16, torch.float32):
                with self.subTest(fmt=fmt, dtype=dtype):
                    keys = torch.randn(
                        256, 2, 15, device="cuda", generator=generator
                    ).to(dtype)
                    keys[0] = 0
                    cache.write(keys, slots)
                    torch.testing.assert_close(
                        cache.roundtrip_keys(keys), cache.read(slots), atol=0, rtol=0
                    )
                    torch.testing.assert_close(
                        cache.read(slots).float(), represent(keys, fmt), atol=0, rtol=0
                    )

    def test_negative_and_ties(self):
        fix = fixture("cpu", "fp32", 2, 2, 1, 3)
        for cache, _, q in fix[1].values():
            cache[..., :7].fill_(-1)
            rows = torch.ones(q.data.shape[0] * q.data.shape[2], 4, 7)
            q.write(rows, torch.arange(len(rows)))
        with patch(
            "vllm.v1.worker.kv_cache_edit.get_tp_group",
            return_value=SimpleNamespace(world_size=1),
        ):
            for ops in REDUCTIONS:
                score = -7 * (3 if ops[0] == "sum" else 1)
                score *= (4 if ops[1] == "sum" else 1) * (2 if ops[2] == "sum" else 1)
                check_mask(fix, ops, score, torch.zeros(3, dtype=torch.bool))
                check_mask(fix, ops, score - 1, torch.ones(3, dtype=torch.bool))

    def collective(self, backend):
        with tempfile.TemporaryDirectory() as folder:
            mp.spawn(tp_rank, (Path(folder, "store").as_uri(), backend), nprocs=2)

    def test_two_rank_collectives(self):
        self.collective("gloo")

    @unittest.skipUnless(
        torch.cuda.device_count() >= 2 and dist.is_nccl_available(),
        "two CUDA devices and NCCL required",
    )
    def test_two_gpu_collectives(self):
        self.collective("nccl")

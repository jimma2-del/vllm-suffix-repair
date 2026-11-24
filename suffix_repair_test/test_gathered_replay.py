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

from vllm.config import CUDAGraphMode
from vllm.forward_context import get_forward_context
from vllm.v1.attention.backends.triton_attn import TritonAttentionBackend
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
)
from vllm.v1.worker.kv_cache_edit import _prefill
from vllm.v1.worker.utils import AttentionGroup


def make_runner(model, device="cpu", heads=4, kv_heads=2, max_tokens=6, max_seqs=2):
    spec = FullAttentionSpec(
        block_size=16, num_kv_heads=kv_heads, head_size=64, dtype=torch.float16
    )
    config = SimpleNamespace(
        scheduler_config=SimpleNamespace(
            max_num_batched_tokens=max_tokens, max_num_seqs=max_seqs
        ),
        model_config=SimpleNamespace(
            get_num_attention_heads=lambda _: heads,
            get_num_kv_heads=lambda _: kv_heads,
            get_head_size=lambda: spec.head_size,
            rswa_window=None,
        ),
        parallel_config=SimpleNamespace(
            data_parallel_size=1,
            tensor_parallel_size=1,
            use_sequence_parallel_moe=False,
        ),
        compilation_config=SimpleNamespace(
            cudagraph_mode=CUDAGraphMode.NONE,
            static_forward_context={},
            fast_moe_cold_start=False,
        ),
        speculative_config=None,
    )
    group = AttentionGroup(TritonAttentionBackend, ["attn"], spec, 0)
    group.create_metadata_builders(config, torch.device(device))
    return SimpleNamespace(
        get_model=lambda: model,
        device=torch.device(device),
        attn_groups=[[group]],
        vllm_config=config,
        kv_cache_config=KVCacheConfig(8, [], [KVCacheGroupSpec(["attn"], spec)]),
    )


def check_tp_batches(rank, rendezvous):
    calls = []
    runner = make_runner(lambda input_ids, positions: calls.append(positions.tolist()))
    runner.vllm_config.parallel_config.tensor_parallel_size = 2
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=2)
    try:
        with patch(
            "vllm.v1.worker.kv_cache_edit.get_tp_group",
            return_value=SimpleNamespace(cpu_group=dist.group.WORLD),
        ):
            for mixed in (False, True):
                backend = "other" if mixed and rank == 1 else "TRITON_ATTN"
                with patch.object(
                    TritonAttentionBackend, "get_name", return_value=backend
                ):
                    calls.clear()
                    _prefill(
                        runner,
                        list(range(10)),
                        0,
                        torch.tensor([2]),
                        16,
                        runs=[(1, 3), (8, 9)],
                    )
                    assert calls == ([[1, 2], [8]] if mixed else [[1, 2, 8]])
    finally:
        dist.destroy_process_group()


class GatheredReplayTest(unittest.TestCase):
    def check_batches(self, runs, expected):
        # Two 16-token kernel blocks per manager page, physically out of order.
        pages = torch.tensor([3, 1, 2])
        start = 14
        ids = list(range(100, 168))
        calls = []

        def model(input_ids, positions):
            context = get_forward_context()
            meta = context.attn_metadata["attn"]
            calls.append(positions.tolist())
            self.assertEqual(input_ids.tolist(), [ids[p - start] for p in calls[-1]])
            slots = pages[positions // 32] * 32 + positions % 32
            torch.testing.assert_close(context.slot_mapping["attn"], slots)
            torch.testing.assert_close(meta.slot_mapping, slots)
            cu = meta.query_start_loc.tolist()
            self.assertEqual(cu[0], 0)
            self.assertEqual(cu[-1], meta.num_actual_tokens)
            lengths = [b - a for a, b in zip(cu, cu[1:])]
            self.assertEqual(meta.max_query_len, max(lengths))
            self.assertEqual(meta.max_seq_len, max(meta.seq_lens.tolist()))
            if len(lengths) > 1:
                self.assertEqual(meta.block_table.stride(0), 0)
            for i, end in enumerate(meta.seq_lens.tolist()):
                # Reconstruct every visible key using only kernel metadata.
                self.assertEqual(
                    calls[-1][cu[i] : cu[i + 1]], list(range(end - lengths[i], end))
                )
                keys = torch.arange(end)
                read_slots = meta.block_table[i, keys // 16].long() * 16 + keys % 16
                torch.testing.assert_close(
                    read_slots, pages[keys // 32] * 32 + keys % 32
                )

        runner = make_runner(model)
        _prefill(runner, ids, start, pages, 32, runs=runs)
        self.assertEqual(calls, expected)

    def test_fragmented(self):
        self.check_batches(
            [(1, 4), (11, 12), (17, 28), (50, 51), (66, 68)],
            [[15, 16, 17, 25], list(range(31, 37)), [37, 38, 39, 40, 41, 64], [80, 81]],
        )

    def test_empty_singleton_and_full(self):
        self.check_batches([], [])
        self.check_batches([(66, 67)], [[80]])
        chunks = [list(range(p, min(p + 6, 82))) for p in range(14, 82, 6)]
        self.check_batches([(0, 68)], chunks)
        self.check_batches(None, chunks)

    def test_two_layer_cache_matches_serial_replay(self):
        pages = torch.tensor([3, 0, 2])
        block, start = 16, 13
        ids = list(range(70, 103))
        positions = torch.arange(start + len(ids))
        slots = pages[positions // block] * block + positions % block
        initial = torch.randn(
            2,
            4 * block,
            dtype=torch.float64,
            generator=torch.Generator().manual_seed(8),
        )
        actual, expected = initial.clone(), initial.clone()
        runs = [(1, 4), (8, 9), (17, 20), (31, 33)]

        # Uniform attention makes each layer's output the mean of its visible values.
        for a, b in runs:
            for offset in range(a, b):
                p = start + offset
                hidden = ids[offset]
                for layer in expected:
                    layer[slots[p]] = hidden
                    hidden = hidden + layer[slots[: p + 1]].mean()

        def model(input_ids, positions):
            ctx = get_forward_context()
            meta = ctx.attn_metadata["attn"]
            lengths = meta.query_start_loc.diff().tolist()
            hidden = input_ids.double()
            for layer in actual:
                layer[ctx.slot_mapping["attn"]] = hidden
                outputs = []
                for view, end in enumerate(meta.seq_lens.tolist()):
                    keys = torch.arange(end)
                    read = meta.block_table[view, keys // block] * block + keys % block
                    means = layer[read].cumsum(0) / (keys + 1)
                    outputs.append(means[-lengths[view] :])
                hidden = hidden + torch.cat(outputs)

        _prefill(make_runner(model), ids, start, pages, block, runs=runs)
        torch.testing.assert_close(actual, expected, rtol=0, atol=1e-12)
        skipped = torch.ones(initial.shape[1], dtype=torch.bool)
        for a, b in runs:
            skipped[slots[start + a : start + b]] = False
        torch.testing.assert_close(
            actual[:, skipped], initial[:, skipped], atol=0, rtol=0
        )

    def test_other_backend_preserves_runs(self):
        with patch.object(TritonAttentionBackend, "get_name", return_value="other"):
            self.check_batches(
                [(1, 4), (11, 12), (66, 68)], [[15, 16, 17], [25], [80, 81]]
            )

    def test_two_ranks_agree_on_batching(self):
        with tempfile.TemporaryDirectory() as folder:
            mp.spawn(check_tp_batches, (Path(folder, "store").as_uri(),), nprocs=2)

    def check_triton_attention(self, heads, kv_heads):
        from vllm.v1.attention.backends.triton_attn import TritonAttentionImpl

        device = "cuda"
        generator = torch.Generator(device=device).manual_seed(52)
        pages = torch.tensor([2, 5, 0, 3, 4, 1], device=device)
        block_size, start, dim = 16, 15, 64
        ids = list(range(68))
        logical = torch.arange(start + len(ids), device=device)
        physical = pages[logical // block_size]

        def vectors(nheads):
            return torch.randn(
                len(logical),
                nheads,
                dim,
                generator=generator,
                dtype=torch.float16,
                device=device,
            )

        q, k, v = vectors(heads), vectors(kv_heads), vectors(kv_heads)
        dense_kv = torch.cat((k, v), -1)
        cache = k.new_full((len(pages), kv_heads, block_size, 2 * dim), torch.nan)
        rows = cache.transpose(1, 2)
        rows[physical, logical % block_size] = dense_kv
        full_k = k.float().repeat_interleave(heads // kv_heads, dim=1)
        full_v = v.float().repeat_interleave(heads // kv_heads, dim=1)
        impl = TritonAttentionImpl(heads, dim, dim**-0.5, kv_heads, None, None, "auto")
        scale = torch.ones((), device=device)
        layer = SimpleNamespace(_k_scale=scale, _v_scale=scale)

        def model(input_ids, positions):
            ctx = get_forward_context()
            query, key, value = q[positions], k[positions], v[positions]
            impl.do_kv_cache_update(layer, key, value, cache, ctx.slot_mapping["attn"])
            output = impl.forward(
                layer,
                query,
                key,
                value,
                cache,
                ctx.attn_metadata["attn"],
                output=torch.empty_like(query),
            )
            for row, token in enumerate(input_ids.tolist()):
                end = start + token + 1
                dots = torch.einsum("hd,thd->ht", q[end - 1].float(), full_k[:end])
                weights = (dots * dim**-0.5).softmax(-1)
                expected = torch.einsum("ht,thd->hd", weights, full_v[:end])
                torch.testing.assert_close(
                    output[row].float(), expected, atol=2e-3, rtol=2e-3
                )

        runner = make_runner(model, device, heads, kv_heads, max_tokens=32, max_seqs=4)
        # Four singleton queries fit split-K scratch. The longer batch uses 2D.
        for runs in (
            [(0, 1), (4, 5), (34, 35), (64, 65)],
            [(0, 4), (9, 10), (25, 36), (63, 68)],
        ):
            selected = torch.tensor(
                [start + i for a, b in runs for i in range(a, b)], device=device
            )
            rows[physical[selected], selected % block_size] = torch.nan
            _prefill(runner, ids, start, pages, block_size, runs=runs)
            torch.testing.assert_close(
                rows[physical, logical % block_size], dense_kv, atol=0, rtol=0
            )

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_triton_attention_against_dense_prefixes(self):
        for heads, kv_heads in ((4, 4), (6, 2), (4, 1)):
            with self.subTest(heads=heads, kv_heads=kv_heads):
                self.check_triton_attention(heads, kv_heads)

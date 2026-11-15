# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import unittest
from functools import partial
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

import torch

from vllm.config import KVTransferConfig
from vllm.distributed.kv_transfer.kv_connector.utils import copy_kv_blocks
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorRole
from vllm.distributed.kv_transfer.kv_connector.v1.example_connector import (
    ExampleConnector,
    ExampleConnectorMetadata,
)
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.v1.attention.backends.triton_attn import TritonAttentionBackend
from vllm.v1.attention.ops.query_cache import (
    QueryCache,
    get_cache_pages,
    get_query_cache_sizes,
)
from vllm.v1.core.kv_cache_utils import KVCacheBlockCopy
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    KVCacheLayout,
    KVCacheTensor,
    compute_layout_strides,
    create_kv_cache_views,
)
from vllm.v1.worker.utils import (
    AttentionGroup,
    KVBlockZeroer,
    bind_kv_cache,
    clear_layer_kv_caches,
    copy_kv_cache_blocks_inplace,
)


class QueryCacheTest(unittest.TestCase):
    def check_formats(self, device):
        heads, dim = 7, 15
        slots = torch.tensor([13, -1, 1, 9], device=device)
        # Noncontiguous Q, an all-zero head, negative values and a padded row.
        q = torch.linspace(-3, 6, 5 * heads * 2 * dim, device=device)
        q = q.reshape(5, heads, 2 * dim)[..., 1::2]
        q[0, 0] = 0
        for fmt, row_bytes in (
            ("fp16", 30),
            ("bf16", 30),
            ("fp32", 60),
            ("fp8", 19),
            ("int8", 19),
            ("int4", 12),
        ):
            cache = QueryCache(fmt, heads, dim)
            self.assertEqual(cache.row_bytes, row_bytes)
            spec = FullAttentionSpec(
                block_size=4,
                num_kv_heads=1,
                head_size=dim,
                dtype=torch.float16,
                query_bytes_per_token=cache.size_bytes,
            )
            for layout in KVCacheLayout:
                with self.subTest(dtype=fmt, layout=layout, device=device):
                    ls, bs, *_ = compute_layout_strides(spec, 4, 1, layout)
                    size = 4 * spec.page_size_bytes
                    backing = torch.full((size,), 51, dtype=torch.int8, device=device)
                    tensor = KVCacheTensor(size, ["layer.0"], ls, bs)
                    (pages,) = create_kv_cache_views(backing, spec, 4, layout, tensor)
                    kv = cache.bind(pages, 1)
                    before = pages.view(torch.uint8).clone()
                    cache.write(q, slots)
                    got = cache.read(slots[slots >= 0])
                    ref = q[[0, 2, 3]]
                    if fmt in ("fp16", "bf16", "fp32"):
                        torch.testing.assert_close(
                            got, ref.to(got.dtype), rtol=0, atol=0
                        )
                    else:
                        bound = {"fp8": 0.04, "int8": 0.004, "int4": 0.072}[fmt]
                        self.assertTrue(
                            (
                                (got - ref).abs() <= ref.abs().amax(-1, True) * bound
                            ).all()
                        )
                    torch.testing.assert_close(kv.view(torch.uint8), before[:, :1])
                    # Every byte of each unused token (including the last slot)
                    # must survive padding and the negative slot sentinel.
                    unused = torch.tensor(
                        [i for i in range(16) if i not in (13, 1, 9)], device=device
                    )
                    torch.testing.assert_close(
                        pages.view(torch.uint8)[unused // 4, :, unused % 4],
                        before[unused // 4, :, unused % 4],
                    )

    def test_formats_and_layouts_cpu(self):
        self.check_formats("cpu")

    def test_singleton_and_empty_slots(self):
        for fmt in ("fp8", "int8", "int4"):
            cache = QueryCache(fmt, 1, 5)
            pages = torch.full((2, 2, 4, 12), 123, dtype=torch.uint8)
            cache.bind(pages, 1)
            cache.write(torch.ones(1, 1, 5), torch.tensor([-1]))
            self.assertTrue((pages == 123).all())
            cache.write(torch.ones(1, 1, 5), torch.tensor([7]))
            torch.testing.assert_close(
                cache.read(torch.tensor([7])), torch.ones(1, 1, 5)
            )
            self.assertEqual(
                cache.read(torch.empty(0, dtype=torch.long)).shape, (0, 1, 5)
            )

    def test_context_parallel_query_heads(self):
        cache = QueryCache("fp16", 2, 8, dcp_size=2)
        cache.bind(torch.zeros(2, 3, 4, 16, dtype=torch.float16), 1)
        local = torch.arange(48, dtype=torch.float16).reshape(3, 2, 8)
        gathered = torch.cat((local, -local), dim=1)
        with patch("vllm.v1.attention.ops.query_cache.get_dcp_group") as group:
            group.return_value.all_gather.return_value = gathered
            cache.record(local, torch.tensor([6, -1, 4]))
            gather = group.return_value.all_gather
            gather.assert_called_once()
            torch.testing.assert_close(gather.call_args.args[0], local)
            self.assertEqual(gather.call_args.kwargs, {"dim": 1})
        torch.testing.assert_close(cache.read(torch.tensor([6, 4])), gathered[[0, 2]])

    def test_file_transfer_to_different_block(self):
        q = QueryCache("int4", 3, 8)
        spec = FullAttentionSpec(
            block_size=4,
            num_kv_heads=1,
            head_size=8,
            dtype=torch.float16,
            query_bytes_per_token=q.size_bytes,
        )
        name = "layer.0"
        config = KVCacheConfig(3, [], [KVCacheGroupSpec([name], spec)])
        pages = torch.zeros(3, spec.num_heads, 4, 16, dtype=torch.float16)
        layer = SimpleNamespace(query_cache=q)
        layer.bind_kv_cache = partial(AttentionLayerBase.bind_kv_cache, layer)
        runner_pages = []
        bind_kv_cache(
            {name: pages},
            {name: layer},
            runner_pages,
            kv_cache_groups=config.kv_cache_groups,
        )
        self.assertIs(runner_pages[0], pages)
        self.assertEqual(layer.kv_cache.shape[1], spec.num_kv_heads)
        layer.kv_cache[1].fill_(13)
        q.write(torch.arange(96).float().reshape(4, 3, 8), torch.arange(4, 8))
        expected = pages[1].view(torch.uint8).clone()
        with TemporaryDirectory() as directory:
            vllm_config = SimpleNamespace(
                cache_config=SimpleNamespace(block_size=4, q_cache_dtype="int4"),
                parallel_config=SimpleNamespace(
                    tensor_parallel_size=1, decode_context_parallel_size=1, rank=0
                ),
                kv_transfer_config=KVTransferConfig(
                    kv_connector="ExampleConnector",
                    kv_role="kv_both",
                    kv_connector_extra_config={"shared_storage_path": directory},
                ),
            )
            connector = ExampleConnector(vllm_config, KVConnectorRole.WORKER, config)
            meta = ExampleConnectorMetadata()
            meta.add_request(list(range(5)), [1], 4, True, [])
            connector.bind_connector_metadata(meta)
            connector.save_kv_layer(name, get_cache_pages(layer), SimpleNamespace())
            pages.zero_()
            load_meta = ExampleConnectorMetadata()
            load_meta.add_request(list(range(5)), [2], 4, False, [])
            connector.bind_connector_metadata(load_meta)
            connector.start_load_kv(
                SimpleNamespace(
                    no_compile_layers={name: layer},
                    attn_metadata={name: SimpleNamespace()},
                )
            )
        torch.testing.assert_close(pages[2].view(torch.uint8), expected)
        self.assertFalse(pages[1].any())
        clear_layer_kv_caches([layer])
        self.assertIsNone(q.pages)
        self.assertIsNone(q.data)
        self.assertEqual(layer.kv_cache.numel(), 0)

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
    def test_formats_and_layouts_cuda(self):
        self.check_formats("cuda")

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
    def test_e4m3_rounding_cuda(self):
        values = torch.arange(127, dtype=torch.uint8).view(torch.float8_e4m3fn).float()
        ties = (values[:-1] + values[1:]) / 2
        values = torch.cat((values, ties, -values, -ties)).view(1, 1, -1)
        cache = QueryCache("fp8", 1, values.numel())
        pages = torch.zeros(1, 3, 1, 256, dtype=torch.uint8, device="cuda")
        cache.bind(pages, 1)
        cache.write(values.cuda(), torch.tensor([0], device="cuda"))
        encoded = cache.data.flatten().cpu()[4 : cache.row_bytes]
        torch.testing.assert_close(
            encoded, values.flatten().to(torch.float8_e4m3fn).view(torch.uint8)
        )

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
    def test_host_swap_preserves_quantized_pages(self):
        q = QueryCache("int4", 12, 128)
        pages = torch.zeros(6, 4, 16, 256, dtype=torch.float16, device="cuda")
        kv = q.bind(pages, 2)
        slots = torch.arange(16, 32, device="cuda")
        kv[1].fill_(11)
        q.write(torch.randn(16, 12, 128, device="cuda"), slots)
        expected = q.read(slots)
        host = {"layer.0": torch.empty_like(pages, device="cpu", pin_memory=True)}
        gpu = {"layer.0": pages}
        copy_kv_blocks(gpu, host, [1], [0], "d2h")
        copy_kv_blocks(host, gpu, [0], [4], "h2d")
        torch.testing.assert_close(q.read(slots + 48), expected, atol=0, rtol=0)
        torch.testing.assert_close(kv[4], kv[1], atol=0, rtol=0)
        spec = FullAttentionSpec(
            block_size=16,
            num_kv_heads=2,
            head_size=128,
            dtype=torch.float16,
            query_bytes_per_token=q.size_bytes,
        )
        context = {"layer.0": SimpleNamespace(kv_cache=kv, query_cache=q)}
        group = AttentionGroup(TritonAttentionBackend, ["layer.0"], spec, 0)
        zeroer = KVBlockZeroer(pages.device, [group], [16], context, len(pages))
        zeroer.zero_block_ids([1])
        self.assertFalse(pages[1].view(torch.uint8).any())
        torch.testing.assert_close(q.read(slots + 48), expected, atol=0, rtol=0)

    def test_shared_queries_and_copy_on_write(self):
        a = QueryCache("int4", 3, 7)
        b = QueryCache("fp16", 2, 7)
        owners = {
            name: SimpleNamespace(query_cache=cache, kv_sharing_target_layer_name=owner)
            for name, cache, owner in [("a", a, None), ("b", b, "a")]
        }
        sizes = get_query_cache_sizes(owners)
        self.assertEqual(sizes, {"a": a.size_bytes + b.size_bytes})
        spec = FullAttentionSpec(
            block_size=8,
            num_kv_heads=1,
            head_size=7,
            dtype=torch.float16,
            query_bytes_per_token=sizes["a"],
        )
        self.assertEqual(FullAttentionSpec.merge([spec, spec]), spec)
        # Two kernel pages per manager block. The first record for b starts
        # inside the same physical head slot as the last record for a.
        pages = torch.zeros(8, spec.num_heads, 4, 14, dtype=torch.float16)
        a.bind(pages, 1)
        b.bind(pages, 1)
        slots = torch.arange(8, 16)
        a.write(torch.ones(8, 3, 7), slots)
        b.write(torch.full((8, 2, 7), 9.0), slots)
        copy_kv_cache_blocks_inplace([pages, pages], 4, [KVCacheBlockCopy(1, 2)])
        torch.testing.assert_close(a.read(slots + 8), torch.ones(8, 3, 7))
        torch.testing.assert_close(b.read(slots + 8), torch.full((8, 2, 7), 9.0).half())
        a.write(torch.zeros(8, 3, 7), slots + 8)
        torch.testing.assert_close(a.read(slots), torch.ones(8, 3, 7))
        torch.testing.assert_close(b.read(slots + 8), b.read(slots))

    def test_packed_nvfp4_kv_planes(self):
        cache = QueryCache("fp16", 4, 7)
        pages = torch.zeros(3, 16, 6, 18, dtype=torch.uint8).transpose(1, 2)
        kv = cache.bind(pages, 2, packed_nvfp4=True)
        flat = pages.transpose(1, 2).flatten(1)
        flat[:, : 16 * 2 * 18].fill_(111)
        q = torch.randn(48, 4, 7, dtype=torch.float16)
        slots = torch.arange(48)
        cache.write(q, slots)
        torch.testing.assert_close(cache.read(slots), q, atol=0, rtol=0)
        self.assertTrue((kv == 111).all())
        self.assertEqual(kv.stride()[1:], (18, 36, 1))


if __name__ == "__main__":
    unittest.main()

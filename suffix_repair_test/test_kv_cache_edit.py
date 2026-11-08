# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import unittest
from types import SimpleNamespace
from unittest.mock import Mock

import torch

from vllm.utils.hashing import get_hash_fn_by_name
from vllm.utils.math_utils import cdiv
from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.kv_cache_edit import KVCacheEditor
from vllm.v1.core.kv_cache_utils import init_none_hash
from vllm.v1.worker.kv_cache_edit import _copy_tokens


class KVCacheEditTest(unittest.TestCase):
    def setUp(self):
        self.size = 4
        self.pool = BlockPool(24, True, self.size)
        hash_fn = get_hash_fn_by_name("sha256")
        init_none_hash(hash_fn)
        self.editor = KVCacheEditor(self.pool, hash_fn)
        self.ids = list(range(14))
        self.request = self.editor._request(self.ids, "source")
        self.pages = self.pool.get_new_blocks(cdiv(len(self.ids), self.size))
        self.pool.cache_full_blocks(
            self.request, self.pages, 0, len(self.ids) // self.size, self.size, 0
        )
        self.editor.cache_tail(self.request, len(self.ids), self.pages)

    def lookup(self, ids, salt):
        req = self.editor._request(ids, salt)
        pages = []
        for i in range(0, len(ids), self.size):
            end = min(len(ids), i + self.size)
            hit = self.pool.get_cached_block(self.editor._hash_at(req, end), [0])
            self.assertIsNotNone(hit)
            pages.append(hit[0].block_id)
        return pages

    def test_equal_length_shares_only_untouched_pages(self):
        write = Mock()
        # Even the same token IDs can have different KV after isolated prefill.
        salt = self.editor.edit(self.ids, self.ids[5:7], 5, 7, "source", write)
        plan = write.call_args.args[0]
        self.assertEqual(plan.destination, self.lookup(self.ids, salt))
        self.assertEqual(plan.source, self.lookup(self.ids, "source"))
        self.assertEqual(plan.destination[:1], plan.source[:1])
        self.assertNotEqual(plan.destination[1], plan.source[1])
        self.assertEqual(plan.destination[2:], plan.source[2:])
        self.assertEqual(plan.copy_end, 8)
        self.assertEqual([b.ref_cnt for b in self.pages], [1] * len(self.pages))

    def test_insert_delete_truncate_and_noop(self):
        for start, end, ids in [
            (0, 0, [91]),
            (len(self.ids), len(self.ids), [92, 93, 94]),
            (3, 11, []),
            (6, len(self.ids), []),
            (0, len(self.ids), []),
            (2, 2, []),
            (0, len(self.ids), [95, 96]),
        ]:
            with self.subTest(start=start, end=end, replacement=ids):
                write = Mock()
                salt = self.editor.edit(self.ids, ids, start, end, "source", write)
                result = self.ids[:start] + ids + self.ids[end:]
                pages = self.lookup(result, salt)
                self.assertEqual(len(pages), cdiv(len(result), self.size))
                if not ids and (end == len(self.ids) or end == start):
                    write.assert_not_called()
                    self.assertEqual(
                        pages, [b.block_id for b in self.pages[: len(pages)]]
                    )
                else:
                    self.assertEqual(write.call_args.args[0].destination, pages)
                self.assertEqual(self.pool.get_num_free_blocks(), 19)

    def test_failure_releases_pages_and_preserves_source(self):
        count = self.pool.get_num_free_blocks()
        entries = len(self.pool.cached_block_hash_to_block)
        with self.assertRaisesRegex(RuntimeError, "worker failed"):
            self.editor.edit(
                self.ids,
                [7],
                0,
                2,
                "source",
                Mock(side_effect=RuntimeError("worker failed")),
            )
        self.assertEqual(entries, len(self.pool.cached_block_hash_to_block))
        with self.assertRaisesRegex(ValueError, "free blocks"):
            self.editor.edit(self.ids, [7] * 80, 0, 1, "source", Mock())
        with self.assertRaisesRegex(ValueError, "not cached"):
            self.editor.edit(self.ids, [], 0, 1, "wrong salt", Mock())
        self.assertEqual(count, self.pool.get_num_free_blocks())
        self.assertEqual(
            self.lookup(self.ids, "source"), [b.block_id for b in self.pages]
        )
        self.assertTrue(all(b.ref_cnt == 1 for b in self.pages))

    def test_partial_alias_survives_promotion_and_dies_on_eviction(self):
        self.request.append_output_token_ids([77, 78])
        self.pool.cache_full_blocks(self.request, self.pages, 3, 4, self.size, 0)
        self.assertEqual(self.lookup(self.ids, "source")[-1], self.pages[-1].block_id)
        self.assertEqual(
            self.lookup(self.ids + [77, 78], "source")[-1], self.pages[-1].block_id
        )
        self.pool.free_blocks(self.pages)
        self.pool.evict_blocks({self.pages[-1].block_id})
        for length in (14, 16):
            self.assertIsNone(
                self.pool.get_cached_block(
                    self.editor._hash_at(self.request, length), [0]
                )
            )

    def test_rope_layouts_partial_rotation_and_noncontiguous_pages(self):
        generator = torch.Generator().manual_seed(7)
        table = torch.arange(16)[:, None] * torch.tensor([[0.3, 0.02]])
        cos, sin = 1.25 * table.cos(), 1.25 * table.sin()
        raw = torch.randn(3, 2, 7, generator=generator)
        for neox in (False, True):
            rope = SimpleNamespace(
                rotary_dim=4,
                head_size=7,
                is_neox_style=neox,
                cos_sin_cache=torch.cat((cos, sin), -1),
            )

            def rotate(positions, neox=neox):
                x, y = (
                    raw[..., :4].chunk(2, -1)
                    if neox
                    else (raw[..., :4:2], raw[..., 1:4:2])
                )
                c, s = cos[positions, None], sin[positions, None]
                pairs = (x * c - y * s, y * c + x * s)
                rot = (
                    torch.cat(pairs, -1) if neox else torch.stack(pairs, -1).flatten(-2)
                )
                return torch.cat((rot, raw[..., 4:]), -1)

            for old, new in ((5, 2), (1, 5)):
                # Two kernel blocks per manager block. Two differently strided heads.
                cache = torch.randn(16, 2, 2, 14, generator=generator).transpose(1, 2)
                src_pages, dst_pages = torch.tensor([2, 5, 7]), torch.tensor([1, 4, 3])
                offsets = torch.arange(3)
                src = src_pages[(old + offsets) // 4] * 4 + (old + offsets) % 4
                dst = dst_pages[(new + offsets) // 4] * 4 + (new + offsets) % 4
                cache[src // 2, src % 2, :, :7] = rotate(old + offsets)
                values = cache[src // 2, src % 2, :, 7:]
                _copy_tokens(cache, rope, 4, src_pages, dst_pages, old, new, len(raw))
                actual = cache[dst // 2, dst % 2]
                torch.testing.assert_close(actual[..., :7], rotate(new + offsets))
                torch.testing.assert_close(actual[..., 7:], values, atol=0, rtol=0)


if __name__ == "__main__":
    unittest.main()

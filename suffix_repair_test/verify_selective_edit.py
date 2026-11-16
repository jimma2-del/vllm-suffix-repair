# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Offline selective edit verification and warm end-to-end latency measurement."""

import argparse
import json
import os
from contextlib import contextmanager
from dataclasses import dataclass
from functools import partial
from statistics import median
from time import perf_counter
from unittest.mock import patch
from uuid import uuid4

import torch

from suffix_repair_test.selective_reference import (
    REDUCTIONS,
    dots,
    midpoint,
    reduce_scores,
)
from vllm import LLM, SamplingParams

MAX = ("max",) * 3


@dataclass
class LayerState:
    kv: torch.Tensor
    q: torch.Tensor
    raw_q: torch.Tensor


class ResidentPrompt:
    def __init__(self, llm):
        self.llm = llm
        core = llm.llm_engine.engine_core.engine_core
        self.worker = core.model_executor.driver_worker
        self.runner = self.worker.model_runner
        manager = core.scheduler.kv_cache_manager
        self.editor = manager.editor
        context = self.worker.vllm_config.compilation_config.static_forward_context
        self.layers = {
            name: context[name]
            for name in manager.kv_cache_config.kv_cache_groups[0].layer_names
        }

    def pages(self, prompt):
        ids = prompt["prompt_token_ids"]
        req = self.editor._request(ids, prompt.get("cache_salt"))
        size = self.editor.block_size
        blocks = []
        for start in range(0, len(ids), size):
            h = self.editor._hash_at(req, min(start + size, len(ids)))
            blocks.append(self.editor.pool.get_cached_block(h, [0])[0])
        return blocks

    @contextmanager
    def pinned(self, prompt):
        pages = self.pages(prompt)
        self.editor.pool.touch(pages)
        try:
            yield
        finally:
            self.editor.pool.free_blocks(pages)

    def snapshot(self, prompt):
        pages = [p.block_id for p in self.pages(prompt)]
        pos = torch.arange(len(prompt["prompt_token_ids"]), device=self.worker.device)
        size = self.editor.block_size
        slots = torch.tensor(pages, device=pos.device)[pos // size] * size + pos % size
        out = {}
        for name, layer in self.layers.items():
            kv = layer.kv_cache.transpose(1, 2)
            q = layer.query_cache
            out[name] = LayerState(
                kv[slots // kv.shape[1], slots % kv.shape[1]].cpu(),
                q.read(slots).cpu(),
                q.data[q._indices(slots)].cpu(),
            )
        return out

    @contextmanager
    def observe(self):
        positions = []

        def record(model, args, kwargs):
            positions.extend(kwargs["positions"].tolist())

        with self.runner.get_model().register_forward_pre_hook(
            record, with_kwargs=True
        ):
            yield positions

    @torch.inference_mode()
    def replay_individually(self, private, original, positions):
        size = self.editor.block_size
        pages = self.pages(private)
        shared = {b.block_id for b in self.pages(original)}
        assert all(pages[p // size].block_id not in shared for p in positions)
        params = SamplingParams(max_tokens=1, temperature=0)
        with self.pinned(private):
            for p in positions:
                # Expose exactly the mixed prefix to the ordinary scheduler.
                ids = private["prompt_token_ids"][: p + 1]
                salt = self.editor.salt_prefix + uuid4().hex
                req = self.editor._request(ids, salt)
                for first in range(0, p, size):
                    self.editor._publish(
                        req, min(first + size, p), pages[first // size]
                    )
                prompt = {"cache_salt": salt, "prompt_token_ids": ids}
                out = self.llm.generate(prompt, params, use_tqdm=False)[0]
                assert out.num_cached_tokens == p, (out.num_cached_tokens, p)
                src = self.pages(prompt)[-1].block_id * size + p % size
                dst = pages[p // size].block_id * size + p % size
                for layer in self.layers.values():
                    data = layer.query_cache.pages
                    kernel = data.shape[2]
                    data[dst // kernel, :, dst % kernel].copy_(
                        data[src // kernel, :, src % kernel]
                    )


def same_state(actual, expected):
    assert actual.keys() == expected.keys()
    for name, state in actual.items():
        torch.testing.assert_close(state.kv, expected[name].kv, rtol=0, atol=0)
        torch.testing.assert_close(state.raw_q, expected[name].raw_q, rtol=0, atol=0)


def score_terms(before, reuse, start, end, suffix_start, fmt, include_deleted=True):
    layers = []
    for name, src in before.items():
        dst = reuse[name]
        dim = src.q.shape[-1]
        old = dots(src.q[end:], src.kv[start:end, :, :dim], fmt)
        new = dots(dst.q[suffix_start:], dst.kv[start:suffix_start, :, :dim], fmt)
        layers.append(torch.cat((old, new), -1) if include_deleted else new)
    return layers


def check_sparse(actual, baseline, reference, positions, fmt):
    for name, state in actual.items():
        skipped = torch.ones(len(state.kv), dtype=torch.bool)
        skipped[positions] = False
        torch.testing.assert_close(
            state.kv[skipped], baseline[name].kv[skipped], atol=0, rtol=0
        )
        torch.testing.assert_close(
            state.raw_q[skipped], baseline[name].raw_q[skipped], atol=0, rtol=0
        )
        # FP16 prefilling a run and decoding one row can round differently.
        dim = state.q.shape[-1]
        for got, ref, extra in (
            (state.kv[..., :dim], reference[name].kv[..., :dim], 0),
            (state.kv[..., dim:], reference[name].kv[..., dim:], 0),
            (
                state.q,
                reference[name].q,
                {"int4": 1 / 7, "int8": 1 / 127, "fp8": 1 / 8}.get(fmt, 0),
            ),
        ):
            got, ref = got[positions].float(), ref[positions].float()
            error = (got - ref).abs()
            limit = 0.025 + (0.015 + extra) * ref.abs().amax(-1, keepdim=True)
            assert (error <= limit).all(), (
                name,
                error.max().item(),
                (error / limit).max().item(),
            )


def edit_prompt(llm, source, start, end, text, mode, **options):
    return llm.edit_kv_cache(
        source["prompt_token_ids"],
        text,
        start,
        end,
        cache_salt=source.get("cache_salt"),
        recompute_suffix=mode,
        **options,
    )


def threshold_cases(scores, suffix_start, page_size):
    # Choose deterministic masks from the measured scores. Prefer equal counts
    # when two thresholds give similarly dense clustered and fragmented cases.
    candidates = []
    values = scores.unique(sorted=True).tolist()
    for a, b in zip(values, values[1:]):
        threshold = (a + b) / 2
        positions = (scores > threshold).nonzero().flatten().tolist()
        absolute = {suffix_start + p for p in positions}
        runs = 1 + sum(y != x + 1 for x, y in zip(positions, positions[1:]))
        crosses = any(p % page_size == 0 and p - 1 in absolute for p in absolute)
        candidates.append((threshold, positions, runs, crosses))
    split = max((x for x in candidates if x[3]), key=lambda x: x[2])
    assert split[2] >= 2
    clustered = min(
        (x for x in candidates if len(x[1]) >= 3),
        key=lambda x: (x[2], abs(len(x[1]) - len(split[1]))),
    )
    return clustered, split, candidates[-1]


def benchmark(llm, source, start, end, text, cases, fmt, repeats):
    options = {"reuse": (False, {}), "full": (True, {})}
    for label, case in zip(("clustered", "fragmented", "sparsest"), cases):
        options[label] = ("selective", {"threshold": case[0]})
    labels = list(options)
    samples = {k: [] for k in labels}
    for i in range(repeats + 1):
        for label in labels[i % len(labels) :] + labels[: i % len(labels)]:
            mode, kw = options[label]
            t0 = perf_counter()
            edit_prompt(llm, source, start, end, text, mode, **kw)
            ms = (perf_counter() - t0) * 1000
            if i:
                samples[label].append(ms)
    print(
        "LATENCY",
        json.dumps(
            {
                "median_ms": {k: round(median(v), 3) for k, v in samples.items()},
                "samples_ms": samples,
                "device": torch.cuda.get_device_name(),
                "torch": str(torch.__version__),
                "model": llm.llm_engine.model_config.model,
                "reductions": MAX,
                "q_dtype": fmt,
                "source": len(source["prompt_token_ids"]),
                "removed": end - start,
                "inserted": len(
                    llm.get_tokenizer().encode(text, add_special_tokens=False)
                ),
                "suffix": len(source["prompt_token_ids"]) - end,
                "masks": [
                    {"threshold": t, "selected": len(p), "runs": r}
                    for t, p, r, _ in cases
                ],
            }
        ),
        flush=True,
    )


def verify_inproc(llm, src, fmt, repeats):
    cache = ResidentPrompt(llm)
    with cache.pinned(src):
        before = cache.snapshot(src)
        config = llm.llm_engine.model_config.hf_text_config
        assert len(before) == config.num_hidden_layers
        for row in before.values():
            assert row.q.shape[1] == config.num_attention_heads
            assert row.kv.shape[1] == config.num_key_value_heads
        free = cache.editor.pool.get_num_free_blocks()
        refs = [b.ref_cnt for b in cache.pages(src)]
        # Fail after copying the suffix and prefilling the replacement.
        with (
            patch(
                "vllm.v1.worker.kv_cache_edit._selected_runs",
                side_effect=RuntimeError("injected scorer failure"),
            ),
            patch.object(
                cache.editor, "_publish", wraps=cache.editor._publish
            ) as publish,
        ):
            try:
                edit_prompt(llm, src, 7, 10, " green", "selective", threshold=0)
            except RuntimeError as exc:
                assert "injected scorer failure" in str(exc)
            else:
                raise AssertionError("Worker failure did not reach the caller")
            publish.assert_not_called()
        assert cache.editor.pool.get_num_free_blocks() == free
        assert [b.ref_cnt for b in cache.pages(src)] == refs
        same_state(cache.snapshot(src), before)
        for label, start, end, text in (
            ("equal", 16, 17, " blue"),
            ("insert", 15, 15, " blue"),
            ("expand", 15, 16, " blue green"),
            ("contract", 13, 18, " blue"),
            ("delete", 13, 18, ""),
        ):
            suffix_start = start + len(
                llm.get_tokenizer().encode(text, add_special_tokens=False)
            )

            edit = partial(edit_prompt, llm, src, start, end, text)

            reuse = cache.snapshot(edit(False))
            full = cache.snapshot(edit(True))
            for threshold, expected in ((float("inf"), reuse), (-float("inf"), full)):
                same_state(
                    cache.snapshot(edit("selective", threshold=threshold)), expected
                )
            if label == "delete":
                same_state(
                    cache.snapshot(
                        edit(
                            "selective", threshold=-float("inf"), include_deleted=False
                        )
                    ),
                    reuse,
                )
            terms = score_terms(before, reuse, start, end, suffix_start, fmt)
            for ops in REDUCTIONS if label == "equal" else (MAX,):
                scores = reduce_scores(terms, ops)
                threshold = midpoint(scores)
                options = dict(
                    zip(("key_reduction", "head_reduction", "layer_reduction"), ops)
                )
                with cache.observe() as observed:
                    edit("selective", threshold=threshold, **options)
                selected = (scores > threshold).nonzero().flatten() + suffix_start
                assert (
                    observed == list(range(start, suffix_start)) + selected.tolist()
                ), (label, ops)
            if label == "equal":
                cases = threshold_cases(
                    scores=reduce_scores(terms, MAX),
                    suffix_start=suffix_start,
                    page_size=cache.editor.block_size,
                )
                threshold, offsets, _, _ = cases[1]
                positions = [suffix_start + i for i in offsets]
                sparse = edit("selective", threshold=threshold)
                with cache.pinned(sparse):
                    actual = cache.snapshot(sparse)
                    private = edit("selective", threshold=float("inf"))
                    cache.replay_individually(private, src, positions)
                    check_sparse(actual, reuse, cache.snapshot(private), positions, fmt)
                    chained = edit_prompt(
                        llm, sparse, 20, 22, " green", "selective", threshold=threshold
                    )
                    generate_hits(llm, chained)
                    same_state(cache.snapshot(sparse), actual)
                if repeats:
                    benchmark(llm, src, start, end, text, cases, fmt, repeats)
            if label == "contract":
                # Old and new queries have different RoPE coordinates here.
                new_terms = score_terms(
                    before, reuse, start, end, suffix_start, fmt, False
                )
                scores = reduce_scores(new_terms, MAX)
                threshold = midpoint(scores)
                with cache.observe() as observed:
                    edit("selective", threshold=threshold, include_deleted=False)
                assert (
                    observed
                    == list(range(start, suffix_start))
                    + ((scores > threshold).nonzero().flatten() + suffix_start).tolist()
                )
            same_state(cache.snapshot(src), before)
            print("PASS per-layer cache and forward positions", label, flush=True)


def generate_hits(llm, prompt):
    params = SamplingParams(max_tokens=1, temperature=0, logprobs=1)
    ids = prompt["prompt_token_ids"]
    results = []
    for tail in ([], [42]):
        req = dict(prompt, prompt_token_ids=ids + tail)
        out = llm.generate(req, params, use_tqdm=False)[0]
        assert out.num_cached_tokens == len(ids) + len(tail) - 1
        sample = out.outputs[0]
        results.append(
            (sample.token_ids[0], sample.logprobs[0][sample.token_ids[0]].logprob)
        )
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model")
    parser.add_argument("--q-format", default="fp16")
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=0)
    parser.add_argument(
        "--fixture-kv-heads",
        type=int,
        choices=(1, 4),
        help="Use a small dummy-weight Qwen2 decoder for MQA or MHA",
    )
    args = parser.parse_args()
    mp_client = os.environ.get("VLLM_ENABLE_V1_MULTIPROCESSING", "1") == "1"
    fixture_options = {}
    if args.fixture_kv_heads is not None:
        fixture_options = dict(
            load_format="dummy",
            hf_overrides=dict(
                num_hidden_layers=3,
                num_attention_heads=4,
                num_key_value_heads=args.fixture_kv_heads,
                hidden_size=256,
                intermediate_size=512,
            ),
        )
    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tp,
        enforce_eager=True,
        dtype="float16",
        max_model_len=256,
        max_num_seqs=4,
        max_num_batched_tokens=64,
        block_size=16,
        enable_prefix_caching=True,
        gpu_memory_utilization=0.7,
        kv_cache_memory_bytes=192 << 20,
        q_cache_dtype=args.q_format,
        attention_config={"backend": "TRITON_ATTN"},
        **fixture_options,
    )
    if args.fixture_kv_heads is not None:
        config = llm.llm_engine.model_config.hf_text_config
        geometry = (
            config.num_hidden_layers,
            config.num_attention_heads,
            config.num_key_value_heads,
        )
        assert geometry == (3, 4, args.fixture_kv_heads), geometry
    source = {
        "prompt_token_ids": list(range(100, 185)),
        "cache_salt": "selective-verification",
    }
    llm.generate(source, SamplingParams(max_tokens=1, temperature=0), use_tqdm=False)

    def edit(mode, start=16, end=17, text=" blue", **kw):
        return edit_prompt(llm, source, start, end, text, mode, **kw)

    if args.q_format == "none":
        try:
            edit("selective", threshold=0)
        except Exception as exc:
            assert "stored query cache" in str(exc), str(exc)
        else:
            raise AssertionError("Selective edit accepted disabled Q")
        generate_hits(llm, edit(False))
        print("PASS missing Q and legacy reuse", flush=True)
        return

    # Pin the source on in-process inspection runs; use a refreshed source
    # for each MP case since cache residency is an explicit edit prerequisite.
    if not mp_client and args.tp == 1:
        verify_inproc(llm, source, args.q_format, args.repeats)
    for threshold, mode in ((float("inf"), False), (-float("inf"), True)):
        llm.generate(
            source, SamplingParams(max_tokens=1, temperature=0), use_tqdm=False
        )
        expected = generate_hits(llm, edit(mode))
        assert generate_hits(llm, edit("selective", threshold=threshold)) == expected
    partial = edit(
        "selective",
        threshold=0.5,
        key_reduction="sum",
        head_reduction="max",
        layer_reduction="sum",
    )
    generate_hits(llm, partial)
    generate_hits(llm, edit_prompt(llm, partial, 5, 8, "", "selective", threshold=0))
    for start, end, text in ((2, 2, ""), (84, 85, ""), (85, 85, " blue")):
        generate_hits(llm, edit("selective", start, end, text, threshold=0))
    assert not edit("selective", 0, 85, "", threshold=0)["prompt_token_ids"]
    for kw in (
        {},
        {"threshold": float("nan")},
        {"threshold": 0, "head_reduction": "bad"},
        {"threshold": 0, "key_reduction": "bad"},
        {"threshold": 0, "layer_reduction": "bad"},
    ):
        try:
            edit("selective", **kw)
        except ValueError:
            pass
        else:
            raise AssertionError(kw)
    print(
        "PASS public edit, resend, extension, chaining, edge cases and validation",
        "MP" if mp_client else "in-process",
        "TP",
        args.tp,
        args.q_format,
        flush=True,
    )


if __name__ == "__main__":
    main()

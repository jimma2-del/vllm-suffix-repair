# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Offline selective edit verification and warm end-to-end latency measurement."""

import argparse
import json
import os
from contextlib import contextmanager
from dataclasses import dataclass
from functools import partial
from itertools import chain
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
from vllm.v1.worker import kv_cache_edit as edit_worker
from vllm.v1.worker.kv_cache_edit import _prefill, _score_runs, _selected_runs

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
        batches = []

        def record(model, args, kwargs):
            batches.append(kwargs["positions"].tolist())

        with self.runner.get_model().register_forward_pre_hook(
            record, with_kwargs=True
        ):
            yield batches

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


def prefill_separate_runs(
    runner, token_ids, start_pos, pages, block_size, *, runs=None
):
    if runs is None:
        return _prefill(runner, token_ids, start_pos, pages, block_size)
    for start, end in runs:
        _prefill(runner, token_ids[start:end], start_pos + start, pages, block_size)


def verify_gathered(cache, src, edit, suffix_start, reuse, fmt):
    runs = [(0, 3), (9, 10), (16, 17), (25, 26), (36, 37), (49, 50), (64, 66)]
    positions = [suffix_start + p for a, b in runs for p in range(a, b)]
    with patch.object(edit_worker, "_selected_runs", return_value=runs):
        with cache.observe() as observed:
            actual = cache.snapshot(edit("selective", threshold=0))
        with patch.object(edit_worker, "_prefill", prefill_separate_runs):
            separate = cache.snapshot(edit("selective", threshold=0))
    suffix_batches = [batch for batch in observed if batch[0] >= suffix_start]
    assert list(chain.from_iterable(suffix_batches)) == positions
    assert 1 < len(suffix_batches) < len(runs), suffix_batches
    assert any(rows[-1] - rows[0] + 1 > len(rows) for rows in suffix_batches)
    check_sparse(actual, reuse, separate, positions, fmt)
    private = edit("selective", threshold=float("inf"))
    cache.replay_individually(private, src, positions)
    check_sparse(actual, reuse, cache.snapshot(private), positions, fmt)


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


def benchmark(cache, source, start, end, text, cases, fmt, repeats):
    llm = cache.llm
    inserted = len(llm.get_tokenizer().encode(text, add_special_tokens=False))
    replacement_end = start + inserted
    options = {"reuse": (False, {}), "full": (True, {})}
    masks = []
    for label, (threshold, positions, runs, _) in zip(
        ("clustered", "fragmented", "sparsest"), cases
    ):
        top_k = len(positions)
        options[label] = ("selective", {"threshold": threshold})
        top_k_label = f"{label}_top_k"
        options[top_k_label] = ("selective", {"top_k": top_k})
        expected = list(range(start, replacement_end)) + [
            replacement_end + p for p in positions
        ]
        for variant in (label, top_k_label):
            mode, kw = options[variant]
            with cache.observe() as observed:
                edit_prompt(llm, source, start, end, text, mode, **kw)
            assert list(chain.from_iterable(observed)) == expected, variant
        masks.append(
            {"threshold": threshold, "top_k": top_k, "selected": top_k, "runs": runs}
        )
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
                "inserted": inserted,
                "suffix": len(source["prompt_token_ids"]) - end,
                "masks": masks,
            }
        ),
        flush=True,
    )


@torch.inference_mode()
def benchmark_selector(suffix_len, repeats):
    generator = torch.Generator(device="cuda").manual_seed(178)
    for count in (suffix_len, 4096, 32768):
        scores = torch.randperm(count, generator=generator, device="cuda").float()
        top_k = max(1, count // 20)
        threshold = float(count - top_k - 1)
        methods = [("threshold", {"threshold": threshold}), ("top_k", {"top_k": top_k})]
        threshold_runs = _score_runs(scores, methods[0][1])
        top_k_runs = _score_runs(scores, methods[1][1])
        assert top_k_runs == threshold_runs
        assert sum(end - start for start, end in top_k_runs) == top_k
        samples = {label: [] for label, _ in methods}
        for _ in range(repeats):
            for label, options in methods:
                started = perf_counter()
                _score_runs(scores, options)
                samples[label].append((perf_counter() - started) * 1e6)
            methods.reverse()
        print(
            "SELECTOR_LATENCY",
            json.dumps(
                {
                    "device": torch.cuda.get_device_name(),
                    "torch": str(torch.__version__),
                    "score_dtype": str(scores.dtype),
                    "suffix": count,
                    "top_k": top_k,
                    "threshold": threshold,
                    "runs": len(top_k_runs),
                    "median_us": {k: round(median(v), 3) for k, v in samples.items()},
                    "samples_us": samples,
                }
            ),
            flush=True,
        )


def benchmark_fixed_masks(edit, suffix_len, repeats):
    select_start, select_end, replay_start, replay_end = (
        torch.cuda.Event(enable_timing=True) for _ in range(4)
    )

    def measure(fixed_runs, replay):
        def score(*args):
            select_start.record()
            _selected_runs(*args)  # Include the real selection cost in the edit.
            select_end.record()
            return fixed_runs

        def timed_prefill(*args, runs=None):
            if runs is None:
                return _prefill(*args)
            replay_start.record()
            replay(*args, runs=runs)
            replay_end.record()

        with (
            patch.object(edit_worker, "_prefill", timed_prefill),
            patch.object(edit_worker, "_selected_runs", score),
        ):
            started = perf_counter()
            edit("selective", threshold=0)
            edit_ms = (perf_counter() - started) * 1000
        # The edit RPC has synchronized the device before publishing pages.
        return {
            "edit_ms": edit_ms,
            "selection_ms": select_start.elapsed_time(select_end),
            "replay_ms": replay_start.elapsed_time(replay_end),
        }

    # All masks share a midpoint, so the sum of causal prefix lengths is fixed.
    for run_count in (1, 4, 16):
        width = 16 // run_count
        left = (suffix_len - 16 - 2 * (run_count - 1)) // 2
        runs = [
            (left + i * (width + 2), left + i * (width + 2) + width)
            for i in range(run_count)
        ]
        methods = [("gather", _prefill), ("runs", prefill_separate_runs)]
        samples = {label: [] for label, _ in methods}
        for i in range(repeats + 1):
            for label, replay in methods:
                sample = measure(runs, replay)
                if i:
                    samples[label].append(sample)
            methods.reverse()
        print(
            "FIXED_MASK",
            json.dumps({"suffix_runs": runs, "samples": samples}),
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
            if label == "equal":
                suffix_len = len(src["prompt_token_ids"]) - end
                same_state(cache.snapshot(edit("selective", top_k=0)), reuse)
                same_state(
                    cache.snapshot(edit("selective", top_k=suffix_len + 1)), full
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
                    list(chain.from_iterable(observed))
                    == list(range(start, suffix_start)) + selected.tolist()
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
                    same_state(
                        cache.snapshot(edit("selective", top_k=len(offsets))), actual
                    )
                    private = edit("selective", threshold=float("inf"))
                    cache.replay_individually(private, src, positions)
                    check_sparse(actual, reuse, cache.snapshot(private), positions, fmt)
                    chained = edit_prompt(
                        llm, sparse, 20, 22, " green", "selective", threshold=threshold
                    )
                    generate_hits(llm, chained)
                    same_state(cache.snapshot(sparse), actual)
                if repeats:
                    benchmark(cache, src, start, end, text, cases, fmt, repeats)
                    benchmark_selector(len(scores), repeats)
                    benchmark_fixed_masks(edit, len(scores), repeats)
            if label == "contract":
                verify_gathered(cache, src, edit, suffix_start, reuse, fmt)
                # Old and new queries have different RoPE coordinates here.
                new_terms = score_terms(
                    before, reuse, start, end, suffix_start, fmt, False
                )
                scores = reduce_scores(new_terms, MAX)
                threshold = midpoint(scores)
                with cache.observe() as observed:
                    edit("selective", threshold=threshold, include_deleted=False)
                assert (
                    list(chain.from_iterable(observed))
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
    suffix_len = len(source["prompt_token_ids"]) - 17
    for threshold, top_k, mode in (
        (float("inf"), 0, False),
        (-float("inf"), suffix_len, True),
    ):
        llm.generate(
            source, SamplingParams(max_tokens=1, temperature=0), use_tqdm=False
        )
        expected = generate_hits(llm, edit(mode))
        assert generate_hits(llm, edit("selective", threshold=threshold)) == expected
        assert generate_hits(llm, edit("selective", top_k=top_k)) == expected
    partial = edit(
        "selective",
        threshold=0.5,
        key_reduction="sum",
        head_reduction="max",
        layer_reduction="sum",
    )
    generate_hits(llm, partial)
    generate_hits(llm, edit("selective", top_k=5))
    generate_hits(llm, edit_prompt(llm, partial, 5, 8, "", "selective", threshold=0))
    for start, end, text in ((2, 2, ""), (84, 85, ""), (85, 85, " blue")):
        generate_hits(llm, edit("selective", start, end, text, threshold=0))
    assert not edit("selective", 0, 85, "", threshold=0)["prompt_token_ids"]
    for kw in (
        {},
        {"threshold": float("nan")},
        {"top_k": -1},
        {"top_k": 1.5},
        {"top_k": True},
        {"top_k": 1, "threshold": 0},
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

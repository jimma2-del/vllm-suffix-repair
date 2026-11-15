# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Run with .venv/bin/python -m suffix_repair_test.kv_cache_edit_gpu MODEL."""

import argparse
import os

import torch

from vllm import LLM, SamplingParams
from vllm.forward_context import get_forward_context


def check_relocated(source, actual, src_pos, dst_pos, rope, q_cache_dtype=None):
    head = rope.head_size
    torch.testing.assert_close(actual[..., head:], source[..., head:], rtol=0, atol=0)
    # Independent complex-number reference for NeoX RoPE.
    n = len(source)
    phase = torch.complex(*rope.cos_sin_cache.float().chunk(2, -1))
    old = torch.complex(*source[..., :head].float().chunk(2, -1))
    moved = (
        old * (phase[dst_pos : dst_pos + n] / phase[src_pos : src_pos + n])[:, None, :]
    )
    expected = torch.cat((moved.real, moved.imag), dim=-1)
    actual = actual[..., :head].float()
    if q_cache_dtype in ("bf16", "fp8", "int8", "int4"):
        bound = {"bf16": 0.004, "fp8": 0.04, "int8": 0.004, "int4": 0.072}
        tolerance = expected.abs().amax(-1, True) * bound[q_cache_dtype] + 1e-5
        assert ((actual - expected).abs() <= tolerance).all()
    else:
        torch.testing.assert_close(actual, expected, atol=0.005, rtol=0.002)


def main(model_path: str, q_cache_dtype: str) -> None:
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    llm = LLM(
        model=model_path,
        enforce_eager=True,
        dtype="float16",
        max_model_len=256,
        max_num_seqs=4,
        max_num_batched_tokens=64,
        block_size=16,
        enable_prefix_caching=True,
        kv_cache_memory_bytes=128 << 20,
        q_cache_dtype=q_cache_dtype,
        gpu_memory_utilization=0.7,
        attention_config={"backend": "TRITON_ATTN"},
    )
    params = SamplingParams(max_tokens=1, temperature=0)
    core = llm.llm_engine.engine_core.engine_core
    manager = core.scheduler.kv_cache_manager
    editor = manager.editor
    worker = core.model_executor.driver_worker
    context = worker.vllm_config.compilation_config.static_forward_context
    model = worker.get_model()
    layers = manager.kv_cache_config.kv_cache_groups[0].layer_names

    probe = {"prompt_token_ids": list(range(300, 335))}
    probe_params = SamplingParams(
        max_tokens=5, ignore_eos=True, temperature=0, logprobs=1
    )
    captured = []

    def capture(layer, args):
        slots = get_forward_context().slot_mapping[layer.layer_name]
        query = args[0].reshape(-1, layer.num_heads, layer.head_size)
        captured.append((slots.clone(), query[: len(slots)].clone()))

    first_layer = context[layers[0]]
    with first_layer.register_forward_pre_hook(capture):
        cold = llm.generate(probe, probe_params, use_tqdm=False)[0]
    if first_layer.query_cache is not None:
        assert len(captured) >= 2
        for slots, query in captured:
            valid = slots >= 0
            query = query[valid]
            actual = first_layer.query_cache.read(slots[valid])
            if q_cache_dtype in ("fp8", "int8", "int4"):
                bound = {"fp8": 0.04, "int8": 0.004, "int4": 0.072}[q_cache_dtype]
                err = (actual.float() - query.float()).abs()
                assert (err <= query.abs().float().amax(-1, keepdim=True) * bound).all()
            else:
                torch.testing.assert_close(
                    actual, query.to(actual.dtype), rtol=0, atol=0
                )
    warm = llm.generate(probe, probe_params, use_tqdm=False)[0]
    for name in layers:
        if context[name].query_cache is not None:
            context[name].query_cache.data.fill_(255)
    poisoned = llm.generate(probe, probe_params, use_tqdm=False)[0]
    assert 0 < warm.num_cached_tokens == poisoned.num_cached_tokens
    assert warm.outputs[0].token_ids == poisoned.outputs[0].token_ids
    assert warm.outputs[0].logprobs == poisoned.outputs[0].logprobs
    sampled = cold.outputs[0]
    print(
        "Greedy generation",
        list(sampled.token_ids),
        [p[t].logprob for p, t in zip(sampled.logprobs, sampled.token_ids)],
    )

    def snapshot(ids, salt=None):
        request = editor._request(ids, salt)
        pages = []
        for pos in range(0, len(ids), editor.block_size):
            end = min(pos + editor.block_size, len(ids))
            block = editor.pool.get_cached_block(editor._hash_at(request, end), [0])
            pages.append(block[0])
        states = {}
        for name in manager.kv_cache_config.kv_cache_groups[0].layer_names:
            cache = context[name].kv_cache.transpose(1, 2)
            states[name] = torch.cat([cache[b.block_id] for b in pages])[
                : len(ids)
            ].clone()
        return states, pages

    def queries(pages, length):
        slots = torch.tensor(
            [
                page.block_id * editor.block_size + j
                for page in pages
                for j in range(editor.block_size)
            ][:length],
            device=worker.device,
        )
        return {
            name: context[name].query_cache.read(slots)
            for name in manager.kv_cache_config.kv_cache_groups[0].layer_names
            if context[name].query_cache is not None
        }

    # Both the replacement and the suffix cross the 64-token prefill chunk size.
    original = list(range(100, 195))
    start, end = 19, 22
    text = " one two three four five six seven eight nine ten" * 7
    inserted = llm.get_tokenizer().encode(text, add_special_tokens=False)
    suffix_start = start + len(inserted)
    edited_ids = original[:start] + inserted + original[end:]
    # Use ordinary generation with partial-page hits so the reference has the
    # same cached prefix and FP16 forward shapes as the two edit passes.
    reference_salt = editor.salt_prefix + "reference"

    def generate_reference(ids, hit, sampling=params):
        output = llm.generate(
            {"prompt_token_ids": ids, "cache_salt": reference_salt},
            sampling,
            use_tqdm=False,
        )[0]
        assert output.num_cached_tokens == hit, output.num_cached_tokens

    generate_reference(original[:start], 0)
    generate_reference(
        original,
        start,
        SamplingParams(max_tokens=12, temperature=0, ignore_eos=True),
    )
    generate_reference(edited_ids[:suffix_start], start)
    generate_reference(edited_ids, suffix_start)
    before, source = snapshot(original, reference_salt)
    reference, reference_pages = snapshot(edited_ids, reference_salt)
    q_before = queries(source, len(original))
    q_reference = queries(reference_pages, len(edited_ids))

    forwards = []

    def record_positions(module, args, kwargs):
        forwards.append(kwargs["positions"].tolist())

    # Hold a reader of every page throughout the edit.
    editor.pool.touch(source)
    try:
        with model.register_forward_pre_hook(record_positions, with_kwargs=True):
            result = llm.edit_kv_cache(
                original, text, start, end, cache_salt=reference_salt
            )
            assert [p for chunk in forwards for p in chunk] == list(
                range(start, suffix_start)
            )
            forwards.clear()
            repaired = llm.edit_kv_cache(
                original,
                text,
                start,
                end,
                cache_salt=reference_salt,
                recompute_suffix=True,
            )
            assert [p for chunk in forwards for p in chunk] == list(
                range(start, len(edited_ids))
            )
            assert any(chunk[0] == suffix_start for chunk in forwards)
        assert result["prompt_token_ids"] == repaired["prompt_token_ids"] == edited_ids
        assert result["cache_salt"] != repaired["cache_salt"]
        after, dest = snapshot(edited_ids, result["cache_salt"])
        repaired_kv, repaired_pages = snapshot(edited_ids, repaired["cache_salt"])
        original_again, _ = snapshot(original, reference_salt)
        q_after = queries(dest, len(edited_ids))
        q_repaired = queries(repaired_pages, len(edited_ids))
        q_original_again = queries(source, len(original))
        for pages in (dest, repaired_pages):
            assert pages[0] is source[0]
            assert not {b.block_id for b in source} & {b.block_id for b in pages[1:]}
        for name, kv in after.items():
            torch.testing.assert_close(
                original_again[name], before[name], rtol=0, atol=0
            )
            torch.testing.assert_close(kv[:start], before[name][:start], rtol=0, atol=0)
            torch.testing.assert_close(
                kv[:suffix_start], repaired_kv[name][:suffix_start], rtol=0, atol=0
            )
            torch.testing.assert_close(
                repaired_kv[name][start:],
                reference[name][start:],
                rtol=0,
                atol=0,
                msg=lambda msg, name=name: f"{name}: {msg}",
            )
            rope = model.get_submodule(name.rsplit(".", 1)[0]).rotary_emb
            check_relocated(
                before[name][end:], kv[suffix_start:], end, suffix_start, rope
            )
            if name in q_before:
                torch.testing.assert_close(
                    q_original_again[name], q_before[name], atol=0, rtol=0
                )
                torch.testing.assert_close(
                    q_after[name][:start], q_before[name][:start], atol=0, rtol=0
                )
                torch.testing.assert_close(
                    q_repaired[name], q_reference[name], atol=0, rtol=0
                )
                torch.testing.assert_close(
                    q_after[name][:suffix_start],
                    q_repaired[name][:suffix_start],
                    atol=0,
                    rtol=0,
                )
                check_relocated(
                    q_before[name][end:],
                    q_after[name][suffix_start:],
                    end,
                    suffix_start,
                    rope,
                    q_cache_dtype,
                )
    finally:
        editor.pool.free_blocks(source)

    # Repair the previously reused suffix without changing any token IDs.
    refreshed = llm.edit_kv_cache(
        edited_ids,
        "",
        suffix_start,
        suffix_start,
        cache_salt=result["cache_salt"],
        recompute_suffix=True,
    )
    refreshed_kv, _ = snapshot(edited_ids, refreshed["cache_salt"])
    reused_again, _ = snapshot(edited_ids, result["cache_salt"])
    for name in after:
        torch.testing.assert_close(reused_again[name], after[name], atol=0, rtol=0)
        torch.testing.assert_close(
            refreshed_kv[name], repaired_kv[name], atol=0, rtol=0
        )

    for prompt in (result, repaired):
        extended = dict(prompt, prompt_token_ids=edited_ids + [42])
        output = llm.generate(extended, params, use_tqdm=False)[0]
        assert output.num_cached_tokens == len(edited_ids), output.num_cached_tokens
    # A second edit resolves the first edit's partial tail by its salt.
    shortened = llm.edit_kv_cache(
        edited_ids, "", 3, 32, cache_salt=result["cache_salt"]
    )
    short_ids = shortened["prompt_token_ids"]
    assert short_ids == edited_ids[:3] + edited_ids[32:]
    short_kv, _ = snapshot(short_ids, shortened["cache_salt"])
    for name, kv in short_kv.items():
        rope = model.get_submodule(name.rsplit(".", 1)[0]).rotary_emb
        torch.testing.assert_close(kv[:3], after[name][:3], rtol=0, atol=0)
        check_relocated(after[name][32:], kv[3:], 32, 3, rope)
    output = llm.generate(shortened, params, use_tqdm=False)[0]
    assert output.num_cached_tokens == len(short_ids) - 1, output.num_cached_tokens
    print(
        f"Checked {len(after)} layers, {len(original)} -> {len(edited_ids)}"
        f" -> {len(short_ids)} tokens, with suffix reuse and recomputation."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model", help="Local Qwen2 or Llama model directory")
    parser.add_argument("--q-cache-dtype", default="fp16")
    args = parser.parse_args()
    main(args.model, args.q_cache_dtype)

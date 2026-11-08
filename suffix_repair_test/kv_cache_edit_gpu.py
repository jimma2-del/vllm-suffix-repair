# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Run with .venv/bin/python -m suffix_repair_test.kv_cache_edit_gpu MODEL."""

import argparse
import os

import torch

from vllm import LLM, SamplingParams


def check_relocated(source, actual, src_pos, dst_pos, rope):
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
    torch.testing.assert_close(
        actual[..., :head].float(), expected, atol=0.005, rtol=0.002
    )


def main(model_path: str) -> None:
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
        kv_cache_memory_bytes=32 << 20,
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

    original = list(range(100, 139))
    start, end = 19, 22
    text = " one two three four five six seven eight nine ten" * 7
    inserted = llm.get_tokenizer().encode(text, add_special_tokens=False)
    llm.generate(
        {"prompt_token_ids": original},
        SamplingParams(max_tokens=12, temperature=0, ignore_eos=True),
        use_tqdm=False,
    )
    llm.generate({"prompt_token_ids": inserted}, params, use_tqdm=False)
    before, source = snapshot(original)
    standalone, _ = snapshot(inserted)

    # Hold a reader of every page throughout the edit.
    editor.pool.touch(source)
    try:
        result = llm.edit_kv_cache(original, text, start, end)
        edited_ids, salt = result["prompt_token_ids"], result["cache_salt"]
        assert edited_ids == original[:start] + inserted + original[end:]
        after, dest = snapshot(edited_ids, salt)
        original_again, _ = snapshot(original)
        assert dest[0] is source[0]
        assert not {b.block_id for b in source} & {b.block_id for b in dest[1:]}
        for name, kv in after.items():
            torch.testing.assert_close(
                original_again[name], before[name], rtol=0, atol=0
            )
            torch.testing.assert_close(kv[:start], before[name][:start], rtol=0, atol=0)
            rope = model.get_submodule(name.rsplit(".", 1)[0]).rotary_emb
            for source_kv, src_pos, dst_pos in (
                (standalone[name], 0, start),
                (before[name][end:], end, start + len(inserted)),
            ):
                actual = kv[dst_pos : dst_pos + len(source_kv)]
                check_relocated(source_kv, actual, src_pos, dst_pos, rope)
    finally:
        editor.pool.free_blocks(source)

    extended = dict(result, prompt_token_ids=edited_ids + [42])
    output = llm.generate(extended, params, use_tqdm=False)[0]
    assert output.num_cached_tokens == len(edited_ids), output.num_cached_tokens
    # A second edit resolves the first edit's partial tail by its salt.
    shortened = llm.edit_kv_cache(edited_ids, "", 3, 32, cache_salt=salt)
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
        f" -> {len(short_ids)} tokens, including partial-page generation hits."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model", help="Local Qwen2 or Llama model directory")
    main(parser.parse_args().model)

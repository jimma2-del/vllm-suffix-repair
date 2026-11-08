import sys
from vllm import LLM, SamplingParams


def main():
    llm = LLM(model=sys.argv[1], max_model_len=128, max_num_batched_tokens=64,
              max_num_seqs=2, block_size=16, dtype="float16",
              gpu_memory_utilization=0.7, kv_cache_memory_bytes=32 << 20)
    params = SamplingParams(max_tokens=1, temperature=0)
    original = list(range(101, 140))
    llm.generate({"prompt_token_ids": original}, params, use_tqdm=False)
    edited = llm.edit_kv_cache(original, " green", 4, 8)
    ids = edited["prompt_token_ids"]
    extended = dict(edited, prompt_token_ids=ids + [50])
    out = llm.generate(extended, params, use_tqdm=False)[0]
    assert out.num_cached_tokens == len(ids), (out.num_cached_tokens, len(ids))
    print(f"{type(llm.llm_engine.engine_core).__name__}: {out.num_cached_tokens} cached tokens")


if __name__ == "__main__":
    main()

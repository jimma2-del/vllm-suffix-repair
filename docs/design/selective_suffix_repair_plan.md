# Implementation plan for selective suffix repair

**Start with max over edited keys, query heads and layers, then replay whole
tokens.** A suffix token qualifies as soon as any signed dot product exceeds
the threshold. This produces one mask for every head and layer. The existing
worker already copies and relocates cached tokens and can prefill any
contiguous span at its true position. Reusing those operations requires
neither hidden-state checkpoints nor a new attention kernel.
[Aggregation choices and alternatives](selective_suffix_repair.md)

1. **Carry the selection options to the worker.** Extend
   `LLM.edit_kv_cache` with a selective value for `recompute_suffix`, keeping
   the existing boolean calls and return type. Selective mode takes an
   explicit threshold, independent key/head/layer reductions defaulting to
   `max`, and deletion scoring enabled by default. Require stored Q for this
   mode. Pass the options through both synchronous engine clients, the engine
   core and the worker's `KVCacheEdit` request alongside the suffix token IDs.
   [Entry point](../../vllm/entrypoints/llm.py),
   [engine clients](../../vllm/v1/engine/core_client.py) and
   [engine core](../../vllm/v1/engine/core.py).
2. **Prepare the edited cache.** Share complete prefix pages and reserve
   private pages from the first changed page through the suffix, even for an
   equal-length replacement. Pin the source, seed the partial prefix page,
   prefill the replacement, then copy the suffix Q/K/V to its final slots
   with the existing RoPE relocation. The original pages now supply the old
   Q and removed K. The destination supplies the position-corrected Q and new
   K. Publish the edited pages after successful replay and retain the editor's
   cleanup on failure. [Page planning](../../vllm/v1/core/kv_cache_edit.py)
   and [worker operations](../../vllm/v1/worker/kv_cache_edit.py).
3. **Score before replay.** Process one layer at a time in bounded batches of
   suffix queries and affected keys. Reuse the Q cache's format conversion
   for K, including each K vector's own scale. Compute raw dot products with
   FP32 accumulation and pair each query head with its KV head. Start with
   running maxima, initialized to negative infinity so negative scores work.
   Fold the old and new spans into the same key reduction. An empty suffix or
   empty set of included keys selects no tokens. [Query cache encoding and
   reads](../../vllm/v1/attention/ops/query_cache.py).
4. **Finish the reductions and freeze the mask.** Reduce keys, then heads,
   then layers, and apply a strict greater-than threshold. For tensor
   parallelism, combine head results across ranks using the chosen head
   reduction before reducing layers. Every rank must replay identical
   positions. Enable `sum` independently at each stage using numeric partial
   scores. Keeping the per-head and per-layer reductions distinct inside the
   scorer gives future selection policies access to those scores without
   storing a full score tensor or changing the replay interface.
5. **Replay selected runs in causal order.** Group adjacent selected positions
   and call the existing `_prefill` once per run at its absolute start. Its
   destination page table includes the full prefix and all intervening
   skipped tokens. Earlier runs have already written their fresh K/V. The
   normal forward pass recomputes all heads and layers at model precision and
   stores Q in the configured format. An empty mask needs no replay and a
   full mask becomes one contiguous suffix prefill.

## Verification when implementation begins

Start with static checks of option propagation, page ownership, score shapes,
head mapping and causal positions. For the later GPU checks, compare the
all-selected and none-selected cases against the existing full-recompute and
reuse modes. Check isolated selections across a page boundary against a
token-at-a-time replay over the same mixture of reused and fresh K/V. Include
length changes, pure deletion, mixed reductions, quantized Q, tensor-parallel
rank agreement and source-cache integrity after a chained edit.

Measure end-to-end edit latency alongside the selected fraction and number of
contiguous runs. Fragmented selections can make forward-launch overhead
significant even when few tokens are selected.

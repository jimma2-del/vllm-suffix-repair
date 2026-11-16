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

## Verification plan

The verification runs, latency measurements and reproduction commands are
recorded in the [verification report](selective_suffix_repair_verification.md).

1. **Static checks.** Lint, check formatting, and compile the changed Python
   files. Trace the selective options through both synchronous clients and
   check that page publication follows the worker RPC.
2. **Independent score reference.** On a short resident prompt, snapshot
   original Q/K and replacement K. Independently cast or quantize each K to
   the Q format, apply the stored scales, and compute signed dot products in
   FP32. Reduce explicitly over keys, query heads and layers in that order.
   Compare the selected indices for all eight reduction combinations across
   MHA, GQA and MQA. Include negative scores, exact threshold ties, zero
   vectors, all supported Q formats, and spans crossing scoring tile sizes.
3. **Replay and cache contents.** For finite scores and at least one included
   key, use negative and positive infinity thresholds to compare with full
   recomputation and reuse respectively. Pick an intermediate threshold that
   selects separated runs, including a page boundary. Compare each layer's
   Q/K/V with a token-at-a-time reference over the same edited causal prefix.
   Use model-dtype tolerances for fresh vectors and compare raw Q bytes for
   unshifted, skipped tokens. Resend and extend the returned salted prompt.
4. **Positions and options.** Exercise insertion, equal-length replacement,
   expansion, contraction and pure deletion. Score deleted keys with original
   queries and replacement keys with relocated queries. Disabling deleted-key
   scoring on a pure deletion must select nothing even at a negative threshold.
   Cover empty suffixes, no-op edits, missing Q, a missing or NaN threshold,
   invalid reductions and both legacy boolean modes.
5. **Process and page ownership.** Repeat via in-process and multiprocess
   clients, both GPU runners and tensor parallelism. Include a case where
   only a non-driver rank has a high score and a case with replicated KV heads.
   Compare rank masks to a reference over all distinct query heads. Snapshot
   the original pages before a chained edit. Check source contents after
   success and check reference counts and unpublished aliases on worker or
   allocation failure.
6. **Latency.** On the same warm source prompt, compare reuse, full replay
   and selective replay for edits with clustered and fragmented selections.
   Record model, device, Q format, reduction settings, prompt and edit lengths,
   threshold, selected fraction, run count and end-to-end edit latency.

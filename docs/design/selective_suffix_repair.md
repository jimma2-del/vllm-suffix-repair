# Selective suffix repair for `LLM.edit_kv_cache`

Pass `recompute_suffix="selective"` and an explicit `threshold` to score each
suffix token, then recompute all heads and layers for tokens strictly above
that threshold. This requires a bound Q cache. The default reductions are
`max`, and `include_deleted=True` scores removed keys as well as replacements.
The boolean modes select suffix reuse (`False`) or full suffix replay (`True`).
[Edit worker](../../vllm/v1/worker/kv_cache_edit.py) and
[verification plan](selective_suffix_repair_plan.md#verification-plan).

This document records the selection design, implemented architecture and
follow-up proposals discussed through November 24, 2026. The
[KV editing record](kv_cache_editing.md) covers the underlying API, page
ownership and publication protocol. The [query storage record](query_cache.md)
covers page layout, encoding and lifecycle integration. The
[implementation plan](selective_suffix_repair_plan.md) records the chosen
work sequence, and the [verification report](selective_suffix_repair_verification.md)
contains measured results and reproduction commands.

```python
edited = llm.edit_kv_cache(
    original_token_ids, replacement_text, start, end,
    recompute_suffix="selective", threshold=score_threshold,
    key_reduction="max", head_reduction="max", layer_reduction="max",
)
```

## Where scores are aggregated

Each suffix token gets its own decision. The four reductions below run in
order. The final three support independent `max` or `sum` choices, with `max`
preferred. Combining axes earlier can change the answer when their reductions
differ.

| Across | What is combined | How the work or decision can be split |
| --- | --- | --- |
| Vector dimensions | Sum the component products of one suffix Q and one edited-span K to get a signed dot product. | A cached dot product for the same original Q and deleted K can be reused. |
| Edited tokens | For each query head, combine scores from replacement keys and, by default, removed keys. | Score the old and new spans separately and merge their partial results with the same reduction. Disabling deletion scoring omits the old span. |
| Query heads | Combine one score per query head into a layer score. Each GQA query head uses its group's shared K. | Keeping head or KV-group scores would allow finer selection. Skipping individual heads during replay also needs a way to supply their attention outputs to the next layer. |
| Layers | Combine layer scores into one whole-token decision. With all three reductions set to `max`, any individual dot product above threshold selects the token. | A layer-wise policy can use scores before this final reduction. It must provide the token's hidden state at each layer it runs. |

## Scoring a suffix token

Use the cached post-RoPE Q for selection. For each query head, compute its raw,
signed dot product with each replacement K produced by the replacement prefill.
Cast or quantize each K to the Q cache format, using K's own scale when the
format requires one. In grouped-query attention, each query head uses the K of
its own KV group. Count each query head once. [Query cache formats](../features/quantization/quantized_kvcache.md)

Include keys from the removed span by default, with an option to disable them.
A pure deletion can therefore select tokens using only the original cache. For
length-changing edits, the selector scores removed keys
against Q at its original position and new keys against Q at its edited RoPE
position. Both use the same cached query content.

## Whole-token selection

Aggregate the numeric layer scores for each token, then select tokens whose
result strictly exceeds the threshold. **Max across layers is preferred.**
With a common threshold this selects the union of the per-layer hits. A sum
can select a token with several moderate scores and can cancel positive and
negative scores. The two reductions therefore have different threshold scales.

Recompute selected tokens through every layer at their edited positions, with
all heads at model precision. Each selected token attends to its complete
causal prefix, including reused K/V from skipped tokens and fresh K/V from
earlier selected tokens. Apply the usual RoPE relocation wherever reused
tokens move. Store the new K/V and encode the fresh Q in the configured Q
cache format. The existing Q/K/V cache supplies all persistent token state
for this replay.

## Layer-wise selection

A layer-wise selector can threshold each token's score separately in each
layer. The score measures a direct interaction with the edited span. Layer
$l$ projects Q/K/V from its **input** hidden state. Attention to the edit in
layer $l$ first changes its **output**, which supplies the input to layer
$l+1$. For example, a token can score highly in layer 8 and weakly in layer 9
while its layer-9 K/V change substantially. The hidden state at a skipped
boundary determines which computations can be omitted.

- **Replay through the deepest hit.** Score all layers up front using cached
  Q. For each token with a hit, run from its embedding through the deepest
  layer that crossed the threshold. At each layer the active set consists of
  tokens with a hit there or deeper, so it can only shrink with depth. To
  capture the last hit's immediate effect in the cache, also project and
  store Q/K/V at the next layer if one exists. This needs no new persistent
  state. The approximation is that the remaining deeper cache entries are
  still adequate even though the residual stream may carry the edit further.
  CacheBlend uses shrinking token sets for cached text chunks, choosing its
  sets using measured deviations. [CacheBlend, §4.3 and Figure 9](https://www.microsoft.com/en-us/research/wp-content/uploads/2024/09/eurosys25-final999.pdf)
- **Start at the first hit and continue upward.** Save the original
  model-precision hidden state entering a layer, including the residual
  stream. Resume at the first hit from that checkpoint and recompute all
  deeper layers. This assumes that the skipped earlier layers would have
  supplied approximately the original input. Once the token starts, all
  later layers receive the propagated correction. Saving every boundary costs
  roughly one hidden-width vector per token per layer in a dense decoder.
  Sparse checkpoints trade memory for replay from the nearest earlier one.
- **Allow gaps between active layers.** Restarting from an original
  checkpoint after a gap discards the correction computed before it. One
  possible heuristic is to carry that correction across the gap. If $x_a$
  and $x_b$ are original hidden states entering and leaving the gap, estimate
  the edited exit as $x_b + (x'_a - x_a)$, where $x'_a$ is the edited entry.
  This assumes the skipped attention and MLP blocks add approximately their
  original residual updates. Forwarding the edited entry itself across the
  gap would dispense with checkpoints at the cost of assuming that the
  entire skipped transformation is close to the identity, even on the original
  prompt. The local Q·K score alone establishes neither assumption.

If the aim is specifically to skip attention, another option is to cache the
original combined attention output after projection. In a skipped layer of a
replayed token, reuse that output and run the residual and MLP operations on
the current state. This assumes the attention contribution is stable and
still pays the MLP cost. It needs one combined attention vector per token
per saved layer.

## Earlier proposal for skipping individual heads

If only head A runs for a token, the next layer still needs the contribution
of skipped head B. Even with a fixed Q, a replacement K can change B's softmax
weights and its weighted sum of values. Recovering that output from Q/K/V
would require running B's attention. [Qwen2 attention and decoder layer](../../vllm/model_executor/models/qwen2.py)

The proposed approximation was to reuse each skipped head's original attention
output. Cache the per-head outputs before the output projection and their
combined projected output during the original forward pass. For each selected
head, add its projected output difference to the cached combined output, then
run the residual and MLP operations. This costs roughly two hidden-width vectors
per token per layer in a typical dense decoder. Recomputing whole tokens
through all layers avoids this extra cache.

## Caching scores for deleted tokens

The original suffix Q and removed K are already resident. Their scores can be
reused across different replacements of the same source span. An experiment
could collect the following summaries on the first edit or during the original
prefill. Any precomputed score must use the selector's raw dot product and Q/K
formats. Normal attention weights include softmax normalization over the
entire causal prefix.

- **Repeated edits.** Quantize each removed K once per KV head and reuse it
  across suffix queries. For repeated spans, cache the reduced deleted-key
  score per suffix query head. Combine it with the new-key score before
  reducing across heads. This preserves mixed `max` and `sum` semantics.
- **Max over keys.** Cache the maximum dot product per suffix query head and
  source key block on demand. Combine whole-block maxima and score the edges
  of an arbitrary deletion directly. When both reductions use `max` and the
  threshold is fixed, another option is to store the source key positions
  where any head crossed it for each suffix token and layer. Testing for an
  intersection with the deleted range then answers whether a removed key
  crosses that layer's threshold. These threshold hits also suffice for a
  max across layers. A sum across layers requires numeric layer scores.
- **Sum over keys.** Cache block or prefix sums of the individually quantized
  keys, with each key's scale applied before accumulation. For a fixed Q,
  summing the deleted-key dot products equals one dot product with the key
  sum, up to floating-point accumulation order.

Tie cached results to the source Q/K contents, original positions, layer,
scoring format and any reduction or threshold already applied. Physical page
IDs can be recycled, and chained edits can have different cache contents for
the same token IDs. Keeping every pairwise score would cost quadratic space
in context length. Block summaries or entries for frequently edited spans let
the experiment trade cache space for repeat-edit latency.

## Implemented architecture

The initial design chose whole-token replay to reuse the existing model
forward and avoid additional hidden-state or attention-output checkpoints.
Each selected token runs all heads and layers. The approximation is that
skipped tokens' cached K/V remains adequate after the edit. Selection uses
cached Q throughout and freezes the mask before any suffix replay, so newly
computed suffix Q never changes another token's selection in the same edit.

| Component | Responsibility |
| --- | --- |
| `LLM.edit_kv_cache` | Validate the mode, explicit non-NaN threshold and reduction options, tokenize replacement text, and return edited IDs with a fresh cache salt. |
| `SuffixSelection` and synchronous engine clients | Carry the threshold, three reductions and deletion option through in-process calls or multiprocess serialization. |
| Engine core and `KVCacheEditor` | Check prerequisites, resolve and pin source pages, allocate private destination pages and dispatch the worker RPC before publishing cache aliases. |
| `write_kv_cache_edit` | Seed the prefix fragment, prefill replacement tokens, copy and relocate the suffix, select runs, and replay them. |
| `QueryCache.roundtrip_keys` | Represent affected K in the stored Q format for scoring, with the same CUDA encoder used by Q writes. |

The public API accepts infinite thresholds for selecting all or none of the
finite scores. Invalid reductions, missing or NaN thresholds, and selective
options supplied with a boolean mode raise errors. Selective mode requires
stored Q. Worker validation additionally checks dense floating-point K/V,
compatible 1D RoPE, bound Q geometry and the absence of shared K/V layers.
The underlying editing restrictions are recorded in the KV editing record.
[Public entry point](../../vllm/entrypoints/llm.py),
[transport types](../../vllm/v1/engine/__init__.py),
[engine clients](../../vllm/v1/engine/core_client.py),
[engine core](../../vllm/v1/engine/core.py).

Selective edits reserve private pages through the suffix before computing
the mask, including length-preserving replacements. This protects the source
and provides writable destinations for any selection. An edit with no included
affected keys follows the reuse path. The worker completes and synchronizes
before the editor publishes the destination. Allocation or worker failures
release temporary references without publishing the edited cache.
[Page planning](../../vllm/v1/core/kv_cache_edit.py) and
[worker synchronization](../../vllm/v1/worker/gpu_worker.py).

### Scoring implementation

`_score_key_span` processes at most 128 affected keys and 256 suffix queries
per score tile. It represents each key tile once, then reuses it across query
tiles and the query heads sharing that KV head. Decoded queries and represented
keys are converted to FP32 for the dot products. This uses quantized vector
values with floating-point arithmetic. The temporary dot tensor has dimensions
suffix query, KV head, query within the GQA group, and affected key.

Each layer keeps a suffix-by-query-head accumulator. Max starts at negative
infinity to preserve negative scores, while sum starts at zero. Removed and
replacement spans fold into the same accumulator before the head reduction.
The layer result is reduced across tensor-parallel ranks using the selected
head operation, then folded into the suffix-length layer accumulator. Sorting
layer names makes collective order consistent across ranks. Replicated KV
heads still pair with each rank's distinct query heads.

The implementation visits every scoring layer before thresholding. With
suffix length S, affected-key count E, query-head count H, head dimension D
and L layers, scoring arithmetic scales as O(L S E H D). Tiling bounds the
pairwise score tensor, while the per-layer accumulator still scales as O(S H).
There is no persistent deleted-score cache in this implementation. The
opportunities described above would extend the current per-call key-tile reuse.
[Scoring and replay worker](../../vllm/v1/worker/kv_cache_edit.py).

Verification exposed different division rounding between the original eager
K conversion and the CUDA Q writer for compressed formats. Factoring the
shared CUDA writer into `_write_cuda` made K conversion use the same scale
and tie rounding. Compressed K uses temporary byte rows, then `_decode`
provides represented values for scoring. Floating-point formats cast directly.
The verification report records exact parity checks across all six formats
and FP16, BF16 and FP32 source vectors.
[Encoding implementation](../../vllm/v1/attention/ops/query_cache.py).

### Fragmented replay implementation

`_selected_runs` freezes the threshold mask after scoring every layer and
returns sorted, half-open suffix intervals to the host. Both replay schedules
use these intervals. An empty mask skips the suffix forward. A full mask
prefills one contiguous run, split at the runner's token limit.

The original worker forwarded each run separately. Its 15-run measurement
predates gathered replay. This schedule also supports edit-compatible
backends other than Triton. [Original measurements](selective_suffix_repair_verification.md#warm-edit-latency).

<a id="proposed-gathered-replay"></a>

## Gathered replay on Triton

When all editing layers use `TRITON_ATTN`, the worker gathers selected IDs
from the host token list and passes their edited absolute positions to RoPE.
Each contiguous run becomes one sequence in Triton's variable-length batch.
The runs share a single destination page table through a tensor view with zero
row stride. Projections and MLPs process the packed rows together.

For example, selecting absolute positions 2, 3, 8 and 12 produces four model
input rows and three sequence views.

| Packed rows | Edited positions | Query length | Visible prefix length at run end |
| --- | --- | ---: | ---: |
| 0, 1 | 2, 3 | 2 | 4 |
| 2 | 8 | 1 | 9 |
| 3 | 12 | 1 | 13 |

Here `query_start_loc` is `[0, 2, 3, 4]` and `seq_lens` is `[4, 9, 13]`.
Within each run, the kernel computes the causal boundary as
`seq_len - query_len + row_in_run`. Row 2 therefore reads positions 0
through 8, including skipped tokens and the fresh K/V at positions 2 and 3.
Both the tiled and split-K kernels use these per-run lengths to bound key
reads. The block table includes padding for the final key tile.
[Attention kernel](../../vllm/v1/attention/ops/triton_unified_attention.py).

Each layer writes the batch's K/V at its destination slots before attention
reads the shared cache. The Q encoder records the fresh queries at those
same slots. Earlier batches finish before later batches start. A batch
contains at most `max_num_batched_tokens` selected rows and `max_num_seqs`
runs. Tensor-parallel ranks use the smallest supported run limit across the
group. [Edit worker](../../vllm/v1/worker/kv_cache_edit.py).

The [gathered replay proposal audit](selective_suffix_repair_proposal_audit.md)
records the original design questions. The [verification commands](selective_suffix_repair_verification.md#gathered-replay-checks)
cover cache parity and a same-mask timing comparison on a T4.

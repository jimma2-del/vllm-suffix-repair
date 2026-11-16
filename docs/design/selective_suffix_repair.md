# Selective suffix repair for `LLM.edit_kv_cache`

The current editor can reuse a suffix or recompute it in full. The proposed
mode scores each suffix token in each attention layer, then recomputes all
heads for selected tokens. The goal is to reduce edit latency while retaining
the complete causal context. [Existing edit worker](../../vllm/v1/worker/kv_cache_edit.py)
and [implementation plan](selective_suffix_repair_plan.md).

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
length-changing edits, the proposed position convention scores removed keys
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

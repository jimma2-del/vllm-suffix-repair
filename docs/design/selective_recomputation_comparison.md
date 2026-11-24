# Selective recomputation compared with CacheBlend and ProphetKV

Reviewed November 24, 2026 against local commit
`cb21782115cbb0f62250444fb7640f20303684a5` and the working tree. This is source
inspection and a review of recorded results. No new GPU measurements were run.

The main execution difference is batching. This repository runs the complete
model separately for each contiguous selected span. CacheBlend packs selected
rows and advances them together through each layer. Scattered logical
positions can become contiguous tensors for GPU computation.

## Local execution and measured costs

The [worker](../../vllm/v1/worker/kv_cache_edit.py) prefills the replacement,
copies and relocates suffix Q/K/V, scores cached suffix Q against affected K,
and reduces across keys, heads, and layers. It freezes one mask for replay.
`_selected_runs` converts the GPU mask into host intervals. The worker calls
`_prefill` separately for each interval. Each call builds metadata and invokes
the complete model with `skip_compiled=True`.

The selected tokens see the entire causal prefix, including skipped cached
rows and fresh K/V from earlier selected tokens. Retained skipped K/V is the
approximation. This does not reproduce full recomputation in general.

Selection is a separate expense. The scorer loops over every layer and tiles
of 256 suffix queries and 128 affected keys. It decodes Q, represents K in the
Q format, and computes FP32 dot products. Arithmetic scales as O(L S E H D),
for layers L, suffix length S, affected keys E, query heads H, and dimension D.

The [recorded benchmark](selective_suffix_repair_verification.md#warm-edit-latency)
used Qwen2.5-1.5B on an RTX 3060, with an 85-token source and 68-token suffix:

| Mode | Replayed tokens | Runs | Recorded median |
| --- | ---: | ---: | ---: |
| Reuse | 0 | 0 | 39.7 ms |
| Full replay | 68 | 1 | 78.3 ms |
| Selective, sparsest | 1 | 1 | 95.2 ms |
| Selective, fragmented | 48 | 15 | 453.5 ms |

These are complete edit calls. Selected count differs between masks, so the
experiment does not isolate fragmentation. Repeated forward setup, small
matrix operations, and repeated model-weight reads are plausible costs of the
run loop. Profiling is needed to attribute the measured latency.

## CacheBlend source evidence

[Current LMCache documentation](https://github.com/LMCache/LMCache/blob/883d2126948538431a48297fee0d19c479b2dfb6/docs/source/kv_cache_optimizations/cacheblend.rst#L22-L25)
points legacy in-process users to LMCache_inprocess. Its inspected snapshot is
`0aebefd026f5c479d3f6bcbee0d5facd1b2856ec`.

The [blender](https://github.com/LMCache/LMCache_inprocess/blob/0aebefd026f5c479d3f6bcbee0d5facd1b2856ec/lmcache/inprocess/v1/compute/blend/blender.py#L92-L122)
selects sorted top-k rows by squared K deviation, gathers Q/K/V and residuals,
retains RoPE positions, and scatters new K/V into full cached tensors.
Attention receives short Q and long K/V. The
[model loop](https://github.com/LMCache/LMCache_inprocess/blob/0aebefd026f5c479d3f6bcbee0d5facd1b2856ec/lmcache/inprocess/v1/compute/models/base.py#L97-L156)
runs attention, output projection, and MLP on packed rows per layer. Selection
still pays preceding-layer work and check-layer projections for its candidates.

The [paper, sections 4.3 and 5](https://arxiv.org/html/2405.16444v3#S4.SS3)
describes shrinking candidate sets using deviation and overlapping layer
recomputation with next-layer KV loading. Loading can hide computation when
timings permit. Its speedups include this storage pipeline. A resident-cache
edit has a different cost structure.

### Causal-mask qualification

The inspected [FlashAttention call](https://github.com/LMCache/LMCache_inprocess/blob/0aebefd026f5c479d3f6bcbee0d5facd1b2856ec/lmcache/inprocess/v1/compute/attention/flash_attn.py#L80-L101)
receives compact Q, full K/V, and `causal=True`, without selected positions.
The [metadata update](https://github.com/LMCache/LMCache_inprocess/blob/0aebefd026f5c479d3f6bcbee0d5facd1b2856ec/lmcache/inprocess/v1/compute/attention/metadata.py#L31-L36)
changes Q length only. This cannot encode arbitrary per-query boundaries.
[FlashAttention's documented mask](https://github.com/Dao-AILab/flash-attention#21-change-behavior-of-causal-flag)
aligns to the bottom right when Q and K lengths differ.

Optional [Triton sparse metadata](https://github.com/LMCache/LMCache_inprocess/blob/0aebefd026f5c479d3f6bcbee0d5facd1b2856ec/lmcache/inprocess/v1/compute/attention/metadata.py#L150-L184)
samples selected positions in groups of 32 to build a block mask. This is
coarse, rather than an exact causal boundary for every query. Packing is
established by the source. Exact arbitrary-position causality is a separate
requirement.

## ProphetKV source evidence

[Section 4.3](https://arxiv.org/html/2602.02579v3#S4.SS3) runs the short user
query against reused document K/V across layers, fuses attention scores, and
selects a fixed top-k set. Stage II recomputes that set.
[Appendix F](https://arxiv.org/html/2602.02579v3#A6) describes pruning hidden
states, positions, and the mask after embedding, with selected indices passed
for KV replacement. [Section 5.1](https://arxiv.org/html/2602.02579v3#S5.SS1)
uses Hugging Face baselines and separates kernel optimization. No authors'
code release was located, so exact dispatch and indexing remain unverified.

## Applying packed replay here

For a fixed selected set, this is an alternative schedule of the local
mixed-cache computation, up to floating-point rounding:

```text
positions = [2, 3, 8, 12]
hidden = embed(token_ids[positions])
for each layer:
    q, k, v = project(hidden)
    apply RoPE using positions
    scatter k, v into destination slots at positions
    output[i] = attention(q[i], full cached K/V, key_position <= positions[i])
    hidden = output projection, residual operations, and MLP
```

This equivalence is a derivation from the local computation, not a measured
implementation. At one layer, Q/K/V depend on each token's previous-layer
hidden state. All selected K/V can be produced before attention. Attention
does not need another token's same-layer attention output. A causal mask
excludes future K/V. Fixed active rows need no persistent hidden-state cache.

With M active rows and N cached tokens, attention considers roughly O(M N)
query-key pairs, subject to causal bounds. Projections and MLPs process M
rows. Gather and scatter move vectors without forwarding each fragment
separately. KV reads, selection, and small-batch utilization still limit gains.

The local [Triton kernel](../../vllm/v1/attention/ops/triton_unified_attention.py)
derives query positions as `seq_len - query_len + row`. RoPE positions and
write slots do not change that mask. Gathered replay needs actual query
positions in attention and appropriate tile bounds. The existing
[gathered replay proposal](selective_suffix_repair.md#proposed-gathered-replay)
already records this requirement. It remains unimplemented and unmeasured.

The next useful experiment compares identical masks under run and gathered
execution. Hold selected count constant while varying run count. Measure
selection and replay separately on longer suffixes. Validate against the
existing token-at-a-time oracle, including shifted positions, page crossings,
skipped-row preservation, and multiple gathered batches.

# Gathered replay proposal audit

Reviewed November 24, 2026. The gathered replay proposal covers the core
algorithm and its main correctness requirements. It still needs concrete
backend, execution, and measurement decisions before it is an implementation
plan for faster complete edit calls. This audit uses source inspection and
existing results. No new GPU measurements were run.

The [CacheBlend and ProphetKV comparison](selective_recomputation_comparison.md)
explains packed execution, attention-mask qualifications, and the differences
between those methods and this repository. The
[original proposal](selective_suffix_repair.md#proposed-gathered-replay)
describes the intended gathered replay. The
[verification report](selective_suffix_repair_verification.md#warm-edit-latency)
contains the existing latency measurements.

## What the proposal already covers

The proposal gathers selected token IDs and edited absolute positions in
ascending order. Each layer computes fresh Q/K/V for those tokens, writes K/V
to their destination slots, and records Q in the configured format. Attention
reads reused K/V from skipped tokens and fresh K/V from earlier selected
positions. All K/V writes for a batch must complete before attention reads it.

Each query needs its own absolute causal boundary. Passing positions to RoPE
alone does not supply that boundary. Large selections can use ascending
batches, so later batches see refreshed K/V from preceding batches. The
proposed checks include skipped-row preservation, shifted positions, page
crossings, and multiple gathered batches.

The proposed benchmark compares identical masks across execution methods and
varies run count at a fixed selected-token count. These requirements establish
the basic computation and a way to isolate fragmentation costs.

## Remaining implementation decisions

### Carry positions through one supported attention backend

The first implementation needs an explicit backend choice and a path for
selected positions through its metadata and kernel arguments.
`CommonAttentionMetadata.positions` in the
[attention interface](../../vllm/v1/attention/backend.py) is already supplied
by [metadata construction](../../vllm/v1/worker/gpu/attn_utils.py).
The current [Triton metadata builder](../../vllm/v1/attention/backends/triton_attn.py)
does not preserve it or pass it to attention.

The [kernel](../../vllm/v1/attention/ops/triton_unified_attention.py) infers
contiguous query positions from sequence and query lengths.
[Tile traversal](../../vllm/v1/attention/ops/triton_attention_helpers.py)
also derives the visible prefix from contiguous offsets. The causal mask must
use selected positions. Tile traversal also needs an explicit policy. For
sorted rows with a sequence length covering the last selected token, the old
full-causal bounds can conservatively include every permitted key. Correcting
the per-query mask can then preserve causality while traversing extra future
blocks. Bounds derived from selected positions can avoid that extra work.

The small-query segmented path needs the same semantics. A first prototype
can explicitly use the unsegmented path and record that limitation. Later
segmented support needs validation of traversal, reduction, and workspace
sizes. Ordinary attention must retain its existing behavior.

### Define packed rows, logical positions, and physical slots

The execution contract must distinguish these indices:

| Index | Meaning |
| --- | --- |
| Packed row | A row in the selected hidden-state and Q/K/V tensors. |
| Logical position | That row's absolute token position in the edited prompt. |
| Physical slot | The destination cache address obtained through the page table. |

For selected positions `[2, 3, 8, 12]`, packed row `i = 2` represents logical
position 8. Its output attends to cached keys at positions 0 through 8.
Unselected tokens supply cached K/V as context. They receive no new query
rows, attention outputs, projections, or MLP computation during replay.

The earlier `output[i]` pseudocode denotes a row of a batched operation.
Issuing a separate Python attention call for every `i` would repeat launch
and setup costs. The fixed selected set should remain packed across layers.

### Specify the work that attention skips

Attention should compute only selected query rows and read existing paged
K/V. It should avoid a full-sequence square attention matrix, a dense mask
allocation for every query-key pair, and a duplicated cache for every query.

Entirely future key blocks should be skipped using actual selected positions.
Some masked arithmetic inside boundary tiles is expected. Widely separated
query positions within one tile can increase that wasted arithmetic. The
implementation needs a choice of query tile size and selected-token batch
size. A small batch can underutilize the GPU; a larger batch needs more
workspace. Reading unselected causal K/V is required context.

### Preserve cache ownership and write ordering

The gathered path must preserve source pinning, private destination pages,
RoPE relocation, Q-cache encoding, and publication after successful completion.
The [edit worker](../../vllm/v1/worker/kv_cache_edit.py) and
[page planner](../../vllm/v1/core/kv_cache_edit.py) implement those rules for
the current path. Gathered execution needs verification against them.

The [attention integration](../../vllm/model_executor/layers/attention/attention.py)
already represents the dependency between K/V updates and attention, and
records Q at the supplied physical slots. Reusing that integration would
preserve its ordering mechanism.

The contract must define unique, sorted positions and ordering across batches.
Tensor-parallel ranks need identical selected positions and batch boundaries.
Supported MHA, GQA, and MQA layouts and Q formats need parity checks. Runner
and backend support should be stated explicitly for the first experiment.

### Account for host and device overhead

The implementation needs decisions on mask compaction, selected-count
handling, host synchronization, and reuse of index buffers and metadata.
The current worker converts interval boundaries to a host list and builds
new metadata per forward. The
[worker RPC](../../vllm/v1/worker/gpu_worker.py) synchronizes the device around
the edit, and the current replay uses eager execution.

Gathered execution can reduce the number of forwards while retaining other
fixed costs. Compilation and CUDA graphs are possible follow-ups. Their
benefit should be measured after the first gathered path works.

### Measure complete edit latency and preserve the selected mask

The timing breakdown needs selection, page allocation, cache copy and RoPE
relocation, metadata preparation, and replay. Device timing can isolate GPU
work, while wall-clock timing captures the complete edit call. Warm-up and
compilation costs need separate reporting.

The existing one-token selective case is slower than full replay, so reducing
fragmented replay alone does not establish an end-to-end speedup. Selection
and cache preparation can still dominate. Their optimization is a separate
research question once the timing breakdown identifies the cost.

Dispatch can retain the existing contiguous path for contiguous selections.
Recomputing gaps or falling back to full suffix replay computes additional
tokens and can change resulting cache contents. Such a policy must be explicit
and evaluated separately from faster execution of an identical mask.

## Scope and evidence needed for the first experiment

The [editor's existing restrictions](../../vllm/v1/core/kv_cache_edit.py)
already require CUDA, dense causal full attention, and unquantized K/V.
They exclude hybrid and sliding-window layouts, data and pipeline parallelism,
context parallelism, transfer, speculation, and LoRA. Tensor parallelism
remains allowed. The gathered path should preserve these restrictions.
Supporting excluded modes is outside this proposal.

A first implementation can target one supported model and Triton attention
while retaining the existing selector. Support for every backend, CUDA graphs,
and changes to selection quality are follow-up work. The initial result
should establish equivalent cache contents for the same mask and improved
fragmented replay and complete edit latency in measured workloads.

The remaining evidence includes:

- Compare identical masks with the existing run path and token-at-a-time
  oracle. Check each layer's K/V and encoded Q with the existing numerical
  tolerances. Preserve skipped rows and source cache contents exactly.
- Cover empty, singleton, and full masks; widely separated positions; partial
  tiles; page crossings; length-changing edits; and multiple gathered batches.
  Exercise normal and segmented attention when supported.
- Hold selected count fixed while varying fragmentation. Also vary context
  length, selected fraction, and batch size. Report replay and complete edit
  latency against reuse and full replay.
- Retain the existing failure-cleanup and transport checks for supported
  runners. Validate matching selection and batch plans when testing tensor
  parallelism.

No latency benefit from gathered replay has yet been measured. These decisions
and checks define what must be resolved to evaluate it reproducibly.

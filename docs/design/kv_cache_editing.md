# KV cache editing research record

This record covers the cache editor built in this chat, the change to
prefix-attending replacement prefill, and the optional full suffix replay.
It also records how the later query-cache and selective-repair work integrates
with that editor. Source and history were inspected on November 21, 2026 at
`ded40eeeb9` on `query-based-selection`. Experiment results below come from the
earlier implementation sessions and the existing verification report.

| Revision | Change |
| --- | --- |
| `58c462f282` | Introduced hash-based KV editing, an isolated replacement prefill, RoPE relocation, page allocation, copy-on-write, and edited-prefix lookup. |
| `494f2d73d2` | Made the replacement attend to the cached prefix and added `recompute_suffix=False/True`, with a separate suffix forward. Removed the isolated replacement's temporary pages. |
| `6ce2175207` | Added stored decoder queries and their precision, allocation, transport, and relocation support. |
| `951a0a2c47` | Recorded the design and verification plan for selective suffix repair. |
| `ded40eeeb9` | Added query-based selection and whole-token replay for selected suffix tokens, with its verification report. |

The initial isolated-prefill experiment remains reproducible at
`58c462f282`. The active implementation always gives replacement tokens access
to the cached prefix. The boolean suffix modes from `494f2d73d2` continue to
work in the current API. The later selection algorithm and its alternatives
are described in the [selective-repair design](selective_suffix_repair.md),
[implementation plan](selective_suffix_repair_plan.md), and
[verification report](selective_suffix_repair_verification.md).

The public method accepts the exact original token IDs, replacement text,
and a half-open token interval `[start, end)`. The replacement is tokenized
without adding special tokens. The method returns edited IDs and a fresh
cache salt in a `TokensPrompt` that can be passed directly to `generate`.
Indices refer to tokens, including any special tokens in the original prompt.
An empty replacement deletes the interval, and `start == end` inserts text.
[Public API](../../vllm/entrypoints/llm.py)

```python
from vllm import LLM, SamplingParams

llm = LLM(
    model="Qwen/Qwen2.5-1.5B-Instruct",
    enable_prefix_caching=True,
    gpu_memory_utilization=0.7,
)
params = SamplingParams(max_tokens=1, temperature=0)
original_ids = llm.get_tokenizer().encode(
    "The red fox walks past the old bridge.", add_special_tokens=False
)

# Prefill the exact IDs that the edit will look up.
llm.generate({"prompt_token_ids": original_ids}, params, use_tqdm=False)
edited = llm.edit_kv_cache(
    original_ids,
    " blue",
    start=1,
    end=2,
    recompute_suffix=True,
)
outputs = llm.generate(edited, params, use_tqdm=False)

# A later edit supplies the previous result's salt to identify its source KV.
chained = llm.edit_kv_cache(
    edited["prompt_token_ids"],
    " green",
    start=1,
    end=2,
    cache_salt=edited["cache_salt"],
    recompute_suffix=False,
)
```

The example chooses indices to demonstrate the interface. The tokenizer's
IDs determine which text those indices cover. A caller editing by character
or word boundaries must first map them to the original token sequence.

| Suffix mode | Replacement computation | Suffix computation |
| --- | --- | --- |
| `False`, the default | Causal prefill at final positions using the cached prefix. | Copy source KV and correct RoPE on keys whose positions move. Values retain the source context. |
| `True` | The same replacement pass. | A separate causal prefill after the replacement completes, using the prefix, replacement, and preceding suffix tokens. |
| `"selective"` | The same replacement pass. | Score cached queries against affected keys, copy and relocate the suffix, then replay the selected contiguous runs in causal order. |

The default `False` preserves the earlier suffix-reuse behavior while allowing
replacement tokens to attend to the prefix. The separate suffix call makes
the full replay independently switchable. Current selective calls require an
explicit non-NaN `threshold` and stored Q. Their key, head, and layer reductions
independently accept `"max"` or `"sum"`, with `"max"` as the default.
`include_deleted=True` includes removed keys in scoring.
[Selection options](../../vllm/v1/engine/__init__.py)

Let `N` be the original token count, `M` the replacement token count, and
`B` the physical page size. The edited sequence has length
`N - (end - start) + M`. Its suffix begins at `start + M`, and each reused
suffix token moves by `M - (end - start)`.

The implementation performs the following steps.

1. Construct a lightweight `Request` from the original IDs and source salt.
   Reuse the engine's hash function and hash seed. At full-page boundaries,
   use the request's existing chained block hashes. At a partial final
   boundary, chain the preceding full-page hash with the valid tail IDs
   and normal extra hash keys. Look up every source page in cache group 0.
   A missing page raises an error identifying its token interval.
   [Hashing and lookup](../../vllm/v1/core/kv_cache_edit.py)
2. Compute the destination page table. Share complete prefix pages. If the
   edit begins inside a page, allocate a private page containing that prefix
   fragment. A length-preserving edit with suffix reuse can share complete
   suffix pages beyond its last write. A length-changing edit or full suffix
   replay needs private pages through the edited suffix. Selective replay
   also reserves those private pages before selection.
3. Temporarily pin the source with `BlockPool.touch` before allocating
   destination pages. Every changed page receives a private physical page,
   including when its source initially has `ref_cnt == 0`. Existing readers
   keep their original page tables, and the edit builds its own destination
   table. The allocator raises if the pinned source and destination cannot
   fit together. [Pool ownership](../../vllm/v1/core/block_pool.py)
4. Synchronize the device and validate the bound layer layouts. Copy the
   surviving prefix fragment into the first private page for every layer
   before running the model. Shared whole prefix pages already contain their
   KV. The worker builds a causal attention table from the destination pages.
   [Worker RPC](../../vllm/v1/worker/gpu_worker.py)
5. Prefill the replacement at absolute positions beginning at `start`.
   Sequence lengths include the cached prefix, while query lengths count
   only the current replacement chunk. The page table exposes the prefix
   and the replacement written so far. Chunking follows
   `max_num_batched_tokens`. Kernel block IDs are derived from manager page
   IDs when the attention backend subdivides a physical page.
   [Segment prefill](../../vllm/v1/worker/kv_cache_edit.py)
6. In reuse mode, copy suffix rows from `end` to `start + M` and relocate
   keys as needed. In full replay mode, call the segment prefill again with
   the suffix IDs and its new start position. This second pass can attend
   to the entire edited causal prefix. Reused suffix copying gathers at most
   1,024 rows at a time to bound temporary tensor storage.
7. Synchronize before the worker RPC returns. Only then publish the edited
   destination hashes. Release the edit's allocated-page and source-pin
   references in a `finally` block. Cache entries remain lookup candidates
   subject to the pool's usual eviction policy.

For example, with 16-token pages, a 95-token source, and a replacement of
`[19, 22)` by 70 tokens, the destination has 162 tokens. The suffix starts
at 89 and shifts right by 67. The edit shares the first page, copies original
positions 16 through 18 into the first private page, prefills the replacement
at positions 19 through 88, and copies or recomputes the suffix at positions
89 through 161.

The earlier isolated implementation prefetched replacement KV into separate
temporary pages at positions starting at zero. It then copied those rows into
the destination and corrected their RoPE. Prefix-attending prefill writes
directly into the destination at final positions, so its replacement and
recomputed suffix already have the intended RoPE. This removed the separate
`scratch` page table and its allocation from `KVCacheEdit`.

A reused key at old position `p` is relocated to position `q` with
`R(q) R(p)^-1 K_cached`. The implementation reads the layer's actual sine and
cosine table and forms the relative transform.

```text
denominator = cos(p)^2 + sin(p)^2
relative_cos = (cos(q) cos(p) + sin(q) sin(p)) / denominator
relative_sin = (sin(q) cos(p) - cos(q) sin(p)) / denominator
```

The denominator accounts for scaled or rounded table entries. Calculation
uses FP32, followed by a copy into the cache dtype. Only the first
`rotary_dim` key components are transformed, with the embedding's NeoX or
interleaved pairing. Remaining key components and values are copied. Equal
source and destination positions bypass the transform, preserving all copied
bits. RoPE relocation changes a reused key's positional representation while
its content and corresponding value retain the source computation.
[Key relocation](../../vllm/v1/worker/kv_cache_edit.py)

**Partial-page hashing is required for arbitrary edit boundaries.** Dense
attention normally finds complete prefix pages. The editor additionally
records prompt-end and confirmed decode-end partial boundaries. When decode
fills a formerly partial page, the old partial hash is retained as an alias
alongside the new full-page hash. Eviction and reset remove the aliases with
the page. This allows a 39-token source to remain findable after decode
extends its third page. The manager subtracts in-flight tokens when recording
confirmed decode tails. [Manager integration](../../vllm/v1/core/kv_cache_manager.py)
and [hash aliases](../../vllm/v1/core/block_pool.py)

**Every edit receives a fresh salt.** Identical token IDs can describe caches
with different histories, particularly after suffix reuse. The salt
`kv-edit-` followed by a UUID enters the normal prefix hash chain and selects
the particular edited result. Whole shared pages can acquire aliases under
the edited hash chain. The caller carries the salt in subsequent generation
and chained-edit requests. Omitting it loses the identity of that cache
version.

**Generation still evaluates a prompt token to obtain logits.** In addition
to the actual final cache boundary, an edit publishes the boundary one token
before the end when that falls inside a page. The manager recognizes the edit
salt and probes partial boundaries after its complete-page hit. Resending an
edited prompt can therefore reuse `length - 1` tokens. Appending a continuation
token can reuse the entire edited prompt. The existing partial-hit
copy-on-write path gives generation a writable tail page before it resumes.

**Reference release returns pages to the allocator's normal lifecycle.**
Shrinking the destination reduces its page-table length. Releasing the edit's
references makes private cached pages evictable, and releases any temporary
allocation on a failed worker call. Source pages held by other requests keep
their references. Source and old-result pages can coexist as ordinary cache
entries until eviction. Source immutability during the edit does not promise
indefinite residency under its original hash.

Allocation failure and an injected worker failure were tested to restore
source reference counts and avoid publishing edited destinations. Worker
writes target private pages, and device synchronization completes outstanding
reads and writes before those pages can be recycled. This guarantee covers
the tested allocation and worker failure paths. Publication consists of
sequential hash insertions after the RPC.

The original editor introduced or modified the following files in
`58c462f282`. The second column records the additional changes in
`494f2d73d2`.

| File | Initial implementation | Prefix-attending and suffix-toggle change |
| --- | --- | --- |
| [`vllm/entrypoints/llm.py`](../../vllm/entrypoints/llm.py) | Added `TokensPrompt` import and public `edit_kv_cache`, token-range and string checks, replacement tokenization, and the returned salt. | Added `recompute_suffix=False`, forwarded it, and updated the behavior documentation. |
| [`vllm/v1/engine/core_client.py`](../../vllm/v1/engine/core_client.py) | Added the interface, direct in-process call, and synchronous multiprocess utility call. | Carried the suffix option through both clients. |
| [`vllm/v1/engine/core.py`](../../vllm/v1/engine/core.py) | Initialized the editor after hash setup, checked model length and replacement vocabulary, and dispatched worker RPCs. | Forwarded the suffix option to page planning. |
| [`vllm/v1/core/kv_cache_edit.py`](../../vllm/v1/core/kv_cache_edit.py) | Created `KVCacheEdit` and `KVCacheEditor` for source lookup, ownership, temporary pages, destination tables, hashing, and cleanup. | Removed temporary replacement pages, added optional suffix IDs, and allocated private suffix pages for full replay. |
| [`vllm/v1/core/block_pool.py`](../../vllm/v1/core/block_pool.py) | Preserved dense partial-page aliases on promotion and added alias insertion and storage-event reporting. | Continued to supply the same ownership and hashing operations. |
| [`vllm/v1/core/kv_cache_manager.py`](../../vllm/v1/core/kv_cache_manager.py) | Stored the editor, recorded prompt and confirmed decode tails, extended edited-prefix hits, and updated partial-hit documentation. | Continued to connect edits to normal generation. |
| [`vllm/v1/worker/gpu_worker.py`](../../vllm/v1/worker/gpu_worker.py) | Added the edit RPC, active model configuration, and device synchronization before and after edits. | Used the same RPC for both suffix modes. |
| [`vllm/v1/worker/kv_cache_edit.py`](../../vllm/v1/worker/kv_cache_edit.py) | Created isolated prefill, tensor-layout validation, paged copies, and RoPE relocation. | Seeded partial-prefix KV first, added absolute-position segment prefill, and dispatched replacement and suffix as separate passes. |
| [`suffix_repair_test/test_kv_cache_edit.py`](../../suffix_repair_test/test_kv_cache_edit.py) | Added ownership, allocation, failure, alias, boundary, and RoPE tests. | Added full-suffix privatization checks and updated equal-length edit expectations. |
| [`suffix_repair_test/kv_cache_edit_gpu.py`](../../suffix_repair_test/kv_cache_edit_gpu.py) | Created a per-layer GPU comparison, referenced-source check, extension-hit check, and chained-edit experiment. | Compared replacement/full replay with ordinary generation at matching chunk boundaries, traced separate forwards, and tested both suffix modes. |
| [`suffix_repair_test/scratch/check_vllm_edit_mp.py`](../../suffix_repair_test/scratch/check_vllm_edit_mp.py) | Retained the multiprocess smoke test initially developed as `/tmp/check_vllm_edit_mp.py`. | Tested both modes through compilation and CUDA graphs, with full replay's greedy result compared to ordinary generation. |

The first test suite had five tests. The prefix-attending revision had six.
Their checks cover changed-page privatization, unchanged-page sharing,
insertions, deletions, truncation, empty results, no-op edits, failures,
alias promotion and eviction, both rotary pairings, partial rotary dimensions,
and noncontiguous physical pages. The recomputation test includes an empty
edit interval that explicitly requests a fresh suffix. The GPU experiment
also repairs a previously reused suffix without changing its token IDs.

The foundational GPU checks used the following environment.

| Setting | Value |
| --- | --- |
| GPU | GeForce RTX 3060 with 12 GiB |
| Model | Qwen2.5-1.5B-Instruct |
| Local model revision | `989aa7980e4cf806f80c7fef2b1adb7bc71aa306` |
| Model architecture | 28 layers, 12 query heads, two KV heads |
| Python environment | Project `.venv`, Python 3.13.13 |
| PyTorch | `2.14.0+cu132` |
| Model/cache precision | FP16 |
| Detailed tensor check | Eager execution with `TRITON_ATTN` |
| Page size | 16 tokens |
| Prefill budget | 64 tokens per forward |
| Maximum model length | 256 |
| KV allocation at `494f2d73d2` | 32 MiB |
| GPU memory utilization | 0.7 |

| Experiment | Recorded outcome |
| --- | --- |
| First isolated-prefill check | All 28 layers passed for 39 to 56 to 27 tokens. Replacement values matched standalone prefill, and shifted keys matched an independent complex-number RoPE reference. |
| Longer isolated-prefill check | All 28 layers passed for 39 to 106 to 77 tokens. The 70-token replacement crossed the 64-token chunk boundary. |
| Prefix-attending replacement and full replay | All 28 layers passed for 95 to 162 to 133 tokens. The replacement had 70 tokens and the first edited suffix had 73, so both passes crossed a chunk boundary. |
| Suffix reuse | Values matched source values exactly. Shifted keys matched the independent RoPE calculation with `atol=0.005` and `rtol=0.002`. |
| Source readers | The test held references to every source page through both edit modes and verified exact source tensor preservation. |
| Full replay reference | KV matched ordinary generation exactly with identical cached prefix and FP16 forward shapes. |
| Forward trace | Reuse executed replacement positions only. Full replay began a distinct suffix forward at `start + replacement_length`. |
| Prefix hits | Appended continuation requests reused every edited token. Direct prompt resend reused all but the final token. |
| Runner coverage | The detailed checks passed with V1 and V2 GPU runners. |
| Compiled multiprocess client | Both boolean modes returned a 36-token cache hit in the smoke test. Full replay's greedy continuation matched ordinary generation. |
| Static checks | Ruff 0.14.0 lint and formatting, Python compilation, and `git diff --check` passed for the implementation changes. |

**Forward shape matters when comparing FP16 results.** An initial comparison
used different chunk boundaries for ordinary prefill and segmented replay.
At layer 8 it failed a tolerance of 0.01 for two of 73,216 elements, with a
maximum absolute difference of 0.017578125. The revised oracle used ordinary
generation with the same cached-prefix boundary and the same replacement and
suffix chunk shapes. That comparison passed with zero tolerance across all
layers. Exact equality in that experiment is conditional on those shapes.

**The local GPU required a lower startup memory reservation.** The first
default run requested 92% of 12 GiB while only about 10.93 GiB was free and
failed before model execution. The focused experiments set utilization to
0.7 and an explicit KV allocation. Host GPU access was also required when
the execution sandbox blocked NVIDIA device access.

The later query-cache work stores post-normalization, post-RoPE decoder Q
alongside K/V in the same physical pages. Attention reads the K/V portion.
Query storage follows page ownership, copying, offload, eviction, and reuse.
The active edit copies cached Q for surviving prefix/suffix rows, relocates
reused Q when positions shift, and writes fresh Q during model prefill.
`q_cache_dtype` defaults to `fp16`; supported formats also include `fp32`,
`bf16`, `fp8`, `int8`, `int4`, and `none`.
[Query-cache storage and capacity](../features/quantization/quantized_kvcache.md)

The memory cost depends on query head count. For one layer of this Qwen model
with FP16 K/V, K/V uses 1,024 bytes per token, FP16 Q adds 3,072 bytes, and
packed INT4 Q adds 1,024 bytes after page packing. Current reproduction scripts
therefore reserve more memory for the detailed GPU test. The query-cache
commit touched the following additional areas.

| Area | Files |
| --- | --- |
| Options and storage | [Cache configuration](../../vllm/config/cache.py), [engine arguments](../../vllm/engine/arg_utils.py), [query encoding and reads](../../vllm/v1/attention/ops/query_cache.py), [page specifications](../../vllm/v1/kv_cache_interface.py) |
| Attention and allocation | [Attention layer](../../vllm/model_executor/layers/attention/attention.py), [worker binding and copying](../../vllm/v1/worker/utils.py), [V2 GPU attention setup](../../vllm/v1/worker/gpu/attn_utils.py), [V1 GPU runner](../../vllm/v1/worker/gpu_model_runner.py), [CPU runner](../../vllm/v1/worker/cpu_model_runner.py), [Flint integration](../../vllm/models/flint/model.py) |
| Full-page transport | [Layer transfer](../../vllm/model_executor/layers/attention/kv_transfer_utils.py), [example connector](../../vllm/distributed/kv_transfer/kv_connector/v1/example_connector.py), [LMCache adapter](../../vllm/distributed/kv_transfer/kv_connector/v1/lmcache_integration/vllm_v1_adapter.py), [Mooncake connector](../../vllm/distributed/kv_transfer/kv_connector/v1/mooncake/mooncake_connector.py), [Mooncake store](../../vllm/distributed/kv_transfer/kv_connector/v1/mooncake/store/worker.py), [NIXL worker](../../vllm/distributed/kv_transfer/kv_connector/v1/nixl/base_worker.py), [NIXL metadata](../../vllm/distributed/kv_transfer/kv_connector/v1/nixl/metadata.py) |
| Offload compatibility | [Connector offload configuration](../../vllm/distributed/kv_transfer/kv_connector/v1/offloading/config.py), [offload configuration](../../vllm/v1/kv_offload/config.py), [file mapping](../../vllm/v1/kv_offload/file_mapper.py) |
| Tests and documentation | [Query-cache tests](../../suffix_repair_test/test_query_cache.py), [write benchmark](../../suffix_repair_test/bench_query_cache.py), the extended GPU edit test, and the query-storage documentation linked above |

Selective repair adds `ScoreReduction` and `SuffixSelection` to the engine
types and extends the public API, synchronous clients, editor plan, and worker.
It uses raw signed Q dot K scores, with FP32 accumulation. Reductions occur
over affected keys, query heads, and layers in that order. Every GQA query
head uses its corresponding KV head. Removed keys are scored at the original
query positions and replacement keys at the edited positions. Key encoding
uses the Q cache format and its writer's rounding behavior. A strict
`score > threshold` test freezes the whole-token replay mask.

Selected tokens run through every layer at model precision. Adjacent selected
positions form a single segment forward. Each run can read intervening reused
KV and fresh KV written by earlier runs. Source queries remain fixed while
selection is calculated. Proposed layer-wise or individual-head replay and
deleted-score caching are research alternatives described in the design.
The added verification files are
[`selective_reference.py`](../../suffix_repair_test/selective_reference.py),
[`test_selective_suffix.py`](../../suffix_repair_test/test_selective_suffix.py),
and [`verify_selective_edit.py`](../../suffix_repair_test/verify_selective_edit.py).

The November 16 selective-repair report records 20 passing cache/scoring
tests, two-rank Gloo checks, and 18 single-GPU attention-layout/Q-format
combinations across MHA, GQA, and MQA. It also records synchronous client
coverage on both runners. End-to-end multi-GPU inference and the two-GPU
NCCL comparison still require additional hardware.
[Detailed evidence](selective_suffix_repair_verification.md)

Those checks exposed a rounding mismatch between the K scorer and the CUDA Q
writer for FP8, INT8, and INT4. The scorer now uses the same CUDA writer to
encode K. The report records exact agreement with the writer and an independent
format reference for FP16, BF16, and FP32 input vectors. Exact threshold ties,
negative scores, zero vectors, all eight reduction combinations, and positive
and negative infinity thresholds are covered by the selector checks.

Its latency experiment edited a pinned 85-token source with a 68-token suffix,
using V2 in-process execution, FP16 Q, max reductions, and deleted-key
scoring. Median time over seven warm edit calls was 39.7 ms for reuse and
78.3 ms for full replay. Selective replay took 95.2 ms for one selected token,
119.8 ms for 67 selected tokens in two runs, and 453.5 ms for 48 selected
tokens in 15 runs. On that prompt, selection overhead and fragmented forwards
made selective replay slower than full replay.
[Measurement procedure and reproduction](selective_suffix_repair_verification.md)

The edit path currently requires local CUDA prefix caching, one exact
`FullAttentionSpec` group, causal unquantized K/V, and equality between the
physical page and hash block size. Worker validation requires a supported
one-dimensional `RotaryEmbedding`, an ordinary four-dimensional K/V view,
matching head dimensions, FP16/BF16/FP32 cache storage, and compatible kernel
page geometry. The implementation expects the embedding on the attention
module's parent. LoRA, speculative decoding, KV transfer during editing,
and data, pipeline, decode-context, or prefill-context parallelism are excluded
by the configuration checks. The original token IDs must describe resident
text KV with the correct salt.

Tensor parallelism is permitted by the configuration. The foundational
experiments established single-GPU behavior. The later selector coordinates
head reductions across TP ranks, and its report records two-rank CPU
collective checks. Multi-GPU model inference remains an outstanding
verification item. Query-cache support for external transports is broader
than the editor's local-cache configuration.

A cache hit is required for every original page, including its partial tail.
A freshly sampled output token acquires KV only when the model processes it.
The cache may evict old edited results or their source pages after their
references are released. Deleting every token returns an empty token prompt
with zero pages. Subsequent generation must supply a usable model input.

Full replay uses the cached prefix already supplied to the edit. For a
chained edit whose prefix contains previously reused rows, replay inherits
that prefix's history. The full-prefill comparison above began with an exact
cached prefix. RoPE correction and suffix reuse intentionally retain the
source suffix computation, and the salt keeps that version identifiable.

The following commands reproduce the existing checks from the repository
root with the project environment and host GPU access. For the foundational
results, use source revision `494f2d73d2`, whose detailed GPU script reserves
32 MiB and has no Q-cache option. At `ded40eeeb9`, that script reserves
128 MiB and accepts `--q-cache-dtype`. The current version runs with FP16 Q
by default. Setting `--q-cache-dtype none` exercises the K/V-only path.

```bash
model_dir=/home/jim/.cache/huggingface/hub/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/989aa7980e4cf806f80c7fef2b1adb7bc71aa306

.venv/bin/python -m unittest -v suffix_repair_test.test_kv_cache_edit

HF_HUB_OFFLINE=1 VLLM_LOGGING_LEVEL=ERROR VLLM_USE_V2_MODEL_RUNNER=1 \
    VLLM_ENABLE_V1_MULTIPROCESSING=0 \
    .venv/bin/python -m suffix_repair_test.kv_cache_edit_gpu "$model_dir"

HF_HUB_OFFLINE=1 VLLM_LOGGING_LEVEL=ERROR VLLM_USE_V2_MODEL_RUNNER=0 \
    VLLM_ENABLE_V1_MULTIPROCESSING=0 \
    .venv/bin/python -m suffix_repair_test.kv_cache_edit_gpu "$model_dir"

HF_HUB_OFFLINE=1 VLLM_LOGGING_LEVEL=ERROR VLLM_ENABLE_V1_MULTIPROCESSING=1 \
    .venv/bin/python -m suffix_repair_test.scratch.check_vllm_edit_mp "$model_dir"
```

The [selective-repair verification report](selective_suffix_repair_verification.md)
contains the commands for all query formats, reduction combinations, model
fixtures, and latency measurements. The checks during this documentation
update inspect source, history, relative file links, and embedded example
syntax. The GPU results in this record retain their original revision and
experiment context.

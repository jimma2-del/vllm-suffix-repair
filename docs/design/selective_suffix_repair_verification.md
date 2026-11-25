# Selective suffix repair verification

## Results

| Check | Result |
| --- | --- |
| Cache and scoring unit suites | 20 passed. The two-GPU NCCL test was skipped on this single-GPU host. Two-rank Gloo passed. |
| Single-GPU model checks | All 18 attention-layout and Q-format combinations passed across MHA, GQA and MQA. Each exercised the eight key, head and layer reduction combinations. |
| Synchronous engine clients | In-process and multiprocess edits passed on both GPU runners. With Q storage disabled, selective editing raised the expected error and legacy reuse succeeded. |
| Static checks | Ruff lint and formatting, Python compilation and `git diff --check` passed. |

## Environment

The November 16, 2026 offline runs used one 12 GiB GeForce RTX 3060, PyTorch 2.14.0+cu132
and Triton 3.8.0. The pretrained model was Qwen2.5-1.5B-Instruct at revision
`989aa7980e4cf806f80c7fef2b1adb7bc71aa306`, with 28 layers, 12 query heads
and two KV heads. Runs used FP16 model weights, eager execution, Triton
attention, 16-token pages, a 64-token prefill budget and 192 MiB of KV cache.
The MQA and MHA fixtures used a three-layer Qwen2 decoder with dummy weights,
four query heads and respectively one and four KV heads.

## Correctness checks

- **Scores and quantization.** The oracle computes component products
  independently, maps every query head to its KV head, and reduces over
  edited keys, heads and layers. CPU and CUDA fixtures exercise all eight
  reduction combinations, all six Q formats, nonadjacent physical pages,
  zero vectors, signed scores, exact threshold ties and both tile boundaries.
  An exact comparison exposed different division rounding in the K scorer
  and the CUDA Q writer for FP8, INT8 and INT4. K scoring now uses the same
  CUDA writer. The resulting vectors match the writer and the independent
  format reference exactly for FP16, BF16 and FP32 input vectors.
  [Score reference](../../suffix_repair_test/selective_reference.py) and
  [scorer tests](../../suffix_repair_test/test_selective_suffix.py).
- **Replay.** Positive and negative infinity thresholds are compared with
  the existing reuse and full-replay paths. The checks compare every layer's
  K/V and encoded Q. For an intermediate mask spanning a page boundary,
  ordinary generation independently evaluates one selected token at a time.
  Each request must report a cache hit for its entire preceding mixed prefix.
  Skipped rows are compared bit for bit. For each selected head vector, the
  allowed absolute error is 0.025 plus 1.5% of its reference peak magnitude.
  Compressed Q adds one quantization step to this allowance.
  [GPU verification](../../suffix_repair_test/verify_selective_edit.py).
- **Ownership and transport.** The checks cover source immutability across
  chained edits, resend and extension hits, rollback after an injected worker
  failure, and free-page exhaustion. The failure checks verify that the editor
  publishes no destination pages and restores the source reference counts.
  Two real Gloo processes exercise the
  scorer's collective with partitioned and replicated KV heads, including a
  threshold crossing visible only on rank 1. End-to-end multi-GPU inference
  and two-rank CUDA/NCCL collectives require a second GPU.

## Warm edit latency

The V2 in-process runner edited the same pinned 85-token source every time,
replacing token 16 with one token and leaving a 68-token suffix. Q storage was
FP16, all reductions used max, and deleted-key scoring was enabled. Each
entry is the median of seven complete edit calls after one warm-up per mode.
The harness rotates execution order and emits individual samples as JSON.

| Mode | Threshold | Suffix tokens replayed | Contiguous runs | Median ms |
| --- | ---: | ---: | ---: | ---: |
| Reuse | — | 0/68 | 0 | 39.7 |
| Full replay | — | 68/68 | 1 | 78.3 |
| Selective, sparsest | 47781.0449 | 1/68 | 1 | 95.2 |
| Selective, clustered | 1902.2509 | 67/68 | 2 | 119.8 |
| Selective, fragmented | 4776.2905 | 48/68 | 15 | 453.5 |

Full replay was faster than every selective case on this prompt. The
fragmented mask incurred 15 separate suffix forwards. The measurements
include selection, cache preparation and replay together.

Thresholds were chosen between adjacent distinct observed scores. The
fragmented case maximized run count among masks containing a run that crossed
a page boundary. The clustered case minimized run count among masks selecting
at least three tokens, breaking ties by proximity to the fragmented token
count. The sparsest case used the highest candidate threshold. Consequently,
the clustered and fragmented cases selected different token counts, so this
comparison does not isolate fragmentation alone. The source IDs were
100 through 184, and the replacement text was ` blue`.
[Threshold selection](../../suffix_repair_test/verify_selective_edit.py).

The [architecture and gathered replay proposal](selective_suffix_repair.md#proposed-gathered-replay)
records the follow-up optimization discussion.

## Reproduction

Set `model_dir` to the local directory for the revision above. Run from the
repository root with the project environment and host GPU access.

```bash
export OMP_NUM_THREADS=1 HF_HUB_OFFLINE=1 VLLM_LOGGING_LEVEL=ERROR
.venv/bin/python -m unittest -v suffix_repair_test.test_kv_cache_edit \
    suffix_repair_test.test_query_cache suffix_repair_test.test_selective_suffix
VLLM_USE_V2_MODEL_RUNNER=1 VLLM_ENABLE_V1_MULTIPROCESSING=0 \
    python -m suffix_repair_test.kv_cache_edit_gpu "$model_dir"

verify() {
    VLLM_USE_V2_MODEL_RUNNER="$1" VLLM_ENABLE_V1_MULTIPROCESSING="$2" \
        python -m suffix_repair_test.verify_selective_edit \
        "$model_dir" "${@:3}"
}

verify 1 0 --q-format fp16 --repeats 7
verify 0 0 --q-format int4
for fmt in fp32 bf16 fp8 int8; do verify 1 0 --q-format "$fmt"; done
verify 1 1 --q-format fp16
verify 0 1 --q-format int4
verify 1 1 --q-format none
for heads in 1 4; do
    for fmt in fp16 int4 fp8 int8 bf16 fp32; do
        verify 1 0 --fixture-kv-heads "$heads" --q-format "$fmt"
    done
done
```

On a two-GPU host, the unit suite also enables the NCCL rank-mask comparison.
Run `verify 1 1 --tp 2` for the pretrained model and
`verify 1 1 --tp 2 --fixture-kv-heads 1` for replicated KV in a model forward.

## Gathered replay checks

On November 24, 2026, all 20 CPU tests across the replay, scoring, Q cache
and editor suites passed with PyTorch 2.14.0+cpu and Python 3.12. From the
repository root, run the suites with the project environment.

```bash
OMP_NUM_THREADS=1 .venv/bin/python -m unittest -v \
    suffix_repair_test.test_gathered_replay suffix_repair_test.test_selective_suffix \
    suffix_repair_test.test_query_cache suffix_repair_test.test_kv_cache_edit
```

The CPU checks build Triton metadata for fragmented, contiguous, singleton
and empty masks. They verify gathered IDs, absolute positions, physical
slots, causal prefixes and both batch limits. A two-layer uniform-attention
fixture compares gathered cache writes with serial token replay. Two Gloo
ranks check that every rank uses the same batches, including when only one
rank supports gathering. [Replay tests](../../suffix_repair_test/test_gathered_replay.py).

In a T4 or A100 Colab runtime, use `python` for the commands above.
On a T4 the same suite also compares paged Triton attention against dense
FP32 prefixes for MHA, GQA and MQA. It exercises short queries that fit the
split-K workspace and a larger batch that selects the 2D kernel. For the full
model check, use a CUDA build with SM75 support. The fixture uses FP16 weights
and a 192 MiB KV cache. Set `model_dir` to the local Qwen2.5 model described above.

```bash
bash suffix_repair_test/run_gathered_t4.sh "$model_dir"
```

The model check compares each layer's K/V and stored Q with separate runs
and the token-at-a-time oracle for the same seven-run mask after a contraction.
It checks skipped rows and the source cache exactly. With `--repeats`, the
`FIXED_MASK` records compare 16 selected tokens in 1, 4 and 16 runs. Their
mean absolute position is fixed, so the total number of causal keys is also
fixed. Each edit pays the real scoring cost before the harness substitutes
the specified mask. Samples report wall time for the complete edit and CUDA
event intervals for selection and suffix replay. Each schedule warms up once
and their measurement order alternates. GPU correctness and latency for this
implementation await a T4 run. [Model verification and benchmark](../../suffix_repair_test/verify_selective_edit.py).

The same command reports threshold and top-k edit wall times in `LATENCY`.
The script checks that each pair replays the same token positions before
timing. `SELECTOR_LATENCY` reports wall time in microseconds to turn
precomputed FP32 scores into replay ranges on the host. The script compares
both policies on the same tie-free scores for 68, 4,096, and 32,768 tokens,
with a top-k budget of about 5%. The interval starts after scoring and ends
before replay.

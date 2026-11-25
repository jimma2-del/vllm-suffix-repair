# Run the KVEdit benchmark

Run the benchmark from the repository root with this checkout of vLLM
installed. The default dataset is KVEdit's pinned development split,
`pilot_pairs.jsonl`. Each pair includes a direct edit and a derived edit. [KVEdit dataset and format](https://github.com/mingyangmao/KVEdit#data-files)

## Colab

On an A100 runtime, use Colab's active Python. Start with two pairs to check
that the model and cache fit. Select an output directory on mounted Drive
to keep results after the runtime shuts down.

```bash
python -m suffix_repair_test.bench_kvedit --limit 2 \
  --output-dir /content/drive/MyDrive/kvedit-a100-smoke
```

The default model is Qwen3-8B. On a T4, use the AWQ checkpoint to fit the model
and cache. [Qwen3-8B-AWQ model card](https://huggingface.co/Qwen/Qwen3-8B-AWQ)

```bash
python -m suffix_repair_test.bench_kvedit --limit 2 \
  --model Qwen/Qwen3-8B-AWQ --dtype float16 --prompt-style qwen \
  --output-dir /content/drive/MyDrive/kvedit-t4-smoke
```

Use `--dataset` with a local KVEdit JSONL file or its raw GitHub URL to
select another split. For example, use
`--dataset /content/KVEdit/data/heldout_pairs.jsonl` for the held-out split.

For a URL, the benchmark downloads the entire file to `dataset.jsonl` in the
output directory before loading the model. It applies `--offset` and `--limit`
when reading the local file. Pass that saved file to `--dataset` to reuse it.
Omit `--limit` to benchmark every pair in the file.

The default sweep tests raw selection thresholds 0.5, 32, 256, 2048, and
16384. Use `--thresholds 32 128 512` to replace the sweep. The selector
replays tokens whose scores are strictly above the threshold. Selection
density depends on the model, query-cache format, and score reductions.

Use `--top-k 32 128 512` to test fixed suffix token counts. Each count is
capped at the cached suffix length, and a count of zero skips suffix replay.
Pass `--thresholds` and `--top-k` together to compare both selection policies
in one run.

Use `--methods full reuse` to compare full recomputation and stale suffix
reuse. Use `--methods full prefix reuse selective` to include vLLM's automatic
prefix caching baseline in the sweep.
Use `--conditions derived` to select one probe. Use `--offset`, `--limit`,
and `--repeats` to control the sample range and repetition count. The score
reductions and query-cache format are configurable through `--help`.

## Local CPU smoke run

Use the workspace virtual environment and a vLLM build with CPU kernels.

```bash
.venv/bin/python -m suffix_repair_test.bench_kvedit \
  --model Qwen/Qwen3-0.6B --prompt-style qwen --methods full \
  --limit 1 --warmups 0
```

The `reuse` and `selective` methods require CUDA.

## Read the results

The default methods are `full`, `reuse`, and `selective`. Each starts
with a separate cold generation on the unedited prompt. The `full` method
reruns the edited prompt cold. The optional `prefix` method uses vLLM's
normal cache lookup to reuse the unchanged prefix.

The `reuse` method computes KV for the replacement tokens and keeps the
cached suffix with zero suffix tokens replayed. The `selective` method also
recomputes suffix tokens selected by the threshold or top-k policy. Both
generate against the edited document with a fresh question and any partial
document block.

Use the `edited` rows in `summary.json` for the second-run median and p95
latency by method, condition, and distance. Second-run latency includes
generation and any cache editing. Accuracy counts outputs that contain the
expected registry value and omit the opposite version's value. The benchmark
also reports exact match after trimming whitespace and folding case.

Each run creates a new directory. The default location is
`data/kvedit-results/kvedit-<timestamp>`. The files contain the
following measurements and reproduction inputs.

- `config.json` records the dataset source and local path, CLI settings,
  engine options, device, software versions, and script hashes.
- `dataset.jsonl` holds the complete input file for a URL source.
- `cases.jsonl` records the exact input token IDs, edit spans, and gold answers.
  It also records cases skipped because they exceed the model length.
- `runs.jsonl` records both stages, their timings, generated answers, cache
  hits, and per-answer scores. For an edited run, `edit_ms` and `generate_ms`
  show the wall-clock components separately. `ttft_ms` includes cache editing
  when applicable. Reuse and selective edits record `repair_selected_positions`
  and `repair_selected_count` for the replayed suffix. Positions are zero-based
  indexes into the case's `new_prefix_ids`. The replacement tokens are
  always recomputed separately. `repair_suffix_tokens` is the cached suffix
  length. Edit time includes returning the replay ranges. The benchmark
  expands the ranges after timing generation and flushes each stage to disk.
- `summary.json` aggregates measured trials and reports the minimum, median,
  and maximum suffix replay counts for `reuse` and each selective variant.
  Selective rows record the requested `threshold` or `top_k` value. Warmup
  trials remain in `runs.jsonl` with `warmup: true`.

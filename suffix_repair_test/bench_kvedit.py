# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Measure old and edited generation on paired KVEdit documents."""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import re
import subprocess
from collections import defaultdict
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from statistics import median
from time import perf_counter
from uuid import uuid4

from suffix_repair_test.kvedit_dataset import (
    CONDITIONS,
    DEFAULT_DATASET,
    PROMPT_STYLES,
    load_cases,
    prepare,
)

DEFAULT_THRESHOLDS = (0.5, 32.0, 256.0, 2048.0, 16384.0)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset", default=DEFAULT_DATASET, help="KVEdit JSONL path or URL"
    )
    parser.add_argument("--model", default="Qwen/Qwen3-8B")
    parser.add_argument("--revision", help="Model and tokenizer revision")
    parser.add_argument(
        "--conditions", nargs="+", choices=CONDITIONS, default=CONDITIONS
    )
    parser.add_argument(
        "--offset", type=int, default=0, help="Skip this many dataset pairs"
    )
    parser.add_argument("--limit", type=int, help="Maximum number of dataset pairs")
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=("full", "prefix", "reuse", "selective"),
        default=("full", "reuse", "selective"),
    )
    parser.add_argument(
        "--thresholds",
        nargs="+",
        type=float,
        default=DEFAULT_THRESHOLDS,
        help="Raw selection thresholds to test (default: %(default)s)",
    )
    for axis in ("key", "head", "layer"):
        parser.add_argument(
            f"--{axis}-reduction", choices=("max", "sum"), default="max"
        )
    parser.add_argument(
        "--include-deleted", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--prompt-style", choices=PROMPT_STYLES, default="auto")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument(
        "--warmups",
        type=int,
        default=1,
        help="Warmup repeats on the first eligible case",
    )
    parser.add_argument("--max-tokens", type=int, default=16)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--max-num-batched-tokens", type=int, default=2048)
    parser.add_argument("--max-num-seqs", type=int, default=8)
    parser.add_argument(
        "--block-size", type=int, help="Defaults to 16 on CUDA and 32 on CPU"
    )
    parser.add_argument("--dtype", default="auto")
    parser.add_argument(
        "--q-format", choices=("none", "fp16", "fp8", "int8", "int4"), default="int4"
    )
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--cpu-offload-gb", type=float, default=0)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()

    for option in (
        "repeats",
        "max_tokens",
        "max_model_len",
        "max_num_batched_tokens",
        "max_num_seqs",
    ):
        if getattr(args, option) < 1:
            parser.error(f"--{option.replace('_', '-')} must be positive")
    if (
        args.offset < 0
        or args.warmups < 0
        or (args.limit is not None and args.limit < 1)
    ):
        parser.error(
            "offset and warmups must be nonnegative, and limit must be positive"
        )
    if args.block_size is not None and args.block_size < 1:
        parser.error("--block-size must be positive")
    if any(not math.isfinite(value) for value in args.thresholds):
        parser.error("thresholds must be finite")
    if "selective" in args.methods and args.q_format == "none":
        parser.error("selective repair requires a stored query cache")
    if (
        not 0 < args.gpu_memory_utilization <= 1
        or not 0 <= args.cpu_offload_gb < math.inf
    ):
        parser.error("invalid GPU memory utilization or CPU offload size")
    return args


def trial_variants(args):
    for method in dict.fromkeys(args.methods):
        if method != "selective":
            yield method, False if method == "reuse" else None
            continue
        for threshold in dict.fromkeys(args.thresholds):
            yield (
                f"selective@{threshold!r}",
                {
                    "threshold": threshold,
                    "key_reduction": args.key_reduction,
                    "head_reduction": args.head_reduction,
                    "layer_reduction": args.layer_reduction,
                    "include_deleted": args.include_deleted,
                },
            )


def git_revision():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[1],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def append_json(file, record):
    file.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
    file.flush()
    os.fsync(file.fileno())


def write_json(path, data):
    pending = path.with_suffix(".tmp")
    pending.write_text(
        json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    pending.replace(path)


def verdict(text, expected, other):
    def contains(answer):
        return bool(
            re.search(rf"(?<!\w){re.escape(answer)}(?!\w)", text, re.IGNORECASE)
        )

    return {
        "correct": contains(expected) and not contains(other),
        "exact_match": text.strip().casefold() == expected.casefold(),
    }


def generation_record(output, latency_ms, expected, other):
    result = output.outputs[0]
    if result.finish_reason not in ("stop", "length"):
        raise RuntimeError(
            f"Unexpected generation finish reason: {result.finish_reason}"
        )
    ttft = getattr(output.metrics, "first_token_latency", None)
    return {
        "generate_ms": latency_ms,
        "generate_ttft_ms": None if ttft is None else ttft * 1000,
        "cached_tokens": output.num_cached_tokens,
        "generated_tokens": len(result.token_ids),
        "generated_token_ids": list(result.token_ids),
        "text": result.text,
        "finish_reason": result.finish_reason,
        "stop_reason": result.stop_reason,
        "expected_answer": expected,
        **verdict(result.text, expected, other),
    }


def run_trial(llm, journal, info, prepared, sampling, variant, block_size):
    name, recompute_suffix = variant
    stage = "unedited"
    selection = recompute_suffix if isinstance(recompute_suffix, dict) else None
    record = dict(info, method=name, selection=selection)
    salt = uuid4().hex
    old_ids = prepared.old_prefix + prepared.suffix
    new_ids = prepared.new_prefix + prepared.suffix
    old_prompt = {"prompt_token_ids": old_ids, "cache_salt": salt}
    new_prompt = {"prompt_token_ids": new_ids, "cache_salt": salt}
    expected_old, expected_new = info["answer_old"], info["answer_new"]
    edit_ms = 0.0
    try:
        started = perf_counter()
        old = llm.generate(old_prompt, sampling, use_tqdm=False)[0]
        elapsed_ms = (perf_counter() - started) * 1000
        if old.num_cached_tokens != 0:
            raise RuntimeError(
                f"Unedited request had {old.num_cached_tokens} cache hits"
            )
        result = generation_record(old, elapsed_ms, expected_old, expected_new)
        append_json(
            journal,
            dict(
                record,
                stage=stage,
                status="ok",
                latency_ms=elapsed_ms,
                ttft_ms=result["generate_ttft_ms"],
                **result,
            ),
        )

        stage = "edited"
        expected_hits = None
        if name == "full":
            # A different salt isolates the new prompt from the unedited request.
            new_prompt["cache_salt"] = uuid4().hex
            expected_hits = 0
        elif recompute_suffix is not None:
            # The unedited request publishes full document blocks. Keep the
            # remaining document tokens and the question fresh for generation.
            source_len = len(prepared.old_prefix) // block_size * block_size
            source_ids = prepared.old_prefix[:source_len]
            start, end = prepared.edit_span
            replacement = prepared.replacement_ids
            expected_hits = source_len + len(replacement) - (end - start)
            if end > source_len:
                raise ValueError("The edit extends beyond the cached document blocks")
            if (
                source_ids[:start] + replacement + source_ids[end:]
                != new_ids[:expected_hits]
            ):
                raise ValueError(
                    "Cache edit would produce different IDs from the full rerun"
                )
            record["repair_source_tokens"] = source_len
            record["repair_suffix_tokens"] = source_len - end

        selected_runs = None
        started = perf_counter()
        if recompute_suffix is not None:
            # The engine accepts exact IDs, including BPE fragments that may
            # tokenize differently when passed as standalone replacement text.
            edit = llm.llm_engine.engine_core.edit_kv_cache(
                source_ids,
                replacement,
                start,
                end,
                salt,
                recompute_suffix,
                return_selected_runs=True,
            )
            assert isinstance(edit, dict)
            new_prompt["cache_salt"] = edit["cache_salt"]
            selected_runs = edit["selected_runs"]
            edit_ms = (perf_counter() - started) * 1000
        generation_started = perf_counter()
        try:
            new = llm.generate(new_prompt, sampling, use_tqdm=False)[0]
            finished = perf_counter()
        finally:
            if selected_runs is not None:
                suffix_start = start + len(replacement)
                positions = [
                    pos
                    for first, last in selected_runs
                    for pos in range(suffix_start + first, suffix_start + last)
                ]
                record.update(
                    repair_selected_positions=positions,
                    repair_selected_count=len(positions),
                )
        if expected_hits is not None and new.num_cached_tokens != expected_hits:
            raise RuntimeError(
                f"Expected {expected_hits} cache hits, got {new.num_cached_tokens}"
            )
        elapsed_ms = (finished - started) * 1000
        generation_ms = (finished - generation_started) * 1000
        result = generation_record(new, generation_ms, expected_new, expected_old)
        ttft = result["generate_ttft_ms"]
        append_json(
            journal,
            dict(
                record,
                stage=stage,
                status="ok",
                latency_ms=elapsed_ms,
                edit_ms=edit_ms,
                ttft_ms=None if ttft is None else ttft + edit_ms,
                **result,
            ),
        )
    except Exception as exc:
        append_json(
            journal,
            dict(
                record,
                stage=stage,
                status="error",
                error=str(exc),
                edit_ms=edit_ms,
            ),
        )
        raise


def summarize(path):
    groups = defaultdict(list)
    with path.open(encoding="utf-8") as journal:
        for line in journal:
            row = json.loads(line)
            if not row["warmup"]:
                row.pop("repair_selected_positions", None)
                groups[
                    row["condition"], row["distance"], row["method"], row["stage"]
                ].append(row)

    summary = []
    for (condition, distance, method, stage), rows in groups.items():
        valid = [row for row in rows if row["status"] == "ok"]
        latencies = sorted(row["latency_ms"] for row in valid)
        count = len(valid)
        metrics = {
            "condition": condition,
            "distance": distance,
            "method": method,
            "stage": stage,
            "samples": count,
            "errors": len(rows) - count,
            "accuracy": sum(row["correct"] for row in valid) / count if count else None,
            "exact_match": sum(row["exact_match"] for row in valid) / count
            if count
            else None,
            "median_ms": median(latencies) if count else None,
            "p95_ms": latencies[math.ceil(count * 0.95) - 1] if count else None,
        }
        if (selection := rows[0].get("selection")) is not None:
            metrics["threshold"] = selection["threshold"]
        selected_counts = [
            row["repair_selected_count"]
            for row in valid
            if "repair_selected_count" in row
        ]
        if selected_counts:
            metrics.update(
                min_selected_tokens=min(selected_counts),
                median_selected_tokens=median(selected_counts),
                max_selected_tokens=max(selected_counts),
            )
        summary.append(metrics)
    return summary


def run(args, llm, sampling_params, block_size, output_dir):
    variants = list(trial_variants(args))
    journal_path = output_dir / "runs.jsonl"
    case_count = 0
    tokenizer = llm.get_tokenizer()
    try:
        with (
            journal_path.open("w", encoding="utf-8") as journal,
            (output_dir / "cases.jsonl").open("w", encoding="utf-8") as cases_file,
        ):
            for case in load_cases(
                args.dataset, dict.fromkeys(args.conditions), args.offset, args.limit
            ):
                info = {
                    "pair_id": case.pair_id,
                    "condition": case.condition,
                    "distance": case.distance,
                    "answer_old": case.answer_old,
                    "answer_new": case.answer_new,
                }
                prepared = prepare(case, tokenizer, args.prompt_style)
                old_len = len(prepared.old_prefix) + len(prepared.suffix)
                new_len = len(prepared.new_prefix) + len(prepared.suffix)
                info.update(old_prompt_tokens=old_len, new_prompt_tokens=new_len)
                case_data = dict(
                    info,
                    old_prefix_ids=prepared.old_prefix,
                    new_prefix_ids=prepared.new_prefix,
                    question_ids=prepared.suffix,
                    edit_span=prepared.edit_span,
                    replacement_ids=prepared.replacement_ids,
                    stop_token_ids=prepared.stop_token_ids,
                    dataset_token_metadata=case.token_metadata,
                )
                if max(old_len, new_len) + args.max_tokens > args.max_model_len:
                    append_json(
                        cases_file,
                        dict(case_data, status="skipped", reason="max_model_len"),
                    )
                    continue
                append_json(cases_file, dict(case_data, status="ready"))
                sampling = sampling_params(
                    temperature=0,
                    max_tokens=args.max_tokens,
                    stop_token_ids=prepared.stop_token_ids,
                )
                if case_count == 0:
                    for repeat in range(args.warmups):
                        for variant in variants:
                            run_trial(
                                llm,
                                journal,
                                dict(info, warmup=True, repeat=repeat),
                                prepared,
                                sampling,
                                variant,
                                block_size,
                            )
                for repeat in range(args.repeats):
                    shift = (case_count + repeat) % len(variants)
                    for variant in variants[shift:] + variants[:shift]:
                        run_trial(
                            llm,
                            journal,
                            dict(info, warmup=False, repeat=repeat),
                            prepared,
                            sampling,
                            variant,
                            block_size,
                        )
                case_count += 1
                print(
                    f"Completed {case.pair_id} {case.condition} ({case_count})",
                    flush=True,
                )
        if case_count == 0:
            raise ValueError(
                "No eligible cases. Check the dataset range and max model length."
            )
    finally:
        if journal_path.is_file():
            summary = summarize(journal_path)
            write_json(output_dir / "summary.json", summary)
    return summary


def main():
    args = parse_args()
    import torch

    import vllm
    from vllm import LLM, SamplingParams
    from vllm.platforms import current_platform

    cuda = current_platform.is_cuda()
    if not cuda and {"reuse", "selective"}.intersection(args.methods):
        raise SystemExit(
            "KV cache editing requires CUDA. For a CPU run use --methods full prefix."
        )
    if not cuda and not current_platform.is_cpu():
        raise SystemExit("This benchmark supports CUDA and CPU runtimes.")

    output_dir = args.output_dir or Path("data/kvedit-results") / (
        f"kvedit-{datetime.now(UTC):%Y%m%dT%H%M%S%fZ}"
    )
    output_dir.mkdir(parents=True, exist_ok=False)
    print(f"Saving benchmark results to {output_dir.resolve()}", flush=True)

    options = {
        "model": args.model,
        "revision": args.revision,
        "tokenizer_revision": args.revision,
        "dtype": args.dtype,
        "max_model_len": args.max_model_len,
        "max_num_batched_tokens": args.max_num_batched_tokens,
        "max_num_seqs": args.max_num_seqs,
        "block_size": args.block_size or (16 if cuda else 32),
        "enable_prefix_caching": True,
        "disable_log_stats": False,
        "enforce_eager": True,
        "generation_config": "vllm",
    }
    if cuda:
        options.update(
            attention_config={"backend": "TRITON_ATTN"},
            q_cache_dtype=args.q_format,
            gpu_memory_utilization=args.gpu_memory_utilization,
            cpu_offload_gb=args.cpu_offload_gb,
        )
    manifest = {
        "args": dict(vars(args), output_dir=str(output_dir.resolve())),
        "engine_options": options,
        "git_revision": git_revision(),
        "script_sha256": {
            name: sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
            for name in ("bench_kvedit.py", "kvedit_dataset.py")
        },
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "device": torch.cuda.get_device_name() if cuda else "CPU",
        "torch_version": str(torch.__version__),
        "cuda_version": torch.version.cuda,
        "vllm_version": vllm.__version__,
        "runner_env": {
            key: os.environ.get(key)
            for key in ("VLLM_USE_V2_MODEL_RUNNER", "VLLM_ENABLE_V1_MULTIPROCESSING")
        },
    }
    write_json(output_dir / "config.json", manifest)
    llm = LLM(**options)
    config = llm.llm_engine.vllm_config
    block_size = config.cache_config.block_size
    manifest.update(
        actual_block_size=block_size,
        actual_dtype=str(config.model_config.dtype),
        model_commit=getattr(config.model_config.hf_config, "_commit_hash", None),
    )
    write_json(output_dir / "config.json", manifest)
    for row in run(args, llm, SamplingParams, block_size, output_dir):
        if row["stage"] == "edited":
            print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()

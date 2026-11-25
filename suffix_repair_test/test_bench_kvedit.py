# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import json
import unittest
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import Mock, patch

from suffix_repair_test.bench_kvedit import (
    parse_args,
    run_trial,
    summarize,
    trial_variants,
)
from suffix_repair_test.kvedit_dataset import PreparedCase, fetch_dataset, load_cases


class KVEditBenchmarkTest(unittest.TestCase):
    def test_downloads_full_development_split_before_limiting(self):
        with patch("sys.argv", ["bench_kvedit"]):
            source = parse_args().dataset
        self.assertTrue(source.endswith("/pilot_pairs.jsonl"))
        pair = dict(
            question_block="Which value?",
            answer_old="old",
            answer_new="new",
            conditions={"direct": {"context_old": "old", "context_new": "new"}},
        )
        payload = "".join(
            json.dumps(dict(pair, pair_id=pair_id)) + "\n" for pair_id in ("a", "b")
        ).encode()
        response = BytesIO(payload)
        response.headers = {"Content-Length": str(len(payload))}
        with (
            TemporaryDirectory() as tmp,
            patch(
                "suffix_repair_test.kvedit_dataset.urlopen", return_value=response
            ) as open_url,
        ):
            local_path = fetch_dataset(source, Path(tmp) / "dataset.jsonl")
            self.assertEqual(local_path.read_bytes(), payload)
            self.assertTrue(response.closed)
            cases = list(load_cases(local_path, ("direct",), limit=1))
            self.assertEqual([case.pair_id for case in cases], ["a"])
            self.assertEqual(
                fetch_dataset(local_path, Path(tmp) / "unused"), local_path
            )
            open_url.assert_called_once_with(source, timeout=30)

    def test_incomplete_download_leaves_no_dataset(self):
        response = BytesIO(b"partial")
        response.headers = {"Content-Length": "100"}
        with (
            TemporaryDirectory() as tmp,
            patch("suffix_repair_test.kvedit_dataset.urlopen", return_value=response),
        ):
            with self.assertRaisesRegex(OSError, "Incomplete dataset download"):
                fetch_dataset(
                    "https://example.com/data.jsonl", Path(tmp) / "dataset.jsonl"
                )
            self.assertEqual(list(Path(tmp).iterdir()), [])

    def test_default_sweep_and_override(self):
        with patch("sys.argv", ["bench_kvedit"]):
            variants = list(trial_variants(parse_args()))
        self.assertEqual(variants[:2], [("full", None), ("reuse", False)])
        self.assertEqual(
            [policy["threshold"] for _, policy in variants[2:]],
            [0.5, 32, 256, 2048, 16384],
        )
        with patch("sys.argv", ["bench_kvedit", "--thresholds", "16", "64", "16"]):
            variants = list(trial_variants(parse_args()))
        self.assertEqual([policy["threshold"] for _, policy in variants[2:]], [16, 64])

    def test_top_k_sweep_and_combined_sweep(self):
        with patch("sys.argv", ["bench_kvedit", "--top-k", "0", "16", "16"]):
            variants = list(trial_variants(parse_args()))
        self.assertEqual(variants[:2], [("full", None), ("reuse", False)])
        self.assertEqual(
            [(label, policy["top_k"]) for label, policy in variants[2:]],
            [("selective@top-k=0", 0), ("selective@top-k=16", 16)],
        )
        with patch("sys.argv", ["bench_kvedit", "--thresholds", "32", "--top-k", "64"]):
            variants = list(trial_variants(parse_args()))
        self.assertEqual(
            [label for label, _ in variants[2:]],
            ["selective@32.0", "selective@top-k=64"],
        )

    def test_rejects_negative_top_k(self):
        with (
            patch("sys.argv", ["bench_kvedit", "--top-k", "-1"]),
            patch("sys.stderr"),
            self.assertRaises(SystemExit),
        ):
            parse_args()

    def test_scores_reuse_and_records_replay_after_length_change(self):
        old = list(range(11))
        prepared = PreparedCase(
            old_prefix=old,
            new_prefix=old[:2] + [40, 41, 42] + old[3:],
            suffix=[50],
            edit_span=(2, 3),
            replacement_ids=[40, 41, 42],
            stop_token_ids=[],
        )
        info = dict(
            condition="direct",
            distance=None,
            warmup=False,
            repeat=0,
            answer_old="H684",
            answer_new="Z349",
        )

        def output(text, cached_tokens):
            return [
                SimpleNamespace(
                    num_cached_tokens=cached_tokens,
                    metrics=None,
                    outputs=[
                        SimpleNamespace(
                            text=text,
                            token_ids=[1],
                            finish_reason="stop",
                            stop_reason=None,
                        )
                    ],
                )
            ]

        old_output, new_output = output("H684", 0), output("Z349", 10)
        core = Mock()
        core.edit_kv_cache.side_effect = [
            {"cache_salt": "edited-reuse", "selected_runs": []},
            {"cache_salt": "edited-32", "selected_runs": [[0, 2], [4, 5]]},
            {"cache_salt": "edited-128", "selected_runs": []},
            {"cache_salt": "edited-top-k", "selected_runs": [[0, 5]]},
            {"cache_salt": "edited-error", "selected_runs": [[1, 2]]},
        ]
        llm = SimpleNamespace(
            llm_engine=SimpleNamespace(engine_core=core),
            generate=Mock(
                side_effect=[
                    old_output,
                    output("H684", 10),
                    old_output,
                    new_output,
                    old_output,
                    new_output,
                    old_output,
                    new_output,
                    old_output,
                    RuntimeError("generation failed"),
                ]
            ),
        )
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "runs.jsonl"
            with path.open("w", encoding="utf-8") as journal:
                run_trial(llm, journal, info, prepared, None, ("reuse", False), 4)
                for threshold in (32, 128):
                    variant = (f"selective@{threshold}", {"threshold": threshold})
                    run_trial(llm, journal, info, prepared, None, variant, block_size=4)
                run_trial(
                    llm,
                    journal,
                    info,
                    prepared,
                    None,
                    ("selective@top-k=64", {"top_k": 64}),
                    block_size=4,
                )
                with self.assertRaisesRegex(RuntimeError, "generation failed"):
                    run_trial(
                        llm,
                        journal,
                        info,
                        prepared,
                        None,
                        ("selective@32", {"threshold": 32}),
                        block_size=4,
                    )
            rows = [
                json.loads(line)
                for line in path.read_text(encoding="utf-8").splitlines()
            ]
            edited = [row for row in rows if row["stage"] == "edited"]
            summary = {
                row["method"]: row
                for row in summarize(path)
                if row["stage"] == "edited"
            }

        self.assertEqual(
            [row["repair_selected_positions"] for row in edited],
            [[], [5, 6, 9], [], [5, 6, 7, 8, 9], [6]],
        )
        self.assertEqual(
            [row["repair_selected_count"] for row in edited], [0, 3, 0, 5, 1]
        )
        reuse = edited[0]
        self.assertIsNone(reuse["selection"])
        self.assertEqual(reuse["expected_answer"], "Z349")
        self.assertFalse(reuse["correct"])
        self.assertEqual(summary["reuse"]["accuracy"], 0)
        self.assertEqual(summary["reuse"]["median_selected_tokens"], 0)
        self.assertIs(core.edit_kv_cache.call_args_list[0].args[-1], False)
        self.assertEqual(
            llm.generate.call_args_list[1].args[0],
            {
                "prompt_token_ids": prepared.new_prefix + prepared.suffix,
                "cache_salt": "edited-reuse",
            },
        )
        self.assertEqual(edited[-1]["status"], "error")
        self.assertEqual(summary["selective@32"]["median_selected_tokens"], 3)
        self.assertEqual(summary["selective@128"]["median_selected_tokens"], 0)
        self.assertEqual(core.edit_kv_cache.call_args_list[3].args[-1], {"top_k": 5})
        self.assertEqual(edited[3]["selection"], {"top_k": 64})
        self.assertEqual(summary["selective@top-k=64"]["top_k"], 64)
        self.assertEqual(summary["selective@top-k=64"]["median_selected_tokens"], 5)
        self.assertTrue(
            all(
                call.kwargs["return_selected_runs"]
                for call in core.edit_kv_cache.call_args_list
            )
        )

"""Run a small offline generation and accuracy benchmark on LongBench v2"""

import argparse
import re
import string
import time
from collections import Counter

from datasets import load_dataset
from vllm import LLM, SamplingParams


### Evaluation code adapted from https://github.com/THUDM/LongBench/blob/main/pred.py

import re

def extract_answer(response):
    response = response.replace('*', '')
    match = re.search(r'The correct answer is \(([A-D])\)', response)
    if match:
        return match.group(1)
    else:
        match = re.search(r'The correct answer is ([A-D])', response)
        if match:
            return match.group(1)
        else:
            return None

PROMPT_TEMPLATE = """
Please read the following text and answer the question below.

<text>
$DOC$
</text>

What is the correct answer to this question: $Q$
Choices:
(A) $C_A$
(B) $C_B$
(C) $C_C$
(D) $C_D$

Format your response as follows: "The correct answer is (insert answer here)".
""".strip()

def make_prompt(item):
    template = PROMPT_TEMPLATE

    return template \
        .replace('$DOC$', item['context'].strip()) \
        .replace('$Q$', item['question'].strip()) \
        .replace('$C_A$', item['choice_A'].strip()) \
        .replace('$C_B$', item['choice_B'].strip()) \
        .replace('$C_C$', item['choice_C'].strip()) \
        .replace('$C_D$', item['choice_D'].strip())

###

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--num-samples", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=64)
    args = parser.parse_args()

    if args.num_samples < 1:
        parser.error("--num-samples must be at least 1")

    dataset = load_dataset('THUDM/LongBench-v2', split='train') # 'train' is the only split
    dataset = dataset.filter(lambda x: x["length"] == 'short') # filter only short samples
    dataset = dataset.filter(lambda x: len(x['context']) <= 60_000) # filter even shorter context samples

    sample = dataset.select(range(min(args.num_samples, len(dataset))))

    prompts = [ make_prompt(item) for item in sample ]
    answers = sample['answer']

    llm = LLM(model=args.model, gpu_memory_utilization=0.7)
    sampling_params = SamplingParams(temperature=0, max_tokens=args.max_tokens)

    start = time.perf_counter()
    outputs = llm.generate(prompts, sampling_params)
    elapsed = time.perf_counter() - start

    scores = [ extract_answer(output.outputs[0].text) == answer for output, answer in zip(outputs, answers)]
    accurary = sum(scores) / len(scores)

    generated_tokens = sum(len(output.outputs[0].token_ids) for output in outputs)
    count = len(outputs)

    print(f"Model: {args.model}")
    print(f"Dataset: THUDM/LongBench-v2 test ({count} prompts)")
    print(f"Generation time: {elapsed:.2f} s")
    print(f"Requests per second: {count / elapsed:.2f}")
    print(f"Generated tokens: {generated_tokens}")
    print(f"Output tokens per second: {generated_tokens / elapsed:.2f}")
    print(f"Accuracy: {accurary:.1%}")
    print("\nExample:")
    print(f"Prompt: {outputs[0].prompt[:300]!r}")
    print(f"Reference answer: {answers[0]!r}")
    print(f"Output: {outputs[0].outputs[0].text!r}")


if __name__ == "__main__":
    main()

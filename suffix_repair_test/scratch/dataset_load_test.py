from datasets import load_dataset

### LongBench-E

# dataset = load_dataset("THUDM/LongBench", "hotpotqa_e", split="test", 
#     revision="refs/pr/7") # dataset scripts are no longer supported; use this unmerged pr
# short_dataset = dataset.filter(lambda x: x["length"] <= 4096) # filter only short samples

# print(short_dataset)

### LongBench-v2

dataset = load_dataset('THUDM/LongBench-v2', split='train') # 'train' is the only split
dataset = dataset.filter(lambda x: x["length"] == 'short') # filter only short samples
dataset = dataset.filter(lambda x: len(x['context']) <= 60_000) # filter even shorter context samples

print(dataset)
# print(max([ len(x['context']) for x in dataset ]))

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

print(make_prompt(dataset[0]))
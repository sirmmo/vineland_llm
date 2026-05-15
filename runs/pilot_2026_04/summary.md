# Pilot Run Summary

## Token Usage per Agent

| Agent | Prompt Tokens | Completion Tokens | Reasoning Tokens |
|-------|--------------|-------------------|-----------------|
| claude-opus-4-5 | 16,840 | 33,607 | 0 |
| deepseek-r1 | 15,425 | 114,501 | 94,055 |
| deepseek-r1-free | 0 | 0 | 0 |
| deepseek-v3-free | 0 | 0 | 0 |
| gemma-27b | 16,040 | 17,652 | 0 |
| gemma-3-12b-free | 8,255 | 0 | 0 |
| gemma-3-1b-free | 0 | 0 | 0 |
| gemma-3-27b-free | 2,881 | 0 | 0 |
| gemma-3-4b-free | 14,865 | 0 | 0 |
| gemma-4-26b-a4b-it-free | 9,709 | 16,235 | 0 |
| gemma-4-31b-it-free | 9,311 | 11,730 | 0 |
| gpt-5 | 14,965 | 101,690 | 92,608 |
| gpt-5-reasoning | 16,963 | 133,938 | 111,674 |
| llama-3.1-8b-free | 0 | 0 | 0 |
| llama-3.2-3b-free | 0 | 0 | 0 |
| llama-3.3-70b-free | 0 | 0 | 0 |
| llama-70b | 15,405 | 21,855 | 0 |
| mistral-7b | 0 | 0 | 0 |
| mistral-7b-free | 0 | 0 | 0 |
| mistral-7b-instruct-v0.1 | 14,172 | 16,696 | 0 |
| mistral-7b-instruct-v0.3 | 0 | 0 | 0 |
| o3 | 14,965 | 59,434 | 36,070 |
| openchat-7b-free | 0 | 0 | 0 |
| phi-3-mini-free | 0 | 0 | 0 |
| qwen-2.5-72b-free | 0 | 0 | 0 |
| qwen-2.5-7b-free | 0 | 0 | 0 |
| zephyr-7b-free | 0 | 0 | 0 |

## Mean Polytomous Score by Agent × Subdomain

| Agent |  | AFD | COP | CTX | DOM | EXP | IPR | PER | REC | TOL | WMM |
|-------|---|---|---|---|---|---|---|---|---|---|---|
| claude-opus-4-5 | - | 1.00 | 2.00 | 2.00 | 2.00 | 1.00 | 2.00 | 2.00 | 1.00 | 1.00 | 2.00 |
| deepseek-r1 | - | 1.50 | 2.00 | 2.00 | 2.00 | 1.00 | 2.00 | 2.00 | 1.00 | 1.50 | 2.00 |
| gemma-27b | - | 1.50 | 2.00 | 2.00 | 2.00 | 1.00 | 2.00 | 2.00 | 1.00 | 1.00 | 1.00 |
| gemma-3-12b-free | 2.00 | 1.50 | - | 2.00 | 0.00 | 2.00 | - | 2.00 | 1.00 | 1.00 | 2.00 |
| gemma-3-27b-free | 2.00 | 0.00 | - | 2.00 | 0.00 | 2.00 | - | - | 2.00 | 1.00 | 0.00 |
| gemma-3-4b-free | 0.00 | 1.00 | 2.00 | 0.00 | 0.00 | 1.00 | 2.00 | 2.00 | 1.00 | 1.00 | 1.00 |
| gemma-4-26b-a4b-it-free | - | 1.50 | 2.00 | 2.00 | 2.00 | 1.00 | 2.00 | 2.00 | 1.00 | 1.00 | 1.50 |
| gemma-4-31b-it-free | - | 1.00 | 2.00 | 2.00 | 2.00 | 1.00 | 2.00 | 2.00 | 1.00 | 1.50 | 1.50 |
| gpt-5 | - | 2.00 | 2.00 | 2.00 | 2.00 | 1.50 | 2.00 | 2.00 | 1.00 | 1.00 | 1.00 |
| gpt-5-reasoning | - | 2.00 | 2.00 | 2.00 | 2.00 | 1.00 | 2.00 | 2.00 | 1.00 | 1.00 | 1.00 |
| llama-70b | - | 1.00 | 2.00 | 2.00 | 2.00 | 1.00 | 2.00 | 2.00 | 1.00 | 1.50 | 2.00 |
| mistral-7b-instruct-v0.1 | - | 1.00 | 2.00 | 0.00 | 2.00 | 1.00 | 2.00 | 2.00 | 1.00 | 1.00 | 1.00 |
| o3 | - | 2.00 | 2.00 | 2.00 | 2.00 | 1.00 | 2.00 | 2.00 | 1.00 | 1.00 | 1.00 |

## Response Integrity by Agent

Critical for detecting silent-failure regressions (e.g., empty responses).

| Agent | Total Runs | Success | Fail | Partial | Null | Empty Response | Empty Rate |
|-------|-----------:|--------:|-----:|--------:|-----:|---------------:|-----------:|
| claude-opus-4-5 | 80 | 59 | 21 | 0 | 0 | 0 | 0.00% |
| deepseek-r1 | 80 | 61 | 19 | 0 | 0 | 0 | 0.00% |
| deepseek-r1-free | 80 | 0 | 0 | 0 | 80 | 0 | 0.00% |
| deepseek-v3-free | 80 | 0 | 0 | 0 | 80 | 0 | 0.00% |
| gemma-27b | 80 | 55 | 25 | 0 | 0 | 0 | 0.00% |
| gemma-3-12b-free | 83 | 33 | 13 | 0 | 37 | 0 | 0.00% |
| gemma-3-1b-free | 80 | 0 | 0 | 0 | 80 | 0 | 0.00% |
| gemma-3-27b-free | 83 | 14 | 7 | 0 | 62 | 0 | 0.00% |
| gemma-3-4b-free | 85 | 44 | 36 | 0 | 5 | 0 | 0.00% |
| gemma-4-26b-a4b-it-free | 80 | 42 | 12 | 0 | 26 | 0 | 0.00% |
| gemma-4-31b-it-free | 80 | 40 | 10 | 0 | 30 | 0 | 0.00% |
| gpt-5 | 80 | 62 | 18 | 0 | 0 | 0 | 0.00% |
| gpt-5-reasoning | 80 | 60 | 20 | 0 | 0 | 0 | 0.00% |
| llama-3.1-8b-free | 80 | 0 | 0 | 0 | 80 | 0 | 0.00% |
| llama-3.2-3b-free | 80 | 0 | 0 | 0 | 80 | 0 | 0.00% |
| llama-3.3-70b-free | 80 | 0 | 0 | 0 | 80 | 0 | 0.00% |
| llama-70b | 80 | 60 | 20 | 0 | 0 | 0 | 0.00% |
| mistral-7b | 80 | 0 | 0 | 0 | 80 | 0 | 0.00% |
| mistral-7b-free | 80 | 0 | 0 | 0 | 80 | 0 | 0.00% |
| mistral-7b-instruct-v0.1 | 80 | 40 | 24 | 0 | 16 | 0 | 0.00% |
| mistral-7b-instruct-v0.3 | 80 | 0 | 0 | 0 | 80 | 0 | 0.00% |
| o3 | 80 | 58 | 22 | 0 | 0 | 0 | 0.00% |
| openchat-7b-free | 80 | 0 | 0 | 0 | 80 | 0 | 0.00% |
| phi-3-mini-free | 80 | 0 | 0 | 0 | 80 | 0 | 0.00% |
| qwen-2.5-72b-free | 80 | 0 | 0 | 0 | 80 | 0 | 0.00% |
| qwen-2.5-7b-free | 80 | 0 | 0 | 0 | 80 | 0 | 0.00% |
| zephyr-7b-free | 80 | 0 | 0 | 0 | 80 | 0 | 0.00% |

## Grader Type Distribution

| Grader Type | Items |
|-------------|------:|
| (unset) | 17 |

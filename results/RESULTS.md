# Results

Effect-conditioned g-mean2 (OpenAI metric, 200 bootstraps), gpt-oss-20b as both the
model and the monitor.

| dataset       | CoT monitor (reads the reasoning) | no-op (reads only the answer) |
| ------------- | --------------------------------- | ----------------------------- |
| gpqa_encoded  | 0.37                              | 0.95                          |
| gpqa          | 0.32                              | 0.94                          |

The no-op reproduces the misbehavior label with 100% fidelity on gpqa_encoded
(94% on gpqa), reading none of the chain of thought. It decodes the cue from the
prompt and checks the final answer. Because the cue decodes to the correct answer,
the misbehavior label is a function of the prompt and the output, so the chain of
thought is irrelevant to it.

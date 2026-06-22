# Results

Effect-conditioned g-mean2 (OpenAI metric, 200 bootstraps), gpt-oss-20b as both
the model and the monitor.

| dataset          | CoT monitor (reads reasoning) | no-op (reads only the answer) | eligible |
| ---------------- | ----------------------------- | ----------------------------- | -------- |
| gpqa             | 0.32 ± 0.18                   | 1.00 ± 0.00                   | 0.08     |
| gpqa_encoded     | 0.37 ± 0.06                   | 0.95 ± 0.02                   | 0.36     |
| daft             | 0.20 ± 0.05                   | 0.99 ± 0.01                   | 0.30     |
| wmdp_sandbagging | 0.34 ± 0.21                   | 0.91 ± 0.07                   | 0.04     |

The no-op detector applies the eval's own grading rule to the final answer and
ignores the chain of thought entirely. Because the misbehavior label Y is a
function of (prompt, grading rule, model answer), the no-op reproduces Y by
construction; no monitor that honestly reads the reasoning can outperform it.

CIs widen where the eligibility fraction is small (gpqa: 0.08; sandbagging: 0.04
because the model resists the sandbag cue, so few instances are eligible). The pattern
persists across all four datasets and across two distinct cue mechanisms:

- **cue → correct answer** (gpqa, gpqa_encoded, daft): the hint points at the
  gold answer. "Got it right" ≈ "used the cue", so reading the answer
  reproduces the label.
- **cue → wrong answer** (sandbagging): the hint asks the model to give a wrong
  answer. The label needs the answer key, but it is still an I/O function, no
  chain of thought required.

Either way, the chain of thought is irrelevant to the label, so the eval cannot
reward reading it.

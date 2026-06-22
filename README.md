# Chain of thought monitorability has a measurement problem

A small, replicable study built on OpenAI's open-sourced monitorability evals.

- OpenAI article: https://alignment.openai.com/monitorability-evals/
- OpenAI evals: https://github.com/openai/monitorability-evals
- Paper: https://arxiv.org/abs/2512.18311 (Monitoring Monitorability, Guan et al.)

## What I wanted to know

Monitorability is about catching a model doing something bad by reading its chain of
thought, ideally before it acts. I am a skeptic of chain of thought monitoring, so the
question I cared about was simple: how good is this monitoring really, and can we trust
the number these evals give it?

## The setup

I loaded gpt-oss-20b on an H200 and served it with vLLM. I used the cheat style
(intervention) tests: take a hard multiple choice science question, run the model two
ways, once clean and once with a planted cheat in the prompt. For gpqa_encoded the cheat
is an encoded hint that decodes to the answer. "Misbehavior" is defined as the model
giving the cheat's answer. I generated rollouts for both versions, 24 samples per
question, at high concurrency, and saved every chain of thought and answer.

## Running the monitor

I ran OpenAI's own monitor (an LLM that reads each chain of thought and judges whether
the hint influenced the answer) and scored it with their own metric (effect conditioned
g-mean2). The monitor scored 0.37. It caught about 40 percent of the misbehavior and
missed 60.

## What makes that number go up

I tried a few ways to push the monitor's score higher and none cleanly worked, so
instead of chasing the number I asked the basic question: why is it missing 60 percent?
Is the model hiding its reasoning, or is the monitor just weak? I read the rollouts. The
model was not hiding anything. It wrote the cheat into its reasoning about 90 percent of
the time. The monitor was missing things sitting in plain text, because it was cautious:
when the model used the cheat but also did some real looking work, it gave it the benefit
of the doubt.

## The no-op baseline

Reading the rollouts made me suspect the chain of thought was not even necessary. So I
built the dumbest possible detector: it ignores the reasoning entirely, decodes the cheat
from the prompt, looks only at the final answer, and flags the row if the answer matches
the cheat. No reasoning, no answer key.

That blind detector reproduced the misbehavior label with 100 percent fidelity and scored
0.95, versus the real monitor's 0.37. Same result on plain gpqa (0.94 vs 0.32).

## What I learned

The cheat points at the correct answer, and the questions are too hard to solve alone, so
"got the answer right" basically is "used the cheat." You can read that off the final
answer in one glance. The chain of thought is irrelevant to the label. So the monitor's
miss was never about hidden reasoning. These tests reward reading the answer, and a
monitor that honestly tries to judge the reasoning does worse than a glance at the output.
They cannot measure whether reading a model's reasoning helps catch misbehavior.

Two separate claims, kept separate:

1. The monitor misses because it is cautious, not because the model hides anything.
2. The no-op baseline beating the monitor is about the test, not the monitor. It says the
   test cannot measure monitoring in the first place.

## Where this points

The fix follows from the cause: make the cheat point at a wrong answer. Then "got it
right" no longer means "cheated", you cannot catch it by reading the answer, and reading
the reasoning has to do real work. More generally, a monitorability test is only
meaningful if you cannot catch the misbehavior by looking at the answer alone. By that
test the cheat style (intervention) datasets fail, but the other two kinds do not: the
process tests (does the reasoning show the right method) genuinely need the chain of
thought, and the agent tests (did the agent do something harmful) are the open question
worth checking next.

## Results

| dataset       | CoT monitor (reads the reasoning) | no-op (reads only the answer) |
| ------------- | --------------------------------- | ----------------------------- |
| gpqa_encoded  | 0.37                              | 0.95                          |
| gpqa          | 0.32                              | 0.94                          |

Effect conditioned g-mean2, OpenAI's metric, gpt-oss-20b as both model and monitor.

## Reproduce

```
bash setup.sh        # clones OpenAI evals (metric, templates, data), installs deps
bash reproduce.sh    # scores the monitor and the no-op on the shipped rollouts, no GPU
```

To regenerate rollouts from scratch (needs a Hopper GPU):

```
bash serve.sh
python src/run_rollouts.py --datasets gpqa_encoded --samples 24
python src/noop_baseline.py --dataset gpqa_encoded \
    --raw results/encoded/rollouts_raw.jsonl --csv results/encoded/rollouts.csv \
    --out results/encoded/noop.csv
```

## Files

- `src/run_rollouts.py` generates both arm rollouts and scores the monitor
- `src/noop_baseline.py` the answer only detector
- `src/analyze.py` OpenAI's effect conditioned g-mean2
- `src/helpers.py` data loading and prompt rendering
- `results/` the rollouts behind the numbers above

## A note on monitor strength

gpt-oss-20b serves as both the model and the monitor here. The finding does not depend on
monitor strength: even a perfect monitor can at most tie the no-op, because the label is
fully determined by the prompt and the answer. The point is about the test, not the
monitor.

Built on OpenAI's monitorability-evals. See their repository for its license and dataset
attribution.

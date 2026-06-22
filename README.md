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

I loaded gpt-oss-20b on an H200 and served it with vLLM. I used the intervention tests:
take a hard question, run the model two ways, once clean and once with a planted hint
in the prompt. "Misbehavior" is defined as the model giving the answer the hint is
trying to push toward. I generated rollouts for both arms, 24 samples per question,
saved every chain of thought and answer, and scored OpenAI's monitor with their own
metric (effect-conditioned g-mean2).

## Running the monitor

The monitor scored 0.37 on gpqa_encoded, 0.32 on gpqa, 0.20 on daft, 0.34 on
wmdp_sandbagging. It caught roughly a third of the misbehavior and missed the rest.

## Making the monitorability number go up

I tried a few ways to push the monitor's score higher and none cleanly worked. So
instead of chasing the number I asked the basic question: why is it missing so much?
Is the model hiding its reasoning, or is the monitor just weak? I read the rollouts.
The model was not hiding anything. On gpqa_encoded it wrote the cheat into its
reasoning about 90% of the time. The monitor was missing things sitting in plain
text, because it was cautious: when the model used the hint but also did some real
looking work, it gave it the benefit of the doubt.

## The no-op baseline

Reading the rollouts made me suspect the chain of thought was not even necessary. So
I built the dumbest possible detector: it never reads the reasoning. It looks at the
final answer and applies the eval's own grading rule. That is, it copies the
misbehavior label Y directly.

That blind detector scored 0.91 to 1.00 on every intervention dataset I tested.
The real monitor scored 0.20 to 0.37.

| dataset          | CoT monitor (reads reasoning) | no-op (reads only the answer) | n eligible |
| ---------------- | ----------------------------- | ----------------------------- | ---------- |
| gpqa             | 0.32 ± 0.18                   | 1.00 ± 0.00                   | 0.08       |
| gpqa_encoded     | 0.37 ± 0.06                   | 0.95 ± 0.02                   | 0.36       |
| daft             | 0.20 ± 0.05                   | 0.99 ± 0.01                   | 0.30       |
| wmdp_sandbagging | 0.34 ± 0.21                   | 0.91 ± 0.07                   | 0.04       |

Effect-conditioned g-mean2 (OpenAI's metric, 200 bootstraps), gpt-oss-20b as both
the model and the monitor. The no-op CIs are tight where the eligibility fraction
is high (encoded, daft); on gpqa and sandbagging the eligible slice is small and CIs
widen accordingly, but the pattern persists.

## The general issue

The misbehavior label Y is defined as a function of (prompt, grading rule, model's
final answer). The chain of thought never enters the definition. So any detector
that reads the answer and applies the grading rule can reproduce Y exactly. The
monitor's job, by contrast, is to predict Y from the chain of thought. As soon as
Y is recoverable from the input/output, a CoT-blind reader is the ceiling, and no
monitor that honestly reads the reasoning can beat it.

This is not a quirk of one dataset. It is a property of the intervention archetype
in this suite: in all four datasets the eval grades the final answer against a
known target (the gold key, a planted hint letter, or a directional rule like
"anything except A"), and that grade *is* the label. The chain of thought is
irrelevant to the label, so reading it cannot help.

## Formalizing the idea

Let Y(prompt, answer, key) be the eval's misbehavior label. Let N(prompt, answer, key)
be a "no-op" detector that ignores the chain of thought. Since Y is itself a function
of (prompt, answer, key), choosing N = Y gives perfect agreement: TPR = 1, FPR = 0,
and so g-mean2 attains its label-distribution-limited maximum. For any CoT monitor M
that reads the reasoning to predict Y, agreement with Y can at best equal the no-op.
Therefore, on any eval whose misbehavior label is an I/O-only function, no monitor
can outperform a no-op outcome reader: the test cannot reward reading the chain of
thought, by construction.

## Where this points

The misbehavior label has to depend on the chain of
thought (or on actions taken), not just on the final answer. Per OpenAI's own
[dataset registry](https://github.com/openai/monitorability-evals/blob/main/dataset_registry.yaml),
two of the three eval archetypes already meet this criterion:

- **process** evals (e.g. `process_gpqa_physics`) label misbehavior as
  "did the reasoning apply the right theorem path", judged from the chain of thought.
  By definition no I/O reader can reproduce this label.
- **agentic / outcome** evals (e.g. `shadearena`, `flaky_tools`) label misbehavior as
  "did the agent take a harmful or deceptive action". Actions are not the final
  answer, so an I/O-only reader cannot reproduce them.

These archetypes pass the validity criterion structurally, without running a single
rollout. The intervention archetype, as currently defined, does not.

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
- `src/noop_baseline.py` a stronger prompt-only no-op for `gpqa_encoded`: decodes
  the encoded cue from the prompt and checks the final answer (no answer key
  needed, it reproduces Y because the cue decodes to the gold answer and the
  model verbalizes the decode ~90% of the time)
- `src/analyze.py` OpenAI's effect-conditioned g-mean2 with 200 bootstraps
- `src/helpers.py` data loading and prompt rendering
- `results/` the rollouts behind the numbers above

## A note on monitor strength

gpt-oss-20b serves as both the model and the monitor here. The finding does not
depend on monitor strength: by the formal statement above, even a perfect monitor
can at most tie the no-op, because the label is fully determined by prompt and
answer. Running this with a frontier monitor changes nothing about the measurability
result. The point is about the test, not the monitor.

Built on OpenAI's monitorability-evals. See their repository for its license and
dataset attribution.

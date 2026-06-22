#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"
export EVALS_DIR="${EVALS_DIR:-$PWD/monitorability-evals}"
echo "### CoT monitor (reads the whole chain of thought)"
python src/analyze.py --csv results/encoded_rollouts.csv --z z_cot
python src/analyze.py --csv results/gpqa_rollouts.csv    --z z_cot
echo
echo "### no-op baseline (reads only the final answer; decodes cue from prompt, checks final answer)"
python src/analyze.py --csv results/encoded_rollouts.csv --z z_noop
python src/analyze.py --csv results/gpqa_rollouts.csv    --z z_noop

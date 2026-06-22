#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"
export EVALS_DIR="${EVALS_DIR:-$PWD/monitorability-evals}"

echo "### CoT monitor (reads the whole chain of thought)"
for ds in encoded gpqa daft sandbag; do
  python src/analyze.py --csv results/${ds}_rollouts.csv --z z_cot
done

echo
echo "### no-op (reads only the answer; applies the eval's own grading rule = the label Y)"
for ds in encoded gpqa daft sandbag; do
  python src/analyze.py --csv results/${ds}_rollouts.csv --z y
done

echo
echo "### stronger no-op for gpqa_encoded: prompt-only (decodes the cue from the prompt,"
echo "    needs no answer key) -- reproduces Y because cue decodes to gold and the model follows it."
python src/analyze.py --csv results/encoded_rollouts.csv --z z_noop

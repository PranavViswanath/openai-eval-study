#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"
# OpenAI evals: provides the g-mean2 metric, the monitor templates, and the datasets.
if [ ! -d monitorability-evals ]; then
  git clone https://github.com/openai/monitorability-evals.git
fi
pip install -r requirements.txt
echo
echo "setup done."
echo "intervention datasets are expected at monitorability-evals/.cache/environments/intervention/"
echo "if your clone does not have them, follow the openai repo README to materialize the environments."

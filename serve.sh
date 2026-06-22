#!/usr/bin/env bash
# Serve gpt-oss-20b for rollout generation. Needs a Hopper GPU (tested on an H200).
# VLLM_USE_FLASHINFER_SAMPLER=0 avoids a JIT-compile (ninja) dependency at startup.
export HF_HOME="${HF_HOME:-$PWD/hf}"
export VLLM_USE_FLASHINFER_SAMPLER=0
vllm serve openai/gpt-oss-20b --port 8000 --max-model-len 32768 --gpu-memory-utilization 0.90

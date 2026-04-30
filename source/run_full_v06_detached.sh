#!/usr/bin/env bash
set -euo pipefail

cd /data/manikm/manik/normbench_v06
mkdir -p /data/manikm/manik/normbench_v06/logs
export OLLAMA_MODELS=/data/manikm/manik/ollama/models

echo "===== FULL V06 RUN START ====="
date

python3 normbench_v06.py \
  --models qwen3:8b llama3.1:8b \
  --envs resource_two_stage turn_taking_two_stage \
  --role_modes explicit_roles light_roles uniform random \
  --prompt_variants v1 v2 v3 \
  --conditions NORMAL TEMPTATION ENFORCEMENT COSTLY_ENFORCEMENT NOISY_ENFORCEMENT \
  --seeds 30 \
  --rounds 5 \
  --temperature 0.3

echo "===== PRINT LATEST SUMMARY ====="
python3 /data/manikm/manik/normbench_v06/print_latest_summary.py

echo "===== FULL V06 RUN END ====="
date

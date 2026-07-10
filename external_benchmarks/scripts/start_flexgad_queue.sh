#!/usr/bin/env bash
set -euo pipefail

cd /root/Flex-GAD
mkdir -p /root/Flex-GAD/results

tag="flexgad_gadbench_v1"
out="/root/Flex-GAD/results/${tag}.csv"
status="/root/Flex-GAD/results/${tag}_status.tsv"
log="/root/Flex-GAD/results/${tag}.out"
pidfile="/root/Flex-GAD/results/${tag}.pid"

if pgrep -af "python3 run_flexgad_gadbench.py .*${tag}" >/tmp/flexgad_existing.txt; then
  echo "Flex-GAD queue is already running:"
  cat /tmp/flexgad_existing.txt
  exit 0
fi

nohup env PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=0 \
  python3 run_flexgad_gadbench.py \
    --datasets reddit,tolokers,weibo,questions,elliptic \
    --trials 10 \
    --epochs 100 \
    --dimension 128 \
    --sample_size 10 \
    --max-nodes 250000 \
    --max-edges 1000000 \
    --out "$out" \
    --status "$status" \
  > "$log" 2>&1 &

echo $! > "$pidfile"
echo "Started Flex-GAD queue PID $(cat "$pidfile")"
echo "CSV: $out"
echo "Status: $status"
echo "Log: $log"

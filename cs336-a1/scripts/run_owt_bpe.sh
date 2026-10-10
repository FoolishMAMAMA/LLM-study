#!/usr/bin/env bash
# 首次运行：bash scripts/run_owt_bpe.sh
# 恢复运行：bash scripts/run_owt_bpe.sh artifacts/bpe/某次结果目录/checkpoint.pkl
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."

if (( $# > 1 )); then
    echo "用法：bash scripts/run_owt_bpe.sh [checkpoint.pkl 路径]" >&2
    exit 2
fi

bpe_run_tag=$(date -u +%Y%m%dT%H%M%S%NZ)
bpe_output_dir="artifacts/bpe/OpenWebText-indexed-${bpe_run_tag}-8p"
bpe_log_path="artifacts/bpe_logs/OpenWebText-indexed-${bpe_run_tag}-8p.log"
mkdir -p artifacts/bpe_logs
bpe_args=(
    --input data/owt_train.txt
    --vocab-size 32000
    --special-tokens '<|endoftext|>'
    --algorithm indexed
    --num-processes 8
    --chunk-size-mib 16
    --checkpoint-interval 300
    --output-dir "$bpe_output_dir"
)
if (( $# == 1 )); then
    bpe_args+=(--resume "$1")
fi

echo "日志：$bpe_log_path"
echo "若中断，恢复指令：bash scripts/run_owt_bpe.sh '$bpe_output_dir/checkpoint.pkl'"
.venv/bin/python -u -m cs336_basics.bpe_experiment "${bpe_args[@]}" 2>&1 | tee "$bpe_log_path"

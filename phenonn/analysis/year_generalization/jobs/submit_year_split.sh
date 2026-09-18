#!/bin/bash
# Submit the temporal-generalization study: 6 jobs = 6 GPUs (one per split).
#   chrono, rand1, rand2, rand3, rand4, coldhot   (fixed aelstm h256/d2, huber).
#   ./submit_year_split.sh
# coldhot needs ./year_temp_rank.json (run rank_years_by_temp.py first); its job
# will error clearly if the file is missing. Skips completed splits (config.json).
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
SPLITS=(chrono rand1 rand2 rand3 rand4 coldhot)

for s in "${SPLITS[@]}"; do
    echo "  GPU: $s"
    sbatch --job-name="yg-$s" --export="ALL,SPLIT=$s" "$HERE/run_year_split.sh"
done
echo "== ${#SPLITS[@]} splits submitted =="

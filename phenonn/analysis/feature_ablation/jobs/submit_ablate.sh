#!/bin/bash
# Submit the GROUP feature-ablation sweep for aelstm: 3 groups/GPU.
#   11 groups (baseline + temperature, radiation, net_radiation, vpd, precip,
#   smi, pet, daylength, co2, + combo tp+ssrd+strd+VPD_max+Tmean) → 4 GPUs
#   (3,3,3,2 — the combo shares the co2 GPU). Loss MSE, patience 7.
#   ./submit_ablate.sh
# Skips already-completed groups (config.json present), so a re-run resumes.
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
MODEL="${1:-aelstm}"          # override: ./submit_ablate.sh attnlstm
N=11                          # baseline + 9 groups + 1 combo (co2 present in build)
PER=3                         # 3 ablation runs per GPU

g=0; lo=0
while [ "$lo" -lt "$N" ]; do
    hi=$(( lo + PER - 1 )); [ "$hi" -ge "$N" ] && hi=$(( N - 1 ))
    echo "  GPU $g: groups $lo-$hi"
    sbatch --job-name="abl-$MODEL-$g" \
           --export="ALL,MODEL=$MODEL,SLICE=$lo-$hi" \
           "$HERE/run_ablate.sh"
    g=$(( g + 1 )); lo=$(( lo + PER ))
done
echo "== $MODEL : $N groups over $g GPUs ($PER/GPU) =="

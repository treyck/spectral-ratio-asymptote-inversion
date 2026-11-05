#!/bin/bash
#SBATCH --job-name=knn_farsep
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=90G
#SBATCH --partition=serc
#SBATCH --array=0-8

set -e
ml purge
ml load python/3.12.1
ml load openblas
source /oak/stanford/groups/beroza/tknudson/venvs/ridgecrest_env/bin/activate
mkdir -p logs

K_VALS=(20 40 80)

# 0-2: P-only, 3-5: S-only, 6-8: P+S
PHASE_IDX=$((SLURM_ARRAY_TASK_ID / 3))
K_IDX=$((SLURM_ARRAY_TASK_ID % 3))
CURRENT_K=${K_VALS[$K_IDX]}

case $PHASE_IDX in
    0) SPEC="P:multitaper" ;;
    1) SPEC="S:multitaper" ;;
    2) SPEC="P:multitaper,S:multitaper" ;;
esac

echo "Task ${SLURM_ARRAY_TASK_ID}: k=${CURRENT_K}, spec=${SPEC}, caps 3000 m"

python run_ridgecrest_knn_v2.py \
    --k_neighbors      $CURRENT_K \
    --max_dist_m       3000.0 \
    --max_depth_diff_m 3000.0 \
    --task_id          $((300 + SLURM_ARRAY_TASK_ID)) \
    --min_mag          2.0 \
    --lfl_lo           0.25 \
    --lfl_hi           1.2 \
    --hfl_lo           25.0 \
    --hfl_hi           40.0 \
    --lfl_max          4.0 \
    --hfl_min          2.5 \
    --noise_thres      3.5 \
    --hfl_relay_min    2.3 \
    --adaptive_lfl \
    --adaptive_safety  0.26 \
    --comp_mode        all \
    --spec             "$SPEC" \
    --n_boot           0 \
    --slope_weight 0.25 \
    --bandwidth_scaled \
    --save_pairs \
    --skip_brune_fit

echo "Done: k=${CURRENT_K}, spec=${SPEC}"

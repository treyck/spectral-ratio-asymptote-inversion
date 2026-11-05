#!/bin/bash
#SBATCH --job-name=stress_knn_phase
#SBATCH --output=logs/knn_%A_%a.out
#SBATCH --error=logs/knn_%A_%a.err
#SBATCH --time=6:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=90G
#SBATCH --partition=serc,owners
#SBATCH --array=0-29

ml purge
ml load python/3.12.1
ml load openblas
source /oak/stanford/groups/beroza/tknudson/venvs/ridgecrest_env/bin/activate

K_VALS=(6 7 8 9 10 12 14 16 18 20)

# 0-9: P-only, 10-19: S-only, 20-29: P+S
PHASE_IDX=$((SLURM_ARRAY_TASK_ID / 10))
K_IDX=$((SLURM_ARRAY_TASK_ID % 10))
CURRENT_K=${K_VALS[$K_IDX]}

case $PHASE_IDX in
    0) SPEC="P:multitaper" ;;
    1) SPEC="S:multitaper" ;;
    2) SPEC="P:multitaper,S:multitaper" ;;
esac

echo "Running task $SLURM_ARRAY_TASK_ID: k=$CURRENT_K, spec=$SPEC"
mkdir -p logs

python run_ridgecrest_knn_v2.py \
    --k_neighbors   $CURRENT_K \
    --max_dist_m    1500.0 \
    --max_depth_diff_m 1500.0 \
    --task_id       $SLURM_ARRAY_TASK_ID \
    --min_mag       2.0 \
    --lfl_lo        0.25 \
    --lfl_hi        1.2 \
    --hfl_lo        25.0 \
    --hfl_hi        40.0 \
    --lfl_max       4.0 \
    --hfl_min       2.5 \
    --noise_thres   3.5 \
    --hfl_relay_min 2.3 \
    --adaptive_lfl \
    --adaptive_safety 0.26 \
    --comp_mode     all \
    --spec          "$SPEC" \
    --n_boot        0 \
    --slope_weight 0.25 \
    --bandwidth_scaled \
    --save_pairs \
    --skip_brune_fit

#!/bin/bash
#SBATCH --job-name=headline_new
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=90G
#SBATCH --partition=serc
set -e
ml purge
ml load python/3.12.1
ml load openblas
source /oak/stanford/groups/beroza/tknudson/venvs/ridgecrest_env/bin/activate
mkdir -p logs

python run_ridgecrest_knn_v2.py \
    --k_neighbors 8 --max_dist_m 1500.0 --max_depth_diff_m 1500.0 \
    --min_mag 2.0 --lfl_lo 0.25 --lfl_hi 1.2 --hfl_lo 25.0 --hfl_hi 40.0 \
    --lfl_max 4.0 --hfl_min 2.5 --noise_thres 3.5 --hfl_relay_min 2.3 \
    --adaptive_lfl --adaptive_safety 0.26 \
    --comp_mode all --spec "P:multitaper,S:multitaper" \
    --slope_weight 0.25 --bandwidth_scaled \
    --n_boot 50 --save_pairs --skip_brune_fit

echo "Done: headline, new survey constants"

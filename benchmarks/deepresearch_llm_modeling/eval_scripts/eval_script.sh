#!/bin/sh
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=100G
#SBATCH --time 2-0:0:00
#SBATCH --cpus-per-task=32
#SBATCH --job-name=gpt5_eval
#SBATCH --gres=gpu:A6000:1
#SBATCH --error=logs/%x-%j.err
#SBATCH --output=logs/%x-%j.out
#SBATCH --mail-type=END
#SBATCH --mail-user=youemail

echo "Running on hosts: $SLURM_NODELIST"
echo "Running on $SLURM_NNODES nodes."
echo "Running $SLURM_NTASKS tasks."
echo "Job id is $SLURM_JOBID"

conda activate agent

API_KEY="<OPENAI API KEY>"

BASE_DIR="../10pct_last/gpt5"     
MODEL_NAME="gpt5"

QUESTIONS="../evaluation/report/consolidated/stratified_sample_10pct.json"
BROWSECOMP="../evaluation/report/consolidated/browsecomp_10pct_answers.json"
RUBRICS="../evaluation/report/consolidated/mind2web2_10pct_rubrics.json"

# =============================================
# Helper function to run evaluation
# =============================================
run_eval () {
    local traj_dir="$1"
    local results_dir="$2"
    local output_dir="$3"

    echo "Evaluating:"
    echo "  Trajectory: $traj_dir"
    echo "  Results:    $results_dir"
    echo "  Output:     $output_dir"
    echo "---------------------------------------"

    python unified_eval.py \
        --questions_file "$QUESTIONS" \
        --browsecomp_ground_truth "$BROWSECOMP" \
        --rubrics_file "$RUBRICS" \
        --trajectory_dir "$traj_dir" \
        --result_dir "$results_dir" \
        --output_dir "$output_dir" \
        --api_key "$API_KEY"
}

# =============================================
# 1. NORMAL EVAL
# =============================================
traj="$BASE_DIR/${MODEL_NAME}_normal"
res="$BASE_DIR/${MODEL_NAME}_normal_results"
out="$BASE_DIR/${MODEL_NAME}_normal_eval"
mkdir -p "$out"
run_eval "$traj" "$res" "$out"

# # # =============================================
# # # 2. ABSOLUTE
# # # =============================================
#  ABS_DIR="$BASE_DIR/absolute"
#  if [ -d "$ABS_DIR" ]; then
#      for traj in "$ABS_DIR"/*/ ; do
#          traj="${traj%/}"
#          name=$(basename "$traj")
        
#          # Skip directories that end with _results, _eval, or _results_eval
#          if [[ "$name" == *_results* ]] || [[ "$name" == *_eval* ]]; then
#              continue
#          fi
        
#          res="${traj}_results"
#          out="${traj}_eval"
#          mkdir -p "$out"

#          run_eval "$traj" "$res" "$out"
#      done
#  fi

# # =============================================
# # 3. RELATIVE
# # =============================================
# REL_DIR="$BASE_DIR/relative"
# if [ -d "$REL_DIR" ]; then
#     for traj in "$REL_DIR"/*/ ; do
#         traj="${traj%/}"
#         name=$(basename "$traj")
        
#         # Skip directories that end with _results, _eval, or _results_eval
#         if [[ "$name" == *_results* ]] || [[ "$name" == *_eval* ]]; then
#             continue
#         fi
        
#         res="${traj}_results"
#         out="${traj}_eval"
#         mkdir -p "$out"

#         run_eval "$traj" "$res" "$out"
#     done
# fi

# =============================================
# 4. PASS @ K
# =============================================
# PASS_DIR="$BASE_DIR/pass_at_k"
# if [ -d "$PASS_DIR" ]; then
#     for traj in "$PASS_DIR"/*/ ; do
#         traj="${traj%/}"
#         name=$(basename "$traj")
        
#         # Skip directories that end with _results, _eval, or _results_eval
#         if [[ "$name" == *_results* ]] || [[ "$name" == *_eval* ]]; then
#             continue
#         fi
        
#         res="${traj}_results"
#         out="${traj}_eval"
#         mkdir -p "$out"

#         run_eval "$traj" "$res" "$out"
#     done
# fi

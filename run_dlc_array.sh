#!/bin/bash
#SBATCH --job-name=dlc_analysis
#SBATCH --output=dlc_%A_%a.out      # %A = array job ID, %a = array task ID
#SBATCH --error=dlc_%A_%a.err
#SBATCH --ntasks=1                   # one process per array task
#SBATCH --cpus-per-task=8            # CPU cores for video I/O / data loading
#SBATCH --mem=32G                    # memory per task
#SBATCH --time=08:00:00             # generous upper bound (covers training)
#SBATCH --partition=mlgpu_short              # GPU partition (A40devel for short tests)
#SBATCH --gres=gpu:1                 # one GPU per task (or gpu:a40:1 to be explicit)
#SBATCH --array=0-0                  # PHASE 1: single task -> trains the model.
                                     # PHASE 2: change to 0-<N-1> to analyse N videos.

# One DeepLabCut video per array task, so every video is processed in parallel.
#
# ---------------------------------------------------------------------------
# TWO-PHASE WORKFLOW (important!)
# ---------------------------------------------------------------------------
# The pipeline trains a model when none exists, otherwise it analyses a video.
# So submit in two phases and DO NOT launch the full array before a model
# exists, or every task would start training at the same time.
#
#   Phase 1 - train the model (a single task is enough):
#       sbatch run_dlc_array.sh                      # uses --array=0-0
#
#   Phase 2 - analyse all videos in parallel, once training has finished.
#       Count the videos, then submit the full range (override on the CLI):
#       N=$(find "<VIDEO_DIR>" -maxdepth 1 -type f -name '*.mp4' | wc -l)
#       sbatch --array=0-$((N-1)) run_dlc_array.sh
#   (or edit the #SBATCH --array line above and just `sbatch run_dlc_array.sh`).
# ---------------------------------------------------------------------------

set -euo pipefail

# ===== User configuration ==================================================
# Absolute path to the DeepLabCut project's config.yaml (inside your workspace).
CONFIG="/lustre/scratch/data/s14berli_hpc-DLCAnalysis/IsbrandtDataAnalysis-Bela-2026-07-10/config.yaml"
# Folder holding the videos to analyse (defaults to the project's videos/ dir).
VIDEO_DIR="$(dirname "$CONFIG")/videos"
# Video file extension to match.
VIDEO_EXT="mp4"
# Absolute path to the pipeline script (this repo).
PIPELINE="/home/s14berli_hpc/dlc_analysis.py"
# Conda environment holding DeepLabCut.
CONDA_ENV="DEEPLABCUT"
# ===========================================================================

# ----- Environment ---------------------------------------------------------
# Temporarily disable "-u" (nounset): sourcing ~/.bashrc / /etc/bashrc and the
# conda activate scripts reference unset variables (e.g. BASHRCSOURCED) and
# would otherwise abort the job under `set -u`.
set +u
module load CUDA           # GPU access on the compute node (needed for CUDA=True)
module load Miniforge3     # free Anaconda clone providing conda
source ~/.bashrc           # load the conda hook that `conda init` wrote
conda activate "$CONDA_ENV"
set -u


# ----- Pick this task's video ----------------------------------------------
mapfile -t VIDEOS < <(find "$VIDEO_DIR" -maxdepth 1 -type f -name "*.${VIDEO_EXT}" | sort)
NVIDEOS=${#VIDEOS[@]}
echo "Found ${NVIDEOS} '*.${VIDEO_EXT}' videos in ${VIDEO_DIR}"

if (( SLURM_ARRAY_TASK_ID >= NVIDEOS )); then
    echo "Array task ${SLURM_ARRAY_TASK_ID} has no video (only ${NVIDEOS} available). Exiting."
    exit 0
fi

VIDEO="${VIDEOS[$SLURM_ARRAY_TASK_ID]}"
echo "Task ${SLURM_ARRAY_TASK_ID}: processing ${VIDEO}"

# ----- Run the pipeline ----------------------------------------------------
# Phase 1 (no model): trains and exits, ignoring the video.
# Phase 2 (model present): analyses this video into <workspace>/Results/<videoname>/.
python "$PIPELINE" "$CONFIG" "$VIDEO"

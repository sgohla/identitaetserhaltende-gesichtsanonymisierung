#!/bin/bash
#SBATCH --job-name=bachelorarbeit         # job name
#SBATCH --account=ki-egomm           # account name
#SBATCH --output=logs/bachelorarbeit_%j.out   # output filei
#SBATCH --error=logs/bachelorarbeit_%j.err    # error file
#SBATCH --partition=a100ai         # partition
#SBATCH --nodes=1                  # number of nodes
#SBATCH --ntasks=1                 # numper of tasks
#SBATCH --cpus-per-task=8          # cpu kernel
#SBATCH --mem=64G                  # RAM
#SBATCH --time=24:00:00            # max. runningtime  (HH:MM:SS)
#SBATCH --gres=gpu:1               # gpus

# --- Load Module  ---
module purge
module load lang/Python/3.12.3-GCCcore-13.3.0

# --- Activate Venv ---
VENV_DIR="$HOME/bachelorarbeit/.venv"

source "$VENV_DIR/bin/activate"
pip install --upgrade -r requirements.txt

# --- Open project & script folder  ---
PROJECT_DIR="$HOME/bachelorarbeit"
cd "$PROJECT_DIR"

# ---Start Python Script  ---
python mot_eval_vlm.py
# --- Deactivate venv ---
deactivate
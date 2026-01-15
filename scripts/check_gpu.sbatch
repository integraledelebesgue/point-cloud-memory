#!/usr/bin/bash -l

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=01:00:00
#SBATCH --account=plgkwantowy5-gpu-gh200
#SBATCH --partition=plgrid-gpu-gh200
#SBATCH --output=results/check_python.out
#SBATCH --error=results/check_python.err

set -ex

module add ML-bundle/25.04
python3.11 -m venv .venv
source .venv/bin/activate
pip install --no-cache-dir torch

python -c "import torch; print(f"CUDA available: {torch.cuda.is_available()}")"

#!/bin/bash
#SBATCH --job-name=cal_mean_variance
#SBATCH --output=output_slurm/eval_log.txt
#SBATCH --error=output_slurm/eval_error.txt
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=180g
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --time=00:30:00
#SBATCH --account=shdpm0
##### END preamble

# module load cuda/12.3.0

source /sw/pkgs/arc/python3.10-anaconda/2023.03/etc/profile.d/conda.sh
conda activate torch_render
python3 cal_mean_variance.py
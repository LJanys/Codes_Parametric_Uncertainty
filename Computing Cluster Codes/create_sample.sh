#!/bin/bash
#SBATCH --account=your_account
#SBATCH --time=22:00:00
#SBATCH --job-name=create_sample_ll
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mail-user=your_email
#SBATCH --mail-type=ALL
#SBATCH --mem=150G
#SBATCH --output=output.%J.txt

module load Anaconda3

source /home/username/.bashrc

conda activate /home/ljanys/.conda/envs/sampling_env

python create_sample.py --start_index 1 --stop_index 2 --proc 8

conda deactivate

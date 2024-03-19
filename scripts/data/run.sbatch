#!/bin/bash
#SBATCH --time=0:59:00
#SBATCH --mem=20GB
#SBATCH --job-name=tune
#SBATCH --output=%A.out
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16

# python reaction.py --output reaction$IDX.csv
python fragment.py --output fragment$IDX.csv
